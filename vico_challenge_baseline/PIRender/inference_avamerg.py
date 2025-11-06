import os
import cv2
import lmdb
import math
import argparse
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from util.logging import init_logging, make_logging_dir
from util.distributed import init_dist
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from data.vox_video_dataset import VoxVideoDataset
from data.avamerg_video_dataset import AvamergVideoDataset
from config import Config
from util.distributed import master_only_print as print


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--cross_id', action='store_true')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--output_dir', type=str)


    args = parser.parse_args()
    return args

def write2video(results_dir, *video_list):
    cat_video=None

    for video in video_list:
        video_numpy = video[:,:3,:,:].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array=[]
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i])
    # print("cat_video shape:", cat_video.shape)  # Expected: (frame_num, height, width, 3)
    # print("number of frames to write:", len(image_array))

    out_name = results_dir+'.mp4'
    _, height, width, layers = cat_video.shape
    size = (width,height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 24, size) # fps adjusted

    for i in range(len(image_array)):
        out.write(image_array[i][:,:,::-1])
    out.release()

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)
    opt.data.path = args.input
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not args.single_gpu:
        opt.local_rank = local_rank
        init_dist(opt.local_rank)
        opt.device = torch.cuda.current_device()
    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)

    # create a model
    net_G, net_G_ema, opt_G, sch_G \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, \
                          opt_G, sch_G, None)

    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter)
    net_G = trainer.net_G_ema.eval()

    output_dir = os.path.join(
        args.output_dir,
        'epoch_{:05}_iteration_{:09}'.format(current_epoch, current_iteration)
        )
    os.makedirs(output_dir, exist_ok=True)
    opt.data.cross_id = args.cross_id
    # dataset = VoxVideoDataset(opt.data, is_inference=True)
    dataset = AvamergVideoDataset(opt.data, is_inference=True)
    with torch.no_grad():
        for video_index in range(dataset.__len__()):
            data = dataset.load_next_video()

            ## Version without source image change (modification start)
            input_source = data['source_image'][None].cuda()
            ## (modification end)

            ## Change start
            # 1) Specify external image path
            your_img_path = "./demo_images/KYK.png"
            # 2) Load and preprocess image (RGB, tensor conversion, range [-1,1])
            your_img = Image.open(your_img_path).convert('RGB')
            your_img_tensor = F.to_tensor(your_img).unsqueeze(0).cuda() * 2 - 1
            # 3) Replace the existing source_image
            input_source = your_img_tensor
            ## Change end
            name = data['video_name']

            output_images, gt_images, warp_images = [], [], []

            # print(f"[RANK {local_rank}] Start processing video_index: {video_index}, video_name: {name}")

            for frame_index in range(len(data['target_semantics'])):
                # print(f"[RANK {local_rank}] Processing frame {frame_index}/{len(data['target_semantics'])}")
                target_semantic = data['target_semantics'][frame_index][None].cuda()

                # print(f"[RANK {local_rank}] Before model forward")
                output_dict = net_G(input_source, target_semantic)
                # print(f"[RANK {local_rank}] After model forward")

                output_images.append(output_dict['fake_image'].cpu().clamp_(-1, 1))
                warp_images.append(output_dict['warp_image'].cpu().clamp_(-1, 1))
                gt_images.append(data['target_image'][frame_index][None])

            gen_images = torch.cat(output_images, 0)
            gt_images = torch.cat(gt_images, 0)
            warp_images = torch.cat(warp_images, 0)

            write2video("{}/{}".format(output_dir, name), gt_images, warp_images, gen_images)
            # write2video("{}/{}".format(output_dir, name), gt_images, gen_images)
            # write2video("{}/{}".format(output_dir, name),  gen_images)
            print("write results to video {}/{}".format(output_dir, name))
