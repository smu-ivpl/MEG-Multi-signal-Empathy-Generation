import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from util.distributed import master_only_print as print


def format_for_lmdb(*args):
    parts = []
    for a in args:
        if isinstance(a, int):
            a = str(a).zfill(7)
        parts.append(a)
    return ('-'.join(parts)).encode('utf-8')

class AvamergVideoDataset(Dataset):
    """
    Dataset that loads the entire video from AVAMERG LMDB.
    - video_name includes person_id (e.g., dia00001utt2_16).
    - LMDB keys:
        * {video_name}-length          : number of frames (string, zfill(7))
        * {video_name}-{0000000..}     : frame image bytes (e.g., JPEG)
        * {video_name}-coeff_3dmm      : float32 raw bytes, reshape(num_frames, D)
        * {video_name}-transform_params: float32 raw bytes, reshape(num_frames, 5)
    """
    def __init__(self, opt, is_inference):
        self.path = opt.path
        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius

        self.env = lmdb.open(
            os.path.join(self.path, str(self.resolution)),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', self.path)

        list_file = "test_list.txt" if is_inference else "train_list.txt"
        list_file = os.path.join(self.path, list_file)
        if not os.path.isfile(list_file):
            fallback = os.path.join(self.path, "test_list.txt")
            if os.path.isfile(fallback):
                list_file = fallback
            else:
                raise FileNotFoundError(f"Missing list file: {list_file}")
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.strip() for line in lines if line.strip()]

        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.cross_id = False
        self.norm_crop_param = False
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5),
                                 inplace=True),
        ])
        self.video_index = -1

    def get_video_index(self, videos):
        video_items = []
        for video_name in videos:
            vi = self.Video_Item(video_name)
            video_items.append(vi)
        person_ids = sorted(list({vi['person_id'] for vi in video_items}))
        return video_items, person_ids

    def group_by_key(self, video_list, key):
        d = collections.defaultdict(list)
        for index, vi in enumerate(video_list):
            d[vi[key]].append(index)
        return d

    def Video_Item(self, video_name):
        person_id = video_name.split('_')[-1]
        with self.env.begin(write=False) as txn:
            k = format_for_lmdb(video_name, 'length')
            v = txn.get(k)
            # print(f"[DEBUG] Checking length key: {k} -> {'FOUND' if v else 'MISSING'}")
            if v is None:
                raise KeyError(f"Missing key: {video_name}-length")
            length = int(v.decode('utf-8'))
        return {
            'video_name': video_name,
            'person_id': person_id,
            'num_frame': length
        }

    def __len__(self):
        return len(self.video_items)
    def find_crop_norm_ratio(self, source_coeff, target_coeffs):
        alpha = 0.3
        exp_diff = np.mean(np.abs(target_coeffs[:,80:144] - source_coeff[:,80:144]), 1)
        angle_diff = np.mean(np.abs(target_coeffs[:,224:227] - source_coeff[:,224:227]), 1)
        index = np.argmin(alpha*exp_diff + (1-alpha)*angle_diff)
        crop_norm_ratio = source_coeff[:,-3] / target_coeffs[index:index+1, -3]
        return crop_norm_ratio

    def load_next_video(self, crop_norm_ratio=None):
        self.video_index += 1
        if self.video_index >= len(self.video_items):
            raise IndexError("No more videos")
        video_item = self.video_items[self.video_index]
        video_name = video_item['video_name']
        if not video_name.endswith('.listener'):
            video_name_key = video_name + '.listener'
        else:
            video_name_key = video_name

        data = {}
        with self.env.begin(write=False) as txn:
            key0 = format_for_lmdb(video_name_key, 0)
            img_bytes_0 = txn.get(key0)
            if img_bytes_0 is None:
                print(f"[ERROR] Frame 0 not found: {key0}")
                raise KeyError(f"Missing frame 0 for {video_name_key}")
            img0 = Image.open(BytesIO(img_bytes_0)).convert('RGB')
            data['source_image'] = self.transform(img0)

            coeff_key = format_for_lmdb(video_name_key, 'coeff_3dmm')
            transf_key = format_for_lmdb(video_name_key, 'transform_params')

            coeff_bytes = txn.get(coeff_key)
            transf_bytes = txn.get(transf_key)

            if coeff_bytes is None:
                print(f"[ERROR] coeff_3dmm not found for: {coeff_key}")
            if transf_bytes is None:
                print(f"[ERROR] transform_params not found for: {transf_key}")

            if coeff_bytes is None or transf_bytes is None:
                raise KeyError(f"Missing coeff or transform_params for {video_name_key}")

            num_frames = video_item['num_frame']
            coeff = np.frombuffer(coeff_bytes, dtype=np.float32).reshape(num_frames, 257)
            transf = np.frombuffer(transf_bytes, dtype=np.float32).reshape(num_frames, 5)

            ex = coeff[:, 80:144]  # 64 dimensions
            ang = coeff[:, 224:227]  # 3 dimensions
            tra = coeff[:, 254:257]  # 3 dimensions
            crp = transf[:, -3:]  # 3 dimensions (scale, center_x, center_y)

            # Apply crop_norm_ratio if needed (uncomment to use)
            if crop_norm_ratio is not None:
                crp[:, -3] = crp[:, -3] * crop_norm_ratio

            semantics_np = np.concatenate([ex, ang, tra, crp], axis=1)  # Merged into 73 dimensions

            data['target_image'], data['target_semantics'] = [], []
            for frame_index in range(num_frames):
                kf = format_for_lmdb(video_name_key, frame_index)
                img_bytes = txn.get(kf)
                if img_bytes is None:
                    print(f"[ERROR] Missing frame {frame_index} for {video_name_key}")
                    raise KeyError(f"Missing frame {frame_index} for {video_name_key}")
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                data['target_image'].append(self.transform(img))
                sem_feat = self.transform_semantic(semantics_np, frame_index)
                data['target_semantics'].append(sem_feat)

            data['video_name'] = video_name
        return data

    def transform_semantic(self, semantic, frame_index):
        index_seq = self.obtain_seq_index(frame_index, semantic.shape[0])
        window = semantic[index_seq, ...]
        sem = torch.tensor(window, dtype=torch.float32).permute(1, 0)
        return sem

    def obtain_seq_index(self, index, num_frames):
        r = self.semantic_radius
        seq = list(range(index - r, index + r + 1))
        seq = [min(max(i, 0), num_frames - 1) for i in seq]
        if len(seq) == 0:
            seq = [min(max(index, 0), num_frames - 1)]
        return seq
