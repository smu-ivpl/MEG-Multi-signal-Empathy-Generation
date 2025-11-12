# MEG: Multi-signal Empathy Generation 

This project proposes a multimodal empathy modeling framework that learns from dyadic interactions by encoding a speaker’s **audio** and **facial dynamics (3DMM coefficients)** to generate **an empathizer’s context-aware and dynamic audiovisual responses** beyond static reactions.

https://github.com/user-attachments/assets/13b9ebac-a837-4b49-bc08-31037ba5fbb9

---
## Empathetic response generation 

This code is composed of five groups:

- `Deep3DFaceRecon_pytorch`: use for extract 3dmm coefficients. Mainly from [sicxu/Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch), modified following [RenYurui/PIRender](https://github.com/RenYurui/PIRender)
- `preprocess`: scripts for making dataset compatible with our method
- `vico`: our method proposed in paper *Responsive Listening Head Generation: A Benchmark Dataset and Baseline* [arXiv](https://arxiv.org/abs/2112.13548)
- `PIRender`: render 3dmm coefficients to video. Mainly from [RenYurui/PIRender](https://github.com/RenYurui/PIRender) with minor modifications.
- `evaluation`: quantitative analysis for generations, including SSIM, CPBD, PSNR, FID, CSIM, etc.
  - code for CSIM is mainly from [deepinsight/insightface](https://github.com/deepinsight/insightface)
  - code for lip sync evaluation is mainly from [joonson/syncnet_python](https://github.com/joonson/syncnet_python)
  - in [Challenge 2023](https://vico.solutions/challenge/2023), we use [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) to extract landmarks for LipLMD and 3DMM reconstruction.

For end-to-end inference, this [repo](https://github.com/dc3ea9f/face_utils) may be useful.

## Train Baseline

### Data Preparation

1. create a workspace

   ```bash
   mkdir vico-workspace
   cd vico-workspace
   ```

2. download dataset from [this link](https://1drv.ms/u/s!Ag220j2nXkVswCXNjZcGk2mGtMnl?e=sArC1M) and unzip `listening_head.zip` to folder `data/`

   ```bash
   unzip listening_head.zip -d data/
   ```

3. reorganize `data/` folder to meet the requirements of [PIRender](https://github.com/RenYurui/PIRender)

   ```bash
   mkdir -p data/listening_head/videos/test
   mv data/listening_head/videos/*.mp4 data/listening_head/videos/test
   ```

4. clone baseline code
   ```bash
   git clone https://github.com/dc3ea9f/vico_challenge_baseline.git
   ```

5. extract 3d coefficients for video ([[reference]](https://github.com/RenYurui/PIRender/blob/main/DatasetHelper.md))

   1. change directory to `vico_challenge_baseline/Deep3DFaceRecon_pytorch/`

      ```bash
      cd vico_challenge_baseline/Deep3DFaceRecon_pytorch/
      ```

   2. prepare environment following [this](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/73d491102af6731bded9ae6b3cc7466c3b2e9e48#installation)

   3. prepare `BFM/` and `checkpoints/` following [these instructions](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/73d491102af6731bded9ae6b3cc7466c3b2e9e48#prepare-prerequisite-models)

   3. extract facial landmarks from videos

      ```bash
      python extract_kp_videos.py \
        --input_dir ../../data/listening_head/videos/ \
        --output_dir ../../data/listening_head/keypoints/ \
        --device_ids 0,1,2,3 \
        --workers 12
      ```

   4. extract coefficients for videos

      ```bash
      python face_recon_videos.py \
        --input_dir ../../data/listening_head/videos/ \
        --keypoint_dir ../../data/listening_head/keypoints/ \
        --output_dir ../../data/listening_head/recons/ \
        --inference_batch_size 128 \
        --name=face_recon_feat0.2_augment \
        --epoch=20 \
        --model facerecon
      ```

6. extract audios features

   1. change directory to `vico_challenge_baseline/preprocess`

      ```bash
      cd ../preprocess
      ```

   2. install python package `librosa`, `torchaudio` and `soundfile`

   3. extract audio features

      ```bash
      python extract_audio_features.py \
        --input_audio_folder ../../data/listening_head/audios/ \
        --input_recons_folder ../../data/listening_head/recons/ \
        --output_folder ../../data/listening_head/example/features/audio_feats
      ```

7. reorganize video features

   ```bash
   python rearrange_recon_coeffs.py \
     --input_folder ../../data/listening_head/recons/ \
     --output_folder ../../data/listening_head/example/features/video_feats
   ```

8. organize data

   1. compute mean and std for features

      ```bash
      python statistics_mean_std.py ../../data/listening_head/example/features
      ```

   2. organize for training

      ```bash
      mkdir ../../data/listening_head/example/metadata
      cp ../../data/listening_head/train.csv ../../data/listening_head/example/metadata/data.csv
      cd ../vico
      ln -s ../../data/listening_head/example/ ./data
      ```

### Train and Inference

#### Empathizer Head Generation

1. train baseline

   ```bash
   python -m torch.distributed.launch --nproc_per_node 4 --master_port 22345 train.py \
     --batch_size 4 \
     --time_size 90 \
     --max_epochs 500 \
     --lr 0.002 \
     --task listener \
     --output_path saved/baseline_listener
   ```
   
* For users who wish to run simple inference without training, we provide the pretrained checkpoint. \
You can download it from the [Google Drive](https://drive.google.com/file/d/1HPEr1ryTObqxRnclfelB93nnRjIil6bB/view?usp=sharing).
2. inference

   ```bash
   python eval.py \
     --batch_size 4 \
     --output_path saved/baseline_listener_E500 \
     --resume saved/baseline_listener/checkpoints/Epoch_500.bin \
     --task listener
   ```


### Render to Videos

1. change directory to render

   ```bash
   cd ../PIRender
   ```

2. prepare environment for PIRender following [this](https://github.com/RenYurui/PIRender#1-installation)

3. download the trained weights of PIRender following [this](https://github.com/RenYurui/PIRender#inference)

#### Empathizer  Head 

1. prepare vox lmdb

   ```bash
   python scripts/prepare_vox_lmdb.py \
     --path ../../data/listening_head/videos/ \
     --coeff_3dmm_path ../vico/saved/baseline_listener_E500/recon_coeffs/ \
     --out ../vico/saved/baseline_listener_E500/vox_lmdb/
   ```

2. render to videos

   ```bash
   python -m torch.distributed.launch --nproc_per_node=1 --master_port 42345 inference_avarmerg.py \
     --config ./config/face_demo.yaml \
     --name face \
     --no_resume \
     --input ../vico/saved/baseline_listener_E500/vox_lmdb/ \
     --output_dir ./vox_result/baseline_listener_E500
   ```

## Processing Empathic Multi-Signal


### Generating Empathetic Audio
1. Change directory to [AnyGPT](https://github.com/OpenMOSS/AnyGPT/tree/main)
   ```
   git clone https://github.com/OpenMOSS/AnyGPT
   cd AnyGPT
   ```
2. Set up the environment
    ```
    conda create --name AnyGPT python=3.9
    conda activate AnyGPT
    pip install -r requirements.txt
    ```
3. Download pre-trained models
   - Check the AnyGPT-base weights in [fnlp/AnyGPT-base](https://huggingface.co/fnlp/AnyGPT-base)
   - Check the AnyGPT-chat weights in [fnlp/AnyGPT-chat](https://huggingface.co/fnlp/AnyGPT-chat)
   - Check the SpeechTokenizer and Soundstorm weights in [fnlp/AnyGPT-speech-modules](https://huggingface.co/fnlp/AnyGPT-speech-modules)
   - Check the SEED tokenizer weights in [AILab-CVC/seed-tokenizer-2](https://huggingface.co/AILab-CVC/seed-tokenizer-2)

    The SpeechTokenizer is used for tokenizing and reconstructing speech, Soundstorm is responsible for completing paralinguistic information, and SEED-tokenizer is used for tokenizing images.

    The model weights of unCLIP SD-UNet which are used to reconstruct the image, and Encodec-32k which are used to tokenize and reconstruct music will be downloaded automatically.


4. Generate Audio (Inference)
    ```bash
    python anygpt/src/infer/cli_infer_chat_model.py 
    \ --model-name-or-path 'path/to/model'
    \ --image-tokenizer-path 'path/to/model'
    \ --speech-tokenizer-path 'path/to/model'
    \ --speech-tokenizer-config 'path/to/config'
    \ --soundstorm-path 'path/to/model'
    \ --output-dir "infer_output/chat"
    ```
    for example
    ```bash
    python anygpt/src/infer/cli_infer_chat_model.py 
    \ --model-name-or-path models/anygpt/chat
    \ --image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt 
    \ --speech-tokenizer-path models/speechtokenizer/ckpt.dev 
    \ --speech-tokenizer-config models/speechtokenizer/config.json 
    \ --soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt 
    \ --output-dir "infer_output/chat"
    ```


### Alignment
1. Change directory to [Wav2Lip](https://github.com/Rudrabha/Wav2Lip/tree/master)
   ```
   git clone https://github.com/Rudrabha/Wav2Lip
   cd Wav2Lip
   ```
2. Set up the environment
    ```
    conda create -n wav2lip python=3.6
    conda activate wav2lip
    sudo apt-get install ffmpeg
    pip install -r requirements.txt
    ```
3. Download pre-trained models
   - Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.

    | Model  | Description |  Link to the model | 
    | :-------------: | :---------------: | :---------------: |
    | Wav2Lip  | Highly accurate lip-sync | [Link](https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM?usp=share_link)  |
    | Wav2Lip + GAN  | Slightly inferior lip-sync, but better visual quality | [Link](https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view?usp=share_link) |
   - Place all your checkpoints (.pth files) `./checkpoints`.


4. Lip-syncing videos using the pre-trained models (Inference)
    ```
    python inference.py --checkpoint_path ./checkpoints/<ckpt> --face <video.mp4> --audio <an-audio-source>
    ```


### Super-Resolution
1. Change directory to [ESRGAN](https://github.com/xinntao/ESRGAN)
    ```
   git clone https://github.com/xinntao/ESRGAN
   cd ESRGAN
   ```

2. Set up the environment
    ```
   conda create -n esrgan python=3.6
   pip install numpy opencv-python
    ```
   - Dependencies
     - Python 3
     - [PyTorch >= 1.0](https://pytorch.org/) (CUDA version >= 7.5 if installing with CUDA. [More details](https://pytorch.org/get-started/previous-versions/))


3. Download pre-trained models
 [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) or [Baidu Drive](https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ). Place the models in `./models`. We provide two models with high perceptual quality and high PSNR performance (see [model list](https://github.com/xinntao/ESRGAN/tree/master/models)).


4. Inference
   ```
   python test.py
   ```


5. The results are in `./results` folder.



## Acknowledgments
* This work is heavily based on [vico-challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline),
[Deep3DFaceRecon_Pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch),
[PIRender](https://github.com/RenYurui/PIRender),
[AnyGPT](https://github.com/OpenMOSS/AnyGPT),
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip),
[ESRGAN](https://github.com/xinntao/ESRGAN),
and [AvaMERG](https://arxiv.org/pdf/2502.04976)
. Thanks to all the authors for their great work.


