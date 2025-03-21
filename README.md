## ___***[CVPR2025] AnyMoLe: Any Character Motion Inbetweening via Video Diffusion Prior***___ :racehorse: :baby_chick:
<!-- ![](./assets/logo_long.png#gh-light-mode-only){: width="50%"} -->
<!-- ![](./assets/logo_long_dark.png#gh-dark-mode-only=100x20) -->

### [[Project Page](https://kwanyun.github.io/AnyMoLe_page/)] [[Paper](https://arxiv.org/abs/2503.08417)]

### Overview of AnyMoLe
![overview_alpha (1)](https://github.com/user-attachments/assets/ad9606fd-8ed2-46ec-9b12-4ecec7ac4285)







### Be aware current version still need some correction and clean-ups. If there are any suggestion for environment setting, let us know for future users

## :gear: Install Environment via Anaconda (Recommended)
    conda create -n anymole python==3.10.0
    conda activate anymole
    pip install -r requirements.txt
    pip install -e anymole-icadapt/EILEV
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@297020a4b1d7492190cb4a909cafbd2c81a12cb5"

We tested on single Nvidia A6000 GPU (48GB VRAM). 

AnyMoLe requires fine-tuning Video Diffusion model and training scene-specific joint estimator, thus using lower memory GPU needs modifications such as lower batch size.

### pretrained checkpoint from [DynamiCrafter checkpoint](https://huggingface.co/Doubiiu/DynamiCrafter_512_Interp/blob/main/model.ckpt) (Required)
### pretrained checkpoint from [Eilev checkpoint](https://huggingface.co/kpyu/eilev-blip2-opt-2.7b) (Recommended, but can be replaced with your text input)

## Get data from context and key frames
Before start, render frames from 2 sec context motion and target keyframes.

To get data from context and key frames, use `visualizer.ipynb` by rendering.

:exclamation: When rendering with `visualizer.ipynb`, ensure the **full motion is within the camera view** and **sufficiently large** to be clearly noticeable.

If your motion data does not match with our data setting, refer to `data/generate_from_fbx.py` and `data/refine_for_anymole_if_needed.py`


## :fast_forward: How to run code in one step (faster version)
    bash run_example.sh {character} {motion}

ex) bash run_example.sh Amy Amy_careful_Walking_input



https://github.com/user-attachments/assets/9217d4cf-d8b6-413e-a774-56e4cb58ec4b


## :arrow_forward: How to run code step by step (slower version)
1. Finetune video model (ICAdapt) and inference
2. Train Scene-specific joint estimator
3. Do Motion video mimicking 

Here, Motion video mimicking is a new terminology for adopting character motion from video generation model.

## Step 1 : ICAdapt and inference in anymole-ic-adapt
    python run_video_inference.py --frames_dir ../anymole-render/images/{Character}/{Motion} --text_input --ICAdapt --interp --onlykey 750 --stage 2

## Step 2 : Train Scene-specific joint estimator in anymole-render
    python pose3d_train.py --char_name {Character} --motion_name {Motion}
    
### Step 3 : Motion video mimicking in anymole-render
    python motion_video_mimicking.py --char_name {Character} --motion_path {Motion} --kpt_pjt


#### We would like to thank Dynamicrafter for open-source video interpolation model

## Citation
```
@article{yun2025anymole,
  title={AnyMoLe: Any Character Motion In-betweening Leveraging Video Diffusion Models},
  author={Yun, Kwan and Hong, Seokhyeon and Kim, Chaelin and Noh, Junyong},
  journal={arXiv preprint arXiv:2503.08417},
  year={2025}
}
```
