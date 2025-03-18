import os 
#os.environ['TRANSFORMERS_CACHE'] = './EILEV/'
#os.environ['HF_HOME'] = './checkpoint_models/'

import cv2
import argparse
from omegaconf import OmegaConf
from EILEV.eilev.model.v2 import VideoBlipForConditionalGeneration
from EILEV.eilev_generate_action_narration import generate_discription
from transformers import Blip2Processor
import glob
from natsort import natsorted
from utils.utils import instantiate_from_config, move_frame_files
import shutil
from pytorch_lightning import seed_everything
from subprocess import call
import torch
from scripts.evaluation.inference import run_inference, load_model_checkpoint
device = 'cuda'


def get_model(args,ckpt_path):
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.to(device)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    return model

## Get folder, make video and make discription with the video
def process_images(video_dir : str) -> None:
    images = glob.glob(f'{video_dir}/frame*.png') + glob.glob(f'{video_dir}/frame*.jpg') #png or jpg
    num_images = len(images)
    assert num_images > 60, "Wrong image dir or too small images. Should be 2sec context"
    context_images = natsorted(images)[1:61]

    height, width, _ = cv2.imread(context_images[0]).shape
    resulting_video = os.path.join("assets/temp_vid.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(resulting_video, fourcc, 30, (width, height))
    
    for img_path in context_images:
        img = cv2.imread(img_path)
        video.write(img)
    video.release()

    model = VideoBlipForConditionalGeneration.from_pretrained('kpyu/eilev-blip2-opt-2.7b').to(device)
    processor = Blip2Processor.from_pretrained('kpyu/eilev-blip2-opt-2.7b')
    prompt = generate_discription(model, processor, resulting_video)
    if prompt == "":
        prompt = generate_discription(model, processor, resulting_video,alt=True)
    
    prompt_file_path = os.path.join(video_dir, 'test_prompts.txt')
    
    # feel free to use different vision-language model
    # We've only tested this, but other model or human prompt will also act simiarly
    with open(prompt_file_path, 'w') as file:
        file.write(prompt)
    return num_images

def images_to_video(dst_path, stages=2,frame_rate=30):
    images = [img for img in os.listdir(dst_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    
    if len(images) == 0:
        print("No images found in the directory.")
        return

    first_image_path = os.path.join(dst_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Initialize the video writer with the output path, codec, frame rate, and frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_base = dst_path.split('/')[-1]
    video_writer = cv2.VideoWriter(f"results/{video_base}.mp4", fourcc, frame_rate, (width, height))
    for frame_idx, image in enumerate(images):
        img_path = os.path.join(dst_path, image)
        frame = cv2.imread(img_path)
        if frame_idx < 61: #note that current AnyMoLe is fixed to 2 sec video.
            video_writer.write(frame)
        else:
            repeat_count = 6 if stages == 1 else 2
            for _ in range(repeat_count):
                video_writer.write(frame)
    # Release the video writer
    video_writer.release()

def run_anymole(args):
    video_base = args.frames_dir.split('/')[-2]
    ckpt_path = os.path.join(f'checkpoint_models/{video_base}/checkpoints/trainstep_checkpoints','epoch=10-step=500.ckpt')
    args.config = 'configs/inference_512_v1.0.yaml'

    #process images to text and number of images
    num_imgs = process_images(args.frames_dir)
    torch.cuda.empty_cache()
    num_additional_keys = num_imgs-61 #note that current AnyMoLe is fixed to 2 sec video
    print(f'Total images : {num_imgs}, Considering {num_additional_keys+2} sec video')
    
    
    if args.ICAdapt:
        #ICAdapt if ckpt_path does not exists
        if not os.path.exists(ckpt_path):
            temp_path =  f"temp2train/"
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            shutil.copytree(args.frames_dir, temp_path)
            left_images = sorted(glob.glob(temp_path+'left*g'))[61:]
            main_images = sorted(glob.glob(temp_path+'frame*g'))[61:]
            right_images = sorted(glob.glob(temp_path+'right*g'))[61:]
            back_images = sorted(glob.glob(temp_path+'back*g'))[61:]
            all_images2delete = left_images+main_images+right_images+back_images
            for img2delete in all_images2delete:
                os.remove(img2delete)
            print('\n')
            print(video_base)
            #for training on linux. Change it into separate steps if you want
            with open('./configs/training_512_v1.0/ICAdapt.sh', 'r') as file:
                script = file.read()
                
                rc = call(f'export name="{video_base}" && {script}', shell=True)
            shutil.rmtree(temp_path)
        model = get_model(args,ckpt_path)
        
    else:
        #default pretrained_model
        model = get_model(args,'checkpoint_models/dynamicrafter_512_interp_v1/model.ckpt')
    
    
    dst_path =  f"{args.frames_dir}_{args.stages}_{args.ICAdapt}_{args.onlykey}_500"
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    
    move_frame_files(args.frames_dir, dst_path)
    args.frames_dir = dst_path
    #sparse generator, 
    for key2gen in range(num_additional_keys):
        run_inference(args,model, key2gen)
    
    #fine generator
    args.frame_stride = 15
    if args.stages==2 :
        for key2gen in range(num_additional_keys):
            run_inference(args,model, key2gen)
    
    images_to_video(dst_path,args.stages)
    #print(discription)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=5, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default='uniform_trailing', help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.7, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", type=bool, default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--repstart", type=int, default=0, help="set if repaint")
    parser.add_argument("--repduration", type=int, default=0, help="set duration")
    parser.add_argument("--saveim", type=str, default='False', help="save_output_image")
    parser.add_argument("--frames_dir", type=str)
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    parser.add_argument("--ICAdapt", action='store_true', default=False, help="Train ICAdapt?")
    parser.add_argument("--stages", type=int, default=2, help="1: only1, 2: two-stage inference all ")
    parser.add_argument("--onlykey", type=int, default=750, help="only keyframe with higher fidelity")

    return parser

if __name__=='__main__':
    #additional arguments
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    
    run_anymole(args)


