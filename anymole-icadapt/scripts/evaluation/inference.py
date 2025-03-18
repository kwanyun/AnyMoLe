import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
import random


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def load_data_prompts(data_dir,key2gen,fs, video_size=(256,256), video_frames=16):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    ## load prompts
    prompt_file = get_filelist(data_dir, ['txt'])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file)-1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist
    
    ## load video
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)

    assert fs==5 or fs==15, "frame stride error"
    if fs ==5:
        if key2gen==0:
            fst_i=0
            frames2use= [6*fn for fn in range(1,11)]
        else:
            fst_i = key2gen * 30 if key2gen < 3 else (key2gen - 2) * fs + 60
            if key2gen==1:
                frames2use= [36,42,48,54,60,61,62,63,64,65]
            else:
                frames2use = [fn+fst_i for fn in range(1,11)]

        image1 = Image.open(file_list[fst_i]).convert('RGB')
        frame_tensor = transform(image1).unsqueeze(1) # [c,1,h,w]
        for f_i, frame_i in enumerate(frames2use):
            image_frame = Image.open(file_list[frame_i]).convert('RGB')
            image_tensor_frame = transform(image_frame).unsqueeze(1) # [c,1,h,w]
            if f_i==9:
                image_tensor_frame = repeat(image_tensor_frame, 'c t h w -> c (repeat t) h w', repeat=3)
            
            frame_tensor = torch.cat([frame_tensor, image_tensor_frame], dim=1)
                        
        image_last = Image.open(file_list[frames2use[-1]+1]).convert('RGB')
        image_tensor_last = transform(image_last).unsqueeze(1) # [c,1,h,w]
        frame_tensor_last = repeat(image_tensor_last, 'c t h w -> c (repeat t) h w', repeat=3)
        frame_tensor = torch.cat([frame_tensor, frame_tensor_last], dim=1)
        

    else:
        fst_i = key2gen* fs + 60 # key2gen*15+60
        
        image1 = Image.open(file_list[fst_i]).convert('RGB')
        frame_tensor = transform(image1).unsqueeze(1) # [c,1,h,w]
        frames2use = [fst_i+(i+2)//3 for i in range(15)]
        
        for f_i, frame_i in enumerate(frames2use):
            image_frame = Image.open(file_list[frame_i]).convert('RGB')
            image_tensor_frame = transform(image_frame).unsqueeze(1) # [c,1,h,w]            
            frame_tensor = torch.cat([frame_tensor, image_tensor_frame], dim=1)
    
    _, filename = os.path.split(file_list[fst_i])
    data_list.append(frame_tensor)
    filename_list.append(filename)
    first_in_30 = int((key2gen+3- 15/fs)*30)
    return filename_list, data_list, prompt_list,first_in_30



def save_results_seperate(prompt, samples, filename, a_cond,a_repstart,a_repduration,first_idx,frames_dir, fps=10, stages=2,guidescale=1.0):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()

        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            basename = "" #os.path.basename(ckpt_path).split('.')[0]
            
            if fps==5:
                for geniter in range(4):
                    torchvision.io.write_png(grid[11+geniter].permute(2,0,1), f'{frames_dir}/frame_{first_idx+60+(geniter+1)*6:04}.png')
            else:
                gen_indices = [k for k in range(1,15)]
                    
                for gen_idx in gen_indices:
                    filename = f'{frames_dir}/frame_{first_idx+ (gen_idx)*2 :04}.png'
                    torchvision.io.write_png(grid[gen_idx].permute(2, 0, 1), filename)

            
def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False,  timestep_spacing='uniform', guidance_rescale=0.0,a_cond=None,a_fs=5,key2gen=0,stage=2,onlykey=300,a_repstart=0,a_repduration=0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size

    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        img_cat_cond = torch.zeros_like(z)
        img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
        img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = z
    if a_cond== "None":
        cond_mask = None
        keymask=None
    else:
        #stage 1
        if a_fs==5:
            cond_mask  = torch.zeros_like(z0)
            keymask = torch.zeros_like(z0)
            cond_mask[:,:,1:11,:,:] = torch.ones(1,4,10,40,64)
            
            if key2gen==0:
                keymask[:,:,1:11,:,:] = torch.ones(1,4,10,40,64)
            elif key2gen==1:
                keymask[:,:,1:6,:,:] = torch.ones(1,4,5,40,64)
                keymask[:,:,10,:,:] = torch.ones(1,4,40,64)
            else:
                keymask[:,:,5,:,:] = torch.ones(1,4,40,64)
                keymask[:,:,10,:,:] = torch.ones(1,4,40,64)
                

        else:
            cond_mask  = torch.zeros_like(z0)
            if stage==2:
                cond_mask[:,:,3,:,:] = torch.ones(1,4,40,64)
                cond_mask[:,:,6,:,:] = torch.ones(1,4,40,64)
                cond_mask[:,:,9,:,:] = torch.ones(1,4,40,64)
                cond_mask[:,:,12,:,:] = torch.ones(1,4,40,64)
            
            
            keymask = torch.zeros_like(z0)
    kwargs.update({"a_cond": a_cond})
    kwargs.update({"a_repstart": a_repstart})
    kwargs.update({"a_repduration": a_repduration})
    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            keymask=keymask,
                                            onlykey=onlykey,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def run_inference(args, model,key2gen, gpu_num=1, gpu_no=0):
    
    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]


    ## prompt file setting
    assert os.path.exists(args.frames_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list,first_idx = load_data_prompts(args.frames_dir,key2gen,fs=args.frame_stride, video_size=(args.height, args.width), video_frames=n_frames)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]
    
    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args.bs]
            videos = data_list_rank[indice:indice+args.bs]
            filenames = filename_list_rank[indice:indice+args.bs]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")
            
            batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.timestep_spacing, args.guidance_rescale,args.cond,args.frame_stride,key2gen,args.stages,args.onlykey,args.repstart,args.repduration)

            ## save each example individually
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                filename = filenames[nn]
                save_results_seperate(prompt, samples, filename, args.cond,args.repstart,args.repduration,first_idx=first_idx,frames_dir=args.frames_dir, fps=args.frame_stride,stages=args.stages,guidescale=args.unconditional_guidance_scale)

    print(f"Saved video. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--cond", type=str, default=None, help="inversion type : ddpm, ddim, x0, None")
    parser.add_argument("--repstart", type=int, default=0, help="repaint start")
    parser.add_argument("--repduration", type=int, default=0, help="repaint duration")
    parser.add_argument("--saveim", type=str, default='False', help="save_output_image")
    parser.add_argument("--start_frame", type=int, default=0, help="starting frame")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@DynamiCrafter cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2 ** 31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)