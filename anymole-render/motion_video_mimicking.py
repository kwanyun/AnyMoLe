import torch
import os
from os.path import join as pjoin
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
from renderer import Renderer
from model import Model
from torchvision import transforms as tfs
import cv2
from anymole_utils2 import normalize_quaternion, transform_image, slerp
from scipy.ndimage import gaussian_filter1d
import random
from pose3d_train import DinoVal,DinoVal3D, AnyMoLeKPT, ResidualBlock 
from ops import quat

transform_dino = tfs.Compose(
    [
        tfs.Resize((560, 896)),  # Ensure this matches your training transform
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

dino_extractor = DinoVal(device='cuda')
dino_extractor3d = DinoVal3D(device='cuda')


def optimize(args,quats2optimize, pos2optmize,interpolated_pos,target_images,model,renderer,kpt,feature,j_level=False):
    
    prev_quats = quats2optimize.clone().detach()
    quats2optimize.requires_grad = True
    
    
    pos2optmize.requires_grad = True
    UPSTEP = 20
    optimizer = torch.optim.Adam([quats2optimize, pos2optmize], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=1e-5, max_lr=args.lr, step_size_up=UPSTEP, mode='triangular2',cycle_momentum=False)
    pbar = tqdm(total=args.iter, leave=False)
    
    target_images = target_images.permute(0,3,1,2)
    mse_loss = torch.nn.MSELoss()
    
    #For reinitializing
    best_loss = float('inf')
    patient = 0


    kpt_criteria = torch.nn.MSELoss()
    with torch.no_grad():
        base_feature = feature(target_images)
        dino_feature = dino_extractor(transform_dino(target_images))
        dino_feature3d = dino_extractor3d(transform_dino(target_images))
        target_kpt = kpt(dino_feature,dino_feature3d)
        target_kpt = torch.cat((target_kpt[...,:2],target_kpt[...,-1].unsqueeze(-1)*320/2), axis=-1)
    criteria=torch.nn.MSELoss()


    
    batch_size, num_joints, = quats2optimize.shape[:2]
    joint_dim = pos2optmize.shape[-1]
    for i in range(args.iter):
        model.deform_by_motion(quats2optimize, pos2optmize)
        meshes = [model.get_mesh(t) for t in range(batch_size)]
        
        pred_images_ori = renderer.render_multiple_frames(meshes)
        pred_images = pred_images_ori[..., :3].permute(0,3,1,2)

        #optimize
        optimizer.zero_grad()
            
        pred_base_feature = feature(pred_images)
        
        joint_pos_batch = model.get_joint_pos_from_quat_pos(quats2optimize,pos2optmize,bs=batch_size)
        pred_kpt = renderer.cam.transform_points_screen(joint_pos_batch)

        
        pred_kpt = torch.cat((pred_kpt[...,:2],pred_kpt[...,-1].unsqueeze(-1)*320/2), axis=-1)
        
        kpt_loss = kpt_criteria(pred_kpt,target_kpt)*args.l_kpt
        base_loss = criteria(pred_base_feature,base_feature)*args.l_base
        reg_loss = mse_loss(interpolated_pos,pos2optmize)*args.l_reg
        reg_loss2 = mse_loss(prev_quats,quats2optimize)*args.l_regq
        loss = kpt_loss + base_loss + reg_loss +  reg_loss2
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_([quats2optimize, pos2optmize], max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        pbar.set_description(f"Iter {i}: kpt loss={kpt_loss.item():.4f} base loss={base_loss.item():.4f} pos_reg={reg_loss.item():.4f} quat_reg={reg_loss2.item():.4f}")
        pbar.update(1)
        
        #adaptive init
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_quat = quats2optimize.clone().detach()
            best_pos = pos2optmize.clone().detach()
            patient = 0
        #Re-initialize
        elif loss.item() > best_loss*2 or patient > UPSTEP*2:
            quats2optimize = best_quat.clone().requires_grad_(True)
            pos2optmize = best_pos.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([quats2optimize, pos2optmize], lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=1e-5, max_lr=args.lr, step_size_up=UPSTEP, mode='triangular2',cycle_momentum=False)
            patient=0
            print(f'adaptive resetting with loss ={best_loss}')
        elif loss > best_loss:
            patient+=1
            
        
        if i % 25 == 0:
            tqdm.write(f"Iter {i}: lr={optimizer.param_groups[0]['lr']:.6f} kpt loss={kpt_loss.item():.4f} base loss={base_loss.item():.4f} reg loss={reg_loss.item():.4f}")
            
    os.makedirs('log/kpt_image', exist_ok=True)
    pbar.close() #close
    return best_quat,best_pos

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char_name", type=str, default="Hippo", help="Name of the character")
    parser.add_argument("--motion_path", type=str, default='Hippo_Attack3_input',  help="Path to motion data")
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_fps", type=int, default=15)
    parser.add_argument("--l_kpt", type=float, default=1)
    parser.add_argument("--l_base", type=float, default=50)
    parser.add_argument("--l_reg", type=float, default=7000)
    parser.add_argument("--l_regq", type=float, default=30000)

    parser.add_argument("--method_param", type=str, default='_2_True_750_500', help="only when not using predefined characters" )
    parser.add_argument("--only_front", action="store_true", help="use only fronts (set to True when specified)")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # model & motion
    try:
        diffuse_name = [x for x in os.listdir(pjoin("data", "characters", args.char_name, "textures")) if "Diffuse" in x][0]
        model = Model(pjoin("data", "characters",args.char_name,f"{args.char_name}.pkl"),
                      texture_paths=[f"./data/characters/{args.char_name}/textures/{diffuse_name}"],
                      device=args.device)
    except:
        model = Model(pjoin("data","characters",args.char_name,f"{args.char_name}.pkl"),
                      device=args.device)
    motion_file = np.load(pjoin("data", 'characters',args.char_name, f'{args.motion_path}.npz'))
    model.set_motion(motion_file)
    # renderer
    cam_params = json.load(open(pjoin("images", args.char_name, args.motion_path, "cam_params.json")))

    default_cam = argparse.Namespace(znear=0.1,zfar=30.0,aspect_ratio=1.0,fov=30.0,dist=2.7,
                                    elev=0.0,azim=0.0,lookat_x=0.0,lookat_y=1.0,lookat_z=0.0)

    renderer = Renderer(args.device, cam_params=default_cam)
    
    source_cam = json.load(open(pjoin("images", args.char_name, args.motion_path, "cam_params.json")))

    at = torch.tensor([[float(source_cam['lookat_x']),float(source_cam['lookat_y']),
                        float(source_cam['lookat_z'])]], dtype=torch.float32)
    renderer.set_camera(float(source_cam['azim']),float(source_cam['elev']),
                        float(source_cam['dist']), at)
    
    # load images (B, H, W, C)
    images = []
    img_path = pjoin('images',args.char_name,args.motion_path + args.method_param)
    img_dir = sorted(glob.glob(f"{img_path}/*.png"))
    assert int(30/args.img_fps) == 30//args.img_fps, "interval should be integer"
    target_intervals = 30//args.img_fps
    target_dir = img_dir[60:]
    for img in target_dir:
        img = Image.open(img)
        img = torch.from_numpy(np.array(img)).float() / 255.0
        images.append(img.to(args.device))
    images_batch = torch.stack(images, dim=0)
    
        
    model.local_quats = model.local_quats
    model.root_pos = model.root_pos
    
    deformation_target_quats = model.local_quats[60::target_intervals]
    deformation_target_pos = model.root_pos[60::target_intervals]
    
    #making the optimization target    
    feature = transform_image
    if args.only_front:
        kpt =  torch.load(f'checkpoint_dir/best_{args.motion_path}_front.pth', map_location=args.device)
    else:
        kpt =  torch.load(f'checkpoint_dir/best_{args.motion_path}.pth', map_location=args.device)

    kpt.eval()
    kpt.to(args.device)

    key_pos = deformation_target_pos[0::args.img_fps][:-1].clone().detach() #first keyframe
    key_pos_next = deformation_target_pos[args.img_fps::args.img_fps].clone().detach() #next keyfrmae
    torch.cuda.empty_cache()
    
    #ex, 1~7 & 14~8 , total 7x2 optimizations
    for numframe in range(1, args.img_fps//2+1):
        #optim init with prev frames
        target_images_forward = images_batch[numframe::args.img_fps]
        target_images_backward = images_batch[args.img_fps-numframe::args.img_fps]
        bs = target_images_forward.shape[0]
        
        quats2optimize_forward = deformation_target_quats[numframe-1::args.img_fps][:bs].clone().detach()
        quats2optimize_backward = deformation_target_quats[args.img_fps-numframe+1::args.img_fps][:bs].clone().detach()
        pos2optmize_forward = deformation_target_pos[numframe-1::args.img_fps][:bs].clone().detach()
        pos2optmize_backward = deformation_target_pos[args.img_fps-numframe+1::args.img_fps][:bs].clone().detach()
        interpolated_pos_forward = (key_pos*(args.img_fps-numframe) + key_pos_next*numframe)/args.img_fps
        interpolated_pos_backward = (key_pos*numframe + key_pos_next*(args.img_fps-numframe))/args.img_fps

        ##### forward optimization #####
        if numframe==args.img_fps//2:
            quats2optimize_forward = slerp(quats2optimize_forward, quats2optimize_backward, 0.3) #if middle of two key frames, init considering both
        opt_quat, opt_pos = optimize(args,quats2optimize_forward, pos2optmize_forward,interpolated_pos_forward,target_images_forward,model,renderer,kpt,feature)
        opt_quat = normalize_quaternion(opt_quat)

        deformation_target_quats[numframe::args.img_fps] = opt_quat.clone().detach()
        deformation_target_pos[numframe::args.img_fps] = opt_pos.clone().detach()
        
        ##### backward optimization #####
        if numframe==args.img_fps//2:
            quats2optimize_backward = slerp(quats2optimize_backward, opt_quat, 0.3) #if middle of two key frames, init considering both
        opt_quat, opt_pos = optimize(args,quats2optimize_backward, pos2optmize_backward,interpolated_pos_backward,target_images_backward,model,renderer,kpt,feature)
        opt_quat = normalize_quaternion(opt_quat)

        deformation_target_quats[args.img_fps-numframe::args.img_fps] = opt_quat.clone().detach()
        deformation_target_pos[args.img_fps-numframe::args.img_fps] = opt_pos.clone().detach()
    
    
    save_dir = pjoin("results", args.char_name,args.motion_path,"raw")
    
    deform_quats_30fps=model.local_quats[:61].clone()
    deform_pos_30fps=model.root_pos[:61].clone()
    generated_quats_15fps = deformation_target_quats.clone() #the 60 is overlapped, interpolate first and then put curr
    generated_pos_15fps = deformation_target_pos.clone()

    for idx_gen in range(1, len(generated_quats_15fps)):
        interp_pos = (generated_pos_15fps[idx_gen-1]+ generated_pos_15fps[idx_gen])/2
        deform_pos_30fps = torch.cat((deform_pos_30fps, interp_pos.unsqueeze(0)), dim=0)
        deform_pos_30fps = torch.cat((deform_pos_30fps, generated_pos_15fps[idx_gen].unsqueeze(0)), dim=0)
        
        interp_rot = slerp(generated_quats_15fps[idx_gen-1], generated_quats_15fps[idx_gen], 0.5)
        deform_quats_30fps = torch.cat((deform_quats_30fps, interp_rot.unsqueeze(0)), dim=0)
        deform_quats_30fps = torch.cat((deform_quats_30fps, generated_quats_15fps[idx_gen].unsqueeze(0)), dim=0)
    
    isbiped = args.char_name in ['Amy','Michelle','Mousey','Mutant','Ybot']
    input_motion_file = np.load(pjoin("data", 'characters',args.char_name, f"{args.motion_path}.npz"))

    in_motion_rot = input_motion_file['local_quats']
    in_motion_pos = input_motion_file['root_pos']

    assert len(deform_quats_30fps) == len(in_motion_rot)
    assert len(deform_quats_30fps) == len(in_motion_pos)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(deform_quats_30fps, pjoin(save_dir, "local_quats.pt"))
    torch.save(deform_pos_30fps, pjoin(save_dir, "root_pos.pt"))
    print('Optimization finished, save images')
    


    deformation_target_quats_cpu = deformation_target_quats.cpu().numpy()
    deformation_target_pos_cpu = deformation_target_pos.cpu().numpy()
    sigma = 1
    # Apply Gaussian 1D smoothing (on CPU using NumPy)
    smoothed_quats_np = gaussian_filter1d(deformation_target_quats_cpu, sigma=sigma, axis=0)
    smoothed_pos_np = gaussian_filter1d(deformation_target_pos_cpu, sigma=sigma, axis=0)
    
    # Normalize the quaternions after smoothing to preserve valid rotations
    smoothed_quats_np = smoothed_quats_np / np.linalg.norm(smoothed_quats_np, axis=-1, keepdims=True)
    
    # Convert the smoothed data back to PyTorch tensors
    smoothed_quats = torch.tensor(smoothed_quats_np, dtype=torch.float32).to(deformation_target_quats.device)
    smoothed_pos = torch.tensor(smoothed_pos_np, dtype=torch.float32).to(deformation_target_pos.device)
    
    gaussian_deform_quats_30fps=model.local_quats[:61].clone()
    gaussian_deform_pos_30fps=model.root_pos[:61].clone()
    generated_quats_15fps = smoothed_quats.clone() #the 60 is overlapped, interpolate first and then put curr
    generated_pos_15fps = smoothed_pos.clone()

    for idx_gen in range(1, len(generated_quats_15fps)):
        interp_pos = (generated_pos_15fps[idx_gen-1]+ generated_pos_15fps[idx_gen])/2
        gaussian_deform_pos_30fps = torch.cat((gaussian_deform_pos_30fps, interp_pos.unsqueeze(0)), dim=0)
        gaussian_deform_pos_30fps = torch.cat((gaussian_deform_pos_30fps, generated_pos_15fps[idx_gen].unsqueeze(0)), dim=0)
    
    
        interp_rot = slerp(generated_quats_15fps[idx_gen-1], generated_quats_15fps[idx_gen], 0.5)
        gaussian_deform_quats_30fps = torch.cat((gaussian_deform_quats_30fps, interp_rot.unsqueeze(0)), dim=0)
        gaussian_deform_quats_30fps = torch.cat((gaussian_deform_quats_30fps, generated_quats_15fps[idx_gen].unsqueeze(0)), dim=0)
    
    
    gaussian_dir = pjoin("results", args.char_name,args.motion_path,"gaussian")
    os.makedirs(gaussian_dir,exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (e.g., 'mp4v' for MP4 format)
    fps = 30  # Define frames per second (30fps)
    frame_width, frame_height = 512, 320  # Set the frame size according to your rendered images (adjust if necessary)
    video_writer = cv2.VideoWriter(f"{gaussian_dir}/gaussian_output_video.mp4", fourcc, fps, (frame_width, frame_height))
    
    for i in tqdm(range(gaussian_deform_quats_30fps.shape[0])):
        model.deform_by_motion(gaussian_deform_quats_30fps, gaussian_deform_pos_30fps)
        
        # Get the mesh and render the image
        mesh = model.get_mesh(i)
        image = renderer.render(mesh)
        image = Image.fromarray(renderer.tensor2image(image))
        
        # Save the rendered image
        image.save(f"{gaussian_dir}/frame_{i:04d}.png")
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        video_writer.write(image_np)
    video_writer.release()
    
    torch.save(gaussian_deform_quats_30fps, pjoin(gaussian_dir, "local_quats.pt"))
    torch.save(gaussian_deform_pos_30fps, pjoin(gaussian_dir, "root_pos.pt"))
    
    

        
