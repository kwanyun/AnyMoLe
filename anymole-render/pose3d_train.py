import torch
import argparse
import os
import glob
import random

from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['TORCH_HOME'] = 'torch_cache'

class DinoVal:
    def __init__(self, device="cuda:0"):
        super().__init__()
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.model = model.to(device).eval()
    
    def get_dino_features(self, img):
        features = self.model.get_intermediate_layers(img, n=1)[0]
        h, w = int(img.shape[2] / 14), int(img.shape[3] / 14)
        dim = features.shape[-1]
        features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
        return features

    def forward(self,im1):
        if len(im1.shape)==3:
            im1= im1.unsqueeze(0)
        im1val = self.get_dino_features(im1)
        return im1val
    
    def __call__(self, im1):
        return self.forward(im1)

class DinoVal3D:
    def __init__(self, device="cuda:0"):
        super().__init__()
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load("ywyue/FiT3D", 'dinov2_reg_small_fine')
        self.model = model.to(device).eval()
    
    def get_dino_features(self, img):
        features = self.model.get_intermediate_layers(img, n=1)[0]
        h, w = int(img.shape[2] / 14), int(img.shape[3] / 14)
        dim = features.shape[-1]
        features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
        return features

    def forward(self,im1):
        if len(im1.shape)==3:
            im1= im1.unsqueeze(0)
        im1val = self.get_dino_features(im1)
        return im1val
    
    def __call__(self, im1):
        return self.forward(im1)



class ResidualBlock(nn.Module):
    """Resblock with zero init"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        nn.init.zeros_(self.conv_block[-2].weight)  # Zero 
        nn.init.zeros_(self.conv_block[-2].bias)    # Zero 

        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv_block(x) + x)

    
class AnyMoLeKPT(nn.Module):
    """estimates 2D keypoints and uplifts to 3D using the previous keypoints and DINO feature."""
    def __init__(self, num_kpts=17):
        super(AnyMoLeKPT, self).__init__()

        self.feat_2d_net = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        self.feat_3d_net = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.SiLU(),
        )
        
        self.fusion_net = nn.Sequential(
            nn.Conv2d(896, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )
        
        # 2D Keypoint estimation branch (output 2D heatmaps w upsample)
        self.kpt_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            ResidualBlock(256),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            ResidualBlock(256),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            ResidualBlock(128),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, num_kpts, kernel_size=1)
        )


        # Global average pooling before depth estimation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Depth estimation branch (output depth values)
        feat_size = 512  # Output channels from fusion_net
        in_features = feat_size + 3  # Including flattened keypoints
        self.depth_mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        self.num_kpts = num_kpts



    def forward(self, feat_2d, feat_3d):

        feat_2d_proc = self.feat_2d_net(feat_2d)          # Shape: (B, 512, 40, 64)
        feat_3d_proc = self.feat_3d_net(feat_3d)
        
        # Fuse features via concatenation
        fused_feats = torch.cat([feat_2d_proc,feat_3d_proc], dim=1)  # Shape: (B, 1024, 40, 64)

        x = self.fusion_net(fused_feats)  # Shape: (B, 512, 40, 64)
        kpt_heatmaps = self.kpt_head(x)  # Shape: (B, num_kpts, 320, 512)

        # Extract estimated 2D keypoints using soft-argmax
        batch_size, num_kpts, height, width = kpt_heatmaps.shape
        kpt_curr_2d = torch.zeros((batch_size, num_kpts, 2), device=kpt_heatmaps.device)
        # Reshape heatmaps for softmax
        heatmaps_reshaped = kpt_heatmaps.view(batch_size, num_kpts, -1)  # Shape: (B, num_kpts, H*W)

        # Apply softmax to get probability distribution over locations
        heatmaps_probs = F.softmax(heatmaps_reshaped, dim=2)  # Shape: (B, num_kpts, H*W)

        # Compute expected coordinates
        coords = torch.arange(height * width, device=kpt_heatmaps.device).float()
        coords_x = (coords % width) # unnormalize x due to non-equal downsamples and upsamples
        coords_y = (coords // width)  # unnormalize y

        coords_x = coords_x.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H*W)
        coords_y = coords_y.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H*W)

        # Calculate expected x and y coordinates
        kpt_curr_2d_x = torch.sum(heatmaps_probs * coords_x, dim=2)
        kpt_curr_2d_y = torch.sum(heatmaps_probs * coords_y, dim=2)
        
        kpt_curr_2d[:, :, 0] = kpt_curr_2d_x
        kpt_curr_2d[:, :, 1] = kpt_curr_2d_y
        
        kpt_curr_2d_x_norm  = 2*(kpt_curr_2d_x /511)-1
        kpt_curr_2d_y_norm  = 2*(kpt_curr_2d_y /319)-1

        grid = torch.stack((kpt_curr_2d_x_norm, kpt_curr_2d_y_norm), dim=2)  # Shape: (B, num_kpts, 2)
        grid = grid.unsqueeze(2)  # Shape: (B, num_kpts, 1, 2)
        grid = grid.permute(0, 2, 1, 3)  # Shape: (B, 1, num_kpts, 2)

        sampled_features = F.grid_sample(x, grid, align_corners=True,padding_mode="border")
        sampled_features = sampled_features.squeeze(2).permute(0, 2, 1)
        
        #addind kpt indices for positional-like joint index enbedding
        keypoint_indices = torch.arange(num_kpts, device=sampled_features.device, dtype=sampled_features.dtype)
        keypoint_indices = keypoint_indices.unsqueeze(0).expand(batch_size, -1).unsqueeze(2)   # Shape: (B, num_kpts, 1)

        per_kpt_features = torch.cat([sampled_features, kpt_curr_2d,keypoint_indices], dim=2)  # Shape: (B, num_kpts, C + 3)
        per_kpt_features_flat = per_kpt_features.view(batch_size * num_kpts, -1)
        per_kpt_depth_flat = self.depth_mlp(per_kpt_features_flat)
        per_kpt_depth = per_kpt_depth_flat.view(batch_size, num_kpts, 1)

        kpt_curr_3d = torch.cat([kpt_curr_2d, per_kpt_depth], dim=2)

        # - Estimated 3D keypoints (B, num_kpts, 3)
        return kpt_curr_3d
    
    
transform_dino = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((560, 896)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

class posedataset(Dataset):
    """KPT and DINO feats."""
    def __init__(self, path,device):
        self.path = path
        self.img_data = sorted(glob.glob(path + "/*.png"))  # png image data
        self.pose_data = sorted(glob.glob(path + "/*.npy"))  # width, height, depth
        added_images=[]
        added_pose=[]
        assert len(self.img_data)==len(self.pose_data), 'Length of data should be same'

        for data_idx, img_file in enumerate(self.img_data):
            img_basename = os.path.basename(img_file).split('.')[0]
            if int(img_basename.split('_')[-1])>60:
                added_images.append(self.img_data[data_idx])
                added_pose.append(self.pose_data[data_idx])
        
        self.img_data= self.img_data+added_images+added_images # weighted keypoints for keyframes
        self.pose_data = self.pose_data + added_pose+added_pose # weighted keypoints for keyframes
        
        self.transform = transform_dino  # Pass transform_dino when initializing the dataset
        self.dino_extractor= DinoVal()
        self.dino3d_extractor= DinoVal3D()
        self.device=device
        
    def get_augmented(self,image_dir,pose_dir):
        
        image = Image.open(image_dir).convert('RGB')
        pose = np.load(pose_dir)


        # Random translation up to 30 pixels
        max_translation = 30
        dx = random.randint(-max_translation, max_translation)
        dy = random.randint(-max_translation, max_translation)

        image = TF.affine(image, angle=0, translate=(dx, dy), scale=1, shear=0)
        
        # Brcause depth is normalized
        pose[:, 0] += dx
        pose[:, 1] += dy
        
        color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.2, saturation=0.2
        )
        image = color_jitter(image)

        if self.transform:
            image = self.transform(image).to(self.device)
        with torch.no_grad():
            feature2d = self.dino_extractor(image).squeeze()
            feature3d = self.dino3d_extractor(image).squeeze()
        return feature2d,feature3d, torch.from_numpy(pose)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        feature2d,feature3d, pose= self.get_augmented(self.img_data[idx],self.pose_data[idx])
            
        return feature2d,feature3d, pose
    
def train(model, dataloader, criterion, optimizer, device, num_epochs,motion_path,lr):
    model.train()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.2,steps_per_epoch=len(dataloader), epochs=num_epochs,final_div_factor=1e3)
    
    best_loss = float('inf')
    os.makedirs('checkpoint_dir', exist_ok=True)
    os.makedirs(f'results/{motion_path.split("_")[0]}', exist_ok=True)
    dino4test = DinoVal()
    dino3d4test = DinoVal3D()
    for epoch in tqdm(range(num_epochs)):
        running_loss=0
        for i, batch in enumerate(dataloader):
            feat2d, feat3d, pose = batch
            feat2d = feat2d.to(device).float()
            feat3d = feat3d.to(device).float()
            pose = pose.to(device).float()
            
            optimizer.zero_grad()
            
            kpt3d = model(feat2d, feat3d)
            kpt3d_unnormalized = torch.cat((kpt3d[...,:2],kpt3d[...,-1].unsqueeze(-1)*320/2), axis=-1) # unnormalize with height

            pose_unnormalized = torch.cat((pose[...,:2],pose[...,-1].unsqueeze(-1)*320/2), axis=-1) # unnormalize with height
            loss = criterion(kpt3d_unnormalized, pose_unnormalized)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}],  Loss: {epoch_loss:.4f}")
        
        if (epoch%5==0) and (epoch>50):
            if (epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_path = os.path.join('checkpoint_dir', f'best_{motion_path}.pth')
                torch.save(model, best_model_path)
                print(f"Best model saved with loss {best_loss:.4f} at {best_model_path}")

                # visualize the scene-specific joint estimator for logging.
                with torch.no_grad():
                    test_image = os.path.join("images", motion_path.split('_')[0], args.motion_name,'frame_0090.png')
                    test_npy = os.path.join("images", motion_path.split('_')[0], args.motion_name,'frame_0090.npy')
                    test_pil_image = Image.open(test_image)
                    test_joint2d =np.load(test_npy)[:,:2]
                    test_image = transform_dino(test_pil_image).to('cuda')
                    
                    feature2dtest = dino4test(test_image)
                    feature3dtess = dino3d4test(test_image)
    
                    kpt2d = model(feature2dtest, feature3dtess)[0,:,:2]
                    image = test_pil_image.copy()
    
                    draw = ImageDraw.Draw(image)
                    radius = 5
                    for kpt_idx, keypoint in enumerate(kpt2d):
                        x, y = keypoint  # Extract x, y coordinates
                        x=int(x)
                        y=int(y)
                        # Draw a small circle (ellipse) at each keypoint location for visualize
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255-255//kpt2d.shape[0]*kpt_idx,0,255//kpt2d.shape[0]*kpt_idx )) 
                    
                    image.save(f'results/{motion_path.split("_")[0]}/{motion_path}_{epoch:03}_test.png')
                    image = test_pil_image.copy()
                    draw = ImageDraw.Draw(image)
                    radius = 5
                    for kpt_idx, keypoint in enumerate(test_joint2d):
                        x, y = keypoint 
                        x=int(x)
                        y=int(y)
    
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255-255//kpt2d.shape[0]*kpt_idx,0,255//kpt2d.shape[0]*kpt_idx))  # Red dots
                    
                    image.save(f'results/{motion_path.split("_")[0]}/{motion_path}_{epoch:03}_gt.png')
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--char_name", type=str, required=True, help="Name of the character")
    parser.add_argument("--motion_name", type=str, required=True, help="Name of the motion data")
    parser.add_argument("--epoch", type=int, default=360, help="Adjust for each character for optimal performace. For CVPR2025 AnyMoLe paper, we used all 400 for generalization")
    parser.add_argument("--lr", type=float, default=3e-4, help="Max Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    args = parser.parse_args()
    
    data_dir = os.path.join("images", args.char_name, args.motion_name)
    sample_data = np.load(os.path.join(data_dir,"frame_0000.npy"))
    n_KPT = sample_data.shape[0]

    dataset = posedataset(data_dir,args.device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = AnyMoLeKPT(num_kpts=n_KPT) 
    model.to(args.device)
    
    criterion = torch.nn.MSELoss() # we use simple mse loss for scene-specific joint estimator
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train(model, dataloader, criterion, optimizer, args.device, num_epochs=args.epoch,motion_path=args.motion_name,lr=args.lr)
