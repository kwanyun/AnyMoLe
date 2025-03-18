import argparse
import json
import torch
from torchvision import transforms as tfs
import sys
import os
#os.environ["TORCH_HOME"] = "./torch_cache"



transform_clip = tfs.Compose(
        [
            tfs.Pad((0, 128)),
            tfs.CenterCrop((448)),
            tfs.Resize((224, 224)),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

transform_dino = tfs.Compose(
        [
            tfs.Resize((280, 448)),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

transform_dino2 = tfs.Compose(
        [
            tfs.Resize((140, 224)),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

transform_dino3 = tfs.Compose(
        [
            tfs.Resize((70, 112)),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

transform_image = tfs.Compose(
        [
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

transform_kpt = tfs.Compose(
        [
            tfs.Resize((800,1280)),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )




class DinoVal:
    def __init__(self, device="cuda:0"):
        super().__init__()
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.model = model.to(device).eval()
    
    def get_dino_features(self, img):
        features = self.model.get_intermediate_layers(img, n=1)[0].half()
        h, w = int(img.shape[2] / 14), int(img.shape[3] / 14)
        dim = features.shape[-1]
        features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
        return features

    def forward(self,im1,dotransform=True):
        if dotransform:
            im1= transform_dino(im1)
        im1val = self.get_dino_features(im1)
        return im1val
    
    def __call__(self, im1):
        return self.forward(im1)
    
    
class DinoVal2:
    def __init__(self, device="cuda:0"):
        super().__init__()
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.model = model.to(device).eval()
    
    def get_dino_features(self, img):
        features = self.model.get_intermediate_layers(img, n=1)[0].half()
        h, w = int(img.shape[2] / 14), int(img.shape[3] / 14)
        dim = features.shape[-1]
        features = features.reshape(-1, h* w, dim).permute(0, 2, 1)
        return features

    def forward(self,im1):
        im1= transform_dino(im1)
        im1val = self.get_dino_features(im1) # BS x FS x X
        im2 = transform_dino2(im1)
        im2val = self.get_dino_features(im2) # BS x FS x Y
        im3 = transform_dino3(im2)
        im3val = self.get_dino_features(im3) # BS x FS x Z
        
        imval = torch.cat([im1val, im2val, im3val], dim=2)   # BS X 768 X (X+Y+Z)
        return imval
    
    def __call__(self, im1):
        return self.forward(im1)


class DinoVal3:
    def __init__(self, device="cuda:0"):
        super().__init__()
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load("ywyue/FiT3D", "dinov2_reg_small_fine")
        self.model = model.to(device).eval()
    
    def get_dino_features(self, img):
        features = self.model.get_intermediate_layers(img, n=1)[0].half()
        h, w = int(img.shape[2] / 14), int(img.shape[3] / 14)
        dim = features.shape[-1]
        features = features.reshape(-1, h* w, dim).permute(0, 2, 1)
        return features

    def forward(self,im1):
        im1= transform_dino(im1)
        im1val = self.get_dino_features(im1) # BS x FS x X
        im2 = transform_dino2(im1)
        im2val = self.get_dino_features(im2) # BS x FS x Y
        im3 = transform_dino3(im2)
        im3val = self.get_dino_features(im3) # BS x FS x Z
        
        imval = torch.cat([im1val, im2val, im3val], dim=2)   # BS X 768 X (X+Y+Z)
        return imval
    
    def __call__(self, im1):
        return self.forward(im1)

    
def load_model_kpt(model_config_path, model_checkpoint_path, cpu_only=False):
    args = Config.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=args.device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model

    
    
def project_3d_to_2d(deformation_target_pos, camera, image_size):
    # Add a batch dimension to the deformation_target_pos if not already present
    if deformation_target_pos.dim() == 2:
        deformation_target_pos = deformation_target_pos.unsqueeze(0)  # (1, 75, 3)
    
    # Transform the 3D points from world space to screen space
    # First, we convert the points to homogeneous coordinates (i.e., add a dimension for w)
    # This projects the points using the camera intrinsic/extrinsic
    projected_points = camera.transform_points_screen(deformation_target_pos, image_size=image_size)

    # Now projected_points is of shape (1, 75, 3), where the last dimension contains (x, y, z_screen)
    # We are interested in (x, y), the 2D coordinates
    projected_2d = projected_points[..., :2]  # Extract x, y

    # Remove the batch dimension if added
    return projected_2d.squeeze(0) 

def slerp(q1, q2, t):
    # Convert t to a tensor if it's a float
    if isinstance(t, float):
        t = torch.tensor(t, dtype=q1.dtype, device=q1.device)

    # Compute the dot product
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # If the dot product is negative, negate one quaternion to take the shorter path
    q2 = torch.where(dot < 0.0, -q2, q2)
    dot = torch.abs(dot)
    
    # Calculate the angle between the quaternions
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    # Handle small angles to avoid division by zero
    epsilon = 1e-6
    is_small_angle = sin_theta_0.abs() < epsilon
    
    # Compute interpolation factors
    sin_theta_t = torch.sin((1.0 - t) * theta_0) / sin_theta_0
    sin_theta   = torch.sin(t * theta_0) / sin_theta_0
    
    # Use linear interpolation for small angles
    sin_theta_t = torch.where(is_small_angle, 1.0 - t, sin_theta_t)
    sin_theta   = torch.where(is_small_angle, t, sin_theta)
    
    # Interpolate
    res = (q1 * sin_theta_t) + (q2 * sin_theta)
    return res


def normalize_quaternion(quat):
    norm = torch.norm(quat, dim=-1, keepdim=True)  # Compute the L2 norm along the last dimension
    return quat / norm  # Normalize quaternions

