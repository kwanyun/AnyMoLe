import argparse
import json
import torch

def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        setattr(namespace, key, value)
    return namespace

def save_cam_to_json(cam, azimuth, elevation, distance, lookat_x, lookat_y, lookat_z, json_path, *args):
    cam_params = {
        "znear": cam.znear,
        "zfar": cam.zfar,
        "aspect_ratio": cam.aspect_ratio,
        "fov": cam.fov,
        "dist": distance,
        "elev": elevation,
        "azim": azimuth,
        "lookat_x": lookat_x,
        "lookat_y": lookat_y,
        "lookat_z": lookat_z,
    }
    with open(json_path, "w") as f:
        json.dump(cam_params, f)

def load_cam_from_json(json_path):
    with open(json_path, "r") as f:
        cam_params = json.load(f)
    return dict_to_namespace(cam_params)




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


def normalize_quaternion(quat):
    norm = torch.norm(quat, dim=-1, keepdim=True)  # Compute the L2 norm along the last dimension
    return quat / norm  # Normalize quaternions

