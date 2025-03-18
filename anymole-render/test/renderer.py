import torch
import numpy as np

from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.utils import checkerboard
from pytorch3d.transforms import Rotate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights,
    Textures, TexturesUV, TexturesAtlas
)



class Renderer:
    def __init__(self, device, cam=None, light=None, raster=None):
        self.device = device
        
        # plane setting
        self.plane = None
        
        # cam setting
        self.cam = cam
        if cam is None:
            R, T = look_at_view_transform(3.0, 30, 20, at=[[0.0, 1.0, 0.0]])
            cam = {"R": R,
                   "T": T,}
            self.cam = FoVPerspectiveCameras(device=self.device, **cam)
            
        
        # light setting
        self.light = light
        if light is None:
            light = {"location": [[0.0, 0.0, 3.0]]}
            self.light = PointLights(device=self.device, **light)
        
        # raster setting
        self.raster_setting = raster
        if raster is None:
            raster = {"image_size": [320, 512],
                      "blur_radius": 0.0,
                      "faces_per_pixel": 5,
                      "bin_size": 0,
                      "cull_backfaces": True}
            self.raster_setting = RasterizationSettings(**raster)
        
        # render setting
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cam,
                raster_settings=self.raster_setting
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cam,
                lights=self.light
            )
        )
        
    def set_plane(self, plane_obj):
        self.plane = plane_obj
    
    def render_test(self, mesh):
        ckboard = self._get_checkerboard()
        
        # change mesh textures to TexturesAtlas (because of the texture of checkboard())
        gray_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        atlas      = gray_color.view(1, 1, 1, 3).expand(1, mesh.faces_packed().shape[0], 1, 1, 3).to(self.device)
        mesh.textures = TexturesAtlas(atlas=atlas)
        
        # combine mesh and checkerboard
        meshes = join_meshes_as_scene([mesh, ckboard])
        
        return self.renderer(meshes)

    def render(self, meshes):        
        meshes = join_meshes_as_scene(meshes + [self.plane])
        
        return self.renderer(meshes)
    
    def _get_checkerboard(self):
        ckboard = checkerboard().to(self.device)
        
        # rotate checkerboard along x-axis (90 degrees)
        angle = 90.0  # Rotation angle in degrees
        angle_rad = torch.tensor(angle * (torch.pi / 180))  # Convert angle to radians
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
            [0, torch.sin(angle_rad), torch.cos(angle_rad)]
        ], dtype=torch.float32)
        rotate_transform = Rotate(R=rotation_matrix, device=self.device)
        rotated_verts = rotate_transform.transform_points(ckboard.verts_packed())
        ckboard._verts_list = [rotated_verts]
        
        return ckboard
    
    def render_frame(self, meshes):
        return self.renderer(meshes)
    
    def render_motion():
        frames = []
        
    def set_meshes(self, meshes):
        self.meshes = meshes
    
    def render_images(self, azimuth, elevation, distance, at):
        R, T = look_at_view_transform(distance, elevation, azimuth, at=at)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.renderer.rasterizer.cameras = cameras
        self.renderer.shader.cameras = cameras

        meshes = join_meshes_as_scene(self.meshes + [self.plane])
        images = self.renderer(meshes)
        image = images[0, ..., :3].cpu().numpy()
        return (image * 255).astype(np.uint8)
        