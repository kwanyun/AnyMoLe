import numpy as np
import torch

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch, Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights,BlendParams
)


def _camera_from_params(params, device="cuda:0"):
    at = torch.tensor([[params.lookat_x, params.lookat_y, params.lookat_z]], device=device)
    R, T = look_at_view_transform(params.dist, params.elev, params.azim, at=at)
    return FoVPerspectiveCameras(
        znear=params.znear,
        zfar=params.zfar,
        aspect_ratio=params.aspect_ratio,
        fov=params.fov,
        R=R,
        T=T,
        device=device
    )

def _light_from_params(params, device="cuda:0"):
    location = torch.tensor([[params.light_x, params.light_y, params.light_z]], device=device)
    return PointLights(device=device, location=location)

def _raster_from_params(params):
    return RasterizationSettings(
        image_size=[params.image_height, params.image_width],
        blur_radius=params.blur_radius,
        faces_per_pixel=params.faces_per_pixel,
        bin_size=params.bin_size,
        cull_backfaces=params.cull_backfaces
    )

class AlphaBlendingShader(SoftPhongShader):
    def __init__(self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None, transparency=0.5):
        super().__init__(device=device, cameras=cameras, lights=lights, materials=materials, blend_params=blend_params)
        self.transparency = transparency

    def blend_colors(self, fragments, shaded_colors, **kwargs):
        """
        Custom blending function that applies transparency.
        """
        background = torch.tensor(self.blend_params.background_color, device=shaded_colors.device)
        N, H, W, K, C = shaded_colors.shape

        # For simplicity, take the closest face (K=0)
        mask = fragments.pix_to_face[..., 0] >= 0

        # Initialize output image with background
        images = background.expand(N, H, W, C).clone()

        # Assign colors to pixels where there is a face
        images[mask] = shaded_colors[..., 0, :][mask] * self.transparency + images[mask] * (1 - self.transparency)

        return images
    
class Renderer:
    def __init__(self, device, cam_params=None, light_params=None, raster_params=None):
        self.device = device
        
        # initialize plane
        self.plane = load_objs_as_meshes(["./data/checkerboard/checkerboard.obj"], device=device)
        
        # cam setting
        if cam_params is None:
            R, T = look_at_view_transform(3.0, 30, 20, at=[[0.0, 1.0, 0.0]])
            cam = {"R": R,
                   "T": T,}
            self.cam = FoVPerspectiveCameras(device=self.device, **cam)
        else:
            self.cam = _camera_from_params(cam_params, device=self.device)
        
        # light setting
        if light_params is None:
            light = {"location": [[0.0, 0.0, 3.0]]}
            self.light = PointLights(device=self.device, **light)
        else:
            self.light = _light_from_params(light_params, device=self.device)
        
        # raster setting
        if raster_params is None:
            raster = {"image_size": [320, 512],
                      "blur_radius": 0.0,
                      "faces_per_pixel": 1,
                      "bin_size": 0,
                      "cull_backfaces": False}
            self.raster_setting = RasterizationSettings(**raster)
        else:
            self.raster_setting = _raster_from_params(raster_params)
        self.cam.image_size = self.raster_setting.image_size
        
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
    
    def set_camera(self, azimuth, elevation, distance, at):
        R, T = look_at_view_transform(distance, elevation, azimuth, at=at)
        self.cam = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.cam.image_size = self.raster_setting.image_size
        self.renderer.rasterizer.cameras = self.cam
        self.renderer.shader.cameras = self.cam

    def render(self, meshes):
        if isinstance(meshes, Meshes):
            meshes = [meshes]
        meshes = join_meshes_as_scene(meshes + [self.plane])
        images = self.renderer(meshes)
        return images
    
    def render_multiple_meshes_with_transparency(self, meshes, trans_meshes, transparency=0.9):
        if isinstance(meshes, Meshes):
            meshes = [meshes]
        opaque_scene = join_meshes_as_scene(meshes + [self.plane])
        opaque_image = self.renderer(opaque_scene)


        if isinstance(trans_meshes, Meshes):
            trans_meshes = [trans_meshes]
        transparent_scene = join_meshes_as_scene(trans_meshes)
        transparent_image = self.renderer(transparent_scene)

        # Composite the images
        final_image = opaque_image*0.8 + transparent_image*0.2

        return final_image
    
    def render_with_transparency(self, meshes, transparency=0.5):
        if isinstance(meshes, Meshes):
            meshes = [meshes]
        opaque_scene = join_meshes_as_scene([self.plane])
        opaque_image = self.renderer(opaque_scene)

        # Create custom shader with transparency
        blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
        transparent_shader = AlphaBlendingShader(
            device=self.device,
            cameras=self.cam,
            lights=self.light,
            blend_params=blend_params,
            transparency=transparency
        )
        transparent_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cam,
                raster_settings=self.raster_setting
            ),
            shader=transparent_shader
        )

        if isinstance(meshes, Meshes):
            meshes = [meshes]
        transparent_scene = join_meshes_as_scene(meshes)
        transparent_image = transparent_renderer(transparent_scene)

        # Composite the images
        final_image = opaque_image + transparent_image

        return final_image

    def render_multiple_frames(self, meshes):
        assert isinstance(meshes, list)
        batch_meshes = []
        for mesh in meshes:
            batch_meshes.append(join_meshes_as_scene([mesh, self.plane]))
        batch_meshes = join_meshes_as_batch(batch_meshes)
        images = self.renderer(batch_meshes)
        return images
    
    def tensor2image(self, tensor):
        return (tensor[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)