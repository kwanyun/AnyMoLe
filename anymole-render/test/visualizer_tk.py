import torch
import numpy as np
from os.path import join as pjoin

import tkinter as tk
from PIL import Image, ImageTk

from pytorch3d.io import load_objs_as_meshes

from model import Model
from renderer import Renderer
from options.camera import get_args as get_camera_args

class RendererApp:
    def __init__(self, root, device):
        self.root = root
        self.root.title("Interactive 3D Renderer")
        
        self.canvas = tk.Canvas(root, width=512, height=320)
        self.canvas.grid(row=0, column=0, columnspan=2)
        
        self.frame_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL,
                                     length=460,
                                     label="Frames", command=self.update_image)
        self.frame_slider.set(0)
        self.frame_slider.grid(row=1, column=0, columnspan=2)
        
        self.azimuth_slider = tk.Scale(root, from_=0, to=360, orient=tk.HORIZONTAL,
                                       length=200,
                                       label="Azimuth", command=self.update_image)
        self.azimuth_slider.set(0)
        self.azimuth_slider.grid(row=2, column=0)

        self.elevation_slider = tk.Scale(root, from_=0, to=360, orient=tk.HORIZONTAL,
                                         length=200,
                                         label="Elevation", command=self.update_image)
        self.elevation_slider.set(10)
        self.elevation_slider.grid(row=3, column=0)

        self.distance_slider = tk.Scale(root, from_=1.0, to=10.0, orient=tk.HORIZONTAL,
                                        length=200, resolution=0.1,
                                        label="Distance", command=self.update_image)
        self.distance_slider.set(2.7)
        self.distance_slider.grid(row=4, column=0)
        
        self.lookat_x_slider = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL,
                                        length=200, resolution=0.1,
                                        label="LookAt X", command=self.update_image)
        self.lookat_x_slider.set(0)
        self.lookat_x_slider.grid(row=2, column=1)
        self.lookat_y_slider = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL,
                                        length=200, resolution=0.1,
                                        label="LookAt Y", command=self.update_image)
        self.lookat_y_slider.set(0)
        self.lookat_y_slider.grid(row=3, column=1)
        self.lookat_z_slider = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL,
                                        length=200, resolution=0.1,
                                        label="LookAt Z", command=self.update_image)
        self.lookat_z_slider.set(0)
        self.lookat_z_slider.grid(row=4, column=1)
        
        
        # set pytorch3d setting
        plane = load_objs_as_meshes(["./data/checkerboard/checkerboard.obj"], device=device)
    
        char_path = "./data/ortiz"
        model = Model(pjoin(char_path, "mesh.pkl"),
                    get_camera_args(),
                    texture_paths=[pjoin(char_path, "textures", "diffuse.png")], 
                    device=device)
        motion = np.load(pjoin(char_path, "motion.npz"))
        local_quats = torch.from_numpy(motion["local_quats"]).float().to(device)
        root_pos = torch.from_numpy(motion["root_pos"]).float().to(device)
        self.renderer = Renderer(device)
        
        self.renderer.set_plane(plane)
        self.renderer.set_meshes(model.get_frame(local_quats, root_pos))

        self.update_image()

    def update_image(self, event=None):
        azimuth = float(self.azimuth_slider.get())
        elevation = float(self.elevation_slider.get())
        distance = float(self.distance_slider.get())
        at = torch.tensor([[
            float(self.lookat_x_slider.get()),
            float(self.lookat_y_slider.get()),
            float(self.lookat_z_slider.get())
        ]], dtype=torch.float32)

        image = self.renderer.render_images(azimuth, elevation, distance, at)
        image = Image.fromarray(image)
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    root = tk.Tk()
    app = RendererApp(root, device)
    root.mainloop()