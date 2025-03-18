import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin

from pytorch3d.io import load_objs_as_meshes

from model import Model
from renderer import Renderer
from options.camera import get_args as get_camera_args

class Timer:
    def __init__(self):
        self.start = time.time()
    
    def set_start_time(self):
        self.start = time.time()
        
    def get_elapsed_time(self):
        return time.time() - self.start

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # debugging
    timer = Timer()
    
    # renderer
    renderer = Renderer(device)
    
    # model
    # model = Model("ybot.npz", device=device)
    plane = load_objs_as_meshes(["./data/checkerboard/checkerboard.obj"], device=device)
    
    char_path = "./data/ortiz"
    model = Model(pjoin(char_path, "mesh.pkl"),
                  get_camera_args(),
                  texture_paths=[pjoin(char_path, "textures", "diffuse.png")], 
                  device=device)
    motion = np.load(pjoin(char_path, "motion.npz"))
    local_quats = torch.from_numpy(motion["local_quats"]).float().to(device)
    root_pos = torch.from_numpy(motion["root_pos"]).float().to(device)
    
    # start timer
    timer.set_start_time()
    
    renderer.set_plane(plane)
    images = renderer.render(model.get_frame(local_quats, root_pos))
    
    # end timer
    print(f"{timer.get_elapsed_time():.2f} seconds")   
    
    # Plot rendered images
    images = images[0, ..., :3].cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(images)
    plt.axis("off")
    plt.show()



if __name__ == "__main__":
    main()