import os
import random
import glob
import natsort
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
os.chdir('./../')

class Move(Dataset):
    """
    AnyMoLe Dataset.
    Assumes AnyMoLe data is structured as follows.
    Move/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
        self.meta_path = meta_path
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.totensor = transforms.ToTensor()
        self._load_metadata()
        
        self.all_data=[]
        fpslist = [10,15,30] # stride will be 3,2,1
        self.num_total_data =0
        
        #view left, right, frame(front), back
        
        for view_idx in range(4):
            for fps_vid in fpslist:
                stride = 30//fps_vid
                numclip_pervid = 61-stride*(video_length-1)
                self.num_total_data += numclip_pervid
                for startframe_i in range(numclip_pervid):
                    frame_indices = [startframe_i + stride*i for i in range(self.video_length)]
                    frames= []
                    for single_index in frame_indices:
                        frames.append(self.metadata['data'][view_idx*61+single_index])
                    stacked_frames = torch.stack(frames).permute(1, 0, 2,3).float() # [t,c,h,w] -> [c,t,h,w]
                    self.all_data.append({'video': stacked_frames, 'caption': self.metadata['caption'], 'path': '', 'fps': fps_vid, 'frame_stride': stride})

            #import pdb; pdb.set_trace()
            #video_path = '/source/kwan/DynamiCrafter/results/dynamicrafter_512_seed12306/samples_separate/frame_0000_sample0_ddpm_0_0.mp4'
            #video_reader = VideoReader(video_path, ctx=cpu(0))
            #frames = video_reader.get_batch(frame_indices)
            #frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        #if spatial_transform is not None:
        #    if spatial_transform == "random_crop":
        #        self.spatial_transform = transforms.RandomCrop(crop_resolution)
        #    elif spatial_transform == "center_crop":
        #        self.spatial_transform = transforms.Compose([
        #            transforms.CenterCrop(resolution),
        #            ])
        #    elif spatial_transform == "resize_center_crop":
        #        # assert(self.resolution[0] == self.resolution[1])
        #        self.spatial_transform = transforms.Compose([
        #            transforms.Resize(min(self.resolution)),
        #            transforms.CenterCrop(self.resolution),
        #            ])
        #    elif spatial_transform == "resize":
        #        self.spatial_transform = transforms.Resize(self.resolution)
        #    else:
        #        raise NotImplementedError
        #else:
        #    self.spatial_transform = None
                
    def _load_metadata(self):
        metadata = {}
        with open(os.path.join(self.meta_path,'test_prompts.txt'), 'r', encoding='utf-8') as file:
            metadata['caption'] = file.readline().strip()

        all_datapath = glob.glob(self.meta_path+'*g') #jpg or png with 2sec for inbetweening
        assert len(all_datapath) == 61*4, 'Must be multi view rendered'
        all_datapath = natsort.natsorted(all_datapath) #sort

        metadata['data'] =[]
        for single_frame_path in all_datapath:
            frame_img = Image.open(single_frame_path)
            frame_tensor = (self.totensor(frame_img) - 0.5) * 2
            assert (frame_tensor.shape[1], frame_tensor.shape[2]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
            
            metadata['data'].append(frame_tensor)
            
            
        self.metadata = metadata #using the first 61 context frames
            
    
    def __getitem__(self, index):
        return self.all_data[index]
    
    def __len__(self):
        return self.num_total_data


if __name__== "__main__":
    meta_path = "" ## path to the meta file
    save_dir = "" ## path to the save directory
    dataset = Move(meta_path,
                 subsample=None,
                 video_length=16,
                 resolution=[256,448],
                 frame_stride=4,
                 spatial_transform="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True
                 )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False)

    
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        video = batch['video']
        name = batch['path'][0].split('videos/')[-1].replace('/','_')
        tensor_to_mp4(video, save_dir+'/'+name, fps=8)

