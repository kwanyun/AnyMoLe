import torch
import numpy as np
import pickle
from PIL import Image

from model import Motion, Model
from ops import quat, skinning
from aPyOpenGL import agl

class Model2:
    def __init__(self, model_pkl_path, texture_paths=None, device="cuda:0"):
        super().__init__()
        self.model_pkl_path = model_pkl_path
        self.device = device

        # load pkl
        with open(model_pkl_path, "rb") as f:
            pkl = pickle.load(f)
        mesh_data = pkl["mesh"]
        skinning_data = pkl["skinning"]
        
        # mesh
        self.skin_weights = []
        self.verts, self.faces_idx = [], []
        self.joint_names = []
        for i in range(len(mesh_data)):
            verts = torch.from_numpy(mesh_data[i]["verts"]).float().to(device)
            faces_idx = torch.from_numpy(mesh_data[i]["faces_idx"]).long().to(device)

            self.verts.append(verts)
            self.faces_idx.append(faces_idx)

            self.skin_weights.append(torch.from_numpy(mesh_data[i]["skinning_weights"]).float().to(device))
            self.joint_names.append(list(mesh_data[i]["joint_names"]))
        
        # skinning
        self.bind_xform_inv = torch.from_numpy(skinning_data["bind_xform_inv"]).float().to(device)

        # deformed vertices
        self.deformed_verts = None
    
    def set_motion(self, motion_npz):
        self.pre_quats = torch.from_numpy(motion_npz["pre_quats"]).float().to(self.device)
        self.local_pos = torch.from_numpy(motion_npz["local_pos"]).float().to(self.device)
        self.parent_idx = motion_npz["parent_idx"]

        self.local_quats = torch.from_numpy(motion_npz["local_quats"]).float().to(self.device)
        self.root_pos = torch.from_numpy(motion_npz["root_pos"]).float().to(self.device)
        # self.local_quats[..., 0] = 1.0
        # self.local_quats[..., 1:] = 0.0

        joint_names = list(motion_npz["joint_names"])
        # breakpoint()
        # assert len(joint_names) == len(self.joint_names[0]), f"{len(joint_names)} != {len(self.joint_names[0])}"
        for i in range(len(self.skin_weights)):
            joint_name_map = []
            for j in range(len(joint_names)):
                src_name = joint_names[j]
                if src_name not in self.joint_names[i]:
                    self.joint_names[i].append(src_name)
                    self.skin_weights[i] = torch.cat([self.skin_weights[i], torch.zeros(self.skin_weights[i].shape[0], 1).to(self.device)], dim=1)
                    self.bind_xform_inv = torch.cat([self.bind_xform_inv, torch.eye(4).to(self.device)[None]], dim=0)
                tgt_idx = self.joint_names[i].index(src_name)
                joint_name_map.append(tgt_idx)
            self.skin_weights[i] = self.skin_weights[i][:, joint_name_map]
        self.bind_xform_inv = self.bind_xform_inv[joint_name_map]


    def deform_by_motion(self, local_quats, root_pos):
        """
        Attributes:
            local_quats: (T, J, 4)
            root_pos: (T, 3)
        """
        global_quats, global_pos = quat.fk(self.pre_quats, self.local_pos, local_quats, root_pos, self.parent_idx)
        xforms = quat.to_xform(global_quats, global_pos)
        deformed_verts = []
        for idx, verts in enumerate(self.verts):
            deformed_verts.append(skinning.lbs(verts, xforms, self.skin_weights[idx], self.bind_xform_inv))
        
        self.deformed_verts = deformed_verts

class DatasetApp(agl.App):
    def __init__(self, motion_path, model_path):
        super().__init__()
        self.motion = Motion(motion_path)
        self.model = Model(model_path)

        self.motion.save(remove_ee=False, blender=False)
        self.model.save()

        self.model2 = Model2(model_path.replace(".fbx", ".pkl"))
        self.model2.set_motion(np.load(motion_path.replace(".bvh", ".npz")) if motion_path.endswith(".bvh") else np.load(motion_path.replace(".fbx", ".npz")))
        self.model2.deform_by_motion(self.model2.local_quats, self.model2.root_pos)

        self.real_motion = agl.BVH(motion_path, scale=0.01).motion() if motion_path.endswith(".bvh") else agl.FBX(motion_path, scale=0.01).motions()[0]
        self.real_model = agl.FBX(model_path, scale=0.01).model()
    
    def start(self):
        super().start()
        self.verts = self.model2.deformed_verts[0].cpu().numpy()
        self.spheres = agl.Render.sphere(0.005).instance_num(100) if len(self.verts) > 100 else agl.Render.sphere(0.005).instance_num(len(self.verts))
        # self.remaining_spheres = agl.Render.sphere(0.005).instance_num(len(self.verts) % 100)

    def render(self):
        super().render()
        verts = self.verts[self.frame % len(self.verts)]
        for i in range(verts.shape[0] // 100):
            self.spheres.position(verts[i*100:(i+1)*100]).draw()
        # self.remaining_spheres.position(verts[(verts.shape[0] // 100) * 100:]).draw()

        self.real_model.set_pose(self.real_motion.poses[self.frame % len(self.real_motion)])
        agl.Render.model(self.real_model).draw()
    
# Run the script
if __name__ == "__main__":
    motion_path = "/home/user/research/vid2mib/GreatDane/GreatDane_Walk.fbx"
    model_path = "/home/user/research/vid2mib/GreatDane/GreatDane_Walk.fbx"
    agl.AppManager.start(DatasetApp(motion_path, model_path))