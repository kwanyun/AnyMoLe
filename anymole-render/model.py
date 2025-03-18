import torch
from pytorch3d.structures import (
    Meshes,
    join_meshes_as_scene,
)
from pytorch3d.renderer import (
    TexturesVertex,
    TexturesUV,
)

from PIL import Image
import pickle
import numpy as np

from ops import quat, skinning


class Model:
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
        self.verts, self.faces_idx, self.textures = [], [], []
        self.joint_names = []
        for i in range(len(mesh_data)):
            verts = torch.from_numpy(mesh_data[i]["verts"]).float().to(device)
            faces_idx = torch.from_numpy(mesh_data[i]["faces_idx"]).long().to(device)
            verts_uvs = torch.from_numpy(mesh_data[i]["verts_uvs"]).float().to(device)
            # if "diffuse" in mesh_data[i]:
            #     texture = torch.from_numpy(mesh_data[i]["diffuse"]).float().to(device)
            if texture_paths is not None:
                texture = Image.open(texture_paths[i]).convert("RGB")
                texture = torch.tensor(np.array(texture) / 255.0, dtype=torch.float32).to(device)
                texture = texture[..., :3]
            else:
                albedo = torch.from_numpy(mesh_data[i]["albedo"]).float().to(device)
                texture = albedo.reshape(1, 1, 3).repeat(256, 256, 1)
            textures = TexturesUV(
                maps=texture[None],
                faces_uvs=faces_idx[None],
                verts_uvs=verts_uvs[None]
            )

            self.verts.append(verts)
            self.faces_idx.append(faces_idx)
            self.textures.append(textures)

            self.skin_weights.append(torch.from_numpy(mesh_data[i]["skinning_weights"]).float().to(device))
            self.joint_names.append(list(mesh_data[i]["joint_names"]))
        
        # skinning
        self.bind_xform_inv = torch.from_numpy(skinning_data["bind_xform_inv"]).float().to(device)

        # deformed vertices
        self.deformed_verts = None
    
    def set_motion_test(self, motion_npz):
        self.pre_quats = torch.from_numpy(motion_npz["pre_quats"]).float().to(self.device)
        self.pre_quats[..., 0] = 1.0
        self.pre_quats[..., 1:] = 0.0
        self.local_pos = torch.from_numpy(motion_npz["local_pos"]).float().to(self.device)
        self.parent_idx = motion_npz["parent_idx"]

        self.local_quats = torch.from_numpy(motion_npz["local_quats"]).float().to(self.device)
        self.root_pos = torch.from_numpy(motion_npz["root_pos"]).float().to(self.device)

        joint_names = list(motion_npz["joint_names"])
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

    def set_motion(self, motion_npz):
        self.pre_quats = torch.from_numpy(motion_npz["pre_quats"]).float().to(self.device)
        self.local_pos = torch.from_numpy(motion_npz["local_pos"]).float().to(self.device)
        self.parent_idx = motion_npz["parent_idx"]

        self.local_quats = torch.from_numpy(motion_npz["local_quats"]).float().to(self.device)
        self.root_pos = torch.from_numpy(motion_npz["root_pos"]).float().to(self.device)

        joint_names = list(motion_npz["joint_names"])
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

    def deform_by_motion_pf(self, local_quats, root_pos):
        """
        Attributes:
            local_quats: (T, J, 4)
            root_pos: (T, 3)
        """
        new_local_quats = quat.mul(quat.inv(self.pre_quats), local_quats)
        global_quats, global_pos = quat.fk(self.pre_quats, self.local_pos, new_local_quats, root_pos, self.parent_idx)
        xforms = quat.to_xform(global_quats, global_pos)
        deformed_verts = []
        for idx, verts in enumerate(self.verts):
            deformed_verts.append(skinning.lbs(verts, xforms, self.skin_weights[idx], self.bind_xform_inv))
        
        self.deformed_verts = deformed_verts

    def get_mesh(self, frame_idx=None):
        meshes = []
        for i in range(len(self.verts)):
            mesh = Meshes(
                verts=[self.verts[i]] if frame_idx is None or self.deformed_verts is None else [self.deformed_verts[i][frame_idx]],
                faces=[self.faces_idx[i]],
                textures=self.textures[i]
            )
            meshes.append(mesh)
        return join_meshes_as_scene(meshes)
    

    def get_joint_pos(self, frame_idx=None):
        if frame_idx is None:
            gq, gp = quat.fk(self.pre_quats, self.local_pos, self.local_quats, self.root_pos, self.parent_idx)
        else:
            gq, gp = quat.fk(self.pre_quats, self.local_pos, self.local_quats[frame_idx], self.root_pos[frame_idx], self.parent_idx)
        
        return gp

    def get_joint_pos_from_quat_pos(self,local_quats,root_pos,bs):
        gq, gp = quat.fk(self.pre_quats, self.local_pos, local_quats, root_pos, self.parent_idx)
        
        return gp