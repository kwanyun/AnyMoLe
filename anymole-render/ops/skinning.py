import torch

def lbs(verts, joint_xform, skin_weights, bind_xform_inv):
    """
    Attributes:
        verts: (V, 3)
        joint_xform: (T, J, 4, 4)
        skin_weights: (V, J)
        bind_xform_inv: (J, 4, 4)
    """

    # frame-wise vertex transformation
    xforms = torch.matmul(joint_xform, bind_xform_inv) # (T, J, 4, 4)
    skin_weights = skin_weights[:, :, None, None] # (V, J, 1, 1)

    xforms = xforms[:, None]                    # (T, 1, J, 4, 4)
    skin_weights = skin_weights[None]           # (1, V, J, 1, 1)
    xforms = (xforms * skin_weights).sum(dim=2) # (T, V, 4, 4)

    # skinning
    v_hom = torch.ones(verts.shape[0], 1, device=verts.device)  # (V, 1)
    verts = torch.cat([verts, v_hom], dim=-1)                   # (V, 4)
    res = torch.matmul(xforms, verts[..., None]).squeeze(-1)    # (T, V, 4)
    res = res[..., :3] # (T, V, 3)

    return res