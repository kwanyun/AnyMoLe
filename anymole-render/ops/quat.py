import torch
import torch.nn.functional as F

def mul(q0, q1):
    r0, i0, j0, k0 = torch.split(q0, 1, dim=-1)
    r1, i1, j1, k1 = torch.split(q1, 1, dim=-1)
    
    res = torch.cat([
        r0*r1 - i0*i1 - j0*j1 - k0*k1,
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1
    ], dim=-1)

    return res

def mul_vec(q, v):
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    res = v + q[..., 0:1] * t + torch.cross(q[..., 1:], t, dim=-1)
    return res

def _split_axis_angle(aaxis):
    angle = torch.norm(aaxis, dim=-1)
    axis = aaxis / (angle[..., None] + 1e-8)
    return angle, axis

"""
Angle-axis to other representations
"""
def from_aaxis(aaxis):
    angle, axis = _split_axis_angle(aaxis)

    cos = torch.cos(angle / 2)[..., None]
    sin = torch.sin(angle / 2)[..., None]
    axis_sin = axis * sin

    return torch.cat([cos, axis_sin], dim=-1)

def inv(q):
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)

def identity(device="cpu"):
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

def interpolate(q_from, q_to, t):
    """
    Args:
        q_from: (..., 4)
        q_to: (..., 4)
        t: (..., t) or (t,), or just a float
    Returns:
        interpolated quaternion (..., 4, t)
    """
    device = q_from.device
    
    # ensure t is a torch tensor
    if isinstance(t, float):
        t = torch.tensor([t], dtype=torch.float32, device=device)
    t = torch.zeros_like(q_from[..., 0:1]) + t # (..., t)
    
    # ensure unit quaternions
    q_from_ = F.normalize(q_from, dim=-1, eps=1e-8) # (..., 4)
    q_to_   = F.normalize(q_to,   dim=-1, eps=1e-8) # (..., 4)

    # ensure positive dot product
    dot = torch.sum(q_from_ * q_to_, dim=-1) # (...,)
    neg = dot < 0.0
    dot[neg] = -dot[neg]
    q_to_[neg] = -q_to_[neg]

    # omega = arccos(dot)
    linear = dot > 0.9999
    omegas = torch.acos(dot[~linear]) # (...,)
    omegas = omegas[..., None] # (..., 1)
    sin_omegas = torch.sin(omegas) # (..., 1)

    # interpolation amounts
    t0 = torch.empty_like(t)
    t0[linear] = 1.0 - t[linear]
    t0[~linear] = torch.sin((1.0 - t[~linear]) * omegas) / sin_omegas # (..., t)

    t1 = torch.empty_like(t)
    t1[linear] = t[linear]
    t1[~linear] = torch.sin(t[~linear] * omegas) / sin_omegas # (..., t)
    
    # interpolate
    q_interp = t0[..., None, :] * q_from_[..., :, None] + t1[..., None, :] * q_to_[..., :, None] # (..., 4, t)
    q_interp = F.normalize(q_interp, dim=-2, eps=1e-8) # (..., 4, t)
    
    return q_interp


def fk(pre_quats, local_pos, local_quats, root_pos, parent_idx):
    """
    Attributes:
        pre_quats: (J, 4)
        local_pos: (J, 3)
        local_quats: (..., J, 4)
        root_pos: (..., 3)
    """
    njoints = local_quats.shape[-2]
    assert len(parent_idx) == njoints

    # match shape
    original_shape = local_quats.shape[:-2]
    local_quats = local_quats.reshape(-1, njoints, 4)
    root_pos = root_pos.reshape(-1, 3)

    pre_quats = pre_quats[None].expand(local_quats.shape[0], -1, -1) # (-1, J, 4)
    local_pos = local_pos[None].expand(local_quats.shape[0], -1, -1) # (-1, J, 3)

    global_quats = [mul(pre_quats[:, 0, :], local_quats[:, 0, :])]
    global_pos = [root_pos]

    for i in range(1, njoints):
        pidx = parent_idx[i]
        global_quats.append(mul(mul(global_quats[pidx], pre_quats[..., i, :]), local_quats[..., i, :]))
        global_pos.append(mul_vec(global_quats[pidx], local_pos[..., i, :]) + global_pos[pidx])
    
    global_quats = torch.stack(global_quats, dim=1) # (-1, J, 4)
    global_pos = torch.stack(global_pos, dim=1) # (-1, J, 3)

    # reshape
    global_quats = global_quats.reshape(original_shape + (njoints, 4))
    global_pos = global_pos.reshape(original_shape + (njoints, 3))

    return global_quats, global_pos

def to_rotmat(quat):
    two_s = 2.0 / torch.sum(quat * quat, dim=-1) # (...,)
    r, i, j, k = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    rotmat = torch.stack([
        1.0 - two_s * (j*j + k*k),
        two_s * (i*j - k*r),
        two_s * (i*k + j*r),
        two_s * (i*j + k*r),
        1.0 - two_s * (i*i + k*k),
        two_s * (j*k - i*r),
        two_s * (i*k - j*r),
        two_s * (j*k + i*r),
        1.0 - two_s * (i*i + j*j)
    ], dim=-1)
    return rotmat.reshape(quat.shape[:-1] + (3, 3)) # (..., 3, 3)


def to_xform(quat, pos):
    rotmat = to_rotmat(quat) # (..., 3, 3)
    xform = torch.cat([rotmat, pos[..., :, None]], dim=-1) # (..., 3, 4)
    xform = torch.cat([xform, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=xform.device)[None].expand(xform.shape[:-2] + (1, 4))], dim=-2) # (..., 4, 4)
    return xform