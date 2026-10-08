import numpy as np
import torch
import torch.nn.functional as F
import MRzeroCore as mr0

def load_brain_phantom(raw_path, target_size, device, permute_order=(2, 1, 0),
                       fov: float = 0.25):
    n_z, n_y, n_x = target_size
    raw_data = np.fromfile(raw_path, dtype=np.uint8).reshape(362, 434, 362)

    pd_signal = np.zeros_like(raw_data, dtype=np.float32)
    pd_signal[raw_data == 1] = 1.0
    pd_signal[raw_data == 2] = 0.8
    pd_signal[raw_data == 3] = 0.7

    pd_map = F.interpolate(
        torch.from_numpy(pd_signal).to(device).view(1, 1, 362, 434, 362),
        size=(n_z, n_y, n_x),
        mode='trilinear'
    ).squeeze().permute(*permute_order)

    t1_map = torch.zeros_like(pd_map)
    t1_map = torch.where(pd_map > 0.9, 4.0, t1_map)
    t1_map = torch.where((pd_map <= 0.9) & (pd_map > 0.75), 1.3, t1_map)
    t1_map = torch.where((pd_map <= 0.75) & (pd_map > 0.1), 0.9, t1_map)

    t2_map = torch.zeros_like(pd_map)
    t2_map = torch.where(pd_map > 0.9, 0.5, t2_map)
    t2_map = torch.where((pd_map <= 0.9) & (pd_map > 0.75), 0.08, t2_map)
    t2_map = torch.where((pd_map <= 0.75) & (pd_map > 0.1), 0.07, t2_map)

    # VoxelGridPhantom expects a 4x4 affine (voxel -> world, in mm).
    # The grid spans an isotropic cube of side `fov` (meters), centered at
    # the origin: voxel i is at (i - N/2) * dv for even N.
    shape = torch.tensor(pd_map.shape, dtype=torch.float32)
    dv = fov * 1000.0 / shape  # mm per voxel
    affine = torch.eye(4, device=device)
    for i in range(3):
        affine[i, i] = dv[i]
        affine[i, 3] = -dv[i] * (shape[i] // 2)

    return mr0.VoxelGridPhantom(
        pd_map,
        t1_map,
        t2_map,
        torch.full_like(pd_map, 0.04),
        torch.zeros_like(pd_map),
        torch.zeros_like(pd_map),
        torch.ones_like(pd_map).unsqueeze(0),
        torch.ones_like(pd_map).unsqueeze(0),
        affine,
    )
