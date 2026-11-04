import numpy as np
import torch
import torch.nn.functional as F
import MRzeroCore as mr0

from .bssfp_params import NAMES

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


def build_sim_data(raw_path, p, device, permute_order=(2, 1, 0)):
    """Build the full simulation dataset for the 3D bSSFP protocol.

    Encapsulates everything 3D_bssfp_sequence.ipynb used to do inline so that
    both that notebook and Simulate.ipynb build an identical ``data`` object
    (and matching tissue masks) from a single code path.

    Parameters
    ----------
    raw_path : str
        Path to ``subject04_crisp_v.rawb``.
    p : BSSFPParams
        Protocol parameters (grid size, FOV, coil count, ...).
    device : torch.device
        Where to build the tensors.

    Returns
    -------
    data : mr0 SimData
        Result of ``phantom.build()`` (PD/T1/T2/T2'/B0/B1/coil_sens/...).
    masks : dict[str, np.ndarray]
        Per-tissue boolean masks (WM/GM/CSF) on the n^3 grid, matching the
        PD thresholds used in load_brain_phantom.
    brain_mask : np.ndarray
        Boolean mask of the whole brain (background + thin rim excluded).
    coil_den : torch.Tensor
        Sum of |coil|^2 over channels - the denominator for the
        SENSE/adaptive combination of the NUFFT previews.
    """
    from .CoilSens import cyclic_head_coil

    # Some tensors inside load_brain_phantom / VoxelGridPhantom.build() are
    # created without an explicit device and therefore inherit the *default*
    # device. The notebooks call torch.set_default_device("cuda"), which would
    # silently place those tensors on GPU while the maps stay on CPU (or
    # vice-versa) and break the boolean indexing in build().  Force the default
    # device to the target for the whole build so everything lands consistently.
    prev_default = torch.get_default_device()
    torch.set_default_device(device)
    try:
        phantom = load_brain_phantom(
            raw_path=raw_path,
            target_size=(p.n, p.n, p.n),
            device=device,
            permute_order=permute_order,
            fov=p.FOV,
        )
        print(type(phantom).__name__, "size =", phantom.size)

        pd_map = phantom.PD.detach().cpu().numpy()
        brain = pd_map > 0.10                       # exclude background + thin rim
        masks = {
            "WM":  brain & (pd_map < 0.75),
            "GM":  (pd_map >= 0.75) & (pd_map < 0.90),
            "CSF": (pd_map >= 0.90),
        }
        brain_mask = brain
        for t in NAMES:
            print(f"mask {t}: {int(masks[t].sum())} voxels")

        # ---- coils (used directly - identity orientation, no auto-match) ------
        scale = tuple(float(s) / p.FOV for s in phantom.size)
        coil_sim = cyclic_head_coil((p.N_channels, p.n, p.n, p.n), fov_scaling=scale).to(torch.complex64)
        phantom.coil_sens = coil_sim.to(device)
        phantom.B0 = 0 * phantom.B0
        phantom.B1 = torch.ones(phantom.B1.shape, dtype=phantom.B1.dtype)
        data = phantom.build()
        print("simulation voxels:", int(data.PD.numel()))

        # combine weights for the SENSE/adaptive combine of the NUFFT previews
        coil_c64 = coil_sim.to(device)
        coil_den = (coil_c64.abs() ** 2).sum(0)
    finally:
        torch.set_default_device(prev_default)

    return data, masks, brain_mask, coil_den, coil_c64
