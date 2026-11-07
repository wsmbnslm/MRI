import torch
import math
from typing import Optional, Tuple
from scipy.sparse.linalg import LinearOperator
from .FourierTransform import nonuniform_fourier_transform_adjoint, nonuniform_fourier_transform_forward 
import numpy as np
import Modules.bssfp_params as p
from Modules import PARAMS, build_sim_data

p=PARAMS

class LinearOperator:
    def __matmul__(self, other):
        if isinstance(other, LinearOperator):
            return _ProductOperator(self, other)
        return self._matvec(other)

    def __rmatmul__(self, other):
        if isinstance(other, LinearOperator):
            return _ProductOperator(other, self)
        return self._rmatvec(other)

    @property
    def H(self) -> "LinearOperator":
        return _AdjointOperator(self)

    def __add__(self, other: "LinearOperator") -> "LinearOperator":
        return _SumOperator(self, other)

    def __mul__(self, scalar) -> "LinearOperator":
        if isinstance(scalar, LinearOperator):
            # e.g. someone writes op1 * op2 expecting composition
            return _ProductOperator(self, scalar)
        return _ScaledOperator(scalar, self)

    __rmul__ = __mul__


class _ProductOperator(LinearOperator):
    def __init__(self, op1: LinearOperator, op2: LinearOperator):
        if op1.shape[1] != op2.shape[0]:
            raise ValueError(f"cannot compose {op1.shape} and {op2.shape}: shape mismatch")
        self.op1, self.op2 = op1, op2
        self.shape = (op1.shape[0], op2.shape[1])
        self.dtype = op1.dtype

    def _matvec(self, x):
        return self.op1._matvec(self.op2._matvec(x))

    def _rmatvec(self, x):
        return self.op2._rmatvec(self.op1._rmatvec(x))


class _AdjointOperator(LinearOperator):
    def __init__(self, op: LinearOperator):
        self.op = op
        self.shape = (op.shape[1], op.shape[0])
        self.dtype = op.dtype

    def _matvec(self, x):
        return self.op._rmatvec(x)

    def _rmatvec(self, x):
        return self.op._matvec(x)


class _SumOperator(LinearOperator):
    def __init__(self, op1: LinearOperator, op2: LinearOperator):
        self.op1, self.op2 = op1, op2
        self.shape = op1.shape
        self.dtype = op1.dtype

    def _matvec(self, x):
        return self.op1._matvec(x) + self.op2._matvec(x)

    def _rmatvec(self, x):
        return self.op1._rmatvec(x) + self.op2._rmatvec(x)


class _ScaledOperator(LinearOperator):
    def __init__(self, scalar, op: LinearOperator):
        self.scalar = scalar   # keep as tensor — don't cast to float/numpy
        self.op = op
        self.shape = op.shape
        self.dtype = op.dtype

    def _matvec(self, x):
        return self.scalar * self.op._matvec(x)

    def _rmatvec(self, x):
        return self.scalar * self.op._rmatvec(x)


class ChannelOperator(LinearOperator):

    def __init__(
        self,
        coil_sensitivities: torch.Tensor,
        data_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ):
        nC = coil_sensitivities.shape[0]

        self.coil_sensitivities = coil_sensitivities

        self.forward_shape = (1, *data_shape)
        self.adjoint_shape = (nC, *data_shape)
        self.shape = (
            int(2 * torch.prod(torch.tensor(self.adjoint_shape))),
            int(2 * torch.prod(torch.tensor(self.forward_shape))),
        )

        self.dtype = torch.float32
        self.device = device

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.forward_shape)
        y = self.coil_sensitivities * x
        return torch.view_as_real(y.reshape(-1)).reshape(-1)

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.adjoint_shape)
        y = torch.sum(torch.conj(self.coil_sensitivities) * x, dim=0, keepdim=True)
        return torch.view_as_real(y.reshape(-1)).reshape(-1)


class GradientOperator(LinearOperator):

    def __init__(
        self,
        dim: int,
        data_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ):
        self.dim = dim
        self.forward_shape = data_shape
        self.adjoint_shape = data_shape
        self.shape = (
            int(2 * torch.prod(torch.tensor(data_shape))),
            int(2 * torch.prod(torch.tensor(data_shape))),
        )

        self.dtype = torch.float32
        self.device = device

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.forward_shape)
        y = x - torch.roll(x, 1, self.dim)
        return torch.view_as_real(y.reshape(-1)).reshape(-1)

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.adjoint_shape)
        y = x - torch.roll(x, -1, self.dim)
        return torch.view_as_real(y.reshape(-1)).reshape(-1)


class IdentityOperator(LinearOperator):

    def __init__(
        self,
        data_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ):

        self.shape = (
            int(2 * torch.prod(torch.tensor(data_shape))),
            int(2 * torch.prod(torch.tensor(data_shape))),
        )

        self.dtype = torch.float32
        self.device = device

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return x

class NonuniformFourierTransformOperator(LinearOperator):

    def __init__(
        self,
        k: torch.Tensor,
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ):
        nD, nK = k.shape
        nC, nX, nY = input_shape

        self.n_modes = (nX, nY)
        self.k = k
        self.forward_shape = (nC, nX, nY, 1)
        self.adjoint_shape = (nC, nK)
        self.shape = (
            int(2 * torch.prod(torch.tensor(self.adjoint_shape))),
            int(2 * torch.prod(torch.tensor(self.forward_shape))),
        )
        self.nufft_ob = KbNufft(im_size=self.n_modes).to(device)
        self.adj_ob = KbNufftAdjoint(im_size=self.n_modes).to(device)

        self.dtype = torch.float32
        self.device = device

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.forward_shape)
        y = nonuniform_fourier_transform_forward(self.k, x, nufft_ob=self.nufft_ob)
        return torch.view_as_real(y.reshape(-1)).reshape(-1)

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.adjoint_shape)
        y = nonuniform_fourier_transform_adjoint(
            self.k, x, self.n_modes, adj_ob=self.adj_ob
        )
        return torch.view_as_real(y.reshape(-1)).reshape(-1)

def nufft_recon(sig, k_rad, n):
    'Per-channel density-compensated adjoint NUFFT.  sig: (C, N), k_rad: (3, N) -> (C, n, n, n).'
    keep = k_rad.norm(dim=0) <= np.pi
    k, s = k_rad[:, keep], sig[:, keep]
    dcomp = k.norm(dim=0).clamp(min=np.pi / n) ** 2   # |k|^2 for radial
    adj = tkbn.KbNufftAdjoint(im_size=(n, n, n), grid_size=(2 * n,) * 3).to(sig.device)
    return adj((s * dcomp)[None], k)[0]

def combine(img_c):
    'adaptive SENSE combine of a (C, n, n, n) multi-coil image -> (n, n, n) complex'
    return (coil_c64.conj() * img_c).sum(0) / coil_den.clamp(min=1e-2 * coil_den.max())

def frame_spoke_blocks(p):
    'ADC block (N_x samples) index of echo p in every segment'
    return [s * N_readout + p for s in range(N_segments)]

def frame_idx(p):
    return torch.cat([torch.arange(b * N_x, (b + 1) * N_x) for b in frame_spoke_blocks(p)])

class NUFFT3DOperator(LinearOperator):
    'F: single-coil image (nX,nY,nZ) -> multi-coil k-space (nC, nK).  k: (3, nK) in cycles/voxel.'
    def __init__(self, k, input_shape, device=None):
        nD, nK = k.shape
        nC, nX, nY, nZ = input_shape
        self.n_modes = (nX, nY, nZ)
        self.k = (-k).unsqueeze(-1)             # (3, nK, 1); sign convention of the module
        self.forward_shape = (nC, nX, nY, nZ, 1)
        self.adjoint_shape = (nC, nK)
        self.shape = (
            int(2 * torch.prod(torch.tensor(self.adjoint_shape))),
            int(2 * torch.prod(torch.tensor(self.forward_shape))),
        )
        self.nufft_ob = KbNufft(im_size=self.n_modes).to(device)
        self.adj_ob = KbNufftAdjoint(im_size=self.n_modes).to(device)
        self.dtype = torch.float32
        self.device = device

    def _matvec(self, x):
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.forward_shape)
        y = nonuniform_fourier_transform_forward(self.k, x, nufft_ob=self.nufft_ob)
        return torch.view_as_real(y.reshape(-1)).reshape(-1)

    def _rmatvec(self, x):
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.adjoint_shape)
        y = nonuniform_fourier_transform_adjoint(self.k, x, self.n_modes, adj_ob=self.adj_ob)
        return torch.view_as_real(y.reshape(-1)).reshape(-1)


def recon_nufft(f):
    'normal NUFFT preview of kwic frame f: (n, n, n) complex'
    idx, w = kwic_idx[f], kwic_w[f]
    s = signal[:, idx] * w[None, :]
    return combine(nufft_recon(s, k_full[:, idx], n))

def recon_cg(f, maxiter=20, x0=None):
    'CG NUFFT reconstruction of kwic frame f: (n, n, n) complex'
    idx = kwic_idx[f]
    F = NUFFT3DOperator(kwic_k[f], (N_channels, n, n, n))
    Q = C_op.H @ F.H @ F @ C_op
    d = (signal[:, idx] * kwic_w[f][None, :]).contiguous().view(torch.float32).ravel()
    b = C_op.H @ F.H @ d
    x = torch.zeros(n**3, dtype=torch.complex64).view(torch.float32) if x0 is None else x0
    x = conjugate_gradient(Q, b, x, maxiter=maxiter, disable_progressbar=True)
    return x.view(torch.complex64).reshape(n, n, n)

device = torch.device("cuda")
prev_default = torch.get_default_device()
torch.set_default_device(device)

def signed_series(mask):
    m = torch.as_tensor(mask, device=device)
    w = (m / m.sum()).to(torch.float32)
    s = []
    for f in fit_frames:
        img = torch.real(frames_fit[f] * torch.exp(-1j * ref_phase))
        s.append(float((img.masked_select(m) * w.masked_select(m)).sum()))
    return np.array(s)

def mag_series(mask):
    m = torch.as_tensor(mask, device=device)
    w = (m / m.sum()).to(torch.float32)
    s = []
    for f in fit_frames:
        img = frames_fit[f].abs()
        s.append(float((img.masked_select(m) * w.masked_select(m)).sum()))
    return np.array(s)

class K3(LinearOperator):
    'coil-batched image (nC,n,n,n) <-> k-space at k (3, nK); shared plans'
    def __init__(self, k):
        k = k.to(device=device, dtype=torch.float32)
        self.k = (-k).unsqueeze(-1)      # sign convention (matches NUFFT3DOperator)
        nK = k.shape[1]
        self.forward_shape = (p.N_channels, p.n, p.n, p.n, 1)
        self.adjoint_shape = (p.N_channels, nK)
        self.shape = (2 * p.N_channels * nK, 2 * p.N_channels * p.n**3)
        self.dtype = torch.float32
        self.device = device
    def _matvec(self, x):
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.forward_shape)
        y = nonuniform_fourier_transform_forward(self.k, x, nufft_ob=nufft3)
        return torch.view_as_real(y.reshape(-1)).reshape(-1)
    def _rmatvec(self, x):
        x = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.adjoint_shape)
        y = nonuniform_fourier_transform_adjoint(self.k, x, (p.n, p.n, p.n), adj_ob=adj3)
        return torch.view_as_real(y.reshape(-1)).reshape(-1)

def get_partition(f, k):
    'deterministic (train, validate, loss) k-position split of frame f, partition k'
    rng = np.random.default_rng(SEED * 1_000_003 + f * 101 + k)
    perm = rng.permutation(nKf[f])
    nv = int(round(0.20 * nKf[f])); ntr = int(round(0.40 * (nKf[f] - nv)))
    return (torch.as_tensor(perm[:ntr], device=device),
            torch.as_tensor(perm[ntr:ntr + nv], device=device),
            torch.as_tensor(perm[ntr + nv:], device=device))

def _sub(d, idx, Kf):
    # d: flat real (2 * N_channels * Kf), coil-major; idx applied to every coil
    return d.view(p.N_channels, Kf, 2)[:, idx].reshape(-1).contiguous()

def A_ops(f, k):                    # per-frame k positions (trajectory advances per frame)
    tr, va, lo = get_partition(f, k)
    kk = kwic_k[f].to(device)
    At = K3(kk[:, tr]) @ C_op
    Al = K3(kk[:, lo]) @ C_op
    Av = K3(kk[:, va]) @ C_op
    inv = 1.0 / (p.N_channels * nKf[f])
    return (At.H @ At) * inv, At, Al, Av