"""FFT and NUFFT operators for MRI reconstruction.

Uniform (Cartesian) FFT via torch.fft, plus non-uniform FFT (NUFFT)
between an image grid and arbitrary k-space samples, using a vendored,
trimmed Kaiser-Bessel gridding NUFFT (see kbnufft_lite.py) -- pure
PyTorch, GPU-capable, no native library dependency.

k is in cycles/voxel, scaled to [-0.5, 0.5]. kbnufft_lite's sign
convention is negated relative to the usual forward/adjoint DFT
convention, so k is negated before being passed to it. Output is
normalized by 1 / sqrt(#grid points).
"""

import math
import warnings
from collections.abc import Sequence
from typing import Optional

import torch

from .kbnufft import KbNufft, KbNufftAdjoint


def fourier_transform_forward(x: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    """Centered forward FFT of `x` along `axes`."""
    x = torch.fft.ifftshift(x, dim=axes)
    x = torch.fft.fftn(x, dim=axes, norm="ortho")
    return torch.fft.fftshift(x, dim=axes)


def fourier_transform_adjoint(x: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    """Centered adjoint FFT of `x` along `axes`."""
    x = torch.fft.ifftshift(x, dim=axes)
    x = torch.fft.ifftn(x, dim=axes, norm="ortho")
    return torch.fft.fftshift(x, dim=axes)


def _match_device(k, x, device):
    """Move k and x onto a common device."""
    if device is not None:
        return k.to(device), x.to(device), device
    device = k.device
    if x.device != device:
        warnings.warn(f"k and x on different devices, moving x to {device}.")
        x = x.to(device)
    return k, x, device


def _split_batch_axes(k_extra: Sequence[int], x_extra: Sequence[int]):
    """Split x's batch axes into those shared with k and those exclusive to x."""
    shared = list(x_extra[: len(k_extra)])
    excl = list(x_extra[len(k_extra) :])
    if list(k_extra) != shared:
        raise ValueError(f"k's extra dims {list(k_extra)} != x's leading {shared}.")
    return shared, excl


def nonuniform_fourier_transform_forward(
    k: torch.Tensor,
    x: torch.Tensor,
    nufft_ob: Optional["KbNufft"] = None,
    device: torch.device | None = None,
    **kbnufft_kwargs,
) -> torch.Tensor:
    """NUFFT forward: image -> non-uniform k-space samples.

    k: (D, N, ...) in [-0.5, 0.5]. x: (C, R, P1, P2, ...).
    Returns (C, N, ...). kbnufft_kwargs go to KbNufft (e.g. numpoints),
    ignored if nufft_ob is given.
    """
    k, x, device = _match_device(k, x, device)
    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)

    x = torch.moveaxis(x, 0, -1)  # channel -> trailing batch axes
    D, N, *k_extra = k.shape
    R, P1, P2, *x_extra = x.shape
    norm = math.sqrt(R * P1 * P2)

    shared, excl = _split_batch_axes(k_extra, x_extra)
    n_shared, n_excl = math.prod(shared), math.prod(excl)

    x = x.reshape(R, P1, P2, n_shared, n_excl).permute(3, 4, 0, 1, 2)
    k = k.reshape(D, N, n_shared).permute(2, 0, 1)
    x = x.squeeze(dim=(-1, -2, -3))  # drop singleton spatial dims

    if nufft_ob is None:
        nufft_ob = KbNufft(im_size=tuple(x.shape[2:]), **kbnufft_kwargs)
        nufft_ob = nufft_ob.to(device)
    y = nufft_ob(x, -k)  # negate k: kbnufft_lite's sign is flipped

    y = (y / norm).reshape(*shared, *excl, N)
    y = y.moveaxis((-2, -1), (0, 1))  # -> (C, N, shared, excl)
    if x_extra[:-1] == [1]:
        y = y[..., 0]
    return y


def nonuniform_fourier_transform_adjoint(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: tuple[int] | tuple[int, int] | tuple[int, int, int],
    adj_ob: Optional["KbNufftAdjoint"] = None,
    device: torch.device | None = None,
    **kbnufft_kwargs,
) -> torch.Tensor:
    """NUFFT adjoint: non-uniform k-space samples -> image.

    k: (D, N, ...). x: (N,), (C, N) or (C, N, ...).
    Returns (C, nx, ny, nz, ...), missing grid dims set to 1.
    kbnufft_kwargs go to KbNufftAdjoint, ignored if adj_ob is given.
    """
    if k.shape[0] != len(n_modes):
        raise ValueError(f"n_modes has {len(n_modes)} dims, k has {k.shape[0]}.")

    if x.ndim == 1:
        x = x[None, :]
    if x.ndim == 2:
        x = x[:, :, None]

    k, x, device = _match_device(k, x, device)
    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)
    norm = math.sqrt(math.prod(n_modes))

    x = torch.moveaxis(x, 0, -1)  # channel -> trailing batch axes
    N_x, *x_extra = x.shape
    D, N_k, *k_extra = k.shape
    if N_x != N_k:
        raise ValueError(f"x and k disagree on N: {N_x} vs {N_k}.")
    N = N_x

    shared, excl = _split_batch_axes(k_extra, x_extra)
    n_shared, n_excl = math.prod(shared), math.prod(excl)

    x = x.reshape(N, n_shared, n_excl).permute(1, 2, 0)
    k = k.reshape(D, N, n_shared).permute(2, 0, 1)

    if adj_ob is None:
        adj_ob = KbNufftAdjoint(im_size=n_modes, **kbnufft_kwargs)
        adj_ob = adj_ob.to(device)
    y = adj_ob(x, -k)  # negate k: kbnufft_lite's sign is flipped

    y = y / norm
    for _ in range(3 - len(n_modes)):
        y = y[..., None]

    y = y.reshape(*shared, *excl, *y.shape[-3:])
    y = y.moveaxis((len(x_extra) - 1, -3, -2, -1), (0, 1, 2, 3))
    if x_extra[:-1] == [1]:
        y = y[..., 0]
    return y