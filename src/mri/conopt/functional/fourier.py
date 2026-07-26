import math
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Optional

import torch

try:
    from pytorch_finufft.functional import finufft_type1, finufft_type2
except ImportError:
    finufft_type1 = finufft_type2 = None

try:
    from torchkbnufft import KbNufft, KbNufftAdjoint
except ImportError:
    KbNufft = KbNufftAdjoint = None


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


def _check_backend(backend: str) -> None:
    if backend not in ("torchkbnufft", "finufft"):
        raise ValueError(
            f"backend must be 'torchkbnufft' or 'finufft', got {backend!r}."
        )
    if backend == "finufft" and finufft_type2 is None:
        raise ImportError("pip install pytorch-finufft")
    if backend == "torchkbnufft" and KbNufft is None:
        raise ImportError("pip install torchkbnufft")


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
    backend: str = "torchkbnufft",
    nufft_ob: Optional["KbNufft"] = None,
    device: torch.device | None = None,
    **backend_kwargs,
) -> torch.Tensor:
    """NUFFT forward: image -> non-uniform k-space samples.

    k: (D, N, ...) in [-0.5, 0.5]. x: (C, R, P1, P2, ...).
    Returns (C, N, ...). backend_kwargs go to KbNufft (e.g. numpoints) or,
    for backend="finufft", to finufft_type2 (e.g. eps, isign, modeord;
    default isign=1, modeord=0).
    """
    _check_backend(backend)
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

    if backend == "torchkbnufft":
        if nufft_ob is None:
            nufft_ob = KbNufft(im_size=tuple(x.shape[2:]), **backend_kwargs)
            nufft_ob = nufft_ob.to(device)
        y = nufft_ob(x, -k)  # negate k: torchkbnufft's sign is flipped
    else:
        kwargs = {"modeord": 0, "isign": 1, **backend_kwargs}
        fn = partial(finufft_type2, **kwargs)
        y = torch.vmap(fn, in_dims=0)(k.cpu().contiguous(), x.cpu().contiguous())
        y = y.to(device)

    y = (y / norm).reshape(*shared, *excl, N)
    y = y.moveaxis((-2, -1), (0, 1))  # -> (C, N, shared, excl)
    if x_extra[:-1] == [1]:
        y = y[..., 0]
    return y


def nonuniform_fourier_transform_adjoint(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: tuple[int] | tuple[int, int] | tuple[int, int, int],
    backend: str = "torchkbnufft",
    adj_ob: Optional["KbNufftAdjoint"] = None,
    device: torch.device | None = None,
    **backend_kwargs,
) -> torch.Tensor:
    """NUFFT adjoint: non-uniform k-space samples -> image.

    k: (D, N, ...). x: (N,), (C, N) or (C, N, ...).
    Returns (C, nx, ny, nz, ...), missing grid dims set to 1. backend_kwargs
    go to KbNufftAdjoint or, for backend="finufft", to finufft_type1 (e.g.
    eps, isign, modeord; default isign=-1, modeord=0).
    """
    _check_backend(backend)
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

    if backend == "torchkbnufft":
        if adj_ob is None:
            adj_ob = KbNufftAdjoint(im_size=n_modes, **backend_kwargs)
            adj_ob = adj_ob.to(device)
        y = adj_ob(x, -k)  # negate k: torchkbnufft's sign is flipped
    else:
        kwargs = {"modeord": 0, "isign": -1, "output_shape": n_modes, **backend_kwargs}
        fn = partial(finufft_type1, **kwargs)
        y = torch.vmap(fn, in_dims=0)(k.cpu().contiguous(), x.cpu().contiguous())
        y = y.to(device)

    y = y / norm
    for _ in range(3 - len(n_modes)):
        y = y[..., None]

    y = y.reshape(*shared, *excl, *y.shape[-3:])
    y = y.moveaxis((len(x_extra) - 1, -3, -2, -1), (0, 1, 2, 3))
    if x_extra[:-1] == [1]:
        y = y[..., 0]
    return y