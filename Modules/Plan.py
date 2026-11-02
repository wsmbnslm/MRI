import itertools
import numpy as np
import torch
import torch.nn.functional as tnnf
from scipy import special
from scipy.sparse import coo_matrix
from torch import Tensor, nn
from torch.autograd import Function

def _build_spmatrix(om, numpoints, im_size, grid_size, n_shift, order, alpha) -> coo_matrix:

    def interp_coeff(om, npts, grdsz, alpha, order):
        gam = 2 * np.pi / grdsz
        interp_dist = om / gam - np.floor(om / gam - npts / 2)
        jvec = np.reshape(np.arange(1, npts + 1), (1, npts))
        kern_in = -1 * jvec + np.expand_dims(interp_dist, 1)

        coef = np.zeros(shape=kern_in.shape, dtype=np.complex128)
        mask = np.absolute(kern_in) < npts / 2
        bess_arg = np.sqrt(1 - (kern_in[mask] / (npts / 2)) ** 2)
        denom = special.iv(order, alpha)
        coef[mask] = special.iv(order, alpha * bess_arg) / denom
        return np.real(coef), kern_in

    coef, kern_in = interp_coeff(om[0], numpoints[0], grid_size[0], alpha[0], order[0])
    gam = 2 * np.pi / grid_size[0]
    phase = np.exp(1j * gam * (im_size[0] - 1) / 2 * kern_in)
    coef = phase * coef

    koff = np.expand_dims(np.floor(om[0] / gam - numpoints[0] / 2), 1)
    jvec = np.reshape(np.arange(1, numpoints[0] + 1), (1, numpoints[0]))
    kk = np.mod(jvec + koff, grid_size[0])

    klength = om.shape[1]
    phase = np.exp(1j * np.dot(om.T, np.expand_dims(n_shift, 1)))
    coef = np.conj(coef) * phase

    traj_ind = np.repeat(np.arange(klength)[:, None], numpoints[0], axis=1)
    return coo_matrix(
        (coef.flatten(), (traj_ind.flatten(), kk.flatten())),
        shape=(klength, grid_size[0]),
    )


def _build_table(im_size, grid_size, numpoints, table_oversamp, order, alpha):
    tables = []
    for d in range(len(im_size)):
        npts, gridsz, oversamp = numpoints[d], grid_size[d], table_oversamp[d]
        t1 = npts / 2 - 1 + np.arange(oversamp) / oversamp
        om1 = t1 * 2 * np.pi / gridsz
        s1 = _build_spmatrix(
            np.expand_dims(om1, 0),
            numpoints=(npts,),
            im_size=(im_size[d],),
            grid_size=(gridsz,),
            n_shift=(0,),
            order=(order[d],),
            alpha=(alpha[d],),
        )
        h = np.array(s1.getcol(npts - 1).todense())
        for col in range(npts - 2, -1, -1):
            h = np.concatenate((h, np.array(s1.getcol(col).todense())), axis=0)
        h = np.concatenate((h.flatten(), np.array([0])))
        tables.append(torch.tensor(h))
    return tables


def _kaiser_bessel_ft(omega, numpoints, alpha, order, d):
    z = np.sqrt((2 * np.pi * (numpoints / 2) * omega) ** 2 - alpha**2 + 0j)
    nu = d / 2 + order
    coef = (
        (2 * np.pi) ** (d / 2)
        * ((numpoints / 2) ** d)
        * (alpha**order)
        / special.iv(order, alpha)
        * special.jv(nu, z)
        / (z**nu)
    )
    return np.real(coef)


def _scaling_coefs(im_size, grid_size, numpoints, alpha, order) -> Tensor:

    def one_dim(n, npts, gridsz, a, o):
        idx = np.arange(n) - (n - 1) / 2
        coef = 1 / _kaiser_bessel_ft(idx / gridsz, npts, a, o, 1)
        return np.ones_like(coef) if npts == 1 else coef

    coef = one_dim(im_size[0], numpoints[0], grid_size[0], alpha[0], order[0])
    for i in range(1, len(im_size)):
        tmp = one_dim(im_size[i], numpoints[i], grid_size[i], alpha[i], order[i])
        coef = np.expand_dims(coef, -1) * tmp.reshape((1,) * i + tmp.shape)
    return torch.tensor(coef)

def _coef_and_indices(
    tm, base_offset, offset, tables, centers, table_oversamp, grid_size, conj=False
):
    dtype, device = tables[0].dtype, tm.device
    grid_ind = base_offset + offset.unsqueeze(1)
    dist_ind = torch.round((tm - grid_ind.to(tm)) * table_oversamp.unsqueeze(1)).long()

    arr_ind = torch.zeros(tm.shape[1], dtype=torch.long, device=device)
    coef = torch.ones(tm.shape[1], dtype=dtype, device=device)
    for d, (table, dist, center, gind, size) in enumerate(
        zip(tables, dist_ind, centers, grid_ind, grid_size)
    ):
        t = table[dist + center]
        coef = coef * (t.conj() if conj else t)
        arr_ind = arr_ind + torch.remainder(gind, size).view(-1) * torch.prod(grid_size[d + 1 :])
    return coef, arr_ind


def _table_interp_one(image, omega, tables, n_shift, numpoints, table_oversamp, offsets):
    grid_size = torch.tensor(image.shape[2:], dtype=torch.long, device=image.device)
    tm = omega / (2 * np.pi / grid_size.to(omega).unsqueeze(-1))
    centers = torch.floor(numpoints * table_oversamp / 2).long()
    base_offset = 1 + torch.floor(tm - numpoints.unsqueeze(-1) / 2.0).long()

    flat_image = image.reshape(*image.shape[:2], -1)
    kdat = torch.zeros(*image.shape[:2], tm.shape[-1], dtype=image.dtype, device=image.device)
    for offset in offsets:
        coef, arr_ind = _coef_and_indices(
            tm, base_offset, offset, tables, centers, table_oversamp, grid_size
        )
        kdat = kdat + coef * flat_image[:, :, arr_ind]

    phase = torch.exp(1j * torch.sum(omega * n_shift.unsqueeze(-1), dim=-2, keepdim=True))
    return kdat * phase


def _table_interp(image, omega, tables, n_shift, numpoints, table_oversamp, offsets):
    if omega.ndim == 3 and omega.shape[0] == 1:
        omega = omega[0]
    if omega.ndim == 3:
        if omega.shape[0] != image.shape[0]:
            raise ValueError("omega batch dimension must match image.")
        return torch.cat(
            [
                _table_interp_one(
                    img.unsqueeze(0), om, tables, n_shift, numpoints, table_oversamp, offsets
                )
                for img, om in zip(image, omega)
            ]
        )
    return _table_interp_one(image, omega, tables, n_shift, numpoints, table_oversamp, offsets)


def _table_interp_adjoint_one(
    data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
):
    tm = omega / (2 * np.pi / grid_size.to(omega).unsqueeze(-1))
    centers = torch.floor(numpoints * table_oversamp / 2).long()
    base_offset = 1 + torch.floor(tm - numpoints.unsqueeze(-1) / 2.0).long()

    image = torch.zeros(
        data.shape[0],
        data.shape[1],
        int(torch.prod(grid_size)),
        dtype=data.dtype,
        device=data.device,
    )
    phase = torch.exp(1j * torch.sum(omega * n_shift.unsqueeze(-1), dim=-2, keepdim=True))
    data = data * phase.conj()

    for offset in offsets:
        coef, arr_ind = _coef_and_indices(
            tm, base_offset, offset, tables, centers, table_oversamp, grid_size, conj=True
        )
        weighted = coef * data
        for b in range(image.shape[0]):
            image[b].index_add_(1, arr_ind, weighted[b])

    return image.reshape(data.shape[0], data.shape[1], *grid_size.tolist())


def _table_interp_adjoint(
    data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
):
    if omega.ndim == 3 and omega.shape[0] == 1:
        omega = omega[0]
    if omega.ndim == 3:
        if omega.shape[0] != data.shape[0]:
            raise ValueError("omega batch dimension must match data.")
        return torch.cat(
            [
                _table_interp_adjoint_one(
                    d.unsqueeze(0),
                    om,
                    tables,
                    n_shift,
                    numpoints,
                    table_oversamp,
                    offsets,
                    grid_size,
                )
                for d, om in zip(data, omega)
            ]
        )
    return _table_interp_adjoint_one(
        data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
    )

class _InterpForward(Function):
    @staticmethod
    def forward(ctx, image, omega, tables, n_shift, numpoints, table_oversamp, offsets):
        grid_size = torch.tensor(image.shape[2:], device=image.device)
        ctx.save_for_backward(
            omega, n_shift, numpoints, table_oversamp, offsets, grid_size, *tables
        )
        return _table_interp(image, omega, tables, n_shift, numpoints, table_oversamp, offsets)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors                       # single access
        omega, n_shift, numpoints, table_oversamp, offsets, grid_size = saved[:6]
        tables = list(saved[6:])
        grad_data = _table_interp_adjoint(
            grad_output, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
        )
        return grad_data, None, None, None, None, None, None, None


class _InterpAdjoint(Function):
    @staticmethod
    def forward(ctx, data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size):
        ctx.save_for_backward(omega, n_shift, numpoints, table_oversamp, offsets, *tables)
        return _table_interp_adjoint(
            data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
        )

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors                       # single access
        omega, n_shift, numpoints, table_oversamp, offsets = saved[:5]
        tables = list(saved[5:])
        grad_data = _table_interp(
            grad_output, omega, tables, n_shift, numpoints, table_oversamp, offsets
        )
        return grad_data, None, None, None, None, None, None, None

def _fft_and_scale(image, scaling_coef, im_size, grid_size):
    pad = []
    for gd, im in zip(grid_size.flip(0).tolist(), im_size.flip(0).tolist()):
        pad += [0, gd - im]
    padded = tnnf.pad(image * scaling_coef, pad)
    return torch.fft.fftn(padded, dim=list(range(-grid_size.numel(), 0)))


def _ifft_and_scale(image, scaling_coef, im_size, grid_size):
    ndim = grid_size.numel()
    image = torch.fft.ifftn(image, dim=list(range(-ndim, 0)), norm="forward")
    crop = (slice(None), slice(None)) + tuple(slice(0, n) for n in im_size.tolist())
    return image[crop] * scaling_coef.conj()

class _KbNufftBase(nn.Module):

    def __init__(
        self,
        im_size,
        grid_size=None,
        numpoints=6,
        n_shift=None,
        table_oversamp=2**10,
        kbwidth=2.34,
        order=0.0,
        device=None,
    ):
        super().__init__()
        im_size = tuple(im_size)
        ndim = len(im_size)
        grid_size = tuple(grid_size) if grid_size is not None else tuple(2 * d for d in im_size)
        numpoints = (numpoints,) * ndim if isinstance(numpoints, int) else tuple(numpoints)
        n_shift = tuple(d // 2 for d in im_size) if n_shift is None else tuple(n_shift)
        table_oversamp = (
            (table_oversamp,) * ndim if isinstance(table_oversamp, int) else tuple(table_oversamp)
        )
        order = (order,) * ndim if isinstance(order, (int, float)) else tuple(order)
        alpha = tuple(kbwidth * p for p in numpoints)

        tables = _build_table(im_size, grid_size, numpoints, table_oversamp, order, alpha)
        for i, table in enumerate(tables):
            self.register_buffer(f"table_{i}", table.to(dtype=torch.complex64, device=device))

        self.register_buffer("im_size", torch.tensor(im_size, dtype=torch.long, device=device))
        self.register_buffer("grid_size", torch.tensor(grid_size, dtype=torch.long, device=device))
        self.register_buffer("n_shift", torch.tensor(n_shift, dtype=torch.float32, device=device))
        self.register_buffer("numpoints", torch.tensor(numpoints, dtype=torch.long, device=device))
        self.register_buffer(
            "table_oversamp", torch.tensor(table_oversamp, dtype=torch.long, device=device)
        )
        offsets = list(itertools.product(*[range(p) for p in numpoints]))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long, device=device))

        scaling_coef = _scaling_coefs(im_size, grid_size, numpoints, alpha, order)
        self.register_buffer("scaling_coef", scaling_coef.to(dtype=torch.complex64, device=device))

    def _tables(self):
        return [getattr(self, f"table_{i}") for i in range(len(self.im_size))]


class KbNufft(_KbNufftBase):

    def forward(self, image: Tensor, omega: Tensor) -> Tensor:
        gridded = _fft_and_scale(image, self.scaling_coef, self.im_size, self.grid_size)
        return _InterpForward.apply(
            gridded,
            omega,
            self._tables(),
            self.n_shift,
            self.numpoints,
            self.table_oversamp,
            self.offsets,
        )


class KbNufftAdjoint(_KbNufftBase):

    def forward(self, data: Tensor, omega: Tensor) -> Tensor:
        gridded = _InterpAdjoint.apply(
            data,
            omega,
            self._tables(),
            self.n_shift,
            self.numpoints,
            self.table_oversamp,
            self.offsets,
            self.grid_size,
        )
        return _ifft_and_scale(gridded, self.scaling_coef, self.im_size, self.grid_size)