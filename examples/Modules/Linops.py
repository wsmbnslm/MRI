import torch
import math
from typing import Optional, Tuple
from scipy.sparse.linalg import LinearOperator

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