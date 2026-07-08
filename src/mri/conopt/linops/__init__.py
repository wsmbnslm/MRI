import torch

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