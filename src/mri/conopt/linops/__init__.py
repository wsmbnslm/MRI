import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch


class LinearOperator:
    """
    A common interface for performing matrix-vector products using PyTorch.

    This class provides methods for matrix-vector and matrix-matrix products, along with
    operator overloading for common linear algebra operations. Subclasses can implement
    specific behaviors for custom matrix-like objects.

    This class is inspired by the `scipy.sparse.linalg.LinearOperator` class, with numpy
    calls replaced by their PyTorch equivalents.
    """

    ndim = 2

    def __new__(cls, *args, **kwargs) -> "LinearOperator":
        if cls is LinearOperator:
            return super(LinearOperator, cls).__new__(_CustomLinearOperator)
        else:
            obj = super(LinearOperator, cls).__new__(cls)
            if (
                type(obj)._matvec == LinearOperator._matvec
                and type(obj)._matmat == LinearOperator._matmat
            ):
                warnings.warn(
                    "LinearOperator subclass should implement at least one of "
                    "_matvec or _matmat.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            return obj

    def __init__(self, dtype: torch.dtype, shape: Tuple[int, int]) -> None:
        """
        Initialize the LinearOperator.

        Parameters:
        ----------
        dtype : torch.dtype, optional
            Data type of the operator.
        shape : tuple of int
            Shape of the operator (must be 2D).
        """
        if dtype is None:
            dtype = torch.float32

        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"invalid shape {shape} (must be 2-d)")

        self.dtype = dtype
        self.shape = shape

    # Core Matrix-Vector and Matrix-Matrix Methods
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the matrix-vector product.

        Parameters:
        ----------
        x : torch.Tensor
            Input vector for the matrix-vector multiplication.

        Returns:
        -------
        torch.Tensor
            Result of the matrix-vector multiplication.
        """
        M, N = self.shape
        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError("dimension mismatch")
        y = self._matvec(x)
        if x.ndim == 1:
            y = y.view(M)
        elif x.ndim == 2:
            y = y.view(M, 1)
        else:
            raise ValueError("invalid shape returned by user-defined matvec()")
        return y

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the adjoint matrix-vector product.

        Parameters:
        ----------
        x : torch.Tensor
            Input vector for the adjoint matrix-vector multiplication.

        Returns:
        -------
        torch.Tensor
            Result of the adjoint matrix-vector multiplication.
        """
        M, N = self.shape
        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError("dimension mismatch")
        y = self._rmatvec(x)
        if x.ndim == 1:
            y = y.view(N)
        elif x.ndim == 2:
            y = y.view(N, 1)
        else:
            raise ValueError("invalid shape returned by user-defined rmatvec()")
        return y

    def matmat(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform the matrix-matrix product.

        Parameters:
        ----------
        X : torch.Tensor
            Input matrix for the matrix-matrix multiplication.

        Returns:
        -------
        torch.Tensor
            Result of the matrix-matrix multiplication.
        """
        if X.ndim != 2:
            raise ValueError(f"expected 2-d tensor, got {X.ndim}-d")
        if X.shape[0] != self.shape[1]:
            raise ValueError(f"dimension mismatch: {self.shape}, {X.shape}")
        return self._matmat(X)

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.matmat(x.view(-1, 1))

    def _matmat(self, X: torch.Tensor) -> torch.Tensor:
        return torch.hstack([self.matvec(col.view(-1, 1)) for col in X.T])

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.H.matvec(x)

    # Operator Overloading
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the matrix-vector or matrix-matrix multiplication.
        """
        return self * x

    def __matmul__(
        self, other: Union[torch.Tensor, "LinearOperator"]
    ) -> Union[torch.Tensor, "LinearOperator"]:
        """
        Overload the @ operator for matrix-matrix multiplication.
        """
        if isinstance(other, LinearOperator):
            return _ProductLinearOperator(self, other)
        elif torch.is_tensor(other):
            return self.dot(other)
        else:
            raise ValueError(
                f"unsupported operand type(s) for @: '{type(self)}' and '{type(other)}'"
            )

    def __rmatmul__(
        self, other: Union[torch.Tensor, "LinearOperator"]
    ) -> Union[torch.Tensor, "LinearOperator"]:
        """
        Overload the right-hand @ operator for matrix-matrix multiplication.
        """
        if isinstance(other, LinearOperator):
            return _ProductLinearOperator(other, self)
        elif torch.is_tensor(other):
            return other.dot(self)
        else:
            raise ValueError(
                f"unsupported operand type(s) for @: '{type(other)}' and '{type(self)}'"
            )

    def dot(
        self, x: Union[torch.Tensor, "LinearOperator"]
    ) -> Union[torch.Tensor, "LinearOperator"]:
        """
        Compute the dot product between the operator and a tensor or another operator.

        Parameters:
        ----------
        x : torch.Tensor or LinearOperator
            Input tensor or operator for the dot product.

        Returns:
        -------
        torch.Tensor or LinearOperator
            Result of the dot product or operator product.
        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif torch.is_tensor(x):
            if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError(f"expected 1-d or 2-d tensor, got {x}")
        else:
            raise ValueError(f"expected a tensor, got {type(x)}")

    # Adjoint and Transpose
    def adjoint(self) -> "LinearOperator":
        """
        Return the Hermitian adjoint (conjugate transpose) of the operator.
        """
        return _AdjointLinearOperator(self)

    H = property(adjoint)

    def transpose(self) -> "LinearOperator":
        """
        Return the transpose of the operator.
        """
        return _TransposedLinearOperator(self)

    T = property(transpose)

    # Miscellaneous Methods
    def __repr__(self) -> str:
        """
        Return a string representation of the operator, showing its shape and data type.
        """
        M, N = self.shape
        dt = "unspecified dtype" if self.dtype is None else f"dtype={self.dtype}"
        return f"<{M}x{N} {self.__class__.__name__} with {dt}>"

    def __neg__(self) -> "LinearOperator":
        """
        Return the negation of the operator (-A).
        """
        return _ScaledLinearOperator(self, -1)

    def __add__(self, other: "LinearOperator") -> "LinearOperator":
        """
        Return the sum of two operators.
        """
        if isinstance(other, LinearOperator):
            return _SumLinearOperator(self, other)
        else:
            raise ValueError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )

    def __radd__(self, other: "LinearOperator") -> "LinearOperator":
        """
        Return the reverse sum of two operators (other + self).
        """
        if isinstance(other, LinearOperator):
            return _SumLinearOperator(other, self)
        else:
            raise ValueError(
                f"unsupported operand type(s) for +: '{type(other)}' and '{type(self)}'"
            )

    def __mul__(
        self, scalar: Union[torch.Tensor, int, float, complex]
    ) -> "LinearOperator":
        """
        Return the operator scaled by a scalar (A * scalar).

        Parameters:
        ----------
        scalar : torch.Tensor, int, float, complex
            The scalar value to multiply the operator by.

        Returns:
        -------
        LinearOperator
            The scaled linear operator.
        """
        if torch.is_tensor(scalar) or isinstance(scalar, (int, float, complex)):
            return _ScaledLinearOperator(self, scalar)
        raise ValueError(
            f"unsupported operand type(s) for *: '{type(self)}' and '{type(scalar)}'"
        )

    def __rmul__(
        self, scalar: Union[torch.Tensor, int, float, complex]
    ) -> "LinearOperator":
        """
        Return the operator scaled by a scalar (scalar * A).

        This method is called when the scalar is on the left of the multiplication.

        Parameters:
        ----------
        scalar : torch.Tensor, int, float, complex
            The scalar value to multiply the operator by.

        Returns:
        -------
        LinearOperator
            The scaled linear operator.
        """
        return self.__mul__(scalar)


class _ScaledLinearOperator(LinearOperator):
    """
    A linear operator scaled by a scalar.
    """

    def __init__(
        self, A: LinearOperator, scalar: Union[torch.Tensor, int, float, complex]
    ) -> None:
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.scalar = scalar

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.scalar * self.A.matvec(x)

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.scalar * self.A.rmatvec(x)

    def _matmat(self, X: torch.Tensor) -> torch.Tensor:
        return self.scalar * self.A.matmat(X)


class _CustomLinearOperator(LinearOperator):
    """
    A linear operator defined by user-specified functions for matrix-vector and
    matrix-matrix products.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        matvec: Callable[[torch.Tensor], torch.Tensor],
        rmatvec: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        matmat: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(dtype, shape)
        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec
        self.__matmat_impl = matmat

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.__matvec_impl(x)

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        if self.__rmatvec_impl is None:
            raise NotImplementedError("rmatvec is not defined")
        return self.__rmatvec_impl(x)

    def _matmat(self, X: torch.Tensor) -> torch.Tensor:
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)
        else:
            return super()._matmat(X)


class _AdjointLinearOperator(LinearOperator):
    """
    Represents the adjoint (Hermitian conjugate) of a linear operator.
    """

    def __init__(self, A: LinearOperator) -> None:
        shape = (A.shape[1], A.shape[0])
        super().__init__(dtype=A.dtype, shape=shape)
        self.A = A

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A._rmatvec(x)

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A._matvec(x)

    def _matmat(self, X: torch.Tensor) -> torch.Tensor:
        return self.A._rmatmat(X)


class _TransposedLinearOperator(LinearOperator):
    """
    Represents the transpose of a linear operator.
    """

    def __init__(self, A: LinearOperator) -> None:
        shape = (A.shape[1], A.shape[0])
        super().__init__(dtype=A.dtype, shape=shape)
        self.A = A

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.A._rmatvec(torch.conj(x)))

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.A._matvec(torch.conj(x)))

    def _matmat(self, X: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.A._rmatmat(torch.conj(X)))


class _ProductLinearOperator(LinearOperator):
    """
    Represents the product of two linear operators.
    """

    def __init__(self, A: LinearOperator, B: LinearOperator) -> None:
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"cannot multiply {A.shape} and {B.shape}: shape mismatch")
        super().__init__(_get_dtype([A, B]), (A.shape[0], B.shape[1]))
        self.A = A
        self.B = B

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.matvec(self.B.matvec(x))

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.B.rmatvec(self.A.rmatvec(x))

    def _matmat(self, X: torch.Tensor) -> torch.Tensor:
        return self.A.matmat(self.B.matmat(X))


class _SumLinearOperator(LinearOperator):
    """
    Represents the sum of two linear operators.
    """

    def __init__(self, A: LinearOperator, B: LinearOperator) -> None:
        if A.shape != B.shape:
            raise ValueError(f"cannot add {A.shape} and {B.shape}: shape mismatch")
        super().__init__(_get_dtype([A, B]), A.shape)
        self.A = A
        self.B = B

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.matvec(x) + self.B.matvec(x)

    def _rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.rmatvec(x) + self.B.rmatvec(x)

    def _matmat(self, X: torch.Tensor) -> torch.Tensor:
        return self.A.matmat(X) + self.B.matmat(X)


def _get_dtype(operators: List[LinearOperator]) -> torch.dtype:
    """
    Get the common data type of a list of operators.

    Parameters:
    ----------
    operators : list of LinearOperator
        List of linear operators to analyze.

    Returns:
    -------
    torch.dtype
        The promoted data type of all the operators.
    """
    dtypes = [op.dtype for op in operators if op is not None and hasattr(op, "dtype")]
    return torch.promote_types(*dtypes)
