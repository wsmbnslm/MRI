from typing import List, Optional, Union

import torch

from . import LinearOperator


class ConcatOperator(LinearOperator):
    """
    Concatenates a list of linear operators vertically (stacking them).
    """

    def __init__(
        self,
        operators: List[Union[torch.Tensor, LinearOperator]],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the ConcatOperator.

        Parameters:
        ----------
        operators : list of torch.Tensor or LinearOperator
            List of linear operators to concatenate.
        """
        self.dtype = torch.float32
        self.device = device
        self.operators = operators

        # Initialize the shape of the concatenated operator
        self.shape = [0, self.operators[0].shape[1]]

        # Check that all operators have the same number of columns and set the final
        # shape
        for operator in self.operators:
            if self.shape[1] != operator.shape[1]:
                raise ValueError("All operators must have the same number of columns.")
            else:
                self.shape[0] += operator.shape[0]

        self.shape = (int(self.shape[0]), int(self.shape[1]))

        # Create indices for concatenation
        indices = torch.cumsum(
            torch.tensor([0] + [op.shape[0] for op in self.operators]), dim=0
        )
        self.indices = [
            slice(indices[i], indices[i + 1]) for i in range(len(indices) - 1)
        ]

    def _matvec(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform matrix-vector multiplication with the concatenated operator.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to be multiplied.

        Returns:
        -------
        torch.Tensor
            The result of the matrix-vector multiplication.
        """
        y = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)

        # Apply each operator to the corresponding part of x
        for index, operator in zip(self.indices, self.operators):
            y[index] = operator @ x

        return y

    def _rmatvec(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adjoint matrix-vector multiplication (Hermitian conjugate).

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor for the adjoint operation.

        Returns:
        -------
        torch.Tensor
            The result of the adjoint matrix-vector multiplication.
        """
        y = torch.zeros(self.shape[1], dtype=self.dtype, device=self.device)

        # Apply the adjoint of each operator to the corresponding part of x
        for index, operator in zip(self.indices, self.operators):
            y += operator.H @ x[index]

        return y


class SumOperator(LinearOperator):
    """
    Represents the sum of a list of linear operators.
    """

    def __init__(
        self,
        operators: List[Union[torch.Tensor, LinearOperator]],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the SumOperator.

        Parameters:
        ----------
        operators : list of torch.Tensor or LinearOperator
            List of linear operators to sum.
        """
        self.dtype = torch.float32
        self.device = device
        self.operators = operators

        # Ensure all operators have the same shape
        self.shape = self.operators[0].shape
        for operator in self.operators:
            if self.shape != operator.shape:
                raise ValueError("All operators must have the same shape.")

    def _matvec(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform matrix-vector multiplication with the sum of operators.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to be multiplied.

        Returns:
        -------
        torch.Tensor
            The result of the matrix-vector multiplication.
        """
        y = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)

        # Apply each operator and sum the results
        for operator in self.operators:
            y += operator @ x

        return y

    def _rmatvec(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adjoint matrix-vector multiplication (Hermitian conjugate).

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor for the adjoint operation.

        Returns:
        -------
        torch.Tensor
            The result of the adjoint matrix-vector multiplication.
        """
        y = torch.zeros(self.shape[1], dtype=self.dtype, device=self.device)

        # Apply the adjoint of each operator and sum the results
        for operator in self.operators:
            y += operator.H @ x

        return y
