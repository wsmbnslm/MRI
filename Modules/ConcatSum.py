from typing import List, Optional, Union

import torch

from .Linops import LinearOperator


class ConcatOperator(LinearOperator):

    def __init__(
        self,
        operators: List[Union[torch.Tensor, LinearOperator]],
        device: Optional[torch.device] = None,
    ):
        self.dtype = torch.float32
        self.device = device
        self.operators = operators

        self.shape = [0, self.operators[0].shape[1]]

        for operator in self.operators:
            if self.shape[1] != operator.shape[1]:
                raise ValueError("All operators must have the same number of columns.")
            else:
                self.shape[0] += operator.shape[0]

        self.shape = (int(self.shape[0]), int(self.shape[1]))

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

        y = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)

        for index, operator in zip(self.indices, self.operators):
            y[index] = operator @ x

        return y

    def _rmatvec(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y = torch.zeros(self.shape[1], dtype=self.dtype, device=self.device)

        for index, operator in zip(self.indices, self.operators):
            y += operator.H @ x[index]

        return y


class SumOperator(LinearOperator):

    def __init__(
        self,
        operators: List[Union[torch.Tensor, LinearOperator]],
        device: Optional[torch.device] = None,
    ):

        self.dtype = torch.float32
        self.device = device
        self.operators = operators

        self.shape = self.operators[0].shape
        for operator in self.operators:
            if self.shape != operator.shape:
                raise ValueError("All operators must have the same shape.")

    def _matvec(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)

        for operator in self.operators:
            y += operator @ x

        return y

    def _rmatvec(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y = torch.zeros(self.shape[1], dtype=self.dtype, device=self.device)

        # Apply the adjoint of each operator and sum the results
        for operator in self.operators:
            y += operator.H @ x

        return y
