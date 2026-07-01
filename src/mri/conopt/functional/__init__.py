import math
from typing import Tuple

import torch


def norm(
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.abs(input_tensor) ** 2))


def crop_tensor(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Crop a tensor to the specified target shape.

    The function crops the input tensor symmetrically around the center, based on
    the provided target shape.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be cropped.
    target_shape : Tuple[int, ...]
        The target shape for the cropped tensor. Must be smaller than or equal to
        the input tensor shape in all dimensions.

    Returns
    -------
    torch.Tensor
        The cropped tensor.

    Raises
    ------
    AssertionError
        If the input and target shapes do not have the same number of dimensions or
        if the target shape is larger than the input tensor shape in any dimension.
    """
    input_shape = tensor.shape

    # Ensure the input and target shapes have the same number of dimensions
    assert len(input_shape) == len(target_shape), (
        "Input and target shapes must have the same number of dimensions."
    )

    # Ensure that the target shape is less than or equal to the input shape in all
    # dimensions
    assert all(
        target_dim <= input_dim
        for target_dim, input_dim in zip(target_shape, input_shape)
    ), "Target shape must be less than or equal to input shape in all dimensions."

    # Calculate the slice indices for cropping
    indices = [
        slice(
            int(math.floor(input_dim / 2.0) + math.ceil(-target_dim / 2.0)),
            int(math.floor(input_dim / 2.0) + math.ceil(target_dim / 2.0)),
        )
        for input_dim, target_dim in zip(input_shape, target_shape)
    ]

    return tensor[tuple(indices)]


def pad_tensor(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Pad a tensor to the specified target shape.

    The function pads the input tensor symmetrically with zeros around the center,
    based on the provided target shape.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be padded.
    target_shape : Tuple[int, ...]
        The target shape for the padded tensor. Must be greater than or equal to
        the input tensor shape in all dimensions.

    Returns
    -------
    torch.Tensor
        The padded tensor.

    Raises
    ------
    AssertionError
        If the input and target shapes do not have the same number of dimensions or
        if the target shape is smaller than the input tensor shape in any dimension.
    """
    input_shape = tensor.shape

    # Ensure the input and target shapes have the same number of dimensions
    assert len(input_shape) == len(target_shape), (
        "Input and target shapes must have the same number of dimensions."
    )

    # Ensure that the target shape is greater than or equal to the input shape in all
    # dimensions
    assert all(
        target_dim >= input_dim
        for target_dim, input_dim in zip(target_shape, input_shape)
    ), "Target shape must be greater than or equal to input shape in all dimensions."

    # Create a new tensor with the target shape, initialized to zero
    padded_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)

    # Calculate the slice indices for padding
    indices = [
        slice(
            int(math.floor(target_dim / 2.0) + math.ceil(-input_dim / 2.0)),
            int(math.floor(target_dim / 2.0) + math.ceil(input_dim / 2.0)),
        )
        for target_dim, input_dim in zip(target_shape, input_shape)
    ]

    padded_tensor[tuple(indices)] = tensor
    return padded_tensor
