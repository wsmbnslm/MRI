import math
from typing import Tuple

import torch
from torch import nn

from .Resnet import ResNet
from .Unet import UNet


class Regularizer(nn.Module):
    def __init__(
        self,
        shape: Tuple[int, ...],
        regularizer: str = "ResNet",
        features: int = 32,
        activation: str = "ReLU",
        kernel_size: Tuple[int, ...] = (3, 3),
        scale_factor: int = 0,
        num_of_resblocks: int = 10,
        Checkpoints: bool = False,
        device: str = None,
        dtype=torch.complex64,
    ):

        super().__init__()

        if len(shape) < 3:
            raise ValueError(f"shape must be at least (nX, nY, nZ), got {shape}")

        self.shape = tuple(shape)
        self.spatial_shape = tuple(shape[:3])   # (nX, nY, nZ)
        self.contrast_shape = tuple(shape[3:])  # (), (nT,), (nTI, nTE), ...
        self.contrasts = math.prod(self.contrast_shape) if self.contrast_shape else 1

        self.kernel_size = kernel_size
        self.internal_dtype = torch.complex64
        self.dtype = torch.float32

        if regularizer == "ResNet":
            self.regularizer = ResNet(
                contrasts=self.contrasts,
                features=features,
                num_of_resblocks=num_of_resblocks,
                activation=activation,
                scale_factor=scale_factor,
                kernel_size=kernel_size,
                ResNetCheckpoints=Checkpoints,
                device=device,
                dtype=dtype,
            )
        elif regularizer == "UNet":
            self.regularizer = UNet(
                contrasts=self.contrasts,
                features=features,
                kernel_size=kernel_size,
                activation=activation,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"unknown regularizer type: {regularizer!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nX, nY, nZ = self.spatial_shape

        image = torch.view_as_complex(x.reshape(-1, 2)).reshape(self.shape)

        image = image.reshape(nX, nY, nZ, self.contrasts)
        image = image.permute(3, 0, 1, 2)   # [C, nX, nY, nZ]
        image = image[None]                  # [1, C, nX, nY, nZ]

        if len(self.kernel_size) == 2:
            if nZ != 1:
                raise ValueError(
                    "z-dimension will be killed when using a 2D kernel on a 3D dataset"
                )
            image = image[..., 0]            # [1, C, nX, nY]

        image = self.regularizer(image)

        if len(self.kernel_size) == 2:
            image = image[..., None]         # [1, C, nX, nY, nZ]

        image = image[0]                     # [C, nX, nY, nZ]
        image = image.permute(1, 2, 3, 0)    # [nX, nY, nZ, C]
        image = image.reshape(*self.spatial_shape, *self.contrast_shape)
        image = image.contiguous()

        return torch.view_as_real(image.reshape(-1)).reshape(-1)