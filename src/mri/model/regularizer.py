import math
from typing import Tuple

import torch
from torch import nn

from .resnet import ResNet
from .unet import UNet


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
        """
        Initializes a Regularizer. For now it is possible to initialize a ResNet and a UNet.

        Parameters
        ----------
        shape : tuple of int, (nX, nY, nZ, *contrast_dims)
            Shape of the image data. The first three entries are always the spatial
            dimensions — set nZ = 1 for a 2D dataset. Any number of additional
            trailing dimensions (e.g. just nT, or nTI and nTE, or none at all) are
            treated as "contrast" axes and flattened into the network's channel
            dimension. Examples:
              - (nX, nY, 1)                -> single image, no contrast axis
              - (nX, nY, nZ)               -> 3D volume, no contrast axis
              - (nX, nY, 1, nTI, nTE)      -> 2D, two contrast axes (old behaviour)
              - (nX, nY, 1, nT)            -> 2D, one contrast axis
        features : int, optional
            Number of features of the neural network (default is 32).
        regularizer : str, optional
            Which regularizer to use: "ResNet" or "UNet" (default is "ResNet").
        activation : str, optional
            Activation function name (default is "ReLU").
        kernel_size : tuple of int, optional
            Convolution kernel size; its length decides whether operations are 2D
            or 3D (default is (3, 3)).
        num_of_resblocks : int, optional
            Number of ResNet blocks (default is 10).
        Checkpoints : bool, optional
            If True, use gradient checkpointing to trade compute for memory
            (default is False).
        device : str, optional
            Device for computation (default is None, current device).

        NOTE: This function is under development and may not be fully functional yet.
        """
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

        # flatten all trailing contrast dims into one channel axis: [nX,nY,nZ,C]
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