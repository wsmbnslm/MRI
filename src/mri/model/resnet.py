from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .common import ConvLayer, DoubleConv


class ResNetBlocksModule(nn.Module):
    def __init__(
        self,
        features,
        kernel_size=(3, 3),
        activation="ReLU",
        num_of_resblocks=15,
        scale_factor=1,
        device=None,
        ResNetCheckpoints=False,
        dtype=torch.complex64,
    ):
        """
        Initializes a ResNetBlocksModule class that calls a variable number of
        DoubleConvs and can let them calculate in its forward pass.

        Parameters
        ----------
        features : int
            number of in and output features of the convolutions.
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions and defines
            whether a 2D or 3D ConvLayer should be initialized.
            dim = len(kernel_size); (default is (3,3)).
        num_of_resblocks : int, optional
            number of DoubleConvs that should be initialized (default is 15).
        scale_factor: float, optional
            defines the impact of the ResNetBlocks on the image (default is 0.1)
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).

        NOTE: This function is under development and may not be fully functional yet.
        """
        super().__init__()
        self.ResNetCheckpoints = ResNetCheckpoints
        self.layers = nn.ModuleList(
            [
                DoubleConv(
                    features,
                    features,
                    features,
                    kernel_size,
                    activation=activation,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_of_resblocks)
            ]
        )

        self.scale_factor = scale_factor
        self.device = device

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if self.ResNetCheckpoints:
            for i, layer in enumerate(self.layers):
                images = checkpoint(self.CalcLayer, images, layer, use_reentrant=False)

        else:
            for i, layer in enumerate(self.layers):
                images = images + self.scale_factor * layer(images)

        return images

    def CalcLayer(
        self,
        images: torch.Tensor,
        layer,
    ) -> torch.Tensor:
        return images + self.scale_factor * layer(images)


class ResNet(nn.Module):
    def __init__(
        self,
        contrasts=1,
        features=32,
        num_of_resblocks=15,
        scale_factor: int = 0,
        kernel_size: Tuple[int] = (3, 3),
        activation="ReLU",
        timing_level=0,
        validation_level=0,
        device=None,
        ResNetCheckpoints=False,
        dtype=torch.complex64,
    ):
        """
        Initializes a ResNet class with a variable number of DoubleConvs.

        Parameters
        ----------
        contrast : int, optional
            number of contrasts that should be respected (default is 1).
        features : int, optional
            number of the features of the ResNet. The more features it has the more
            things it can learn (default is 32).
        num_of_resblocks : int, optional
            number of DoubleConvs that should be initialized (default is 15).
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions and defines
            whether a 2D or 3D ConvLayer should be initialized.
            dim = len(kernel_size); (default is (3,3)).
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).

        NOTE: This function is under development and may not be fully functional yet.
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.layer1 = ConvLayer(
            contrasts,
            features,
            kernel_size,
            activation="Identity",
            device=device,
            dtype=dtype,
        )

        self.layer2 = ResNetBlocksModule(
            features,
            kernel_size,
            activation=activation,
            num_of_resblocks=num_of_resblocks,
            scale_factor=scale_factor,
            device=device,
            ResNetCheckpoints=ResNetCheckpoints,
            dtype=dtype,
        )

        self.layer3 = ConvLayer(
            features,
            features,
            kernel_size,
            activation="Identity",
            device=device,
            dtype=dtype,
        )

        self.layer4 = ConvLayer(
            features,
            contrasts,
            kernel_size,
            activation="Identity",
            device=device,
            dtype=dtype,
        )

        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device

    def forward(
        self,
        image: torch.Tensor,  # shape: [nX,nY,nZ,nTI,nTE]
    ) -> torch.Tensor:
        image = image.to(self.device)

        l1_out = self.layer1(image)
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        image = self.layer4(l3_out + l1_out)

        return image
