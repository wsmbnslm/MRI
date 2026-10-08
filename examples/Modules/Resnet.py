from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .ModelconvLayer import ConvLayer, DoubleConv


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
        features=16,
        num_of_resblocks=4,
        scale_factor: int = 1,
        kernel_size: Tuple[int] = (3, 3, 3),
        activation="ReLU",
        timing_level=0,
        validation_level=0,
        device=None,
        ResNetCheckpoints=False,
        dtype=torch.complex64,
    ):

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

class DeepADMM(nn.Module):

    def __init__(self, shape, rho_init=0.1, maxiter_admm=10, maxiter_cg=5):

        super().__init__()
        self.rho = nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))
        self.regularizer = Regularizer(shape)
        self.I = IdentityOperator(shape)
        self.maxiter_admm = maxiter_admm
        self.maxiter_cg = maxiter_cg

    def forward(self,Q,b,x, disable_progressbar=True):
        z = torch.zeros(self.I.shape[0], dtype=x.dtype)
        u = torch.zeros(self.I.shape[0], dtype=x.dtype)

        for iteration in range(self.maxiter_admm):
    
            x = conjugate_gradient(
                Q + self.rho * self.I,
                b=b + self.rho * (z - u),
                x=x,
                maxiter=self.maxiter_cg,
                disable_progressbar=True,
            )
    
            z = self.regularizer(x + u)
            u = u + x - z
    
        return x