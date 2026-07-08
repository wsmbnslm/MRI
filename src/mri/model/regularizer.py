import torch
from torch import nn

from .resnet import ResNet
from .unet import UNet


class Regularizer(nn.Module):
    def __init__(
        self,
        shape: tuple[int],
        regularizer: str = "ResNet",
        features: list = 32,
        activation: str = "ReLU",
        kernel_size: tuple[int] = (3, 3),
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
        shape : torch.Tensor, shape (nX, nY, nZ, nTI, nTE)
            Shape of the image data.
        features : int, optional
            Number of the features of the neural network (default is 128).
        regularizer: str, optional
            decides which regularizer should be used. For now there are ResNet and UNet
            (default is ResNet).
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        kernel_size: Tuple[int], optional
            changes the size of the kernel used in the convolutional layers and  its length
            decides whether all operations should be 2D or 3D. (default is (3,3))
        num_of_res_blocks : int, optional
            Number of ResNetBlocks that should be added to the second layer of the
            ResNet (default is 10).
        Checkpoints: bool, optional
            If true then checkpoints will be added in the regularizer, providing lower memory
            usage to the cost of higher computing time (default is False).
        device : str, optional
            Device on which to perform the computation
            (default is None, which uses the current device).
            It is also possible to give a list of strings. The first
            item is the DataConsistency device and the second one is
            the device used for the regularizer.

        NOTE: This function is under development and may not be fully functional yet.
        """

        _, _, _, nTI, nTE = shape
        contrasts = nTI * nTE
        super().__init__()

        self.shape = shape
        self.kernel_size = kernel_size
        self.internal_dtype = torch.complex64
        self.dtype = torch.float32

        if regularizer == "ResNet":
            self.regularizer = ResNet(
                contrasts=contrasts,
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
                contrasts=contrasts,
                features=features,
                kernel_size=kernel_size,
                activation=activation,
                device=device,
                dtype=dtype,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        image = x.view(self.internal_dtype).reshape(self.shape)
    
        # image = image.clone().detach()
        image = image.contiguous()

        nTI, nTE = image.shape[-2:]

        image = image.flatten(start_dim=3, end_dim=4)  # [nX,nY,nZ,nTI*nTE]
        image = image.permute((3, 0, 1, 2))  # [nTI*nTE,nX,nY,nZ]
        image = image[None, :, :, :, :]  # [blank,nTI*nTE,nX,nY,nZ]

        if len(self.kernel_size) == 2:
            if image.shape[-1] == 1:
                image = image[:, :, :, :, 0]  # [blank,nTI*nTE,nX,nY]

            else:
                raise ValueError(
                    "z-dimension will be killed when using a 2D kernel on a 3D dataset"
                )

        image = self.regularizer(image)
        
        if len(self.kernel_size) == 2:
            image = image[:, :, :, :, None]  # [blank,nTI*nTE,nX,nY,nZ]

        image = image[0, :, :, :, :]
        image = image.unflatten(
            0, (nTI, nTE)
        )  # switches shape to [blank,nTI,nTE,nX,nY,nZ]
        image = image.permute((2, 3, 4, 0, 1))  # switches shape to [nX,nY,nZ,nTI,nTE]

        image = image.contiguous()
        
        return image.ravel().view(self.dtype)
