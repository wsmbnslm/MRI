import torch
import torch.nn as nn

from .common import ConvLayer, ConvTransposeLayer, DoubleConv


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: tuple[int] = (3, 3),
        activation: str = "ReLU",
        bias: bool = False,
        device=None,
        dtype=torch.complex64,
    ):
        """
        Initializes a DownConvBlock which is used for the encoding of the unet.
        (downward pass)

        Parameter
        ---------
        in_channles : int
            number of features in the input image (contrasts).
        out_channels: int
            number of features produced by the convolution.
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions and defines
            whether a 2D or 3D ConvLayer should be initialized.
            dim = len(kernel_size); (default is (3,3)).
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).
        """
        super().__init__()
        self.down = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.conv = DoubleConv(
            out_channels,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            activation=activation,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        images,
    ):
        images = self.down(images)
        images = self.conv(images)
        return images


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: tuple[int] = (2, 2),
        activation: str = "ReLU",
        device=None,
        dtype=torch.complex64,
    ):
        """
        Initializes an UpConvBlock which is used for the decoding of the unet.
        (upward pass)

        Parameter
        ---------
        in_channles : int
            number of features in the input image (contrasts).
        out_channels: int
            number of features produced by the convolution.
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions and defines
            whether a 2D or 3D ConvLayer should be initialized.
            dim = len(kernel_size); (default is (3,3)).
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).
        """
        super().__init__()
        self.up = ConvTransposeLayer(
            in_channels,
            out_channels,
            kernel_size=tuple(4 for _ in range(len(kernel_size))),
            stride=2,
            device=device,
            dtype=dtype,
        )
        self.conv = DoubleConv(
            2 * out_channels,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            activation=activation,
            device=device,
            dtype=dtype,
        )  # *2 for skip connection

    def forward(
        self,
        images,
        skip,
    ):
        images = self.up(images, images.shape[0] * 2)
        images = torch.cat([images, skip], dim=1)  # Concatenate skip connection
        images = self.conv(images)

        return images


class UNet(nn.Module):
    def __init__(
        self,
        contrasts: int = 1,
        features: list[int] = [16, 32, 64],
        kernel_size: tuple[int] = (3, 3),
        activation: str = "ReLU",
        device: str = None,
        dtype=torch.complex64,
    ):
        """
        Initializes a UNet as regularizer.

        Parameter
        ---------
        contrasts: int, optional
            defines the number of contrasts which the UNet should take into account
            (default is 1).
        features: list[int], optional
            defines the number of layers in the unet and their number of features.
            (default is [16,32,64]).
        kernel_size: Tuple[int], optional
            changes the size of the kernel used in the convolutional layers and  its length
            decides whether all operations should be 2D or 3D. (default is (3,3))
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.dtype = dtype
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Initial Conv
        self.init_conv = ConvLayer(
            contrasts,
            features[0],
            kernel_size=kernel_size,
            activation=activation,
            device=device,
            dtype=dtype,
        )

        # Encoder (Down Path)
        for i in range(len(features)):
            self.encoder.append(
                DoubleConv(
                    features[i],
                    features[i],
                    features[i],
                    kernel_size=kernel_size,
                    activation=activation,
                    device=device,
                    dtype=dtype,
                ),
            )

        # Down Conv blocks for downsampling
        self.down_convs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_convs.append(
                DownConvBlock(
                    features[i],
                    features[i + 1],
                    activation=activation,
                    kernel_size=kernel_size,
                    device=device,
                    dtype=dtype,
                ),
            )

        # Bottleneck
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1],
            features[-1],
            activation=activation,
            kernel_size=kernel_size,
            device=device,
            dtype=dtype,
        )

        # Decoder (Up Path)
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(
                UpConvBlock(
                    features[i],
                    features[i - 1],
                    kernel_size=kernel_size,
                    activation=activation,
                    device=device,
                    dtype=dtype,
                ),
            )

        # Final Conv
        self.final_conv = ConvLayer(
            features[0],
            contrasts,
            kernel_size=kernel_size,
            activation=activation,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        images,
    ):
        skips = []

        # Initial Conv
        images = self.init_conv(images)

        # Encoder
        for i, layer in enumerate(self.encoder):
            images = layer(images)
            if i < len(self.encoder) - 1:  # Skip the last layer (bottleneck input)
                skips.append(images)
                images = self.down_convs[i](images)  # Down Conv for downsampling

        # Bottleneck
        images = self.bottleneck(images)

        # Decoder
        skips = skips[::-1]  # Reverse for upsampling
        for i, layer in enumerate(self.decoder):
            images = layer(images, skips[i])

        # Final Conv
        images = self.final_conv(images)

        return images
