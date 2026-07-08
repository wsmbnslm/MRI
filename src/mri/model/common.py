import torch
from torch import nn


class ComplexActivation(nn.Module):
    def __init__(self, activation):
        """
        Initializes a complex activation function. Its necessary because
        the original torch activation functions cant handle complex values.

        Parameters
        ----------
        activation: str
            Defines which activation function should be initialized.
            Currently supported: ReLU, ELU, Identity (default is ReLU)
        """
        super().__init__()
        match activation:
            case "ReLU":
                self.functional = nn.ReLU()

            case "ELU":
                self.functional = nn.ELU()

            case "Identity":
                self.functional = nn.Identity()

            case _:
                raise ValueError(f"activation function {activation} is unknown.")

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return torch.complex(
            self.functional(images.real),
            self.functional(images.imag),
        )


class ConvTransposeLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: tuple[int] = (
            3,
            3,
        ),
        padding=1,
        stride: int = 1,
        bias=False,
        activation: str = "ReLU",
        device=None,
        dtype=torch.complex64,
    ):
        """
        Initializes a Convolutional Transpose Layer. Its dimension will be determined
        by the length of the kernel_size. The other properties
        are the same for 2D and 3D.

        Parameters
        ----------
        in_channles : int
            number of features in the input image (contrasts).
        out_channels: int
            number of features produced by the convolution.
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions and defines
            whether a 2D or 3D ConvLayer should be initialized.
            dim = len(kernel_size); (default is (3,3)).
        padding: int, optional
            The Padding thats added to all four sides of the input (default is 1).
        stride: int, optional
            defines in how many pixels every pixel of the input image should be
            split. A stride of 2 will double the image shape. Sometimes it is
            necessary to finetune some padding operations to get the correct size.
        bias: bool, optional
            If True it will add a learnable bias to the output (default is False)
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).

        NOTE: This function is under development and may not be fully functional yet.
        """

        super().__init__()

        if len(kernel_size) == 2:
            self.convtranspose = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
                device=device,
                dtype=dtype,
            )

        elif len(kernel_size) == 3:
            self.convtranspose = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
                device=device,
                dtype=dtype,
            )

        self.activation = ComplexActivation(activation)

    def forward(self, image, target_size):
        image = self.convtranspose(image)
        image = self.activation(image)

        return image


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: tuple[int] = (
            3,
            3,
        ),
        padding=1,
        stride: int = 1,
        bias=False,
        activation="ReLU",
        device=None,
        dtype=torch.complex64,
    ):
        """
        Initializes a Convolutional Layer. Its dimension will be determined
        by the length of the kernel_size. The other properties
        are the same for 2D and 3D.

        Parameters
        ----------
        in_channles : int
            number of features in the input image (contrasts).
        out_channels: int
            number of features produced by the convolution.
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions and defines
            whether a 2D or 3D ConvLayer should be initialized.
            dim = len(kernel_size); (default is (3,3)).
        padding: int, optional
            The Padding thats added to all four sides of the input (default is 1).
        stride: int, optional
            defines how many pixels of the input image should be
            merged in one pixel of the output image. A stride of 2 will half the
            image shape.
        bias: bool, optional
            If True it will add a learnable bias to the output (default is False)
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).

        NOTE: This function is under development and may not be fully functional yet.
        """

        super().__init__()
        self.dtype = dtype

        if len(kernel_size) == 2:
            self.convlayer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
                device=device,
                dtype=dtype,
            )

        elif len(kernel_size) == 3:
            self.convlayer = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
                device=device,
                dtype=dtype,
            )

        self.activation = ComplexActivation(activation)

    def forward(self, image):
        image = self.convlayer(image)
        image = self.activation(image)

        return image


class DoubleConv(nn.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        kernel_size: tuple[int] = (3, 3),
        activation: list[str] = "ReLU",
        padding: int = 1,
        stride: list[int] = [1, 1],
        bias: bool = False,
        device: str = None,
        dtype=torch.complex64,
    ):
        """
        Initializes a DoubleConvolution Class and defines all its parameters used for convolution.

        Parameters
        ----------
        input_size: int
            defines the number of input features of the image.
        intermediate_size: int
            defines the number of features the image will have between the two convolutional layers.
        output_size: int
            defines the number of output features of the image.
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions and defines
            whether a 2D or 3D ConvLayer should be initialized.
            dim = len(kernel_size); (default is (3,3)).
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        stride: list[int,int]
            defines how many pixels of the input image should be
            merged in one pixel of the output image. A stride of 2 will half the
            image shape. The first input of the list will be the stride of the first layer and so on.
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).

        NOTE: This function is under development and may not be fully functional yet.
        """

        if type(activation) is str:
            activation = [activation, activation]

        super().__init__()

        self.double_conv = nn.Sequential(
            ConvLayer(
                in_channels=input_size,
                out_channels=intermediate_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride[0],
                bias=bias,
                activation=activation[0],
                device=device,
                dtype=dtype,
            ),
            ConvLayer(
                in_channels=intermediate_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride[1],
                bias=bias,
                activation=activation[1],
                device=device,
                dtype=dtype,
            ),
        )

    def forward(
        self,
        images,
    ):
        return self.double_conv(images)
