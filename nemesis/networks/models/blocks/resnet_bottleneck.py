from torch import Tensor, nn

from ..conv_bn_act import ConvBNAct


class ResNetBottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        activation: nn.Module | None = nn.ReLU(inplace=False),
    ):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.activation = activation

        self.net: nn.Module | None = None
        self.skip: nn.Module | None = None

    def forward(self, x: Tensor) -> Tensor:
        input_channels = x.shape[1]

        if self.net is None:
            self.net = nn.Sequential(
                ConvBNAct(
                    input_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=self.bias,
                    device=x.device,
                ),
                ConvBNAct(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=self.dilation,
                    bias=self.bias,
                    device=x.device,
                ),
                ConvBNAct(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=self.bias,
                    activation=None,
                    device=x.device,
                ),
            )

        residual: Tensor

        if self.stride > 1 or input_channels != self.out_channels:
            if self.skip is None:
                self.skip = ConvBNAct(
                    input_channels,
                    self.out_channels,
                    kernel_size=1,
                    activation=None,
                    stride=self.stride,
                    bias=self.bias,
                    device=x.device,
                )

            residual = self.skip(x)
        else:
            residual = x

        out: Tensor = self.net(x)

        out += residual

        if self.activation is not None:
            out = self.activation(out)

        return out
