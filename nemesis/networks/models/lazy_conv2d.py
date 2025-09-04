from torch import nn, Tensor


class LazyConv2d(nn.Module):
    def __init__(
        self,
        out_channels: int | str,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: str | int = 0,
        groups: int | str = 1,
        bias: bool = True,
        device: str | None = None,
    ) -> None:
        super().__init__()

        self.conv: nn.Module | None = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        if self.conv is None:
            input_channels = x.shape[1]

            out_channels: int

            if self.out_channels == "input":
                out_channels = input_channels
            elif isinstance(self.out_channels, int):
                out_channels = self.out_channels
            else:
                raise ValueError(self.out_channels)

            groups: int

            if self.groups == "input":
                groups = input_channels
            elif isinstance(self.groups, int):
                groups = self.groups
            else:
                raise ValueError(self.groups)

            self.conv = nn.Conv2d(
                input_channels,
                out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                groups,
                self.bias,
                device=self.device,
            )

        out: Tensor = self.conv(x)

        return out
