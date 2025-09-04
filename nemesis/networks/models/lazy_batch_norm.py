from torch import nn, Tensor


class LazyBatchNorm(nn.Module):
    def __init__(
        self,
        out_channels: int | str,
        device: str | None = None,
    ):
        super().__init__()

        self.layer: nn.Module | None = None
        self.out_channels = out_channels
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        if self.layer is None:
            input_channels = x.shape[1]

            out_channels: int

            if self.out_channels == "input":
                out_channels = input_channels
            elif isinstance(self.out_channels, int):
                out_channels = self.out_channels
            else:
                raise ValueError(self.out_channels)

            self.layer = nn.BatchNorm2d(out_channels, device=self.device)

            self.layer.eval()

        out: Tensor = self.layer(x)

        return out
