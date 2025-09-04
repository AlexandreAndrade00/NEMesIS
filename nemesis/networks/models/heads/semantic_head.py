from torch import nn, Tensor

from ..lazy_conv2d import LazyConv2d


class SemanticHead(nn.Module):
    def __init__(self, number_classes: int, device: str | None = None):
        super().__init__()

        self.layer: nn.Module = LazyConv2d(number_classes, kernel_size=1, bias=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.layer(x)

        return out
