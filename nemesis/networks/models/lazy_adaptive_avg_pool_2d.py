from typing import Tuple

from torch import nn, Tensor


class LazyAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: int | None | Tuple[int | None, int | None]):
        super().__init__()
        self.output_size = output_size
        self.layer: nn.Module | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.layer is None:
            self.layer = nn.AdaptiveAvgPool2d(self.output_size)

        out: Tensor = self.layer(x)

        return out
