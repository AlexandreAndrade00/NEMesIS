from torch import nn, Tensor

from ..lazy_adaptive_avg_pool_2d import LazyAdaptiveAvgPool2d
from ..lazy_linear import LazyLinear


class ClassificationHead(nn.Module):
    def __init__(self, number_classes: int, device: str | None = None):
        super().__init__()
        self.pool = LazyAdaptiveAvgPool2d(1)
        self.fc = LazyLinear(number_classes, bias=False, device=device)
        self.softmax = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.pool(x)
        out = self.fc(out)
        out = self.softmax(out)

        return out
