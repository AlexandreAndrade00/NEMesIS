from torch import nn, Tensor


class LazyLinear(nn.Module):
    def __init__(self, out_features: int, bias: bool = True, device: str | None = None):
        super().__init__()

        self.layer: nn.Module | None = None
        self.out_features = out_features
        self.bias = bias
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        input_tensor: Tensor = x.flatten(1)

        if self.layer is None:
            input_feats = input_tensor.shape[1]

            self.layer = nn.Linear(input_feats, self.out_features, bias=self.bias, device=self.device)

        out: Tensor = self.layer(input_tensor)

        return out
