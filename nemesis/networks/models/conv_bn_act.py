from typing import Optional, Union

import torch
from torch import nn, Tensor


class ConvBNAct(nn.Module):
    """
    Regular Convolution-BatchNormalization-Activation layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[Union[str, int]] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        batch_normalization: bool = True,
        activation: nn.Module | None = nn.ReLU(inplace=False),
        device: torch.device | None = None,
    ):
        super(ConvBNAct, self).__init__()

        if activation is not None and device is not None:
            activation = activation.to(device=device)

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=kernel_size // 2 if padding is None else padding,
            bias=bias,
            device=device,
        )

        self.net: nn.Module

        if not batch_normalization and activation is None:
            self.net = conv
        elif activation is None:
            bn = nn.BatchNorm2d(
                out_channels,
                device=device,
            )

            self.net = nn.Sequential(conv, bn)
        else:
            bn = nn.BatchNorm2d(
                out_channels,
                device=device,
            )

            self.net = nn.Sequential(conv, bn, activation)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)

        return out
