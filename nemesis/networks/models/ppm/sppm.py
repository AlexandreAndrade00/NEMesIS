from typing import Tuple, Callable

import torch
from torch import nn, Tensor, Size
import torch.nn.functional as F

from ..conv_bn_act import ConvBNAct


class SPPM(nn.Module):
    """
    Simple Pyramid Pooling Module.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(
        self,
        inter_channels: int,
        out_channels: int,
        bin_sizes: Tuple[int, ...] = (1, 3),
    ):
        super().__init__()

        self.inter_channels = inter_channels
        self.bin_sizes = bin_sizes

        self.stages: nn.ModuleList | None = None

        self.conv_out: Callable[[Tensor], Tensor] = ConvBNAct(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def _make_stage(self, in_channels: int, out_channels: int, size: int, device: torch.device) -> nn.Module:
        prior = nn.AdaptiveAvgPool2d(output_size=size).to(device=device)

        conv = (
            ConvBNAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="valid",
            )
            .to(device=device)
            .eval()
        )

        return nn.Sequential(prior, conv).to(device=device)

    def forward(self, input: Tensor) -> Tensor:
        out: Tensor | None = None
        input_shape: Size = input.shape[2:]
        in_channels: int = input.shape[1]

        if self.stages is None:
            self.stages = nn.ModuleList(
                [self._make_stage(in_channels, self.inter_channels, size, input.device) for size in self.bin_sizes]
            )

        out = torch.zeros((input.shape[0], self.inter_channels, *input_shape), device=input.device)

        for stage in self.stages:
            x = stage(input)

            x = F.interpolate(x, input_shape, mode="bilinear")

            out += x

        return self.conv_out(out)
