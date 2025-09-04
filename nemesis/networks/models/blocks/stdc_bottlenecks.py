import math

import torch
from torch import nn, Tensor

from ..conv_bn_act import ConvBNAct


class STDCAddBottleneck(nn.Module):
    def __init__(
        self,
        out_channels: int,
        block_num: int,
        stride: int,
    ):
        super().__init__()

        assert block_num > 1, "block number should be larger than 1."

        self.conv_list = nn.ModuleList()

        self.stride = stride
        self.out_channels = out_channels
        self.block_num = block_num
        self._initialised: bool = False

    def _make_layers(self, in_channels: int, device: torch.device) -> None:
        if self.stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    self.out_channels // 2,
                    self.out_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.out_channels // 2,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(
                    self.out_channels // 2,
                    device=device,
                ),
            )

            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(
                    in_channels,
                    device=device,
                ),
                nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(
                    self.out_channels,
                    device=device,
                ),
            )
            self.stride = 1
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(
                    self.out_channels,
                    device=device,
                ),
            )

        channels = 0

        for idx in range(self.block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNAct(
                        in_channels,
                        self.out_channels // 2,
                        kernel_size=1,
                        device=device,
                    )
                )

                channels += self.out_channels // 2
            elif idx == 1 and self.block_num == 2:
                channels += self.out_channels // 2

                fix_channels = self.out_channels - channels

                assert fix_channels >= 0

                self.conv_list.append(
                    ConvBNAct(
                        self.out_channels // 2,
                        self.out_channels // 2 + fix_channels,
                        stride=self.stride,
                        device=device,
                    )
                )
            elif idx == 1 and self.block_num > 2:
                self.conv_list.append(
                    ConvBNAct(
                        self.out_channels // 2,
                        self.out_channels // 4,
                        stride=self.stride,
                        device=device,
                    )
                )

                channels += self.out_channels // 4
            elif idx < self.block_num - 1:
                channels += self.out_channels // int(math.pow(2, idx + 1))

                self.conv_list.append(
                    ConvBNAct(
                        self.out_channels // int(math.pow(2, idx)),
                        self.out_channels // int(math.pow(2, idx + 1)),
                        device=device,
                    )
                )
            else:
                channels += self.out_channels // int(math.pow(2, idx))

                fix_channels = self.out_channels - channels

                self.conv_list.append(
                    ConvBNAct(
                        self.out_channels // int(math.pow(2, idx)),
                        self.out_channels // int(math.pow(2, idx)) + fix_channels,
                        device=device,
                    )
                )

    def forward(self, x: Tensor) -> Tensor:
        if not self._initialised:
            self._initialised = True

            self._make_layers(x.shape[1], x.device)

        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)

            out_list.append(out)

        x = self.skip(x)

        final = torch.cat(out_list, dim=1) + x

        return final


class STDCCatBottleneck(nn.Module):
    def __init__(self, out_channels: int, block_num: int, stride: int):
        super().__init__()

        assert block_num > 1, "block number should be larger than 1."

        self.conv_list = nn.ModuleList()

        self.stride = stride
        self.out_channels = out_channels
        self.block_num = block_num
        self._initialised: bool = False

    def _make_layers(self, in_channels: int, device: torch.device) -> None:
        if self.stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    self.out_channels // 2,
                    self.out_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.out_channels // 2,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(
                    self.out_channels // 2,
                    device=device,
                ),
            )

            self.skip = nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            )

            self.stride = 1

        channels = 0

        for idx in range(self.block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNAct(
                        in_channels,
                        self.out_channels // 2,
                        kernel_size=1,
                        device=device,
                    )
                )

                channels += self.out_channels // 2
            elif idx == 1 and self.block_num == 2:
                channels += self.out_channels // 2

                fix_channels = self.out_channels - channels

                assert fix_channels >= 0

                self.conv_list.append(
                    ConvBNAct(
                        self.out_channels // 2,
                        self.out_channels // 2 + fix_channels,
                        stride=self.stride,
                        device=device,
                    )
                )
            elif idx == 1 and self.block_num > 2:
                self.conv_list.append(
                    ConvBNAct(
                        self.out_channels // 2,
                        self.out_channels // 4,
                        stride=self.stride,
                        device=device,
                    )
                )

                channels += self.out_channels // 4
            elif idx < self.block_num - 1:
                channels += self.out_channels // int(math.pow(2, idx + 1))

                self.conv_list.append(
                    ConvBNAct(
                        self.out_channels // int(math.pow(2, idx)),
                        self.out_channels // int(math.pow(2, idx + 1)),
                        device=device,
                    )
                )
            else:
                channels += self.out_channels // int(math.pow(2, idx))

                fix_channels = self.out_channels - channels

                self.conv_list.append(
                    ConvBNAct(
                        self.out_channels // int(math.pow(2, idx)),
                        self.out_channels // int(math.pow(2, idx)) + fix_channels,
                        device=device,
                    )
                )

    def forward(self, x: Tensor) -> Tensor:
        if not self._initialised:
            self._initialised = True

            self._make_layers(x.shape[1], device=x.device)

        out_list = []

        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)

            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)

        out_list.insert(0, out1)

        final = torch.cat(out_list, dim=1)

        return final
