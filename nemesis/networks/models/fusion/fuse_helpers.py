from typing import List, Tuple

import torch
from torch import nn, Tensor


class Interpolate(nn.Module):
    def __init__(self, size: Tuple[int, int], mode: str):
        super().__init__()

        self.size = size
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class Conv2dStride2(nn.Module):
    def __init__(self, input_dimensions: Tuple[int, int, int], target_dimensions: Tuple[int, int, int]):
        super().__init__()

        h_ratio: int = input_dimensions[1] // target_dimensions[1]
        w_ratio: int = input_dimensions[2] // target_dimensions[2]

        number_convs = min(w_ratio // 2, h_ratio // 2)

        convs: List[nn.Module] = []

        while number_convs > 0:
            convs.append(
                nn.Conv2d(
                    in_channels=input_dimensions[0],
                    out_channels=input_dimensions[0],
                    stride=2,
                    kernel_size=3,
                    padding=0,
                    groups=input_dimensions[0],
                    bias=False,
                ),
            )

            convs.append(nn.BatchNorm2d(input_dimensions[0]))
            convs.append(nn.ReLU(inplace=False))

            number_convs -= 1

        self.net = nn.Sequential(*convs, Interpolate(target_dimensions[1:], "bilinear"))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore


def avg_max_reduce_channel_helper(x: Tensor, use_concat: bool = True) -> Tensor | List[Tensor]:
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))

    mean_value: Tensor = torch.mean(x, dim=1, keepdim=True)
    max_value: Tensor = torch.max(x, dim=1, keepdim=True).values

    if use_concat:
        return torch.concat([mean_value, max_value], dim=1)
    else:
        return [mean_value, max_value]


def avg_max_reduce_channel(x: List[Tensor]) -> Tensor:
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if len(x) == 1:
        result = avg_max_reduce_channel_helper(x[0])

        assert isinstance(result, Tensor)

        return result
    else:
        results: List[Tensor] = []

        for xi in x:
            results.extend(avg_max_reduce_channel_helper(xi, False))

        return torch.cat(results, dim=1)


def avg_max_reduce_hw_helper(x: Tensor, is_training: bool, use_concat: bool = True) -> Tensor | List[Tensor]:
    assert not isinstance(x, (list, tuple))

    avg_pool = nn.functional.adaptive_avg_pool2d(x, 1)

    if is_training:
        max_pool = nn.functional.adaptive_max_pool2d(x, 1)
    else:
        max_pool = torch.amax(input=x, dim=[2, 3], keepdim=True)

    if use_concat:
        return torch.cat([avg_pool, max_pool], dim=1)
    else:
        return [avg_pool, max_pool]


def avg_max_reduce_hw(x: List[Tensor], is_training: bool) -> Tensor:
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if len(x) == 1:
        result = avg_max_reduce_hw_helper(x[0], is_training)

        assert isinstance(result, Tensor)

        return result
    else:
        res_avg = []
        res_max = []

        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
            res_avg.append(avg)
            res_max.append(max)

        res = res_avg + res_max

        return torch.cat(res, dim=1)
