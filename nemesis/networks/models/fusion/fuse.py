from typing import List, Dict, Tuple
from abc import abstractmethod

import logging
from uuid import UUID

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import ModuleDict, Identity, AdaptiveAvgPool2d, Module

from nemesis.misc.enums import ResizeTarget, UpsampleType, DownsampleType
from ..conv_bn_act import ConvBNAct
from .fuse_helpers import Interpolate, Conv2dStride2, avg_max_reduce_channel, avg_max_reduce_hw

logger = logging.getLogger(__name__)


class Fuse(nn.Module):
    def __init__(
        self,
        resize_target: ResizeTarget,
        upsample_type: UpsampleType,
        downsample_type: DownsampleType,
        direct_input_layer_ids: List[UUID],
        skip_input_layer_ids: List[UUID],
    ) -> None:
        super().__init__()
        self.resize_target = resize_target
        self.upsample_type = upsample_type
        self.downsample_type = downsample_type
        self.direct_input_layer_ids: List[str] = [str(layer_id) for layer_id in direct_input_layer_ids]
        self.skip_input_layer_ids: List[str] = [str(layer_id) for layer_id in skip_input_layer_ids]

        self.fix_dimensions_fn = ModuleDict()

    def _fix_size_net(
        self, layer_id: str, input_dimension: Tuple[int, int, int], target_dimension: Tuple[int, int, int]
    ) -> None:
        if input_dimension[2] == target_dimension[2] and input_dimension[1] == target_dimension[1]:
            self.fix_dimensions_fn[layer_id] = Identity()

        elif input_dimension[1] > target_dimension[1] and input_dimension[2] > target_dimension[2]:
            h_ratio: int = input_dimension[1] // target_dimension[1]
            w_ratio: int = input_dimension[2] // target_dimension[2]

            if self.downsample_type == DownsampleType.AVG_POOL or h_ratio < 2 or w_ratio < 2:
                self.fix_dimensions_fn[layer_id] = AdaptiveAvgPool2d((target_dimension[1], target_dimension[2]))
            elif self.downsample_type == DownsampleType.INTERPOLATION:
                self.fix_dimensions_fn[layer_id] = Interpolate(size=target_dimension[1:], mode="bilinear")
            else:
                self.fix_dimensions_fn[layer_id] = Conv2dStride2(input_dimension, target_dimension)

        elif input_dimension[1] < target_dimension[1] and input_dimension[2] < target_dimension[2]:
            self.fix_dimensions_fn[layer_id] = Interpolate(size=target_dimension[1:], mode="bilinear")

        else:
            raise ValueError(f"Mismatch shapes: {target_dimension} {input_dimension}")

    @abstractmethod
    def fuse(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def prepare_data(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(tensors) == 1:
            return tensors

        updated_tensors: Dict[str, torch.Tensor] = {}

        for layer_id, tensor in tensors.items():
            try:
                fn = self.fix_dimensions_fn[layer_id]
            except KeyError:
                target_dimensions = self._get_target_dimensions_forward(tensors)

                logger.info("Added shape fix layer")

                self._fix_size_net(layer_id, input_dimension=(*tensor.shape[1:],), target_dimension=target_dimensions)

                self.fix_dimensions_fn[layer_id] = self.fix_dimensions_fn[layer_id].to(device=tensor.device)

                fn = self.fix_dimensions_fn[layer_id]

            fixed_tensor: Tensor = fn(tensor)

            updated_tensors[layer_id] = fixed_tensor

        return updated_tensors

    def _get_target_dimensions_forward(self, tensors: Dict[str, Tensor]) -> Tuple[int, int, int]:
        match self.resize_target:
            case ResizeTarget.FIRST:
                _, tensor = next(iter(tensors.items()))

                return (*tensor.shape[1:],)
            case ResizeTarget.LARGEST:
                _, largest_tensor = next(iter(tensors.items()))

                for layer_id, tensor in tensors.items():
                    if tensor.shape[2] * tensor.shape[3] > largest_tensor.shape[2] * largest_tensor.shape[3]:
                        largest_tensor = tensor
                return (*largest_tensor.shape[1:],)
            case _:
                raise ValueError("Unkwown enum value")

    def forward(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        fixed_tensors = self.prepare_data(tensors)

        return self.fuse(fixed_tensors)


class FuseCat(Fuse):
    def __init__(
        self,
        resize_target: ResizeTarget,
        upsample_type: UpsampleType,
        downsample_type: DownsampleType,
        direct_input_layer_ids: List[UUID],
        skip_input_layer_ids: List[UUID],
    ) -> None:
        super().__init__(
            resize_target,
            upsample_type,
            downsample_type,
            direct_input_layer_ids,
            skip_input_layer_ids,
        )

    def fuse(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused_tensor: torch.Tensor = torch.cat(list(tensors.values()), dim=1)

        return fused_tensor


class FuseAdd(Fuse):
    def __init__(
        self,
        resize_target: ResizeTarget,
        upsample_type: UpsampleType,
        downsample_type: DownsampleType,
        direct_input_layer_ids: List[UUID],
        skip_input_layer_ids: List[UUID],
    ) -> None:
        super().__init__(
            resize_target,
            upsample_type,
            downsample_type,
            direct_input_layer_ids,
            skip_input_layer_ids,
        )

        self.fix_channels_net = nn.ModuleDict()

    def fuse(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_tensors: List[torch.Tensor] = []

        for layer_id, tensor in tensors.items():
            try:
                input_tensor = self.fix_channels_net[layer_id](tensor)

                input_tensors.append(input_tensor)
            except KeyError:
                logger.info("Added shape fix layer")

                self.fix_channels_net.add_module(
                    str(layer_id),
                    ConvBNAct(
                        tensor.shape[1], tensors[self.direct_input_layer_ids[0]].shape[1], 3, padding="same", bias=False
                    ).to(device=tensor.device),
                )

                self.fix_channels_net[layer_id].eval()

                input_tensor = self.fix_channels_net[layer_id](tensor)

                input_tensors.append(input_tensor)

        return torch.sum(
            torch.stack(input_tensors),
            dim=0,
        )


class UAFM_SpAtten(Fuse):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(
        self,
        resize_target: ResizeTarget,
        upsample_type: UpsampleType,
        downsample_type: DownsampleType,
        direct_input_layer_ids: List[UUID],
        skip_input_layer_ids: List[UUID],
    ):
        super().__init__(
            resize_target,
            upsample_type,
            downsample_type,
            direct_input_layer_ids,
            skip_input_layer_ids,
        )

        self.register_parameter(
            name="scale",
            param=nn.Parameter(torch.tensor(1, dtype=torch.float32)),
        )

        self.fix_channels_net = nn.ModuleDict()

        tensors_num: int = len(self.direct_input_layer_ids + self.skip_input_layer_ids)

        self.conv_xy_atten = nn.Sequential(
            ConvBNAct(tensors_num * 2, tensors_num, kernel_size=3, padding=1, bias=False),
            ConvBNAct(tensors_num, 1, kernel_size=3, padding=1, bias=False, activation=None),
        )

    def fuse(self, tensors: Dict[str, torch.Tensor]) -> Tensor:
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """

        assert len(tensors) == 2

        input_tensors: List[torch.Tensor] = []

        for layer_id, tensor in tensors.items():
            try:
                input_tensor = self.fix_channels_net[layer_id](tensor)

                input_tensors.append(input_tensor)
            except KeyError:
                logger.info("Added shape fix layer")

                self.fix_channels_net.add_module(
                    str(layer_id),
                    ConvBNAct(
                        tensor.shape[1], tensors[self.direct_input_layer_ids[0]].shape[1], 3, padding="same", bias=False
                    ).to(device=tensor.device),
                )

                self.fix_channels_net[layer_id].eval()

                input_tensor = self.fix_channels_net[layer_id](tensor)

                input_tensors.append(input_tensor)

        atten: Tensor = avg_max_reduce_channel(input_tensors)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out: torch.Tensor = input_tensors[0] * atten + input_tensors[1] * (self.scale - atten)

        return out


class UAFM_ChAtten(Fuse):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(
        self,
        resize_target: ResizeTarget,
        upsample_type: UpsampleType,
        downsample_type: DownsampleType,
        direct_input_layer_ids: List[UUID],
        skip_input_layer_ids: List[UUID],
    ):
        super().__init__(
            resize_target,
            upsample_type,
            downsample_type,
            direct_input_layer_ids,
            skip_input_layer_ids,
        )

        self.register_parameter(
            name="scale",
            param=nn.Parameter(torch.tensor(1, dtype=torch.float32)),
        )

        self.fix_channels_net = nn.ModuleDict()

        self.conv_xy_atten: Module | None = None

    def fuse(self, tensors: Dict[str, torch.Tensor]) -> Tensor:
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """

        assert len(tensors) == 2

        input_tensors: List[torch.Tensor] = []

        for layer_id, tensor in tensors.items():
            try:
                input_tensor = self.fix_channels_net[layer_id](tensor)

                input_tensors.append(input_tensor)
            except KeyError:
                logger.info("Added shape fix layer")

                self.fix_channels_net.add_module(
                    str(layer_id),
                    ConvBNAct(
                        tensor.shape[1], tensors[self.direct_input_layer_ids[0]].shape[1], 3, padding="same", bias=False
                    ).to(device=tensor.device),
                )

                self.fix_channels_net[layer_id].eval()

                input_tensor = self.fix_channels_net[layer_id](tensor)

                input_tensors.append(input_tensor)

        atten: Tensor = avg_max_reduce_hw(input_tensors, self.training)

        if self.conv_xy_atten is None:
            logger.info("Added conv out layer")

            target_channels: int = tensors[self.direct_input_layer_ids[0]].shape[1]

            self.conv_xy_atten = nn.Sequential(
                ConvBNAct(
                    4 * target_channels,
                    target_channels // 2,
                    kernel_size=1,
                    bias=False,
                    padding="same",
                    activation=nn.LeakyReLU(inplace=False),
                ),
                ConvBNAct(
                    target_channels // 2,
                    target_channels,
                    kernel_size=1,
                    bias=False,
                    padding="same",
                    activation=None,
                ),
            ).to(device=input_tensors[0].device)

            self.conv_xy_atten.eval()

        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = input_tensors[0] * atten + input_tensors[1] * (self.scale - atten)

        return out
