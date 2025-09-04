# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .rtmdet_head import RTMDetSepBNHeadModule
from ..conv_bn_act import ConvBNAct


class MaskFeatModule(nn.Module):
    """Mask feature head used in RTMDet-Ins. Copy from mmdet.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        stacked_convs (int): Number of convs in mask feature branch.
        num_levels (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        num_prototypes (int): Number of output channel of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=False)
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        num_levels: int = 3,
        num_prototypes: int = 8,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2d(num_levels * in_channels, in_channels, 1)

        convs = []

        for i in range(stacked_convs):
            in_c = in_channels if i == 0 else feat_channels
            convs.append(ConvBNAct(in_c, feat_channels, 3, padding=1))

        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(feat_channels, num_prototypes, kernel_size=1)

    def forward(self, features: Tuple[Tensor, ...]) -> Tensor:
        # multi-level feature fusion
        fusion_feats_list: List[Tensor] = [features[0]]
        size = features[0].shape[-2:]

        for i in range(1, self.num_levels):
            f = F.interpolate(features[i], size=size, mode="bilinear")
            fusion_feats_list.append(f)

        fusion_feats = torch.cat(fusion_feats_list, dim=1)

        fusion_feats = self.fusion_conv(fusion_feats)

        # pred mask feats
        mask_features = self.stacked_convs(fusion_feats)
        mask_features = self.projection(mask_features)

        return  # type: ignore


class RTMDetInsSepBNHeadModule(RTMDetSepBNHeadModule):
    """Detection and Instance Segmentation Head of RTMDet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_prototypes (int): Number of mask prototype features extracted
            from the mask head. Defaults to 8.
        dyconv_channels (int): Channel of the dynamic conv layers.
            Defaults to 8.
        num_dyconvs (int): Number of the dynamic convolution layers.
            Defaults to 3.
        use_sigmoid_cls (bool): Use sigmoid for class prediction.
            Defaults to True.
    """

    def __init__(
        self,
        num_classes: int,
        num_prototypes: int = 8,
        dyconv_channels: int = 8,
        num_dyconvs: int = 3,
        use_sigmoid_cls: bool = True,
    ):
        self.num_prototypes: int = num_prototypes
        self.num_dyconvs: int = num_dyconvs
        self.dyconv_channels: int = dyconv_channels
        self.use_sigmoid_cls: bool = use_sigmoid_cls

        self.cls_out_channels: int = num_classes
        if not self.use_sigmoid_cls:
            self.cls_out_channels += 1

        super().__init__(num_classes=num_classes)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_kernel = nn.ModuleList()
        self.rtm_obj = nn.ModuleList()

        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append((self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        pred_pad_size = self.pred_kernel_size // 2

        for n in range(len(self.featmap_strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            kernel_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvBNAct(chn, self.feat_channels, 3, stride=1, padding=1, activation=nn.SiLU(inplace=False))
                )
                reg_convs.append(
                    ConvBNAct(chn, self.feat_channels, 3, stride=1, padding=1, activation=nn.SiLU(inplace=False))
                )
                kernel_convs.append(
                    ConvBNAct(chn, self.feat_channels, 3, stride=1, padding=1, activation=nn.SiLU(inplace=False))
                )
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(cls_convs)
            self.kernel_convs.append(kernel_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=pred_pad_size,
                )
            )
            self.rtm_reg.append(
                nn.Conv2d(self.feat_channels, self.num_base_priors * 4, self.pred_kernel_size, padding=pred_pad_size)
            )
            self.rtm_kernel.append(
                nn.Conv2d(self.feat_channels, self.num_gen_params, self.pred_kernel_size, padding=pred_pad_size)
            )

        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv  # type: ignore
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv  # type: ignore

        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(self.featmap_strides),
            num_prototypes=self.num_prototypes,
        )

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - kernel_preds (list[Tensor]): Dynamic conv kernels for all scale
              levels, each is a 4D-tensor, the channels number is
              num_gen_params.
            - mask_feat (Tensor): Mask prototype features.
                Has shape (batch_size, num_prototypes, H, W).
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, (x, stride) in enumerate(zip(feats, self.featmap_strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs[idx]:  # type: ignore
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for kernel_layer in self.kernel_convs[idx]:  # type: ignore
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.rtm_kernel[idx](kernel_feat)

            for reg_layer in self.reg_convs[idx]:  # type: ignore
                reg_feat = reg_layer(reg_feat)
            reg_dist = self.rtm_reg[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat
