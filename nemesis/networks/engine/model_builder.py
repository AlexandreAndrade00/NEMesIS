from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple
from copy import deepcopy

import torch
from torch import Tensor, nn, optim
from uuid import UUID

from nemesis.misc.enums import (
    ActivationType,
    Device,
    OptimiserType,
    FuseType,
    PPMAlgorithm,
    HeadType,
    BackboneType,
    TrainType,
)
from .evolved_networks import EvolvedNetwork, ConnectionModule
from .learning_parameters import LearningParams
from ..models import (
    FuseCat,
    FuseAdd,
    UAFM_ChAtten,
    UAFM_SpAtten,
    SPPM,
    DAPPM,
    STDCAddBottleneck,
    STDCCatBottleneck,
    RTMDetInsSepBNHeadModule,
    LazyConv2d,
    LazyBatchNorm,
    LazyLinear,
    LazyAdaptiveAvgPool2d,
    ResNetBasicBlock,
    ResNetBottleneck,
    ClassificationHead,
    SemanticHead,
)
from .parsed_phenotype import ParsedLayer, ParsedBackbone, ParsedHead


warnings.filterwarnings("ignore")


if TYPE_CHECKING:
    from .parsed_phenotype import ParsedOptimiser, ParsedNetwork, ParsedFusion


logger = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(
        self,
        parsed_network: ParsedNetwork,
        device: Device,
        model_saving_dir: str,
        input_tensor: torch.Tensor,
        train_type: TrainType,
    ) -> None:
        self.parsed_network: ParsedNetwork = parsed_network
        self.device = device
        self.model_saving_dir = model_saving_dir
        self.input_tensor = input_tensor
        self.train_type = train_type

    @classmethod
    def assemble_optimiser(cls, model_parameters: Iterable[Tensor], optimiser: ParsedOptimiser) -> LearningParams:
        epochs: int = optimiser.optimiser_parameters.pop("epochs")
        torch_optimiser: optim.Optimizer
        if optimiser.optimiser_type == OptimiserType.RMSPROP:
            torch_optimiser = optim.RMSprop(params=model_parameters, **optimiser.optimiser_parameters)
        elif optimiser.optimiser_type == OptimiserType.GRADIENT_DESCENT:
            torch_optimiser = optim.SGD(params=model_parameters, **optimiser.optimiser_parameters)
        elif optimiser.optimiser_type == OptimiserType.ADAM:
            optimiser.optimiser_parameters["betas"] = (
                optimiser.optimiser_parameters.pop("beta1"),
                optimiser.optimiser_parameters.pop("beta2"),
            )
            torch_optimiser = optim.Adam(params=model_parameters, **optimiser.optimiser_parameters)
        else:
            raise ValueError(f"Invalid optimiser name found: {optimiser.optimiser_type}")
        return LearningParams(epochs=epochs, torch_optimiser=torch_optimiser)

    def assemble_network(self) -> EvolvedNetwork:
        torch_layers: List[Tuple[ParsedLayer, nn.Module]] = []

        connections_to_use: Dict[UUID, ConnectionModule] = {}

        layers_ordered = self.parsed_network.get_layers_in_order()

        for layer in layers_ordered:
            layer_to_add = self._create_torch_layer(layer)

            torch_layers.append((layer, layer_to_add))

        # skip connections added only after layers creation
        for layer in layers_ordered:
            connections_to_use[layer.id] = self._create_layer_connection(layer)

        model = EvolvedNetwork(torch_layers, connections_to_use)

        model.set_train_mode(self.train_type)

        model = model.to(self.device.to_torch_device())

        input_tensor = self.input_tensor.to(device=self.device.to_torch_device(), dtype=torch.float)

        model.eval()

        model(input_tensor)

        return model

    def _create_activation_layer(self, activation: ActivationType) -> nn.Module:
        if activation == ActivationType.RELU:
            return nn.ReLU()
        if activation == ActivationType.SIGMOID:
            return nn.Sigmoid()
        if activation == ActivationType.SOFTMAX:
            return nn.Softmax()
        raise ValueError(f"Unexpected activation function found: {activation}")

    def _create_layer_connection(self, layer: ParsedLayer) -> ConnectionModule:
        if isinstance(layer, ParsedBackbone):
            layer_connection = self.parsed_network.layers[layer.id].connection

            connection_to_add = ConnectionModule(
                layer.direct_input_layers_ids + layer.skip_input_layers_ids,
                self._build_fusion_layer(
                    layer_connection,
                    layer.direct_input_layers_ids,
                    layer.skip_input_layers_ids,
                ),
                layer_connection.fuse_type,
            )

            return connection_to_add
        elif isinstance(layer, ParsedHead):
            return ConnectionModule([layer.input_layer_id], nn.Identity(), None)
        else:
            raise RuntimeError()

    def _create_torch_layer(self, layer: ParsedLayer) -> nn.Module:
        match layer.type:
            case BackboneType.CONV:
                layer_to_add = self._build_convolutional_layer(layer)
            case BackboneType.IDENTITY:
                layer_to_add = nn.Identity()
            case BackboneType.PPM:
                layer_to_add = self._build_ppm(layer)
            case BackboneType.STDC:
                layer_to_add = self._build_stdc(layer)
            case BackboneType.RESNET_BASIC:
                layer_to_add = self._build_resnet_basic(layer)
            case BackboneType.RESNET_BOTTLENECK:
                layer_to_add = self._build_resnet_bottleneck(layer)
            case BackboneType.DSCONV:
                layer_to_add = self._build_depthwise_separable_conv_layer(layer)
            case HeadType.RTMDET_INS_HEAD:
                layer_to_add = self._build_rtmdet_ins_head(layer)
            case BackboneType.FC:
                layer_to_add = self._build_linear(layer)
            case BackboneType.ADAPTIVE_AVG_POOL:
                layer_to_add = self._build_adaptive_avg_pool(layer)
            case HeadType.SEMANTIC_HEAD:
                layer_to_add = self._build_semantic_head(layer)
            case HeadType.CLASSIFICATION_HEAD:
                layer_to_add = self._build_classification_head(layer)
            case _:
                raise ValueError(f"Unexpected layer type found: {layer.type}")

        return layer_to_add

    def _build_convolutional_layer(self, layer: ParsedLayer) -> nn.Module:
        layer_parameters = deepcopy(layer.parameters)

        layer_to_add: nn.Module
        activation: ActivationType = ActivationType(layer_parameters.pop("act"))

        use_batch_norm: bool = layer_parameters.pop("batch_norm")

        if use_batch_norm:
            layer_parameters["bias"] = False

        padding: int | str = layer_parameters.pop("padding")

        if padding == "stride_dep":
            padding = "same" if layer_parameters["stride"] == 1 else "valid"

        convolution = LazyConv2d(
            **layer_parameters,
            padding=padding,
            device=self.device.value,
        )

        if not use_batch_norm and activation == ActivationType.LINEAR:
            return convolution

        layer_to_add = nn.Sequential(convolution)

        if use_batch_norm:
            conv_out_channels: int = layer.parameters["out_channels"]

            layer_to_add.append(LazyBatchNorm(conv_out_channels, device=self.device.value))

        if activation != ActivationType.LINEAR:
            layer_to_add.append(self._create_activation_layer(activation))

        return layer_to_add

    def _build_depthwise_separable_conv_layer(self, layer: ParsedLayer) -> nn.Module:
        depthwise_layer = deepcopy(layer)

        depthwise_layer.parameters["out_channels"] = "input"
        depthwise_layer.parameters["groups"] = "input"

        pointwise_layer = deepcopy(layer)

        pointwise_layer.parameters["kernel_size"] = 1
        pointwise_layer.parameters["dilation"] = 1
        pointwise_layer.parameters["padding"] = 0
        pointwise_layer.parameters["stride"] = 1

        depthwise_conv = self._build_convolutional_layer(depthwise_layer)
        pointwise_conv = self._build_convolutional_layer(pointwise_layer)

        return nn.Sequential(depthwise_conv, pointwise_conv)

    def _build_resnet_basic(self, layer: ParsedLayer) -> nn.Module:
        return ResNetBasicBlock(**layer.parameters)

    def _build_resnet_bottleneck(self, layer: ParsedLayer) -> nn.Module:
        return ResNetBottleneck(**layer.parameters)

    def _build_rtmdet_ins_head(self, layer: ParsedLayer) -> nn.Module:
        num_classes: int = layer.parameters["num_classes"]

        return RTMDetInsSepBNHeadModule(num_classes=num_classes)

    def _build_fusion_layer(
        self,
        connection: ParsedFusion,
        direct_input_ids: List[UUID],
        skip_input_ids: List[UUID],
    ) -> nn.Module:
        match connection.fuse_type:
            case FuseType.ADD:
                return FuseAdd(
                    connection.resize_target,
                    connection.upsample_type,
                    connection.downsample_type,
                    direct_input_ids,
                    skip_input_ids,
                )
            case FuseType.CAT:
                return FuseCat(
                    connection.resize_target,
                    connection.upsample_type,
                    connection.downsample_type,
                    direct_input_ids,
                    skip_input_ids,
                )
            case FuseType.UAFM_SP:
                return UAFM_SpAtten(
                    connection.resize_target,
                    connection.upsample_type,
                    connection.downsample_type,
                    direct_input_ids,
                    skip_input_ids,
                )
            case FuseType.UAFM_CH:
                return UAFM_ChAtten(
                    connection.resize_target,
                    connection.upsample_type,
                    connection.downsample_type,
                    direct_input_ids,
                    skip_input_ids,
                )

    def _build_ppm(self, layer: ParsedLayer) -> nn.Module:
        algorithm = PPMAlgorithm(layer.parameters.pop("ppm_algorithm"))

        match algorithm:
            case PPMAlgorithm.SPPM:
                bin_sizes_len: int = layer.parameters.pop("bin_size")

                bin_sizes: Tuple[int, ...]

                match bin_sizes_len:
                    case 1:
                        bin_sizes = (1,)
                    case 2:
                        bin_sizes = (1, 3)
                    case 3:
                        bin_sizes = (1, 2, 4)
                    case _:
                        raise ValueError("Unsupported bin length")

                return SPPM(
                    inter_channels=layer.parameters["inter_channels"],
                    out_channels=layer.parameters["out_channels"],
                    bin_sizes=bin_sizes,
                )
            case PPMAlgorithm.DAPPM:
                return DAPPM(
                    branch_planes=layer.parameters["inter_channels"],
                    outplanes=layer.parameters["out_channels"],
                )

    def _build_stdc(self, layer: ParsedLayer) -> nn.Module:
        block: str = layer.parameters.pop("stdc_block")

        if block == "add":
            return STDCAddBottleneck(**layer.parameters)
        elif block == "cat":
            return STDCCatBottleneck(**layer.parameters)
        else:
            raise ValueError(f"Unknown STDC bottleneck block: {block}")

    def _build_linear(self, layer: ParsedLayer) -> nn.Module:
        return LazyLinear(
            **layer.parameters,
            device=self.device.value,
        )

    def _build_adaptive_avg_pool(self, layer: ParsedLayer) -> nn.Module:
        return LazyAdaptiveAvgPool2d(**layer.parameters)

    def _build_classification_head(self, layer: ParsedLayer) -> nn.Module:
        return ClassificationHead(
            **layer.parameters,
            device=self.device.value,
        )

    def _build_semantic_head(self, layer: ParsedLayer) -> nn.Module:
        return SemanticHead(
            **layer.parameters,
            device=self.device.value,
        )
