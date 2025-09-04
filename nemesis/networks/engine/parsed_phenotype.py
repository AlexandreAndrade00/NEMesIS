from abc import ABC
from typing import Any, Dict, List

from uuid import UUID

from nemesis.misc.enums import (
    LayerType,
    OptimiserType,
    FuseType,
    ResizeTarget,
    UpsampleType,
    DownsampleType,
    HeadType,
    TrainType,
    BackboneType,
)
from nemesis.misc.bfs import bfs


class ParsedOptimiser:
    def __init__(self, optimiser_type: OptimiserType, optimiser_parameters: Dict[str, Any]) -> None:
        self.optimiser_type: OptimiserType = optimiser_type
        self.optimiser_parameters: Dict[str, Any] = optimiser_parameters

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ParsedOptimiser):
            return self.__dict__ == other.__dict__
        return False


class ParsedFusion:
    def __init__(
        self,
        fuse_type: FuseType,
        resize_target: ResizeTarget,
        upsample_type: UpsampleType,
        downsample_type: DownsampleType,
    ) -> None:
        self.fuse_type = fuse_type
        self.resize_target = resize_target
        self.upsample_type = upsample_type
        self.downsample_type = downsample_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedFusion):
            return False

        return self.__dict__ == other.__dict__


class ParsedLayer(ABC):
    id: UUID
    type: LayerType
    parameters: Dict[str, Any]


class ParsedBackbone(ParsedLayer):
    def __init__(
        self,
        id: UUID,
        type: BackboneType,
        parameters: Dict[str, Any],
        connection: ParsedFusion,
        direct_input_layers_ids: List[UUID],
        skip_input_layers_ids: List[UUID],
    ) -> None:
        self.id: UUID = id
        self.type: LayerType = type
        self.parameters: Dict[str, Any] = parameters
        self.connection = connection
        self.direct_input_layers_ids = direct_input_layers_ids
        self.skip_input_layers_ids = skip_input_layers_ids

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ParsedLayer):
            return self.__dict__ == other.__dict__

        return False

    def __str__(self) -> str:
        return (
            f"Layer [{self.type}]; id: [{self.id}]; direct inputs: {self.direct_input_layers_ids}; "
            + f"skip inputs: {self.skip_input_layers_ids}; params: {self.parameters}"
        )


class ParsedHead(ParsedLayer):
    def __init__(
        self, id: UUID, type: HeadType, parameters: Dict[str, Any], input_layer_id: UUID, train_type: TrainType
    ) -> None:
        self.id: UUID = id
        self.type: HeadType = type
        self.parameters: Dict[str, Any] = parameters
        self.input_layer_id: UUID = input_layer_id
        self.train_type = train_type

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ParsedHead):
            return self.__dict__ == other.__dict__

        return False


class ParsedNetwork:
    def __init__(self, layers: Dict[UUID, ParsedLayer]):
        self.layers: Dict[UUID, ParsedBackbone] = {}
        self.heads: Dict[UUID, ParsedHead] = {}

        for id, layer in layers.items():
            if isinstance(layer, ParsedBackbone):
                self.layers[id] = layer
            elif isinstance(layer, ParsedHead):
                self.heads[id] = layer

    def get_layers_in_order(self) -> List[ParsedLayer]:
        inputs_map: Dict[UUID, List[UUID]] = {}

        modules: Dict[UUID, ParsedLayer] = {**self.layers, **self.heads}

        for x in modules.values():
            inputs_map.setdefault(x.id, [])

            if isinstance(x, ParsedBackbone):
                for layer_id in x.direct_input_layers_ids:
                    inputs_map.setdefault(layer_id, [])

                    inputs_map[layer_id].append(x.id)
            elif isinstance(x, ParsedHead):
                inputs_map.setdefault(x.input_layer_id, [])

                inputs_map[x.input_layer_id].append(x.id)
            else:
                raise RuntimeError()

        ordered_modules: List[UUID] = bfs(inputs_map)

        ordered_modules.remove(UUID(int=0))

        return [modules[x] for x in ordered_modules]
