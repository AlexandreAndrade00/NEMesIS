import os
import logging
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from uuid import UUID

from nemesis.misc.enums import FuseType, TrainType
from nemesis.misc.constants import MODEL_FILENAME, WEIGHTS_FILENAME
from .parsed_phenotype import ParsedLayer, ParsedHead

logger = logging.getLogger(__name__)


class ConnectionModule:
    def __init__(self, input_layers_ids: List[UUID], module: nn.Module, fuse_type: FuseType | None):
        self.input_layers_ids = input_layers_ids
        self.module = module
        self.fuse_type = fuse_type


class _ConnectionModule:
    def __init__(self, input_layers_ids: List[str], module: nn.Module, fuse_type: FuseType | None):
        self.input_layers_ids = input_layers_ids
        self.module = module
        self.fuse_type = fuse_type


class EvolvedNetwork(nn.Module):
    input_id = "00000000-0000-0000-0000-000000000000"

    def __init__(
        self,
        evolved_layers: List[Tuple[ParsedLayer, nn.Module]],
        layers_connections: Dict[UUID, ConnectionModule],
        mode: TrainType = TrainType.NORMAL,
    ) -> None:
        super().__init__()

        self._mode: str = mode.value

        self.layers_connections_adapted: Dict[str, _ConnectionModule] = {
            str(key): _ConnectionModule([str(uuid) for uuid in value.input_layers_ids], value.module, value.fuse_type)
            for key, value in layers_connections.items()
        }

        self.output_layers_ids: Dict[str, str] = {
            parsed_module.train_type.value: str(parsed_module.id)
            for parsed_module, _ in evolved_layers
            if isinstance(parsed_module, ParsedHead)
        }

        self.layers = nn.ModuleDict()
        self.layers_connections = nn.ModuleDict()

        for layer_pheno, layer in evolved_layers:
            self.layers.add_module(str(layer_pheno.id), layer)

        for input_layer_id, connection in self.layers_connections_adapted.items():
            self.layers_connections.add_module(f"connection_{input_layer_id}", connection.module)

    @classmethod
    def load_from_path(cls, path_str: str, with_weights: bool) -> "EvolvedNetwork":
        torch_model: EvolvedNetwork = torch.load(os.path.join(path_str, MODEL_FILENAME), weights_only=False)

        if with_weights:
            torch_model.load_state_dict(torch.load(os.path.join(path_str, WEIGHTS_FILENAME), weights_only=False))
        else:
            torch_model.reset_all_weights()

        return torch_model

    def _process_forward_pass(self, x: Tensor, layer_id: str, connection: _ConnectionModule) -> Tensor:
        assert len(connection.input_layers_ids) > 0

        final_input_tensor: Tensor
        input_tensor: Tensor
        output_tensor: Tensor

        layer_inputs: Dict[str, Tensor] = {}

        for i in connection.input_layers_ids:
            if i == self.input_id:
                input_tensor = x
            else:
                input_tensor = self._process_forward_pass(x, i, self.layers_connections_adapted[i])

            layer_inputs[i] = input_tensor

        if len(layer_inputs) > 1:
            final_input_tensor = self.layers_connections[f"connection_{layer_id}"](layer_inputs)
        else:
            final_input_tensor = next(iter(layer_inputs.values()))

        output_tensor = self.layers[layer_id](final_input_tensor)

        return output_tensor

    def forward(self, x: Tensor) -> List[Tensor]:
        output_layer_id = self.output_layers_ids[self._mode]

        connection: _ConnectionModule = self.layers_connections_adapted[output_layer_id]

        output = self._process_forward_pass(x, output_layer_id, connection)

        result: List[Tensor]

        if not isinstance(output, list):
            result = [output]
        else:
            result = output

        return result

    def pretrain_mode(self) -> "EvolvedNetwork":
        self._mode = TrainType.PRETRAIN.value

        return self

    def train_mode(self) -> "EvolvedNetwork":
        self._mode = TrainType.NORMAL.value

        return self

    def set_train_mode(self, train_type: TrainType) -> "EvolvedNetwork":
        self._mode = train_type.value

        return self

    def reset_all_weights(self) -> None:
        @torch.no_grad()
        def weight_reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()  # type: ignore

        self.apply(fn=weight_reset)

    def set_output_layer(self, new_output_layer: nn.Module) -> None:
        output_layer_id = self.output_layers_ids[self._mode]

        self.layers[output_layer_id] = new_output_layer
