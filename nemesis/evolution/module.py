import logging
import random
from typing import Dict, List

from uuid import UUID, uuid4

from nemesis.config import ModuleConfig
from nemesis.misc.enums import ResizeTarget, UpsampleType, DownsampleType, BackboneType, HeadType, TrainType
from nemesis.networks import ParsedLayer, ParsedFusion, ParsedBackbone, ParsedHead
from .grammar import Genotype, Grammar
from .phenotype import Layer, Fusion


logger = logging.getLogger(__name__)


class ConnectionGenotype:
    def __init__(
        self, layer_id: UUID, fusion: Genotype, direct_input_layers_ids: List[UUID], skip_input_layers_ids: List[UUID]
    ) -> None:
        self.layer_id = layer_id
        self.fusion = fusion
        self.direct_input_layers_ids = direct_input_layers_ids
        self.skip_input_layers_ids = skip_input_layers_ids

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConnectionGenotype):
            return False

        return self.__dict__ == other.__dict__


class Module:
    def __init__(self, module_configuration: ModuleConfig) -> None:
        self.module_name: str = module_configuration.module_name
        self.module_configuration: ModuleConfig = module_configuration
        self.layers: Dict[UUID, Genotype] = {}
        self.connections: Dict[UUID, ConnectionGenotype] = {}
        self.input_layer_id: UUID
        self.output_layer_id: UUID

    def initialise(self, grammar: Grammar, reuse: float, previous_modules: Dict[str, "Module"]) -> None:
        num_expansions = random.choice(self.module_configuration.initial_network_structure)

        # Initialise layers
        previous_layer_id: UUID | None = None
        for idx in range(num_expansions):
            layer_id = uuid4()

            if idx > 0 and random.random() <= reuse:
                r_id: UUID = random.choice(list(self.layers.keys()))

                self.layers[layer_id] = self.layers[r_id]

                assert previous_layer_id is not None

                self.connections[layer_id] = ConnectionGenotype(
                    layer_id, grammar.initialise(self.module_configuration.fusion_rule), [previous_layer_id], []
                )
            else:
                self.layers[layer_id] = grammar.initialise(self.module_name)

                if idx == 0:
                    self.input_layer_id = layer_id

                    self.connections[layer_id] = ConnectionGenotype(
                        layer_id,
                        grammar.initialise(self.module_configuration.input_fusion_rule),
                        [
                            previous_modules[module_name].output_layer_id if module_name != "input" else UUID(int=0)
                            for module_name in self.module_configuration.input_modules
                        ],
                        [],
                    )
                else:
                    assert previous_layer_id is not None

                    self.connections[layer_id] = ConnectionGenotype(
                        layer_id, grammar.initialise(self.module_configuration.fusion_rule), [previous_layer_id], []
                    )

            previous_layer_id = layer_id

        assert previous_layer_id is not None

        self.output_layer_id = previous_layer_id

        max_layers_back = self.module_configuration.self_skip_connection_max_layers_back

        if self.module_configuration.init_with_skip_connections is False:
            return

        if max_layers_back == 0 and self.module_name in self.module_configuration.skip_connections_source_modules:
            raise ValueError(
                f"Module {self.module_name} can't have inner skip connections and "
                + "self_skip_connection_max_layers_back set to 0"
            )

        for layer_id in self.layers.keys():
            connection_possibilities: List[UUID] = []

            for source_module in self.module_configuration.skip_connections_source_modules:
                if source_module == self.module_name:
                    previous_layers = self.get_previous_layers(layer_id)

                    max_layers_back = self.module_configuration.self_skip_connection_max_layers_back

                    previous_layers.pop()

                    if len(previous_layers) == 0:
                        continue

                    init_idx = max(0, len(previous_layers) - max_layers_back)

                    connection_possibilities.extend(previous_layers[init_idx:])
                else:
                    module_layers_ids = previous_modules[source_module].layers.keys()

                    connection_possibilities.extend(module_layers_ids)

                    module_output_layer = previous_modules[source_module].output_layer_id

                    if layer_id == self.input_layer_id and module_output_layer in connection_possibilities:
                        connection_possibilities.remove(module_output_layer)

            sample_size = random.randint(
                0, min(len(connection_possibilities), self.module_configuration.max_skip_connections)
            )

            if sample_size > 0:
                self.connections[layer_id].skip_input_layers_ids += random.sample(connection_possibilities, sample_size)

            assert len(
                set(self.connections[layer_id].direct_input_layers_ids)
                ^ set(self.connections[layer_id].skip_input_layers_ids)
            ) == len(self.connections[layer_id].direct_input_layers_ids) + len(
                self.connections[layer_id].skip_input_layers_ids
            )

    def get_previous_layers(self, start_layer: UUID) -> List[UUID]:
        layers: List[UUID] = []

        current_layer: UUID | None = start_layer

        while current_layer is not None:
            if current_layer == self.input_layer_id:
                current_layer = None
            else:
                assert len(self.connections[current_layer].direct_input_layers_ids) == 1

                layers.insert(0, self.connections[current_layer].direct_input_layers_ids[0])

                current_layer = layers[0]

        return layers

    def decode(self, grammar: Grammar, heads_mapping: Dict[HeadType, TrainType]) -> Dict[UUID, ParsedLayer]:
        layers: Dict[UUID, ParsedLayer] = {}

        current_layer_id: UUID | None = self.output_layer_id

        while current_layer_id is not None:
            layer_pheno = grammar.decode(self.module_name, self.layers[current_layer_id])

            if not isinstance(layer_pheno, Layer):
                raise ValueError(f"Expecting Layer, got {type(layer_pheno)}")

            fusion_pheno = grammar.decode(
                self.module_configuration.fusion_rule
                if self.input_layer_id != current_layer_id
                else self.module_configuration.input_fusion_rule,
                self.connections[current_layer_id].fusion,
            )

            if not isinstance(fusion_pheno, Fusion):
                raise ValueError(f"Expecting Fusion, got {type(fusion_pheno)}")

            fusion = ParsedFusion(
                fusion_pheno.fuse_type,
                ResizeTarget(fusion_pheno.fusion_parameters["resize_target"]),
                UpsampleType(fusion_pheno.fusion_parameters["fusion_upsample"]),
                DownsampleType(fusion_pheno.fusion_parameters["fusion_downsample"]),
            )

            if layer_pheno.type in BackboneType.enum_values():
                layers[current_layer_id] = ParsedBackbone(
                    current_layer_id,
                    BackboneType(layer_pheno.type),
                    layer_pheno.parameters,
                    fusion,
                    self.connections[current_layer_id].direct_input_layers_ids,
                    self.connections[current_layer_id].skip_input_layers_ids,
                )
            elif layer_pheno.type in HeadType.enum_values():
                assert len(self.connections[current_layer_id].direct_input_layers_ids) == 1
                assert len(self.connections[current_layer_id].skip_input_layers_ids) == 0

                head_type = HeadType(layer_pheno.type)

                layers[current_layer_id] = ParsedHead(
                    current_layer_id,
                    head_type,
                    layer_pheno.parameters,
                    self.connections[current_layer_id].direct_input_layers_ids[0],
                    heads_mapping[head_type],
                )

            if current_layer_id == self.input_layer_id:
                break

            assert len(self.connections[current_layer_id].direct_input_layers_ids) == 1

            current_layer_id = self.connections[current_layer_id].direct_input_layers_ids[0]

        return layers

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Module):
            return self.__dict__ == other.__dict__

        return False
