import math

from uuid import uuid4, UUID

from nemesis.evolution import (
    Grammar,
    Individual,
    Module,
    Genotype,
    NonTerminal,
    Terminal,
    Attribute,
    ConnectionGenotype,
)
from nemesis.config import ConfigItem

from .individual_creator import IndividualCreator


class PPLiteSegIndividuaCreator(IndividualCreator):
    def __init__(
        self,
        grammar: Grammar,
        network_architecture_config: ConfigItem,
        track_mutations: bool,
        seed: int,
        reuse: float,
    ):
        super().__init__(grammar, network_architecture_config, track_mutations, seed, reuse)
        self.base = 64
        self.layers = [2, 2, 2]
        self.block_num = 4
        self.ppm_bin_size = 2
        self.ppm_out_channels = 128

    def build(self) -> Individual:
        individual_id = uuid4()

        individual = Individual.uniniatlised(
            self.network_architecture_config, individual_id, self.track_mutations, self.seed, self.grammar, self.reuse
        )

        previous_layer_id = UUID(int=0)

        decoder_1_skip_id: UUID
        decoder_2_skip_id: UUID

        for module_config in individual.modules_configurations:
            module = Module(module_config)

            if module_config.module_name == "conv_bn_relu_enc":
                layer_1_id = uuid4()
                layer_1 = Genotype(
                    expansions={
                        NonTerminal(name="kernel_size"): [[Terminal(name="kernel_size:3", attribute=None)]],
                        NonTerminal(name="batch_norm"): [[Terminal(name="batch_norm:False", attribute=None)]],
                        NonTerminal(name="bias"): [[Terminal(name="bias:False", attribute=None)]],
                        NonTerminal(name="conv_bn_relu_enc"): [
                            [
                                Terminal(name="layer:conv", attribute=None),
                                Terminal(
                                    name="out_channels",
                                    attribute=Attribute.from_values(
                                        var_type="int",
                                        num_values=1,
                                        min_value=32,
                                        max_value=1024,
                                        values=[self.base // 2],
                                    ),
                                ),
                                NonTerminal(name="kernel_size"),
                                Terminal(
                                    name="stride",
                                    attribute=Attribute.from_values(
                                        var_type="int", num_values=1, min_value=1, max_value=2, values=[2]
                                    ),
                                ),
                                Terminal(name="padding:stride_dep", attribute=None),
                                NonTerminal(name="batch_norm"),
                                Terminal(name="act:relu", attribute=None),
                                NonTerminal(name="bias"),
                                Terminal(name="condition:None", attribute=None),
                            ]
                        ],
                    },
                    codons={
                        NonTerminal(name="kernel_size"): [1],
                        NonTerminal(name="batch_norm"): [1],
                        NonTerminal(name="bias"): [1],
                        NonTerminal(name="conv_bn_relu_enc"): [0],
                    },
                )

                module.layers[layer_1_id] = layer_1
                module.input_layer_id = layer_1_id
                module.connections[layer_1_id] = ConnectionGenotype(
                    layer_1_id, self.grammar.initialise(module_config.fusion_rule), [previous_layer_id], []
                )

                previous_layer_id = layer_1_id

                layer_2_id = uuid4()
                layer_2 = Genotype(
                    expansions={
                        NonTerminal(name="kernel_size"): [[Terminal(name="kernel_size:3", attribute=None)]],
                        NonTerminal(name="batch_norm"): [[Terminal(name="batch_norm:False", attribute=None)]],
                        NonTerminal(name="bias"): [[Terminal(name="bias:False", attribute=None)]],
                        NonTerminal(name="conv_bn_relu_enc"): [
                            [
                                Terminal(name="layer:conv", attribute=None),
                                Terminal(
                                    name="out_channels",
                                    attribute=Attribute.from_values(
                                        var_type="int", num_values=1, min_value=32, max_value=1024, values=[self.base]
                                    ),
                                ),
                                NonTerminal(name="kernel_size"),
                                Terminal(
                                    name="stride",
                                    attribute=Attribute.from_values(
                                        var_type="int", num_values=1, min_value=1, max_value=2, values=[2]
                                    ),
                                ),
                                Terminal(name="padding:stride_dep", attribute=None),
                                NonTerminal(name="batch_norm"),
                                Terminal(name="act:relu", attribute=None),
                                NonTerminal(name="bias"),
                                Terminal(name="condition:None", attribute=None),
                            ]
                        ],
                    },
                    codons={
                        NonTerminal(name="kernel_size"): [1],
                        NonTerminal(name="batch_norm"): [1],
                        NonTerminal(name="bias"): [1],
                        NonTerminal(name="conv_bn_relu_enc"): [0],
                    },
                )

                module.layers[layer_2_id] = layer_2
                module.output_layer_id = layer_2_id
                module.connections[layer_2_id] = ConnectionGenotype(
                    layer_2_id, self.grammar.initialise(module_config.fusion_rule), [previous_layer_id], []
                )

                previous_layer_id = layer_2_id

            elif module_config.module_name == "stdc_bottleneck":
                for i, layer in enumerate(self.layers):
                    for j in range(layer):
                        if i == 0 and j == 0:
                            new_layer = self._stdc_block_genotype(self.base * 4, self.block_num, 2)
                        elif j == 0:
                            new_layer = self._stdc_block_genotype(
                                self.base * int(math.pow(2, i + 2)), self.block_num, 2
                            )
                        else:
                            new_layer = self._stdc_block_genotype(
                                self.base * int(math.pow(2, i + 2)), self.block_num, 1
                            )

                        new_layer_id = uuid4()

                        if i == 0 and j == 0:
                            module.input_layer_id = new_layer_id

                        module.layers[new_layer_id] = new_layer
                        module.connections[new_layer_id] = ConnectionGenotype(
                            new_layer_id, self.grammar.initialise(module_config.fusion_rule), [previous_layer_id], []
                        )
                        previous_layer_id = new_layer_id

                    if i == 0:
                        decoder_2_skip_id = previous_layer_id
                    elif i == 1:
                        decoder_1_skip_id = previous_layer_id

                module.output_layer_id = previous_layer_id

            elif module_config.module_name == "ppm":
                new_layer = Genotype(
                    expansions={
                        NonTerminal(name="ppm_algorithm"): [[Terminal(name="ppm_algorithm:sppm", attribute=None)]],
                        NonTerminal(name="ppm"): [
                            [
                                Terminal(name="layer:ppm", attribute=None),
                                NonTerminal(name="ppm_algorithm"),
                                Terminal(
                                    name="bin_size",
                                    attribute=Attribute.from_values(
                                        var_type="int",
                                        num_values=1,
                                        min_value=2,
                                        max_value=3,
                                        values=[self.ppm_bin_size],
                                    ),
                                ),
                                Terminal(name=f"inter_channels:{self.ppm_out_channels}", attribute=None),
                                Terminal(name=f"out_channels:{self.ppm_out_channels}", attribute=None),
                            ]
                        ],
                    },
                    codons={NonTerminal(name="ppm_algorithm"): [0], NonTerminal(name="ppm"): [0]},
                )

                new_layer_id = uuid4()

                module.input_layer_id = new_layer_id

                module.layers[new_layer_id] = new_layer
                module.connections[new_layer_id] = ConnectionGenotype(
                    new_layer_id, self.grammar.initialise(module_config.fusion_rule), [previous_layer_id], []
                )
                previous_layer_id = new_layer_id

                module.output_layer_id = previous_layer_id

            elif module_config.module_name == "conv_bn_relu_dec":
                for i in range(2):
                    new_layer_id = uuid4()

                    if i == 0:
                        module.input_layer_id = new_layer_id

                    module.layers[new_layer_id] = Genotype(
                        expansions={
                            NonTerminal(name="conv_bn_relu_dec"): [[Terminal(name="layer:identity", attribute=None)]]
                        },
                        codons={NonTerminal(name="conv_bn_relu_dec"): [0]},
                    )

                    module.connections[new_layer_id] = ConnectionGenotype(
                        new_layer_id,
                        self.grammar.initialise(module_config.fusion_rule),
                        [previous_layer_id],
                        [decoder_1_skip_id if i == 0 else decoder_2_skip_id],
                    )

                    previous_layer_id = new_layer_id

                    if i == 1:
                        module.output_layer_id = previous_layer_id

            elif module_config.module_name == "pre_seg_head":
                new_layer_id = uuid4()

                module.input_layer_id = new_layer_id

                module.layers[new_layer_id] = Genotype(
                    expansions={
                        NonTerminal(name="kernel_size"): [[Terminal(name="kernel_size:3", attribute=None)]],
                        NonTerminal(name="batch_norm"): [[Terminal(name="batch_norm:True", attribute=None)]],
                        NonTerminal(name="bias"): [[Terminal(name="bias:False", attribute=None)]],
                        NonTerminal(name="pre_seg_head"): [
                            [
                                Terminal(name="layer:conv", attribute=None),
                                Terminal(
                                    name="out_channels",
                                    attribute=Attribute.from_values(
                                        var_type="int", num_values=1, min_value=35, max_value=64, values=[64]
                                    ),
                                ),
                                NonTerminal(name="kernel_size"),
                                Terminal(
                                    name="stride",
                                    attribute=Attribute.from_values(
                                        var_type="int", num_values=1, min_value=1, max_value=2, values=[1]
                                    ),
                                ),
                                Terminal(name="padding:stride_dep", attribute=None),
                                NonTerminal(name="batch_norm"),
                                Terminal(name="act:relu", attribute=None),
                                NonTerminal(name="bias"),
                                Terminal(name="condition:None", attribute=None),
                            ]
                        ],
                    },
                    codons={
                        NonTerminal(name="kernel_size"): [1],
                        NonTerminal(name="batch_norm"): [0],
                        NonTerminal(name="bias"): [1],
                        NonTerminal(name="pre_seg_head"): [0],
                    },
                )

                module.connections[new_layer_id] = ConnectionGenotype(
                    new_layer_id, self.grammar.initialise(module_config.fusion_rule), [previous_layer_id], []
                )

                previous_layer_id = new_layer_id

                module.output_layer_id = previous_layer_id

            elif module_config.module_name == "seg_head":
                new_layer_id = uuid4()

                module.input_layer_id = new_layer_id

                module.layers[new_layer_id] = Genotype(
                    expansions={
                        NonTerminal(name="fuse_type_enc"): [[Terminal(name="fuse:cat", attribute=None)]],
                        NonTerminal(name="resize_target"): [[Terminal(name="resize_target:first", attribute=None)]],
                        NonTerminal(name="fusion_upsample"): [
                            [Terminal(name="fusion_upsample:interpolation", attribute=None)]
                        ],
                        NonTerminal(name="fusion_downsample"): [
                            [Terminal(name="fusion_downsample:interpolation", attribute=None)]
                        ],
                        NonTerminal(name="fusion_enc"): [
                            [
                                NonTerminal(name="fuse_type_enc"),
                                NonTerminal(name="resize_target"),
                                NonTerminal(name="fusion_upsample"),
                                NonTerminal(name="fusion_downsample"),
                            ]
                        ],
                        NonTerminal(name="seg_head"): [
                            [
                                Terminal(name="layer:conv", attribute=None),
                                Terminal(name="out_channels:35", attribute=None),
                                Terminal(name="kernel_size:1", attribute=None),
                                Terminal(name="padding:0", attribute=None),
                                Terminal(name="stride:1", attribute=None),
                                Terminal(name="batch_norm:False", attribute=None),
                                Terminal(name="act:linear", attribute=None),
                                Terminal(name="bias:False", attribute=None),
                                Terminal(name="condition:None", attribute=None),
                                NonTerminal(name="fusion_enc"),
                            ]
                        ],
                    },
                    codons={
                        NonTerminal(name="fuse_type_enc"): [1],
                        NonTerminal(name="resize_target"): [0],
                        NonTerminal(name="fusion_upsample"): [0],
                        NonTerminal(name="fusion_downsample"): [1],
                        NonTerminal(name="fusion_enc"): [0],
                        NonTerminal(name="seg_head"): [0],
                    },
                )

                module.connections[new_layer_id] = ConnectionGenotype(
                    new_layer_id, self.grammar.initialise(module_config.fusion_rule), [previous_layer_id], []
                )

                previous_layer_id = new_layer_id

                module.output_layer_id = previous_layer_id
            else:
                raise RuntimeError("Can't build PPLiteSeg individual, double check the configurations")

            individual.modules[module_config.module_name] = module

            for rule in individual.macro_rules:
                individual.macro.append(self.grammar.initialise(rule))

        return individual

    @staticmethod
    def _stdc_block_genotype(out_channels: int, block_num: int, stride: int) -> Genotype:
        return Genotype(
            expansions={
                NonTerminal(name="stdc_block"): [[Terminal(name="stdc_block:add", attribute=None)]],
                NonTerminal(name="stdc_bottleneck"): [
                    [
                        Terminal(name="layer:stdc", attribute=None),
                        NonTerminal(name="stdc_block"),
                        Terminal(
                            name="out_channels",
                            attribute=Attribute.from_values(
                                var_type="int",
                                num_values=1,
                                min_value=32,
                                max_value=1024,
                                values=[out_channels],
                            ),
                        ),
                        Terminal(
                            name="block_num",
                            attribute=Attribute.from_values(
                                var_type="int", num_values=1, min_value=2, max_value=5, values=[block_num]
                            ),
                        ),
                        Terminal(
                            name="stride",
                            attribute=Attribute.from_values(
                                var_type="int", num_values=1, min_value=1, max_value=2, values=[stride]
                            ),
                        ),
                    ]
                ],
            },
            codons={NonTerminal(name="stdc_block"): [1], NonTerminal(name="stdc_bottleneck"): [0]},
        )
