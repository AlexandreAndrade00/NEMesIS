from dataclasses import dataclass
from typing import List


@dataclass
class ModuleConfig:
    module_name: str
    min_expansions: int
    max_expansions: int
    input_modules: List[str]
    initial_network_structure: List[int]
    skip_connections_source_modules: List[str]
    self_skip_connection_max_layers_back: int
    max_skip_connections: int
    init_with_skip_connections: bool
    fusion_rule: str
    input_fusion_rule: str

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModuleConfig):
            return self.__dict__ == other.__dict__
        return False
