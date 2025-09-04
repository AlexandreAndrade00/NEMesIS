from abc import ABC, abstractmethod

from nemesis.config import ConfigItem
from nemesis.evolution import Individual, Grammar


class IndividualCreator(ABC):
    def __init__(
        self,
        grammar: Grammar,
        network_architecture_config: ConfigItem,
        track_mutations: bool,
        seed: int,
        reuse: float,
    ):
        self.grammar = grammar
        self.network_architecture_config = network_architecture_config
        self.track_mutations = track_mutations
        self.seed = seed
        self.reuse = reuse

    @abstractmethod
    def build(self) -> Individual:
        raise NotImplementedError
