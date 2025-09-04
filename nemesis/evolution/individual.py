import logging
from typing import Any, List, Optional, Dict, Tuple

from uuid import UUID

from nemesis.config import ConfigItem, ModuleConfig
from nemesis.misc.enums import Mutation, MutationType, HeadType, TrainType
from nemesis.misc.evaluation_metrics import EvaluationMetrics
from nemesis.misc.fitness_metrics import Fitness
from nemesis.networks import ParsedNetwork, ParsedLayer, ParsedOptimiser, Evaluator
from .module import Module
from .grammar import Genotype, Grammar
from .phenotype import Optimiser


__all__ = ["Individual"]


logger = logging.getLogger(__name__)


class Individual:
    """
    Candidate solution.


    Attributes
    ----------
    network_structure : list
        ordered list of tuples formated as follows
        [(non-terminal, min_expansions, max_expansions), ...]

    output_rule : str
        output non-terminal symbol

    macro_rules : list
        list of non-terminals (str) with the marco rules (e.g., learning)

    modules : list
        list of Modules (genotype) of the layers

    output : dict
        output rule genotype

    macro : list
        list of Modules (genotype) for the macro rules

    phenotype : str
        phenotype of the candidate solution

    fitness : float
        fitness value of the candidate solution

    metrics : dict
        training metrics

    num_epochs : int
        number of performed epochs during training

    trainable_parameters : int
        number of trainable parameters of the network

    time : float
        network training time

    current_time : float
        performed network training time

    train_time : float
        maximum training time

    id : int
        individual unique identifier


    Methods
    -------
        initialise(grammar, reuse)
            Randomly creates a candidate solution

        decode(grammar)
            Maps the genotype to the phenotype

        evaluate(grammar, cnn_eval, weights_save_path, parent_weights_path='')
            Performs the evaluation of a candidate solution
    """

    def __init__(
        self,
        network_architecture_config: ConfigItem,
        ind_id: UUID,
        track_mutations: bool,
        seed: int,
        grammar: Grammar,
        reuse: float,
        initialise: bool = True,
    ) -> None:
        self.seed: int = seed
        self.modules_configurations: List[ModuleConfig] = network_architecture_config.modules  # type: ignore
        self.macro_rules: List[str] = network_architecture_config.macro_structure  # type: ignore
        self.static_modules: List[str] = network_architecture_config.static_modules  # type: ignore
        self.modules: Dict[str, Module] = {}
        self.mutation_tracker: Optional[List[Mutation]] = [] if track_mutations else None
        self.macro: List[Genotype] = []
        self.phenotype: Optional[str] = None
        self.fitness: Optional[Fitness] = None
        self.metrics: Optional[EvaluationMetrics] = None
        self.num_epochs: int = 0
        self.current_time: float = 0.0
        self.total_allocated_train_time: float = 0.0
        self.total_training_time_spent: float = 0.0
        self.id: UUID = ind_id

        self.heads_mapping: Dict[HeadType, TrainType] = {
            HeadType(network_architecture_config.head_module): TrainType.NORMAL,
            HeadType(network_architecture_config.pretrain_head_module): TrainType.PRETRAIN,
        }

        if initialise:
            self._initialise(grammar, reuse)

    @classmethod
    def uniniatlised(
        cls,
        network_architecture_config: ConfigItem,
        ind_id: UUID,
        track_mutations: bool,
        seed: int,
        grammar: Grammar,
        reuse: float,
    ) -> "Individual":
        return cls(network_architecture_config, ind_id, track_mutations, seed, grammar, reuse, False)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Individual):
            return self.__dict__ == other.__dict__
        return False

    def _decode(self, grammar: Grammar) -> Tuple[ParsedNetwork, ParsedOptimiser]:
        """
        Maps the genotype to the phenotype

        Parameters
        ----------
        grammar : Grammar
            grammar instaces that stores the expansion rules

        Returns
        -------
        phenotype : str
            phenotype of the individual to be used in the mapping to the keras model.
        """

        parsed_layers: Dict[UUID, ParsedLayer] = {}

        for module in self.modules.values():
            parsed_layers.update(module.decode(grammar, self.heads_mapping))

        optimiser = grammar.decode(self.macro_rules[0], self.macro[0])

        assert isinstance(optimiser, Optimiser)

        parsed_optimiser = ParsedOptimiser(optimiser.optimiser_type, optimiser.optimiser_parameters)

        return ParsedNetwork(parsed_layers), parsed_optimiser

    def _initialise(self, grammar: Grammar, reuse: float) -> None:
        """
        Randomly creates a candidate solution

        Parameters
        ----------
        grammar : Grammar
            grammar instaces that stores the expansion rules

        reuse : float
            likelihood of reusing an existing layer

        Returns
        -------
        candidate_solution : Individual
            randomly created candidate solution
        """
        self._initialise_modules(grammar, reuse)

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))

    def _initialise_modules(
        self,
        grammar: Grammar,
        reuse: float,
    ) -> None:
        for module_config in self.modules_configurations:
            new_module: Module = Module(module_config)

            new_module.initialise(grammar, reuse, self.modules)

            self.modules[module_config.module_name] = new_module

    def reset_keys(self, *keys: str) -> None:
        for key in keys:
            attr = getattr(self, key)
            if isinstance(attr, (int, float)):
                setattr(self, key, 0)
            else:
                setattr(self, key, None)

    def get_num_layers(self) -> int:
        return sum(len(m.layers) for m in self.modules.values())

    def track_mutation(self, mutation_type: MutationType, gen: int, data: dict[str, Any]) -> None:
        if self.mutation_tracker is None:
            return
        self.mutation_tracker.append(Mutation(mutation_type, gen, data))

    def evaluate(
        self,
        grammar: Grammar,
        cnn_eval: Evaluator,
        generation: int,
        exploration: bool,
        model_saving_dir: str,
        parent_dir: Optional[str] = None,
    ) -> Fitness:  # pragma: no cover
        parsed_network, optimiser = self._decode(grammar)

        allocated_train_time: float = self.total_allocated_train_time - self.current_time

        logger.info("-----> Starting evaluation for individual %s for %d secs", str(self.id), allocated_train_time)

        self.metrics = cnn_eval.evaluate(
            parsed_network,
            optimiser,
            model_saving_dir,
            parent_dir,
            allocated_train_time,
            self.num_epochs,
        )

        self.fitness = self.metrics.fitness
        self.num_epochs += self.metrics.n_epochs
        self.current_time += allocated_train_time
        self.total_training_time_spent += self.metrics.training_time_spent

        assert self.fitness is not None

        return self.fitness
