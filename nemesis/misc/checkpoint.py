from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from torch import Tensor

if TYPE_CHECKING:
    from nemesis.evolution import Individual
    from nemesis.misc.fitness_metrics import Fitness
    from nemesis.networks.evaluators import Evaluator


@dataclass
class Checkpoint:
    run: int
    random_state: Any
    numpy_random_state: Dict[str, Any]
    torch_random_state: Tensor
    last_processed_generation: int
    total_epochs: int
    best_fitness: Optional[Fitness]
    evaluator: Evaluator
    best_gen_ind_test_accuracy: float
    population: Optional[List[Individual]] = field(default=None)
    parents: List[Individual] = field(default_factory=lambda: [])
    statistics_format: Optional[str] = field(default=None)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Checkpoint):
            return self.__dict__ == other.__dict__
        return False
