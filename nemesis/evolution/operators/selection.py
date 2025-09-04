import logging
from copy import deepcopy
from typing import List

import numpy as np

from nemesis.evolution import Individual

logger = logging.getLogger(__name__)


def select(
    method: str,
    population: List[Individual],
    elite_size: int,
    train_time: float,
) -> List[Individual]:
    parents: List[Individual] = []
    np_population = np.asarray(population)
    idx_max: np.ndarray

    logger.info(f"Parents fitness: {[ind.fitness for ind in population]}")

    if method == "fittest":
        # Get best individual just according to fitness
        idx_sorted = np.argsort([ind.fitness for ind in np_population])
        idx_max = idx_sorted[-elite_size:]
        parents = list(np_population[idx_max])
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    logger.info(f"Selected parents fitness: {[ind.fitness for ind in parents]}")

    assert len(parents) == elite_size

    logger.info("Parents: idx: %s, id: %s", idx_max, [parent.id for parent in parents])
    logger.info("Training times: %s", str([ind.current_time for ind in population]))
    logger.info("ids: %s", str([ind.id for ind in population]))

    selected_parents: List[Individual] = []

    for parent in parents:
        parent.total_allocated_train_time = train_time

        selected_parents.append(deepcopy(parent))

    return selected_parents
