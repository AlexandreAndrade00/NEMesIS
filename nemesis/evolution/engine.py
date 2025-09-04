import logging
import random
from copy import deepcopy
from typing import List, Dict
from functools import reduce
from shutil import copytree

import numpy as np
from uuid import uuid4, UUID
import torch

from nemesis.config import Config
from nemesis.evolution import Grammar, Individual
from nemesis.evolution.operators import selection, mutation
from nemesis.misc import Checkpoint, persistence
from .individuals import PPLiteSegIndividuaCreator, IndividualCreator

logger = logging.getLogger(__name__)


@persistence.SaveCheckpoint
def evolve(
    run: int,
    grammar: Grammar,
    generation: int,
    checkpoint: Checkpoint,
    train_time: float,
    config: Config,
) -> Checkpoint:
    logger.info("Performing generation: %d", generation)

    elite_lambda: int = config.evolutionary.elite_lambda  # type: ignore

    population: List[Individual]

    if generation == 0:
        logger.info("Creating the initial population")

        individual_seed = config.evolutionary.individual_seed  # type: ignore

        if individual_seed is not None:
            creator: IndividualCreator

            match individual_seed:
                case "ppliteseg":
                    creator = PPLiteSegIndividuaCreator(
                        grammar,
                        config.network.architecture,  # type: ignore
                        config.evolutionary.track_mutations,  # type: ignore
                        run,
                        config.network.architecture.reuse_layer,  # type: ignore
                    )
                case _:
                    raise ValueError(individual_seed, "Unknown individual seed")

            population = [
                creator.build()
                for _id_ in range(config.evolutionary.lambda_)  # type: ignore
            ]
        else:
            population = [
                Individual(
                    config.network.architecture,  # type: ignore
                    uuid4(),
                    config.evolutionary.track_mutations,  # type: ignore
                    run,
                    grammar,
                    config.network.architecture.reuse_layer,  # type: ignore
                )
                for _id_ in range(config.evolutionary.lambda_)  # type: ignore
            ]

        # set initial population variables and evaluate population
        for ind in population:
            ind.total_allocated_train_time = train_time  # type: ignore

            ind.reset_keys("current_time", "num_epochs", "total_training_time_spent")

            ind.evaluate(
                grammar,
                checkpoint.evaluator,
                generation,
                True,
                persistence.build_individual_path(config.checkpoints_path, run, generation, ind.id),  # type: ignore
            )  # type: ignore

    else:
        assert len(checkpoint.parents) != 0

        logger.info("Applying mutation operators")

        lambda_: int = config.evolutionary.lambda_  # type: ignore

        exploration: bool = checkpoint.parents[0].fitness.accuracy_metric < 0.5  # type: ignore

        # generate offspring (by mutation)
        offspring: List[Individual] = []

        parents_ids_mapping: Dict[UUID, UUID] = {}

        for idx in range(lambda_ - len(checkpoint.parents)):
            parent = checkpoint.parents[idx % len(checkpoint.parents)]
            new_ind: Individual = deepcopy(parent)
            new_ind.total_training_time_spent = 0.0
            new_ind.id = uuid4()

            parents_ids_mapping[new_ind.id] = parent.id

            mutation.mutate(
                new_ind,
                grammar,
                generation,
                config.evolutionary.mutation,  # type: ignore
                train_time,  # type: ignore
                config.network.learning.max_train_time,  # type: ignore
            )

            offspring.append(new_ind)

        population = [deepcopy(parent) for parent in checkpoint.parents] + offspring

        # set elite variables to re-evaluation
        for i in range(elite_lambda):
            new_id = uuid4()

            parents_ids_mapping[new_id] = population[i].id

            population[i].reset_keys("id", "num_epochs", "current_time")

            population[i].id = new_id

        # evaluate population
        for i, ind in enumerate(population):
            ind_dir = persistence.build_individual_path(config.checkpoints_path, run, generation, ind.id)  # type: ignore
            parent_dir = persistence.build_individual_path(
                config.checkpoints_path,  # type: ignore
                run,
                generation - 1,
                parents_ids_mapping[ind.id],
            )

            if i < elite_lambda:
                copytree(parent_dir, ind_dir)

                continue

            ind.evaluate(
                grammar,
                checkpoint.evaluator,
                generation,
                exploration,
                ind_dir,
                parent_dir,
            )

    assert all(map(lambda x: x.id is not None, population))

    selection_method: str = "fittest"

    # select parent
    parents: List[Individual] = selection.select(
        selection_method,
        population,
        elite_lambda,
        train_time,
    )

    best_individual_path: str = ""

    for parent in parents:
        assert parent.fitness is not None

        # update best individual
        individual_path: str = persistence.build_individual_path(
            config.checkpoints_path,  # type: ignore
            run,
            generation,
            parent.id,
        )

        if best_individual_path == "":
            best_individual_path = individual_path

        if checkpoint.best_fitness is None or parent.fitness > checkpoint.best_fitness:
            best_individual_path = individual_path
            checkpoint.best_fitness = parent.fitness
            persistence.save_overall_best_individual(best_individual_path, parent)

    best_individual = reduce(
        lambda ind1, ind2: ind1 if ind1.fitness.fitness > ind2.fitness.fitness else ind2,  # type: ignore
        population,
    )

    logger.info(
        "Best Individual of generation %d - Fitness: %f; All metrics: %s",
        generation,
        best_individual.fitness.fitness,  # type: ignore
        best_individual.fitness.all_values,  # type: ignore
    )
    logger.info("Best overall fitness: %f\n\n\n", checkpoint.best_fitness.fitness)  # type: ignore

    return Checkpoint(
        run=run,
        random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        torch_random_state=torch.get_rng_state(),
        last_processed_generation=generation,
        total_epochs=checkpoint.total_epochs + sum(ind.num_epochs for ind in population),
        best_fitness=checkpoint.best_fitness,
        evaluator=checkpoint.evaluator,
        population=population,
        parents=parents,
        best_gen_ind_test_accuracy=best_individual.fitness.accuracy_metric,  # type: ignore
        statistics_format=checkpoint.statistics_format,
    )
