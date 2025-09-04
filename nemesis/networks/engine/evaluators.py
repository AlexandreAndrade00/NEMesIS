import logging
import os
import traceback
from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import timedelta

import torch
from torch import nn, OutOfMemoryError
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from nemesis.networks.inference import evaluate_profiler
from nemesis.misc.constants import WEIGHTS_FILENAME
from nemesis.misc.enums import Device, Task, TrainType
from nemesis.misc.evaluation_metrics import EvaluationMetrics
from nemesis.misc.fitness_metrics import Fitness, FitnessMetric
from nemesis.misc.utils import InvalidNetwork, count_parameters
from .callbacks import (
    Callback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    TimedStoppingCallback,
)
from .trainers import Trainer
from .parsed_phenotype import ParsedNetwork, ParsedOptimiser
from .learning_parameters import LearningParams
from .model_builder import ModelBuilder
from .evolved_networks import EvolvedNetwork

__all__ = ["Evaluator", "LegacyEvaluator"]

logger = logging.getLogger(__name__)


def _ddp_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "3"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

    # initialize the process group
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=5))


def _ddp_cleanup() -> None:
    destroy_process_group()


class Evaluator(ABC):
    def __init__(
        self,
        user_chosen_device: Device,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        dataset_num_classes: int,
        task: Task,
        fitness: type[FitnessMetric],
        train_type: TrainType,
    ) -> None:
        self.user_chosen_device: Device = user_chosen_device
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.dataset_num_classes = dataset_num_classes
        self.task: Task = task
        self.fitness: type[FitnessMetric] = fitness
        self.train_type = train_type

    @classmethod
    def create_evaluator(
        cls,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        dataset_num_classes: int,
        is_gpu_run: bool,
        task: Task,
        train_type: TrainType,
        fitness: type[FitnessMetric] | None = None,
    ) -> "Evaluator":
        fitness = fitness if fitness is not None else task.default_fitness_metric()

        user_chosen_device: Device = Device.GPU if is_gpu_run else Device.CPU

        return LegacyEvaluator(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            dataset_num_classes=dataset_num_classes,
            user_chosen_device=user_chosen_device,
            task=task,
            fitness=fitness,
            train_type=train_type,
        )

    @abstractmethod
    def evaluate(
        self,
        parsed_network: ParsedNetwork,
        optimiser: ParsedOptimiser,
        model_saving_dir: str,
        parent_dir: Optional[str],
        train_time: float,
        num_epochs: int,
    ) -> EvaluationMetrics:
        raise NotImplementedError()

    @staticmethod
    def _build_callbacks(model_saving_dir: str, train_time: float, early_stop: Optional[int]) -> list[Callback]:
        callbacks: list[Callback] = [
            ModelCheckpointCallback(model_saving_dir),
            TimedStoppingCallback(max_seconds=train_time),
        ]

        if early_stop is not None:
            callbacks.append(EarlyStoppingCallback(patience=early_stop))

        return callbacks


class LegacyEvaluator(Evaluator):
    def __init__(
        self,
        user_chosen_device: Device,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        dataset_num_classes: int,
        task: Task,
        fitness: type[FitnessMetric],
        train_type: TrainType,
    ) -> None:
        super().__init__(
            user_chosen_device=user_chosen_device,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            dataset_num_classes=dataset_num_classes,
            task=task,
            fitness=fitness,
            train_type=train_type,
        )

    def evaluate(
        self,
        parsed_network: ParsedNetwork,
        optimiser: ParsedOptimiser,
        model_saving_dir: str,
        parent_dir: Optional[str],
        train_time: float,
        num_epochs: int,
    ) -> EvaluationMetrics:  # pragma: no cover
        device: Device = self.user_chosen_device.decide_device()
        torch_model: nn.Module
        fitness_value: Fitness

        os.makedirs(model_saving_dir, exist_ok=True)

        torch.cuda.synchronize(device.to_torch_device())
        torch.cuda.empty_cache()

        try:
            model_builder: ModelBuilder = ModelBuilder(
                parsed_network,
                device,
                model_saving_dir,
                next(iter(self.train_data_loader))[0],
                self.train_type,
            )

            torch_model = model_builder.assemble_network()

            if parent_dir is not None:
                current_model_dict = torch_model.state_dict()

                parent_state_dict = torch.load(os.path.join(parent_dir, WEIGHTS_FILENAME), weights_only=True)

                new_state_dict = {
                    k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
                    for k, v in zip(current_model_dict.keys(), parent_state_dict.values())
                }

                torch_model.load_state_dict(new_state_dict, strict=False)

            trainable_params_count: int = count_parameters(torch_model)

            if trainable_params_count == 0:
                raise InvalidNetwork("Network does not contain any trainable parameters.")

            learning_params: LearningParams = ModelBuilder.assemble_optimiser(torch_model.parameters(), optimiser)

            n_layers: int = len(parsed_network.layers)

            callbacks = self._build_callbacks(
                model_saving_dir, train_time, max(int(trainable_params_count / 2000000), 3)
            )

            mp.set_start_method("spawn", force=True)

            queue: mp.SimpleQueue = mp.SimpleQueue()

            if torch.cuda.is_available() and device == Device.GPU and torch.cuda.device_count() > 1:
                world_size = torch.cuda.device_count()

                context: mp.ProcessContext = mp.spawn(
                    _train_distributed,
                    args=(
                        queue,
                        torch_model,
                        device,
                        world_size,
                        True,
                        learning_params,
                        num_epochs,
                        callbacks,
                        n_layers,
                        self.train_data_loader,
                        self.val_data_loader,
                        self.dataset_num_classes,
                        self.fitness,
                        self.task,
                    ),
                    nprocs=world_size,
                    join=False,
                )

                while not context.join(timeout=60, grace_period=10):
                    logger.info(context.pids())

                for i in range(world_size):
                    metrics: EvaluationMetrics = queue.get()

                queue.close()

                return metrics
            else:
                metrics = _train_distributed(
                    None,
                    queue,
                    torch_model,
                    device,
                    None,
                    False,
                    learning_params,
                    num_epochs,
                    callbacks,
                    n_layers,
                    self.train_data_loader,
                    self.val_data_loader,
                    self.dataset_num_classes,
                    self.fitness,
                    self.task,
                )

                logger.info(metrics.fitness)

                return metrics

        except (InvalidNetwork, ValueError, IndexError, RuntimeError, OutOfMemoryError, KeyError):
            traceback.print_exc()
            logger.warning("Invalid model. Fitness will be computed as invalid individual.")
            fitness_value = self.fitness.worst_fitness()
            return EvaluationMetrics.default(fitness_value)


def _train_distributed(
    rank: int | None,
    connection: mp.SimpleQueue,
    model: EvolvedNetwork,
    device: Device,
    world_size: int | None,
    distributed: bool,
    learning_params: LearningParams,
    num_epochs: int,
    callbacks: List[Callback],
    n_layers: int,
    train_data_loader: torch.utils.data.DataLoader,
    val_data_loader: torch.utils.data.DataLoader,
    dataset_num_classes: int,
    fitness: type[FitnessMetric],
    task: Task,
) -> EvaluationMetrics:
    if distributed:
        assert rank is not None and world_size is not None

        _ddp_setup(rank, world_size)

    trainer = Trainer(
        model=model,
        optimiser=learning_params.torch_optimiser,
        train_data_loader=train_data_loader,
        validation_data_loader=val_data_loader,
        n_epochs=learning_params.epochs,
        initial_epoch=num_epochs,
        device=device,
        gpu_id=rank,
        callbacks=callbacks,
        validation_metrics_calculator=task.validation_metrics_calculator(),
        num_classes=dataset_num_classes,
    )

    try:
        trainer.train()
    except Exception as e:
        logger.warning(traceback.format_exc())
        logger.warning(e)
    finally:
        if distributed:
            _ddp_cleanup()

    trainer.load_best_state_dict()

    if len(trainer.validation_metrics) > 0:
        validation_metric = evaluate_profiler(
            model,
            task.validation_metrics_calculator()(dataset_num_classes, device.to_torch_device()),
            val_data_loader,
            device.to_torch_device(),
        )

        metric: FitnessMetric = fitness(validation_metric)  # type: ignore

        fitness_value = Fitness(metric.compute_metric(), metric.accuracy_metric(), metric.__dict__, fitness)
    else:
        fitness_value = fitness.worst_fitness()

    metrics = EvaluationMetrics(
        is_valid_solution=True,
        fitness=fitness_value,
        n_trainable_parameters=count_parameters(model),
        n_layers=n_layers,
        n_epochs=trainer.trained_epochs,
        train_losses=trainer.train_loss,
        training_time_spent=-1,
        total_epochs_trained=num_epochs + trainer.trained_epochs,
        max_epochs_reached=num_epochs + trainer.trained_epochs >= learning_params.epochs,
    )

    connection.put(metrics)

    return metrics
