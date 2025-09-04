from __future__ import annotations

import logging
import os
import random
import signal
import sys
from argparse import ArgumentParser
from typing import Any, Optional, Dict
from pathlib import Path

import numpy as np
import torch

import nemesis
from nemesis.config import Config
from nemesis.evolution import engine, Grammar
from nemesis.networks import Evaluator, DatasetSubsets, EvolvedNetwork, Trainer, EarlyStoppingCallback
from nemesis.networks.inference import evaluate_profiler
from nemesis.misc import Checkpoint
from nemesis.misc.constants import STATS_FOLDER_NAME
from nemesis.misc.enums import Task, Device, TrainType
from nemesis.misc.persistence import RestoreCheckpoint, build_overall_best_path, build_overall_best_trained_path
from nemesis.misc.utils import is_valid_config_file, is_valid_file
from nemesis.misc.export import export_tensorrt, export_torch


logger: logging.Logger
sigint_original_handler = signal.getsignal(signal.SIGINT)


def setup_logger(file_path: str, run: str) -> logging.Logger:
    file_path = f"{file_path}/run_{run}/file.log"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logging.setLogRecordFactory(nemesis.logger_record_factory(run))
    logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="{asctime} :: {levelname} :: {name} :: [{run}] -- {message}",
        handlers=[logging.StreamHandler(), logging.FileHandler(file_path)],
        force=True,
    )
    return logging.getLogger(__name__)


def create_initial_checkpoint(
    dataset: DatasetSubsets, train_type: TrainType, config: Config, run: int, is_gpu_run: bool, task: Task
) -> Checkpoint:
    distributed = torch.cuda.is_available() and is_gpu_run and torch.cuda.device_count() > 1

    evaluator: Evaluator = Evaluator.create_evaluator(
        train_data_loader=dataset.train_data_loader(distributed),
        val_data_loader=dataset.validation_data_loader(distributed),
        dataset_num_classes=dataset.num_classes,
        is_gpu_run=is_gpu_run,
        task=task,
        train_type=train_type,
    )

    os.makedirs(os.path.join(config.checkpoints_path, f"run_{run}"), exist_ok=True)  # type: ignore
    os.makedirs(os.path.join(config.checkpoints_path, f"run_{run}", STATS_FOLDER_NAME), exist_ok=True)  # type: ignore

    return Checkpoint(
        run=run,
        random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        torch_random_state=torch.get_rng_state(),
        last_processed_generation=-1,
        total_epochs=0,
        best_fitness=None,
        evaluator=evaluator,
        best_gen_ind_test_accuracy=0.0,
        statistics_format=config.statistics_format,
    )


def after_evolution(sig: int | None, frame: Any, checkpoint: Checkpoint, pre_train: bool) -> Checkpoint:
    signal.signal(signal.SIGINT, sigint_original_handler)

    best_network_path: str = build_overall_best_path(config.checkpoints_path, args.run)  # type: ignore
    best_model = EvolvedNetwork.load_from_path(best_network_path, False)

    best_trained_path = build_overall_best_trained_path(config.checkpoints_path, args.run)  # type: ignore
    Path(best_trained_path).mkdir(parents=True, exist_ok=True)

    torch_device = (Device.GPU if args.gpu_enabled else Device.CPU).to_torch_device()

    if pre_train:
        best_model.pretrain_mode()

        train_data_loader = datasets[TrainType.PRETRAIN].train_data_loader(False)
        val_data_loader = datasets[TrainType.PRETRAIN].validation_data_loader(False)

        trainer = Trainer(
            best_model,
            optimiser=torch.optim.SGD(
                best_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True
            ),
            train_data_loader=train_data_loader,
            validation_data_loader=val_data_loader,
            num_classes=datasets[TrainType.PRETRAIN].num_classes,
            n_epochs=1000,
            multiclass_loss_fn=torch.nn.CrossEntropyLoss(),
            binary_loss_fn=torch.nn.BCEWithLogitsLoss(),
            initial_epoch=0,
            device=Device.GPU if args.gpu_enabled else Device.CPU,
            validation_metrics_calculator=datasets[TrainType.PRETRAIN].task.validation_metrics_calculator(),
            callbacks=[EarlyStoppingCallback(patience=100)],
        )

        trainer.train()

        trainer.load_best_state_dict()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    best_model.train_mode()

    train_data_loader = datasets[TrainType.NORMAL].train_data_loader(False)
    val_data_loader = datasets[TrainType.NORMAL].validation_data_loader(False)

    trainer = Trainer(
        best_model,
        optimiser=torch.optim.SGD(best_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True),
        train_data_loader=train_data_loader,
        validation_data_loader=val_data_loader,
        num_classes=datasets[TrainType.NORMAL].num_classes,
        n_epochs=1000,
        multiclass_loss_fn=torch.nn.CrossEntropyLoss(),
        binary_loss_fn=torch.nn.BCEWithLogitsLoss(),
        initial_epoch=0,
        device=Device.GPU if args.gpu_enabled else Device.CPU,
        validation_metrics_calculator=datasets[TrainType.NORMAL].task.validation_metrics_calculator(),
        callbacks=[EarlyStoppingCallback(patience=30)],
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    trainer.train()

    trainer.load_best_state_dict()

    model_input = next(iter(val_data_loader))[0]

    export_torch(best_model, best_trained_path, model_input.shape[1:])
    export_tensorrt(best_model, best_trained_path, model_input.shape[1:])

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    test_data_loader = datasets[TrainType.NORMAL].test_data_loader(False)

    metrics = evaluate_profiler(
        best_model,
        data_loader=test_data_loader,
        metrics_constructor=datasets[TrainType.NORMAL].task.validation_metrics_calculator()(
            datasets[TrainType.NORMAL].num_classes, torch_device
        ),
        device=torch_device,
    )

    logger.info(metrics)

    logging.shutdown()

    sys.exit(0)


@RestoreCheckpoint
def main(
    run: int,
    config: Config,
    grammar: Grammar,
    is_gpu_run: bool,
    pre_train: bool,
    datasets: Dict[TrainType, DatasetSubsets],
    evaluation_train_type: TrainType,
    possible_checkpoint: Optional[Checkpoint] = None,
) -> None:  # pragma: no cover
    if not logging.getLogger(__name__).hasHandlers():
        global logger
        logger = setup_logger(config.checkpoints_path, str(run))  # type: ignore

    checkpoint: Checkpoint
    if possible_checkpoint is None:
        logger.info("Starting fresh run")

        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)

        checkpoint = create_initial_checkpoint(
            datasets[evaluation_train_type],
            evaluation_train_type,
            config,
            run,
            is_gpu_run,
            datasets[evaluation_train_type].task,
        )
    else:
        logger.info("Loading previous checkpoint")
        checkpoint = possible_checkpoint
        random.setstate(checkpoint.random_state)
        np.random.set_state(checkpoint.numpy_random_state)
        torch.set_rng_state(checkpoint.torch_random_state)

    total_generations: int = config.evolutionary.generations  # type: ignore
    max_epochs: int = config.evolutionary.max_epochs  # type: ignore
    train_time: float = config.network.learning.default_train_time + 0.0  # type: ignore
    train_time_multiplier: float = config.network.learning.train_time_multiplier + 0.0  # type: ignore

    signal.signal(signal.SIGINT, lambda a, b: after_evolution(a, b, checkpoint, pre_train))

    for gen in range(checkpoint.last_processed_generation + 1, total_generations):
        # check the total number of epochs (stop criteria)
        if checkpoint.total_epochs is not None and checkpoint.total_epochs >= max_epochs:
            break

        checkpoint = engine.evolve(
            run,
            grammar,
            gen,
            checkpoint,
            train_time,
            config=config,
        )

        train_time *= 1.0 + train_time_multiplier

        signal.signal(signal.SIGINT, lambda a, b: after_evolution(a, b, checkpoint, pre_train))

    after_evolution(None, None, checkpoint, pre_train)


if __name__ == "__main__":  # pragma: no cover
    parser: ArgumentParser = ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config-path",
        "-c",
        required=True,
        help="Path to the config file to be used",
        type=lambda x: is_valid_config_file(parser, x),
    )
    parser.add_argument(
        "--grammar-path",
        "-g",
        required=True,
        help="Path to the grammar to be used",
        type=lambda x: is_valid_file(parser, x),
    )
    parser.add_argument(
        "--run", "-r", required=False, help="Identifies the run id and seed to be used", type=int, default=0
    )
    parser.add_argument("--gpu-enabled", required=False, help="Runs the experiment in the GPU", action="store_true")
    parser.add_argument("--pre-train", required=False, help="Pre-train the final model", action="store_true")
    args: Any = parser.parse_args()

    config: Config = Config(args.config_path)

    logger = setup_logger(config.checkpoints_path, args.run)  # type: ignore

    DatasetSubsets.seed = args.run

    train_dataset_params = config.dataset.normal  # type: ignore
    pretrain_dataset_params = config.dataset.pretrain  # type: ignore

    datasets = {
        TrainType.NORMAL: DatasetSubsets.from_name(
            train_dataset_params.name,
            train_dataset_params.normalise,
            train_dataset_params.shape,
            train_dataset_params.augment,
            train_dataset_params.batch_size,
            config.dataset.num_workers,  # type: ignore
        ),
    }

    if args.pre_train:
        datasets[TrainType.PRETRAIN] = DatasetSubsets.from_name(
            pretrain_dataset_params.name,
            pretrain_dataset_params.normalise,
            pretrain_dataset_params.shape,
            pretrain_dataset_params.augment,
            train_dataset_params.batch_size,
            config.dataset.num_workers,  # type: ignore
        )

    evaluation_train_type = TrainType(config.dataset.evolution)  # type: ignore

    main(
        run=args.run,
        config=config,
        grammar=Grammar(args.grammar_path, backup_path=config.checkpoints_path),
        is_gpu_run=args.gpu_enabled,
        datasets=datasets,
        evaluation_train_type=evaluation_train_type,
        pre_train=args.pre_train,
    )
