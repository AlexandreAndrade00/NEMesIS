import csv
import glob
import json
import os
import shutil
from typing import Any, Callable, Optional, Dict

import torch
import dill
from uuid import UUID

from nemesis.evolution import Individual, Grammar, Layer, Optimiser
from nemesis.networks import DatasetSubsets
from nemesis.config import Config

from .checkpoint import Checkpoint
from .constants import MODEL_FILENAME, OVERALL_BEST_FOLDER, STATS_FOLDER_NAME, OVERALL_BEST_TRAINED_FOLDER
from .enums import Mutation, TrainType
from .fitness_metrics import Fitness


__all__ = [
    "RestoreCheckpoint",
    "SaveCheckpoint",
    "save_overall_best_individual",
    "build_individual_path",
    "build_overall_best_path",
]


class RestoreCheckpoint:
    def __init__(self, f: Callable) -> None:
        self.f: Callable = f

    def __call__(
        self,
        run: int,
        config: Config,
        grammar: Grammar,
        is_gpu_run: bool,
        pre_train: bool,
        datasets: Dict[TrainType, DatasetSubsets],
        evaluation_train_type: TrainType,
    ) -> None:
        self.f(
            run,
            config,
            grammar,
            is_gpu_run,
            pre_train,
            datasets,
            evaluation_train_type,
            possible_checkpoint=self.restore_checkpoint(config, run, datasets[evaluation_train_type], is_gpu_run),
        )

    def restore_checkpoint(
        self, config: Config, run: int, dataset: DatasetSubsets, is_gpu_run: bool
    ) -> Optional[Checkpoint]:
        distributed = torch.cuda.is_available() and is_gpu_run and torch.cuda.device_count() > 1

        checkpoint_path = os.path.join(config.checkpoints_path, f"run_{run}", "checkpoint.pkl")  # type: ignore

        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as handle_checkpoint:
                checkpoint: Checkpoint = dill.load(handle_checkpoint)

            checkpoint.evaluator.train_data_loader = dataset.train_data_loader(distributed)
            checkpoint.evaluator.val_data_loader = dataset.validation_data_loader(distributed)

            return checkpoint

        return None


class SaveCheckpoint:
    def __init__(self, f: Callable) -> None:
        self.f: Callable = f

    def __call__(self, *args: Any, **kwargs: Any) -> Checkpoint:
        new_checkpoint: Checkpoint = self.f(*args, **kwargs)
        # we assume the config is the last parameter in the function decorated
        self._save_checkpoint(new_checkpoint, kwargs["config"].checkpoints_path)
        return new_checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint, save_path: str) -> None:
        assert checkpoint.population is not None
        assert len(checkpoint.parents) != 0

        train_data_loader = checkpoint.evaluator.train_data_loader
        val_data_loader = checkpoint.evaluator.val_data_loader

        checkpoint.evaluator.train_data_loader = None
        checkpoint.evaluator.val_data_loader = None

        with open(os.path.join(save_path, f"run_{checkpoint.run}", "checkpoint.pkl"), "wb") as handle_checkpoint:
            dill.dump(checkpoint, handle_checkpoint)

        assert train_data_loader is not None and val_data_loader is not None

        checkpoint.evaluator.train_data_loader = train_data_loader
        checkpoint.evaluator.val_data_loader = val_data_loader

        self._delete_unnecessary_files(checkpoint, save_path)

        if checkpoint.statistics_format == "csv":
            self._save_statistics_csv(save_path, checkpoint)
        elif checkpoint.statistics_format == "json":
            self._save_statistics_json(save_path, checkpoint)
        else:
            raise ValueError(f"Unknown statistics format: {checkpoint.statistics_format}")

    # pylint: disable=unused-argument
    def _delete_unnecessary_files(self, checkpoint: Checkpoint, save_path: str) -> None:
        assert checkpoint.population is not None
        # remove temporary files to free disk space
        files_to_delete = glob.glob(
            f"{save_path}/"
            f"run_{checkpoint.run}/"
            f"ind=*_generation={checkpoint.last_processed_generation}/*{MODEL_FILENAME}"
        )
        for file in files_to_delete:
            os.remove(file)
        gen: int = checkpoint.last_processed_generation - 2
        if checkpoint.last_processed_generation > 1:
            folders_to_delete = glob.glob(f"{save_path}/run_{checkpoint.run}/ind=*_generation={gen}")
            for folder in folders_to_delete:
                shutil.rmtree(folder)

    def _save_statistics_csv(self, save_path: str, checkpoint: Checkpoint) -> None:
        assert checkpoint.population is not None

        stats_path = os.path.join(save_path, f"run_{checkpoint.run}", STATS_FOLDER_NAME)

        with open(
            os.path.join(stats_path, f"generation_{checkpoint.last_processed_generation}.csv"), "w", encoding="utf-8"
        ) as csvfile:
            csvwriter = csv.writer(csvfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_MINIMAL)

            fields: tuple[str, ...] = (
                "id",
                "phenotype",
                "num_epochs",
                "total_training_time_allocated",
                "modules",
                "mutation_tracker",
            )

            assert checkpoint.population[0].metrics is not None

            csvwriter.writerow(fields + tuple(checkpoint.population[0].metrics.to_dict().keys()))

            for ind in checkpoint.population:
                assert ind.metrics is not None

                csvwriter.writerow(
                    [
                        ind.id,
                        ind.phenotype,
                        ind.num_epochs,
                        ind.total_allocated_train_time,
                        ind.mutation_tracker,
                        *ind.metrics.to_dict().values(),
                    ]
                )

        test_accuracies_path = os.path.join(stats_path, "test_accuracies.csv")
        file_exists: bool = os.path.isfile(test_accuracies_path)
        with open(test_accuracies_path, "a", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            if file_exists is False:
                csvwriter.writerow(["generation", "test_accuracy"])
            csvwriter.writerow([checkpoint.last_processed_generation, checkpoint.best_gen_ind_test_accuracy])

    def _save_statistics_json(self, save_path: str, checkpoint: Checkpoint) -> None:
        assert checkpoint.population is not None

        json_obj = [
            {
                "id": ind.id,
                "phenotype": ind.phenotype,
                "num_epochs": ind.num_epochs,
                "total_training_time_allocated": ind.total_allocated_train_time,
                "mutation_tracker": ind.mutation_tracker,
                **ind.metrics.to_dict(),  # type: ignore
            }
            for ind in checkpoint.population
        ]

        stats_path = os.path.join(save_path, f"run_{checkpoint.run}", STATS_FOLDER_NAME)

        with open(
            os.path.join(stats_path, f"generation_{checkpoint.last_processed_generation}.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(json_obj, file, indent=2, cls=Encoder)

        test_accuracies_path = os.path.join(stats_path, "test_accuracies.json")
        file_exists: bool = os.path.isfile(test_accuracies_path)

        if file_exists:
            with open(test_accuracies_path, encoding="utf-8") as json_file:
                test_accuracies = json.load(json_file)
        else:
            test_accuracies = {}

        test_accuracies[checkpoint.last_processed_generation] = checkpoint.best_gen_ind_test_accuracy

        with open(test_accuracies_path, "w", encoding="utf-8") as file:
            json.dump(test_accuracies, file, indent=4)


class Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> dict[str, Any] | Any:
        if isinstance(obj, Mutation):
            return {"mutation_type": str(obj.mutation_type), "gen": obj.gen, "data": obj.data}
        elif isinstance(obj, Fitness) or isinstance(obj, Layer) or isinstance(obj, Optimiser):
            return obj.to_dict()
        elif isinstance(obj, UUID):
            return str(obj)

        return super().default(obj)


def save_overall_best_individual(best_individual_path: str, parent: Individual) -> None:
    # pylint: disable=unexpected-keyword-arg
    shutil.copytree(
        best_individual_path, os.path.join(best_individual_path, "..", OVERALL_BEST_FOLDER), dirs_exist_ok=True
    )
    with open(os.path.join(best_individual_path, "..", OVERALL_BEST_FOLDER, "parent.pkl"), "wb") as handle:
        dill.dump(parent, handle)


def build_individual_path(checkpoint_base_path: str, run: int, generation: int, individual_id: UUID) -> str:
    return os.path.join(f"{checkpoint_base_path}", f"run_{run}", f"ind={individual_id}_generation={generation}")


def build_overall_best_path(checkpoint_base_path: str, run: int) -> str:
    return os.path.join(f"{checkpoint_base_path}", f"run_{run}", OVERALL_BEST_FOLDER)


def build_overall_best_trained_path(checkpoint_base_path: str, run: int) -> str:
    return os.path.join(f"{checkpoint_base_path}", f"run_{run}", OVERALL_BEST_TRAINED_FOLDER)
