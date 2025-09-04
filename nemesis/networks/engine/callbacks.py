import logging
import os
import math
from abc import ABC, abstractmethod
from time import time

import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nemesis.misc.constants import MODEL_FILENAME, WEIGHTS_FILENAME
from . import trainers

logger = logging.getLogger(__name__)


class Callback(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def on_train_begin(self, trainer: trainers.Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_train_end(self, trainer: trainers.Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_epoch_begin(self, trainer: trainers.Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_epoch_end(self, trainer: trainers.Trainer) -> None:
        raise NotImplementedError()


class ModelCheckpointCallback(Callback):
    def __init__(
        self,
        model_saving_dir: str,
        model_filename: str = MODEL_FILENAME,
        weights_filename: str = WEIGHTS_FILENAME,
    ) -> None:
        super().__init__()
        self.model_saving_dir: str = model_saving_dir
        self.model_filename: str = model_filename
        self.weights_filename: str = weights_filename

    def on_train_begin(self, trainer: trainers.Trainer) -> None:
        pass

    def on_train_end(self, trainer: trainers.Trainer) -> None:
        if trainer.gpu_id is not None and trainer.gpu_id != 0:
            return

        if isinstance(trainer.model, DDP):
            model = trainer.model.module
        else:
            model = trainer.model

        highest_accuracy_idx = np.argmax([metric.get_early_stopping_metric() for metric in trainer.validation_metrics])

        best_state_dict = trainer.state_dicts[highest_accuracy_idx]

        torch.save(model, os.path.join(self.model_saving_dir, self.model_filename))

        torch.save(best_state_dict, os.path.join(self.model_saving_dir, self.weights_filename))

    def on_epoch_begin(self, trainer: trainers.Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: trainers.Trainer) -> None:
        pass


class TimedStoppingCallback(Callback):
    def __init__(self, max_seconds: float) -> None:
        super().__init__()
        self.start_time: float = 0.0
        self.max_seconds: float = max_seconds

    def on_train_begin(self, trainer: trainers.Trainer) -> None:
        self.start_time = time()

    def on_train_end(self, trainer: trainers.Trainer) -> None:
        pass

    def on_epoch_begin(self, trainer: trainers.Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: trainers.Trainer) -> None:
        if time() - self.start_time > self.max_seconds:
            trainer.stop_training = True


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int) -> None:
        super().__init__()
        self.patience: int = patience
        self.best_score: float = 0
        self.counter: int = 0

    def on_train_begin(self, trainer: trainers.Trainer) -> None:
        self.counter = 0

    def on_train_end(self, trainer: trainers.Trainer) -> None:
        pass

    def on_epoch_begin(self, trainer: trainers.Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: trainers.Trainer) -> None:
        early_stop_metric: float = trainer.validation_metrics[-1].get_early_stopping_metric()

        early_stop_coefficient = math.log(1 / (self.best_score + 1e-7), 10) / 30

        if self.best_score > 0 and early_stop_metric < self.best_score * (1 + early_stop_coefficient):
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %d out of %d. Best score %f, current: %f",
                self.counter,
                self.patience,
                self.best_score,
                early_stop_metric,
            )
            if self.counter >= self.patience:
                trainer.stop_training = True
        else:
            self.best_score = early_stop_metric
            self.counter = 0


class PrintMetricsCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_begin(self, trainer: trainers.Trainer) -> None:
        pass

    def on_train_end(self, trainer: trainers.Trainer) -> None:
        pass

    def on_epoch_begin(self, trainer: trainers.Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: trainers.Trainer) -> None:
        print(str(trainer.validation_metrics[-1]))
