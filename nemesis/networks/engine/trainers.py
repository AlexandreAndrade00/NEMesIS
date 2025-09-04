import logging
import gc
from typing import List, Optional, Dict, Any, Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from nemesis.misc import ValidationMetrics
from nemesis.misc.enums import Device
from nemesis.misc.validation_metrics import ValidationMetricsCalculator
from nemesis.networks.engine import callbacks as cb
from nemesis.networks.inference import train as train_epoch, evaluate


from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimiser: optim.Optimizer,
        train_data_loader: DataLoader,
        num_classes: int,
        validation_data_loader: DataLoader,
        n_epochs: int,
        initial_epoch: int,
        device: Device,
        validation_metrics_calculator: type[ValidationMetricsCalculator],
        binary_loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
        multiclass_loss_fn: nn.Module = nn.CrossEntropyLoss(),
        callbacks: List["cb.Callback"] | None = None,
        scheduler: Optional[LRScheduler] = None,
        gpu_id: int | None = None,
        timeout: bool = True,
        load_dataset: bool = False,
    ) -> None:
        self.gpu_id = gpu_id
        self.model: nn.Module = model
        self.optimiser: optim.Optimizer = optimiser
        self.binary_loss_fn: nn.Module = binary_loss_fn
        self.multiclass_loss_fn: nn.Module = multiclass_loss_fn
        self.train_data_loader: DataLoader = train_data_loader
        self.num_classes: int = num_classes
        self.validation_data_loader: DataLoader = validation_data_loader
        self.n_epochs: int = n_epochs
        self.initial_epoch: int = initial_epoch
        self.device: torch.device
        self.validation_metrics_calculator: type[ValidationMetricsCalculator] = validation_metrics_calculator
        self.stop_training: bool = False
        self.trained_epochs: int = 0
        self.train_loss: List[float] = []
        self.state_dicts: List[Dict[str, Any]] = []
        self.validation_metrics: List[ValidationMetrics] = []
        self.callbacks: List["cb.Callback"] = [] if callbacks is None else callbacks
        self.timeout = timeout
        self.load_dataset = load_dataset

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.mode = "max"

        self.scheduler: Optional[LRScheduler] = scheduler

        self._model: nn.Module

        if device == Device.GPU and torch.cuda.device_count() > 1:
            assert gpu_id is not None
            self.device = torch.device(gpu_id)

            self._model = model.to(self.device)

            self._model = DDP(self._model, device_ids=[gpu_id], find_unused_parameters=True)
        else:
            self.device = device.to_torch_device()

            self._model = model.to(self.device)

    def _call_on_train_begin_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_train_begin(self)

    def _call_on_train_end_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_train_end(self)

    def _call_on_epoch_begin_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_epoch_begin(self)

    def _call_on_epoch_end_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_epoch_end(self)

    def train(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        logging.info("Initiating supervised training")

        self.train_loss = []
        self.state_dicts = []
        self.validation_metrics = []
        self.stop_training = False

        epoch: int = self.initial_epoch

        self.train_data: Iterable
        self.val_data: Iterable | None = None

        if self.load_dataset:
            self.train_data = [
                (
                    image.to(device=self.device, dtype=torch.float, non_blocking=True),
                    label.to(device=self.device, dtype=torch.long, non_blocking=True),
                )
                for image, label in self.train_data_loader
            ]

            if self.validation_data_loader is not None:
                self.val_data = [
                    (
                        image.to(device=self.device, dtype=torch.float, non_blocking=True),
                        label.to(device=self.device, dtype=torch.long, non_blocking=True),
                    )
                    for image, label in self.validation_data_loader
                ]
        else:
            self.train_data = self.train_data_loader
            self.val_data = self.validation_data_loader

        self._call_on_train_begin_callbacks()

        loss_fn = self.binary_loss_fn if self.num_classes == 1 else self.multiclass_loss_fn

        while epoch < self.n_epochs and self.stop_training is False:
            logger.info(f"Epoch: {epoch}  Stop training: {self.stop_training}")
            self._call_on_epoch_begin_callbacks()

            # train
            train_loss = train_epoch(self.model, self.optimiser, loss_fn, self.train_data, self.device, self.timeout)

            self.train_loss.append(train_loss)

            self.state_dicts.append({k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()})

            # evaluate
            if self.val_data is not None:
                self.model.eval()

                validation_metrics = evaluate(
                    self.model,
                    self.validation_metrics_calculator(self.num_classes, self.device),
                    self.val_data,
                    self.device,
                )

                self.validation_metrics.append(validation_metrics)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if validation_metrics is not None:
                        self.scheduler.step(validation_metrics.get_early_stopping_metric())
                else:
                    self.scheduler.step()

            epoch += 1

            self._call_on_epoch_end_callbacks()

        self._call_on_train_end_callbacks()

        self.trained_epochs = epoch - self.initial_epoch

    def load_best_state_dict(self) -> None:
        if self.validation_data_loader is not None:
            best_idx = np.argmax([metric.get_early_stopping_metric() for metric in self.validation_metrics])
        else:
            best_idx = np.argmin(self.train_loss)

        self.model.load_state_dict(self.state_dicts[best_idx])
