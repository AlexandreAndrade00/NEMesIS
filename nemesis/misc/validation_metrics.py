from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nemesis.misc.metrics import accuracy as acc
from nemesis.misc.metrics import (
    accuracy_multiclass,
    calculate_dice,
    calculate_iou,
)


class ValidationMetrics(ABC):
    r"""Validation metrics base class"""

    @abstractmethod
    def get_early_stopping_metric(self) -> float: ...


class ValidationMetricsCalculator(ABC):
    r"""Validation metrics calculator base class"""

    @abstractmethod
    def __init__(self, num_classes: int, device: torch.device): ...

    @abstractmethod
    def step(self, true: torch.Tensor, predicted: torch.Tensor) -> None: ...

    @abstractmethod
    def get_metrics(self) -> ValidationMetrics: ...

    @classmethod
    @abstractmethod
    def get_empty_metrics(cls) -> ValidationMetrics: ...


@dataclass
class SemanticSegmentationMetrics(ValidationMetrics):
    r"""Semantic segmentation metrics

    Parameters:
        iou (float): Intersection over Union (IoU) score, a metric for evaluating the accuracy of object detection or
            segmentation models.  Ranges from 0 to 1, with higher values indicating better overlap between predicted
            and ground truth regions.
        dice_coefficient (float): Dice coefficient (also known as Sørensen–Dice coefficient), another metric for
            evaluating the similarity between two sets, often used in image segmentation. Ranges from 0 to 1, with
            higher values indicating better agreement.
    """

    def __init__(self, iou: float, dice_coefficient: float) -> None:
        self.iou = iou
        self.dice_coefficient = dice_coefficient

    def get_early_stopping_metric(self) -> float:
        return self.iou


class SemanticSegmentationCalculator(ValidationMetricsCalculator):
    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.steps: torch.Tensor = torch.zeros(1, device=device)
        self.iou_total: torch.Tensor = torch.zeros(1, device=device)
        self.dice_total: torch.Tensor = torch.zeros(1, device=device)

    def step(self, true: torch.Tensor, predicted: torch.Tensor) -> None:
        metric_iou: torch.Tensor
        metric_dice: torch.Tensor

        if self.num_classes == 1:
            assert true.min() >= 0 and true.max() <= 1, "True mask indices should be in [0, 1]"

            masks_pred = (F.sigmoid(predicted) > 0.5).float()

            metric_iou = calculate_iou(masks_pred.detach().clone(), true, 1).mean()
            metric_dice = calculate_dice(masks_pred.detach().clone(), true, 1).mean()
        else:
            metric_iou = calculate_iou(predicted.detach().clone(), true, self.num_classes).mean()
            metric_dice = calculate_dice(predicted.detach().clone(), true, self.num_classes).mean()

        self.iou_total += metric_iou
        self.dice_total += metric_dice

        self.steps += 1

    def get_metrics(self) -> SemanticSegmentationMetrics:
        if self.steps == 0:
            return SemanticSegmentationMetrics(iou=0, dice_coefficient=0)

        return SemanticSegmentationMetrics(
            iou=(self.iou_total / self.steps).cpu().item(), dice_coefficient=(self.dice_total / self.steps).cpu().item()
        )

    @classmethod
    def get_empty_metrics(cls) -> SemanticSegmentationMetrics:
        return SemanticSegmentationMetrics(iou=0, dice_coefficient=0)


@dataclass
class ClassificationMetrics(ValidationMetrics):
    def __init__(self, accuracy: float) -> None:
        self.accuracy = accuracy

    def get_early_stopping_metric(self) -> float:
        return self.accuracy


class ClassificationCalculator(ValidationMetricsCalculator):
    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.steps: int = 0
        self.accuracy: torch.Tensor = torch.zeros(1, dtype=torch.float, device=device)

    def step(self, true: torch.Tensor, predicted: torch.Tensor) -> None:
        accuracy_score: float

        if self.num_classes == 1:
            assert true.min() >= 0 and true.max() <= 1, "True mask indices should be in [0, 1]"

            masks_pred = (F.sigmoid(predicted) > 0.5).float()

            accuracy_score = acc(true, masks_pred)
        else:
            assert true.min() >= 0 and true.max() < self.num_classes, "True mask indices should be in [0, n_classes["

            accuracy_score = accuracy_multiclass(true, predicted.argmax(dim=1), self.num_classes)

        self.accuracy += accuracy_score

        self.steps += 1

    def get_metrics(self) -> ClassificationMetrics:
        return ClassificationMetrics(accuracy=(self.accuracy / self.steps).cpu().item())

    @classmethod
    def get_empty_metrics(cls) -> ClassificationMetrics:
        return ClassificationMetrics(accuracy=0)


@dataclass
class EfficiencyMetrics(ValidationMetrics):
    r"""Efficiency metrics

    Parameters:
        throughput (float): The processing speed of the model, measured in images per second. Higher values indicate
            faster processing.
        latency (float): The delay between input and output of the model, measured in milliseconds. Lower values
            indicate faster response times.
        gpu_peak_memory (float): The maximum amount of GPU memory used during validation, measured in GB.
        gpu_allocated_memory (float): The amount of GPU memory allocated for the model during validation, measured
            in GB.
        cpu_peak_memory (float): The maximum amount of CPU memory used during validation, measured in GB.
        cpu_allocated_memory (float): The amount of CPU memory allocated for the model during validation, measured
            in GB.
        gpu_power_drawn (float): The amount of power consumed by the GPU during validation, measured in Watts.
    """

    def get_early_stopping_metric(self) -> float:
        raise NotImplementedError

    def __init__(
        self,
        gpu_peak_memory: float,
        gpu_power_drawn: float,
        throughput: float,
        latency: float,
        number_parameters: int,
    ) -> None:
        self.gpu_peak_memory = gpu_peak_memory
        self.gpu_power_drawn = gpu_power_drawn
        self.throughput = throughput
        self.latency = latency
        self.number_parameters = number_parameters


class EfficiencyCalculator(ValidationMetricsCalculator):
    def get_metrics(self) -> EfficiencyMetrics:
        raise NotImplementedError()

    def step(self, true_masks: torch.Tensor, predicted_masks: torch.Tensor) -> None:
        raise NotImplementedError()

    @classmethod
    def get_empty_metrics(cls) -> EfficiencyMetrics:
        return EfficiencyMetrics(
            gpu_peak_memory=0.0,
            throughput=0.0,
            latency=0.0,
            gpu_power_drawn=0.0,
            number_parameters=0,
        )


class EfficientRealTimeSemanticSegmentationMetrics(ValidationMetrics):
    def __init__(
        self,
        iou: float,
        dice_coefficient: float,
        gpu_peak_memory: float,
        gpu_power_drawn: float,
        throughput: float,
        latency: float,
        number_parameters: int,
    ):
        self.iou = iou
        self.dice_coefficient = dice_coefficient
        self.gpu_peak_memory = gpu_peak_memory
        self.gpu_power_drawn = gpu_power_drawn
        self.throughput = throughput
        self.latency = latency
        self.number_parameters = number_parameters

    def get_early_stopping_metric(self) -> float:
        return self.iou

    def __str__(self) -> str:
        return str(self.__dict__)


class EfficientRealTimeSemanticSegmentationCalculator(ValidationMetricsCalculator):
    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.semantic_segmentation_calculator = SemanticSegmentationCalculator(num_classes, device)

    def step(self, true_masks: torch.Tensor, predicted_masks: torch.Tensor) -> None:
        self.semantic_segmentation_calculator.step(true_masks, predicted_masks)

    def get_metrics(self) -> EfficientRealTimeSemanticSegmentationMetrics:
        semantic_metrics: SemanticSegmentationMetrics = self.semantic_segmentation_calculator.get_metrics()

        return EfficientRealTimeSemanticSegmentationMetrics(
            dice_coefficient=semantic_metrics.dice_coefficient,
            iou=semantic_metrics.iou,
            throughput=0.0,
            latency=0.0,
            gpu_power_drawn=0.0,
            gpu_peak_memory=0.0,
            number_parameters=0,
        )

    @classmethod
    def get_empty_metrics(cls) -> EfficientRealTimeSemanticSegmentationMetrics:
        return EfficientRealTimeSemanticSegmentationMetrics(
            gpu_power_drawn=0.0,
            gpu_peak_memory=0.0,
            throughput=0.0,
            latency=0.0,
            dice_coefficient=0.0,
            iou=0.0,
            number_parameters=0,
        )


class EfficientRealTimeClassificationMetrics(ValidationMetrics):
    def __init__(
        self,
        accuracy: float,
        gpu_peak_memory: float,
        gpu_power_drawn: float,
        throughput: float,
        latency: float,
        number_parameters: int,
    ):
        self.accuracy = accuracy
        self.gpu_peak_memory = gpu_peak_memory
        self.gpu_power_drawn = gpu_power_drawn
        self.throughput = throughput
        self.latency = latency
        self.number_parameters = number_parameters

    def get_early_stopping_metric(self) -> float:
        return self.accuracy

    def __str__(self) -> str:
        return str(self.__dict__)


class EfficientRealTimeClassificationCalculator(ValidationMetricsCalculator):
    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.classification_calculator = ClassificationCalculator(num_classes, device)

    def step(self, true_masks: torch.Tensor, predicted_masks: torch.Tensor) -> None:
        self.classification_calculator.step(true_masks, predicted_masks)

    def get_metrics(self) -> EfficientRealTimeClassificationMetrics:
        classification_metrics: ClassificationMetrics = self.classification_calculator.get_metrics()

        return EfficientRealTimeClassificationMetrics(
            accuracy=classification_metrics.accuracy,
            throughput=0.0,
            latency=0.0,
            gpu_power_drawn=0.0,
            gpu_peak_memory=0.0,
            number_parameters=0,
        )

    @classmethod
    def get_empty_metrics(cls) -> EfficientRealTimeClassificationMetrics:
        return EfficientRealTimeClassificationMetrics(
            gpu_power_drawn=0.0,
            gpu_peak_memory=0.0,
            throughput=0.0,
            latency=0.0,
            accuracy=0.0,
            number_parameters=0,
        )
