from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


from nemesis.misc.validation_metrics import (
    EfficientRealTimeClassificationMetrics,
    EfficientRealTimeSemanticSegmentationMetrics,
)


class Fitness:
    def __init__(
        self,
        fitness: float,
        accuracy_metric: float,
        all_values: Dict[str, float],
        metric_type: type[FitnessMetric],
    ) -> None:
        if accuracy_metric < 0 or accuracy_metric > 1:
            raise ValueError("Lambda control metric should be between 0 and 1")

        self.fitness: float = fitness
        self.all_values: Dict[str, float] = all_values
        self.accuracy_metric: float = accuracy_metric
        self.metric_type: type[FitnessMetric] = metric_type

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fitness):
            return self.__dict__ == other.__dict__
        return False

    def __lt__(self, other: Fitness) -> bool:
        return self.metric_type.worse_than(self, other)

    def __gt__(self, other: Fitness) -> bool:
        return self.metric_type.better_than(self, other)

    def __leq__(self, other: Fitness) -> bool:
        return self.metric_type.worse_or_equal_than(self, other)

    def __geq__(self, other: Fitness) -> bool:
        return self.metric_type.better_or_equal_than(self, other)

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        return {"fitness": self.fitness, "all_values": str(self.all_values), "metric": str(self.metric_type.__name__)}


class FitnessMetric(ABC):
    @abstractmethod
    def compute_metric(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def accuracy_metric(self) -> float:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worst_fitness(cls) -> Fitness:
        raise NotImplementedError()


class EfficientRTClassificationMetric(FitnessMetric):
    def __init__(self, validation_metric: EfficientRealTimeClassificationMetrics) -> None:
        super().__init__()
        self.accuracy: float = validation_metric.accuracy
        self.gpu_power_drawn: float = validation_metric.gpu_power_drawn
        self.gpu_peak_memory: float = validation_metric.gpu_peak_memory
        self.throughput: float = validation_metric.throughput
        self.number_parameters: int = validation_metric.number_parameters

    def compute_metric(self) -> float:
        return self.accuracy

    def accuracy_metric(self) -> float:
        return self.accuracy

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.fitness < other.fitness

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.fitness > other.fitness

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.fitness <= other.fitness

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.fitness >= other.fitness

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(-1.0, 0, {}, cls)


class EfficientRTSemSegFitness(FitnessMetric):
    def __init__(self, validation_metric: EfficientRealTimeSemanticSegmentationMetrics) -> None:
        self.iou: float = validation_metric.iou
        self.dice_coefficient = validation_metric.dice_coefficient
        self.gpu_power_drawn: float = validation_metric.gpu_power_drawn
        self.gpu_peak_memory: float = validation_metric.gpu_peak_memory
        self.throughput: float = validation_metric.throughput
        self.number_parameters: int = validation_metric.number_parameters

    def __str__(self) -> str:
        return str(self.__dict__)

    def compute_metric(self) -> float:
        # return self.iou * 3000 - self.gpu_peak_memory * 20 - self.gpu_power_drawn * 0.2 + self.throughput * 0.1  # 2
        return self.iou * 3000 - self.gpu_peak_memory * 30 - self.gpu_power_drawn * 0.3 + self.throughput * 0.15  # 1

    def accuracy_metric(self) -> float:
        return self.iou

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(-1.0, 0, {}, cls)

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.fitness >= other.fitness

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.fitness <= other.fitness

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.fitness > other.fitness

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.fitness < other.fitness
