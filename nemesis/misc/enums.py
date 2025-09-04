from dataclasses import astuple, dataclass
from enum import Enum, unique
from typing import Any, Iterator, List, Literal, TypeVar

import torch

from nemesis.misc.fitness_metrics import (
    EfficientRTSemSegFitness,
    FitnessMetric,
    EfficientRTClassificationMetric,
)
from nemesis.misc.validation_metrics import (
    ClassificationCalculator,
    EfficientRealTimeSemanticSegmentationCalculator,
    SemanticSegmentationCalculator,
    ValidationMetricsCalculator,
    EfficientRealTimeClassificationCalculator,
)


@unique
class AttributeType(Enum):
    INT = "int"
    INT_POWER2 = "int_power2"
    INT_POWER2_INV = "inv_power2"
    FLOAT = "float"


T = TypeVar("T")


class ExtendedEnum(Enum):
    @classmethod
    def enum_values(cls) -> List[T]:
        return list(map(lambda c: c.value, cls))  # type: ignore


@unique
class Entity(ExtendedEnum):
    LAYER = "layer"
    OPTIMISER = "learning"
    FUSION = "fuse"


@unique
class Device(Enum):
    CPU = "cpu"
    GPU = "mps" if torch.backends.mps.is_available() else "cuda"

    def to_torch_device(self) -> torch.device:
        return torch.device(self.value)

    def decide_device(self) -> Literal[CPU, GPU]:
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            return Device.CPU

        return self


@unique
class TrainType(ExtendedEnum):
    NORMAL = "normal"
    PRETRAIN = "pretrain"


@unique
class LayerType(ExtendedEnum):
    pass


@unique
class BackboneType(LayerType):
    CONV = "conv"
    DSCONV = "depthwise_separable_conv"
    IDENTITY = "identity"
    FC = "fc"
    ADAPTIVE_AVG_POOL = "adaptive_avg_pool"
    PPM = "ppm"
    STDC = "stdc"
    RESNET_BASIC = "resnet_basic"
    RESNET_BOTTLENECK = "resnet_bottleneck"


@unique
class HeadType(LayerType):
    RTMDET_INS_HEAD = "rtmdet_ins"
    CLASSIFICATION_HEAD = "classification_head"
    SEMANTIC_HEAD = "semantic_head"


@unique
class OptimiserType(str, Enum):
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    LARS = "lars"


@unique
class FuseType(Enum):
    ADD = "add"
    CAT = "cat"
    UAFM_SP = "uafm_sp"
    UAFM_CH = "uafm_ch"

    @classmethod
    def from_string(cls, condition: str) -> Literal[ADD, CAT, UAFM_SP, UAFM_CH]:
        match condition:
            case "add":
                return cls.ADD
            case "cat":
                return cls.CAT
            case "uafm_sp":
                return cls.UAFM_SP
            case "uafm_ch":
                return cls.UAFM_CH
            case _:
                raise ValueError("Unknown enum value")


@unique
class ActivationType(Enum):
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


class MutationType(ExtendedEnum):
    TRAIN_LONGER = "train_longer"
    ADD_LAYER = "add_layer"
    REUSE_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    DSGE_LAYER = "dsge_layer"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    DSGE_MACRO = "dsge_macro"

    def __str__(self) -> str:
        return self.value


@dataclass
class Mutation:
    mutation_type: MutationType
    gen: int
    data: dict

    def __str__(self) -> str:
        return f"Mutation type [{self.mutation_type}] on gen [{self.gen}] with data: {self.data}"

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))


@unique
class Task(Enum):
    CLASSIFICATION = "classification"
    REAL_TIME_CLASSIFICATION = "real_time_classification"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    REAL_TIME_SEMANTIC_SEGMENTATION = "real_time_semantic_segmentation"

    @classmethod
    def from_string(cls, task: str) -> Literal[CLASSIFICATION, SEMANTIC_SEGMENTATION, REAL_TIME_SEMANTIC_SEGMENTATION]:
        match task:
            case "classification":
                return cls.CLASSIFICATION
            case "semantic_segmentation":
                return cls.SEMANTIC_SEGMENTATION
            case "real_time_semantic_segmentation":
                return cls.REAL_TIME_SEMANTIC_SEGMENTATION
            case _:
                raise ValueError("Unknown enum value")

    def default_fitness_metric(self) -> type[FitnessMetric]:
        match self:
            case Task.CLASSIFICATION:
                raise NotImplementedError()
            case Task.SEMANTIC_SEGMENTATION:
                raise NotImplementedError()
            case Task.REAL_TIME_SEMANTIC_SEGMENTATION:
                return EfficientRTSemSegFitness
            case Task.REAL_TIME_CLASSIFICATION:
                return EfficientRTClassificationMetric
            case _:
                raise ValueError("Unexpected task")

    def validation_metrics_calculator(self) -> type[ValidationMetricsCalculator]:
        match self:
            case Task.CLASSIFICATION:
                return ClassificationCalculator
            case Task.SEMANTIC_SEGMENTATION:
                return SemanticSegmentationCalculator
            case Task.REAL_TIME_SEMANTIC_SEGMENTATION:
                return EfficientRealTimeSemanticSegmentationCalculator
            case Task.REAL_TIME_CLASSIFICATION:
                return EfficientRealTimeClassificationCalculator
            case _:
                raise ValueError("Unexpected task")


@unique
class ResizeTarget(Enum):
    FIRST = "first"
    LARGEST = "largest"


@unique
class UpsampleType(Enum):
    INTERPOLATION = "interpolation"


@unique
class DownsampleType(Enum):
    CONV = "conv_stride_2"
    INTERPOLATION = "interpolation"
    AVG_POOL = "avg_pool"


@unique
class PPMAlgorithm(Enum):
    SPPM = "sppm"
    DAPPM = "dappm"
