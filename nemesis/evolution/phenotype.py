from typing import Any, Dict, Tuple

from nemesis.misc.enums import FuseType, OptimiserType


class Fusion:
    def __init__(
        self,
        fuse_type: FuseType,
        fusion_parameters: Dict[str, str],
    ) -> None:
        self.fuse_type = fuse_type
        self.fusion_parameters: Dict[str, Any] = dict(self._convert(k, v) for k, v in fusion_parameters.items())

    def _convert(self, key: str, value: str) -> Tuple[str, Any]:
        if key in [
            "fuse",
            "resize_target",
            "fusion_upsample",
            "fusion_downsample",
        ]:
            return key, value

        raise ValueError(f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fusion):
            return False

        return self.__dict__ == other.__dict__


class Layer:
    def __init__(
        self,
        type: str,
        parameters: Dict[str, str],
    ) -> None:
        self.type: str = type
        self.parameters: Dict[str, Any] = dict(self._convert(k, v) for k, v in parameters.items())

    def _convert(self, key: str, value: str) -> Tuple[str, Any]:
        if key in ["bias", "align_corners", "batch_norm"]:
            return key, value.title() == "True"
        if key in ["rate"]:
            return key, float(value)
        if (
            key
            in [
                "out_channels",
                "out_features",
                "kernel_size",
                "stride",
                "bin_size",
                "inter_channels",
                "block_num",
                "dilation",
                "number_classes",
            ]
            or (key == "padding" and not (value == "same" or value == "valid" or value == "stride_dep"))
            or (key == "back_layers" and value != "last_module")
        ):
            return key, int(value)
        if (
            key in ["act", "mode", "condition", "ppm_algorithm", "stdc_block"]
            or (key == "padding" and (value == "same" or value == "valid" or value == "stride_dep"))
            or (key == "back_layers" and value == "last_module")
        ):
            return key, value
        if key == "input":
            return key, list(map(int, value))
        if key == "kernel_size_fix":
            return "kernel_size", tuple(map(int, value[1:-1].split(",")))
        if key == "padding_fix":
            return "padding", tuple(map(int, value[1:-1].split(",")))
        raise ValueError(f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Layer):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        return f"Module [{self.type}] with params: {self.parameters}"

    def to_dict(self) -> Dict[str, str]:
        return {"type": str(self.type), "parameters": str(self.parameters)}


class Optimiser:
    def __init__(self, optimiser_type: OptimiserType, optimiser_parameters: Dict[str, str]) -> None:
        self.optimiser_type: OptimiserType = optimiser_type
        self.optimiser_parameters: Dict[str, Any] = {k: self._convert(k, v) for k, v in optimiser_parameters.items()}

    def _convert(self, key: str, value: str) -> Any:
        if key == "nesterov":
            return value.title() == "True"
        if key in ["lr", "lr_weights", "lr_biases", "alpha", "weight_decay", "momentum", "beta1", "beta2"]:
            return float(value)
        if key in ["epochs"]:
            return int(value)
        raise ValueError(f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Optimiser):
            return self.__dict__ == other.__dict__
        return False

    def to_dict(self) -> Dict[str, str]:
        return {"optimiser_type": str(self.optimiser_type), "optimiser_parameters": str(self.optimiser_parameters)}
