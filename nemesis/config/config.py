import filecmp
import json
import keyword
import os
import shutil
from typing import Any, Iterator, Optional, Tuple, Dict, List

import yaml  # type: ignore
from jsonschema import validate  # type: ignore

from .module_config import ModuleConfig


class ConfigItem:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigItem(**value))
            else:
                if key in keyword.kwlist:
                    key = f"{key}_"
                setattr(self, key, value)

    def __getattr__(self, name: str) -> Optional[Any]:
        if name not in self.__dict__:
            return None
        return self.__dict__[name]

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def items(self) -> Iterator[Tuple[Any, Any]]:
        for key, value in self.__dict__.items():
            yield key, value

    def keys(self) -> Iterator[Any]:
        for key in self.__dict__.keys():
            yield key


class Config:
    def __init__(self, path: str) -> None:
        self._config: Any = self._load(path)
        self._validate_config()
        self._config["network"]["architecture"]["modules"] = self._convert_modules_configs()
        self._config = ConfigItem(**self._config)

        os.makedirs(self._config.checkpoints_path, exist_ok=True)

        self._backup_used_config(path, self._config.checkpoints_path)

    def __getattr__(self, name: str) -> Optional[Any]:
        return self._config.__getattr__(name)

    def _convert_modules_configs(self) -> List[ModuleConfig]:
        modules_configurations: List[ModuleConfig] = []

        for module_info in self._config["network"]["architecture"]["modules"]:
            assert (
                isinstance(module_info, Dict)
                and isinstance(module_info["network_structure"], list)
                and all(isinstance(x, int) for x in module_info["network_structure"])
                and isinstance(module_info["network_structure_init"], list)
                and all(isinstance(x, int) for x in module_info["network_structure_init"])
            )

            modules_configurations.append(
                ModuleConfig(
                    module_name=module_info["name"],
                    min_expansions=module_info["network_structure"][0],  # type: ignore
                    max_expansions=module_info["network_structure"][1],  # type: ignore
                    initial_network_structure=module_info["network_structure_init"],  # type: ignore
                    self_skip_connection_max_layers_back=module_info["self_skip_connection_max_layers_back"],
                    max_skip_connections=module_info["max_skip_connections"],
                    init_with_skip_connections=module_info["init_with_skip_connections"],
                    fusion_rule=module_info["fusion_rule"],
                    input_fusion_rule=module_info["input_fusion_rule"],
                    input_modules=module_info["input_modules"],
                    skip_connections_source_modules=module_info["skip_connections_source_modules"],
                )
            )

        return modules_configurations

    def _load(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                return json.load(f)
            if path.endswith(".yaml"):
                return yaml.safe_load(f)

            raise ValueError(f"File extension not supported: {path.split('.')[-1]}")

    def _backup_used_config(self, origin_filepath: str, destination: str) -> None:
        extension = origin_filepath.split(".")[-1]
        destination_filepath: str = os.path.join(destination, f"used_config.{extension}")

        # if there is a config file backed up already and it is different than the one we are trying to backup
        if os.path.isfile(destination_filepath) and filecmp.cmp(origin_filepath, destination_filepath) is False:
            raise ValueError(
                "You are probably trying to continue an experiment "
                "with a different config than the one you used initially. "
                "This is a gentle reminder to double-check the config you "
                "just passed as parameter."
            )

        # pylint: disable=protected-access
        if not shutil._samefile(origin_filepath, destination_filepath):  # type: ignore
            shutil.copyfile(origin_filepath, destination_filepath)

    def _validate_config(self) -> None:
        schema_path: str = os.path.join("nemesis", "config", "schema.json")

        schema: Any = self._load(schema_path)

        validate(self._config, schema)
