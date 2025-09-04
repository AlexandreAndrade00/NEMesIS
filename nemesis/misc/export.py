import os
from typing import Tuple

import torch
import torch_tensorrt


def export_torch(model: torch.nn.Module, path: str, input_shape: Tuple[int, ...]) -> None:
    exported_program = torch.export.export(
        model.eval().cuda(),
        (torch.randn((2, *input_shape), device="cuda"),),
        strict=True,
        dynamic_shapes={
            "x": (torch.export.Dim.AUTO, torch.export.Dim.STATIC, torch.export.Dim.STATIC, torch.export.Dim.STATIC)  # type: ignore[attr-defined]
        },
    )

    exported_training = torch.export.export_for_training(
        model.eval().cuda(),
        (torch.randn((2, *input_shape), device="cuda"),),
        strict=True,
        dynamic_shapes={
            "x": (torch.export.Dim.AUTO, torch.export.Dim.STATIC, torch.export.Dim.STATIC, torch.export.Dim.STATIC)  # type: ignore[attr-defined]
        },
    )

    torch.export.save(exported_program, os.path.join(path, "model.pt2"))
    torch.export.save(exported_training, os.path.join(path, "model_training.pt2"))
    # saved_exported_program = torch.export.load("exported_program.pt2")


def export_tensorrt(model: torch.nn.Module, path: str, input_shape: Tuple[int, ...]) -> None:
    inputs = [torch.randn((1, *input_shape), device="cuda")]

    trt_gm = torch_tensorrt.compile(model.eval().cuda(), ir="dynamo", inputs=inputs)

    torch_tensorrt.save(trt_gm, os.path.join(path, "model.ep"), inputs=inputs)

    # Later, you can load it and run inference
    # model = torch.export.load("trt.ep").module()
