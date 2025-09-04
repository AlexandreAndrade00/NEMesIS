from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from nemesis.misc.validation_metrics import ValidationMetricsCalculator, ValidationMetrics


def evaluate(
    model: torch.nn.Module,
    metrics_constructor: ValidationMetricsCalculator,
    data_loader: Iterable,
    device: torch.device,
) -> ValidationMetrics:
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(tqdm(data_loader, desc="Validation progress")):
            images, true = vdata

            images = images.to(device=device, dtype=torch.float, non_blocking=True)

            true = true.to(device=device, dtype=torch.long, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = model(images)[0]

                if true.dim() > 2 and pred.shape[:-2] != true.shape[:-2]:
                    pred = F.interpolate(pred, true.shape[-2:], align_corners=False, mode="bilinear")

                if true.dim() == 4:
                    true = true.squeeze(1)

            metrics_constructor.step(true, pred)

        metrics = metrics_constructor.get_metrics()

    return metrics
