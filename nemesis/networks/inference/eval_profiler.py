from timeit import default_timer as timer

import torch
import torch.nn.functional as F

# import torch_tensorrt  # noqa: F401
from tqdm import tqdm
import logging

from nemesis.misc.validation_metrics import (
    ValidationMetricsCalculator,
    ValidationMetrics,
    EfficientRealTimeSemanticSegmentationMetrics,
    EfficientRealTimeSemanticSegmentationCalculator,
    EfficientRealTimeClassificationMetrics,
    EfficientRealTimeClassificationCalculator,
)
from nemesis.misc.utils import count_parameters

logger = logging.getLogger(__name__)


def evaluate_profiler(
    model: torch.nn.Module,
    metrics_constructor: ValidationMetricsCalculator,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> ValidationMetrics:
    power_drawn_step: int = 5
    total_power_drawn = 0.0

    smi_error_logged = False

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    torch._dynamo.reset()

    logging.getLogger("torch_tensorrt").setLevel(logging.WARNING)
    logging.getLogger("torch_tensorrt [TensorRT Conversion Context]").setLevel(logging.WARNING)

    # compiled_model = torch.compile(
    #     model,
    #     fullgraph=True,
    #     backend="torch_tensorrt",
    #     dynamic=False,
    #     options={
    #         "truncate_long_and_double": True,
    #         "precision": torch.half,  # type: ignore[dict-item]
    #         "debug": False,
    #         "use_python_runtime": False,
    #     },
    # )

    warmup_runs = 10

    with torch.no_grad():
        data = [
            (
                image.to(device=device, dtype=torch.float, non_blocking=True),
                label.to(device=device, dtype=torch.long, non_blocking=True),
            )
            for image, label in data_loader
        ]

        for i, vdata in enumerate(tqdm(data, desc="Validation progress")):
            images, true = vdata

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                predicted = model(images)[0]

                try:
                    if i % power_drawn_step == 0:
                        total_power_drawn += torch.cuda.power_draw(device=device)
                except ModuleNotFoundError:
                    if not smi_error_logged:
                        logger.warning("Can't get GPU power draw")

                        smi_error_logged = True

                if true.dim() > 2 and predicted.shape[:-2] != true.shape[:-2]:
                    predicted = F.interpolate(predicted, true.shape[-2:], align_corners=False, mode="bilinear")

                if true.dim() == 4:
                    true = true.squeeze(1)

            # metrics computing
            metrics_constructor.step(true, predicted)

        metrics = metrics_constructor.get_metrics()

        if not isinstance(metrics_constructor, EfficientRealTimeSemanticSegmentationCalculator) and not isinstance(
            metrics_constructor, EfficientRealTimeClassificationCalculator
        ):
            return metrics

        with torch.no_grad():
            for i, vdata in enumerate(tqdm(data, desc="Validation progress")):
                if i == warmup_runs:
                    start = timer()

                images, true = vdata

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    model(images)

        torch.cuda.synchronize()
        time_elapsed = timer() - start

        fps = (len(data_loader.dataset) - warmup_runs * data_loader.batch_size) / time_elapsed  # type: ignore[arg-type,operator]
        latency = time_elapsed / float((len(data_loader.dataset) - warmup_runs * data_loader.batch_size))  # type: ignore[arg-type,operator]

        total_power_drawn /= len(data_loader) // power_drawn_step

        assert isinstance(metrics, EfficientRealTimeClassificationMetrics) or isinstance(
            metrics, EfficientRealTimeSemanticSegmentationMetrics
        )

        metrics.gpu_power_drawn = total_power_drawn / 1000
        metrics.gpu_peak_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"] / (1000**3)
        metrics.throughput = fps
        metrics.latency = latency
        metrics.number_parameters = count_parameters(model)

    return metrics
