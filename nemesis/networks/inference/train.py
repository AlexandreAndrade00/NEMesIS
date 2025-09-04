from typing import Callable, Iterable
from timeit import default_timer as timer

import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: Iterable,
    device: torch.device,
    timeout: bool = True,
) -> float:
    total_loss: torch.Tensor = torch.zeros(1, device=device)

    model.train(True)

    scaler = torch.amp.GradScaler(device.type)

    start = timer()

    for i, data in enumerate(tqdm(data_loader, desc="Training progress")):
        images, true = data

        images = images.to(device=device, dtype=torch.float, non_blocking=True)
        true = true.to(device=device, dtype=torch.long, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            pred = model(images)[0]

            if true.dim() > 2 and pred.shape[:-2] != true.shape[:-2]:
                pred = F.interpolate(pred, true.shape[-2:], align_corners=False, mode="bilinear")

            if true.dim() == 4:
                true = true.squeeze(1)

            loss = loss_fn(pred, true)

        total_loss += loss

        scaler.scale(loss).backward()

        scaler.step(optimiser)

        scaler.update()

        if timeout and timer() - start > 900:
            raise TimeoutError("Model taking too long to train")

    loss_avg: float = total_loss.item() / sum(1 for _ in data_loader)

    return loss_avg
