from pathlib import Path
from typing import Callable, Tuple, Any

from torchvision.datasets import VisionDataset


class ImageNet21KProcessed(VisionDataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        raise NotImplementedError()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.images)
