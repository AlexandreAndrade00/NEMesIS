import os
from typing import List, Callable, Any, Tuple
from pathlib import Path

from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import Image, Mask
import PIL.Image


class SceneParsing150(VisionDataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "training",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        assert split in ["training", "validation"]

        self.images_dir = os.path.join(self.root, "images", split)
        self.targets_dir = os.path.join(self.root, "annotations", split)

        assert os.path.isdir(self.images_dir)
        assert os.path.isdir(self.targets_dir)

        self.split = split
        self.images: List[str] = []
        self.targets: List[str] = []

        for image_file in os.listdir(self.images_dir):
            image_dir = os.path.join(self.images_dir, image_file)
            target_dir = os.path.join(self.targets_dir, image_file.split(".")[0] + ".png")

            self.images.append(image_dir)
            self.targets.append(target_dir)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image(PIL.Image.open(self.images[index]).convert("RGB"))

        target = Mask(PIL.Image.open(self.targets[index]))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
