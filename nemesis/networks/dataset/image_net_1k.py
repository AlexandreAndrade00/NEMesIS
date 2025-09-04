import os
from pathlib import Path
from typing import Callable, List, Tuple, Any, Dict

from torch import tensor
from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import Image
import PIL.Image


class ImageNet1K(VisionDataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        assert split in ["train", "val"]

        self.images_dir = os.path.join(self.root, "Data", "CLS-LOC", split)
        self.targets_dir = os.path.join(self.root, "Annotations", "CLS-LOC", split)

        self.split = split
        self.images: List[str] = []
        self.targets: List[int] = []

        classes_mapping: Dict[str, int] = {}

        if split == "train":
            classes_dir = os.path.join(self.root, "devkit", "data", "map_clsloc.txt")

            with open(classes_dir, "r") as fp:
                for meta in fp.readlines():
                    folder_name, class_idx, class_name = meta.split(" ")

                    classes_mapping[folder_name] = int(class_idx) - 1
        elif split == "val":
            classes_dir = os.path.join(self.root, "devkit", "data", "ILSVRC2015_clsloc_validation_ground_truth.txt")

            with open(classes_dir, "r") as fp:
                self.targets = [int(line) - 1 for line in fp.readlines()]

        for dir in os.listdir(self.images_dir):
            images_dir_path = os.path.join(self.images_dir, dir)

            if split == "train":
                for image_file in os.listdir(images_dir_path):
                    image_dir = os.path.join(images_dir_path, image_file)

                    self.images.append(image_dir)
                    self.targets.append(classes_mapping[dir])
            elif split == "val":
                self.images.append(images_dir_path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image(PIL.Image.open(self.images[index]).convert("RGB"))

        target = tensor(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
