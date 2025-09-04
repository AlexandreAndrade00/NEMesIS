import os
from typing import List, Callable, Any, Tuple
from pathlib import Path

from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import Image, Mask
import PIL.Image


class ADE20K(VisionDataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "training",
        target_type: List[str] | str = "instance",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        assert split in ["training", "validation"]

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        else:
            self.target_type = target_type

        for t_type in self.target_type:
            assert t_type in ["semantic", "instance"]

        if "instance" in self.target_type:
            raise NotImplementedError()

        self.images_dir = os.path.join(self.root, "images", "ADE", split)

        assert os.path.isdir(self.images_dir)

        self.split = split
        self.images: List[str] = []
        self.targets: List[List[str]] = []

        for image_type in os.listdir(self.images_dir):
            image_type_dir = os.path.join(self.images_dir, image_type)

            for image_subtype in os.listdir(image_type_dir):
                image_subtype_dir = os.path.join(image_type_dir, image_subtype)

                images_path = [
                    os.path.join(image_subtype_dir, file)
                    for file in os.listdir(image_subtype_dir)
                    if "jpg" in file.split(".")
                ]

                for image_path in images_path:
                    targets: List[str] = []

                    if "semantic" in self.target_type:
                        targets.append(image_path.split(".")[0] + "_seg.png")

                    if "instance" in self.target_type:
                        instaces_mask_dir = image_path.split(".")[0]

                        targets.extend(
                            [
                                os.path.join(instaces_mask_dir, instance_file)
                                for instance_file in os.listdir(instaces_mask_dir)
                            ]
                        )

                    self.targets.append(targets)

                self.images.extend(images_path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image(PIL.Image.open(self.images[index]).convert("RGB"))

        targets = [Mask(PIL.Image.open(target)) for target in self.targets[index]]

        target = tuple(targets) if len(targets) > 1 else targets[0]  # type: ignore[assignment]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
