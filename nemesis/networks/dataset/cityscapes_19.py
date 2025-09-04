from typing import Tuple

import torch
from torchvision import datasets
from torchvision.tv_tensors import Image, Mask
import PIL.Image


class CityscapesDataset19(datasets.Cityscapes):
    mapping_20 = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 3,
        12: 4,
        13: 5,
        14: 0,
        15: 0,
        16: 0,
        17: 6,
        18: 0,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 0,
        30: 0,
        31: 17,
        32: 18,
        33: 19,
        -1: 0,
    }

    def _encode_labels(self, mask: torch.Tensor) -> Mask:
        label_mask = torch.zeros_like(mask)

        for k in self.mapping_20:
            label_mask[mask == k] = self.mapping_20[k]

        return Mask(label_mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image(PIL.Image.open(self.images[idx]))

        target = Mask(PIL.Image.open(self.targets[idx][0]))

        target = self._encode_labels(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
