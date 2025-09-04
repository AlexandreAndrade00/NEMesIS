import os
from os.path import join
from typing import ClassVar, List, Tuple

from torch import Generator, float32, uint8
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import v2

from nemesis.misc.enums import Task
from .cityscapes_19 import CityscapesDataset19
from .ade20k import ADE20K
from .scene_parsing_150 import SceneParsing150
from .image_net_1k import ImageNet1K


class DatasetSubsets:
    seed: ClassVar[int] = 1
    data_relative_root_dir: ClassVar[str] = "data"
    supported_datasets: ClassVar[List[str]] = [
        "cityscapes",
        "cifar10",
        "cifar100",
        "mnist",
        "fashion-mnist",
        "ade20k",
        "scene-parsing-150",
        "image-net-1k",
    ]

    def __init__(
        self,
        train: Dataset,
        validation: Dataset,
        test: Dataset,
        batch_size: int,
        num_classes: int,
        num_workers: int,
        input_shape: Tuple[int, int, int],
        dataset_name: str,
        task: Task,
    ) -> None:
        self.train = train
        self.validation = validation
        self.test = test
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.input_shape = input_shape
        self.dataset_name = dataset_name
        self.task = task

    @classmethod
    def from_name(
        cls,
        name: str,
        normalise: bool,
        shape: List[int],
        augment: bool,
        batch_size: int,
        num_workers: int,
    ) -> "DatasetSubsets":
        assert len(shape) == 2

        match name:
            case "cityscapes":
                return cls.cityscapes(normalise, shape, augment, batch_size, num_workers)
            case "cifar10":
                return cls.cifar10()
            case "cifar100":
                return cls.cifar100()
            case "mnist":
                return cls.mnist()
            case "fashion-mnist":
                return cls.fashion_mnist()
            case "ade20k":
                return cls.ade20k(normalise, augment, batch_size, num_workers)
            case "scene-parsing-150":
                return cls.scene_parsing_150(normalise, augment, batch_size, num_workers)
            case "image-net-1k":
                return cls.image_net_1k(normalise, shape, augment, batch_size, num_workers)
            case _:
                raise ValueError("Dataset not available")

    @classmethod
    def cityscapes(
        cls, normalise: bool, shape: List[int], augment: bool, batch_size: int, num_workers: int
    ) -> "DatasetSubsets":
        dataset_name = "Cityscapes"

        scaled_img_w = shape[0]
        scaled_img_h = shape[1]

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        transforms_train_list = [
            v2.ToDtype(uint8, scale=True),
        ]

        if augment:
            transforms_train_list += [
                v2.RandomResizedCrop(size=(scaled_img_h, scaled_img_w), scale=(0.125, 1.5), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(),
            ]

        if normalise:
            transforms_train_list += [
                v2.ToDtype(float32, scale=True),
                v2.Normalize(mean=[0.2869, 0.3251, 0.2839], std=[0.1761, 0.1810, 0.1777]),
            ]

        if not augment and not normalise:
            transforms_train_list += [
                v2.Resize(size=(scaled_img_h, scaled_img_w), antialias=True),
                v2.ToDtype(float32, scale=True),
            ]
        elif not normalise:
            transforms_train_list += [
                v2.ToDtype(float32, scale=True),
            ]

        transforms_train = v2.Compose(transforms_train_list)

        transforms_val_test_list = [
            v2.ToDtype(uint8, scale=True),
            v2.Resize(size=(scaled_img_h, scaled_img_w), antialias=True),
            v2.ToDtype(float32, scale=True),
        ]

        if normalise:
            transforms_val_test_list += [
                v2.Normalize(mean=[0.2949, 0.3334, 0.2945], std=[0.1839, 0.1868, 0.1822]),
            ]

        transforms_val_test = v2.Compose(transforms_val_test_list)

        train_data: datasets.VisionDataset = CityscapesDataset19(
            root=data_relative_path,
            split="train",
            mode="fine",
            target_type="semantic",
            transforms=transforms_train,
        )

        val_data: datasets.VisionDataset = CityscapesDataset19(
            root=data_relative_path,
            split="val",
            mode="fine",
            target_type="semantic",
            transforms=transforms_val_test,
        )

        test_data = val_data

        return cls(
            train_data,
            val_data,
            test_data,
            batch_size,
            20,
            num_workers,
            (3, scaled_img_h, scaled_img_w),
            dataset_name,
            Task.REAL_TIME_SEMANTIC_SEGMENTATION,
        )

    @classmethod
    def ade20k(cls, normalise: bool, augment: bool, batch_size: int, num_workers: int) -> "DatasetSubsets":
        dataset_name = "ADE20K"

        scaled_img_w = 512
        scaled_img_h = 512

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        transforms_train_list = [
            v2.ToDtype(dtype=float32, scale=True),
        ]

        if augment:
            transforms_train_list += [
                v2.RandomResizedCrop(size=(scaled_img_h, scaled_img_w), scale=(0.4, 1.6)),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(),
            ]

        if normalise:
            transforms_train_list += [
                v2.Normalize(mean=[0.4853, 0.4652, 0.4309], std=[0.2301, 0.2304, 0.2418]),
            ]

        if not augment and not normalise:
            transforms_train_list.append(v2.Resize(size=(scaled_img_h, scaled_img_w)))

        transforms_train = v2.Compose(transforms_train_list)

        transforms_val_test_list = [v2.ToDtype(dtype=float32, scale=True), v2.Resize(size=(scaled_img_h, scaled_img_w))]

        if normalise:
            transforms_val_test_list += [
                v2.Normalize(mean=[0.4853, 0.4652, 0.4309], std=[0.2301, 0.2304, 0.2418]),
            ]

        transforms_val_test = v2.Compose(transforms_val_test_list)

        train_data: datasets.VisionDataset = ADE20K(
            root=data_relative_path,
            split="training",
            target_type="semantic",
            transforms=transforms_train,
        )

        val_data: datasets.VisionDataset = ADE20K(
            root=data_relative_path,
            split="validation",
            target_type="semantic",
            transforms=transforms_val_test,
        )

        test_data = val_data

        return cls(
            train_data,
            val_data,
            test_data,
            batch_size,
            3688,
            num_workers,
            (3, scaled_img_h, scaled_img_w),
            dataset_name,
            Task.REAL_TIME_SEMANTIC_SEGMENTATION,
        )

    @classmethod
    def scene_parsing_150(cls, normalise: bool, augment: bool, batch_size: int, num_workers: int) -> "DatasetSubsets":
        dataset_name = "SceneParsing150"

        scaled_img_w = 1024
        scaled_img_h = 512

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        transforms_train_list = [v2.ToDtype(dtype=float32, scale=True)]

        if augment:
            transforms_train_list += [
                v2.RandomResizedCrop(size=(scaled_img_h, scaled_img_w), scale=(0.4, 1.6)),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(),
            ]

        if normalise:
            transforms_train_list += [
                v2.Normalize(mean=[0.4890, 0.4653, 0.4292], std=[0.2286, 0.2296, 0.2405]),
            ]

        if not augment and not normalise:
            transforms_train_list.append(v2.Resize(size=(scaled_img_h, scaled_img_w)))

        transforms_train = v2.Compose(transforms_train_list)

        transforms_val_test_list = [v2.ToDtype(dtype=float32, scale=True), v2.Resize(size=(scaled_img_h, scaled_img_w))]

        if normalise:
            transforms_val_test_list += [
                v2.Normalize(mean=[0.4853, 0.4652, 0.4309], std=[0.2301, 0.2304, 0.2418]),
            ]

        transforms_val_test = v2.Compose(transforms_val_test_list)

        train_data: datasets.VisionDataset = SceneParsing150(
            root=data_relative_path,
            split="training",
            transforms=transforms_train,
        )

        val_data: datasets.VisionDataset = SceneParsing150(
            root=data_relative_path,
            split="validation",
            transforms=transforms_val_test,
        )

        test_data = val_data

        return cls(
            train_data,
            val_data,
            test_data,
            batch_size,
            151,
            num_workers,
            (3, scaled_img_h, scaled_img_w),
            dataset_name,
            Task.REAL_TIME_SEMANTIC_SEGMENTATION,
        )

    @classmethod
    def image_net_1k(
        cls, normalise: bool, shape: List[int], augment: bool, batch_size: int, num_workers: int
    ) -> "DatasetSubsets":
        dataset_name = "ImageNet1K"

        scaled_img_w = shape[0]
        scaled_img_h = shape[1]

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        transforms_train_list = [
            v2.ToDtype(dtype=float32, scale=True),
        ]

        if augment:
            transforms_train_list += [
                v2.RandomResizedCrop(size=(scaled_img_h, scaled_img_w), scale=(0.125, 1.5)),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(),
            ]

        if normalise:
            transforms_train_list += [
                v2.Normalize(mean=[0.4813, 0.4574, 0.4077], std=[0.2334, 0.2293, 0.2301]),
            ]

        if not augment and not normalise:
            transforms_train_list.append(v2.Resize(size=(scaled_img_h, scaled_img_w)))

        transforms_train = v2.Compose(transforms_train_list)

        transforms_val_test_list = [v2.ToDtype(dtype=float32, scale=True), v2.Resize(size=(scaled_img_h, scaled_img_w))]

        if normalise:
            transforms_val_test_list += [
                v2.Normalize(mean=[0.4813, 0.4574, 0.4077], std=[0.2334, 0.2293, 0.2301]),
            ]

        transforms_val_test = v2.Compose(transforms_val_test_list)

        train_data: datasets.VisionDataset = ImageNet1K(
            root=data_relative_path,
            split="train",
            transforms=transforms_train,
        )

        val_data: datasets.VisionDataset = ImageNet1K(
            root=data_relative_path,
            split="val",
            transforms=transforms_val_test,
        )

        test_data = val_data

        return cls(
            train_data,
            val_data,
            test_data,
            batch_size,
            1000,
            num_workers,
            (3, scaled_img_h, scaled_img_w),
            dataset_name,
            Task.REAL_TIME_CLASSIFICATION,
        )

    @classmethod
    def fashion_mnist(cls) -> "DatasetSubsets":
        dataset_name = "Fashion-MNIST"

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        train_data = datasets.FashionMNIST(root=data_relative_path, train=True, download=True)
        train_subset, val_subset = random_split(train_data, [0.7, 0.3], generator=Generator().manual_seed(cls.seed))
        test_data = datasets.FashionMNIST(root=data_relative_path, train=False, download=True)

        return cls(
            train_subset,
            val_subset,
            test_data,
            64,
            10,
            2,
            (3, 32, 32),
            dataset_name,
            Task.CLASSIFICATION,
        )

    @classmethod
    def mnist(cls) -> "DatasetSubsets":
        dataset_name = "MNIST"

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        train_data = datasets.MNIST(root=data_relative_path, train=True, download=True)
        train_subset, val_subset = random_split(train_data, [0.7, 0.3], generator=Generator().manual_seed(cls.seed))
        test_data = datasets.MNIST(root=data_relative_path, train=False, download=True)

        return cls(
            train_subset,
            val_subset,
            test_data,
            64,
            10,
            2,
            (3, 32, 32),
            dataset_name,
            Task.CLASSIFICATION,
        )

    @classmethod
    def cifar10(cls) -> "DatasetSubsets":
        dataset_name = "Cifar-10"

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        train_data = datasets.CIFAR10(root=data_relative_path, train=True, download=True)
        train_subset, val_subset = random_split(train_data, [0.7, 0.3], generator=Generator().manual_seed(cls.seed))
        test_data = datasets.CIFAR10(root=data_relative_path, train=False, download=True)

        return cls(
            train_subset,
            val_subset,
            test_data,
            64,
            10,
            2,
            (1, 32, 32),
            dataset_name,
            Task.CLASSIFICATION,
        )

    @classmethod
    def cifar100(cls) -> "DatasetSubsets":
        dataset_name = "Cifar-100"

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        train_data = datasets.CIFAR100(root=data_relative_path, train=True, download=True)
        train_subset, val_subset = random_split(train_data, [0.7, 0.3], generator=Generator().manual_seed(cls.seed))
        test_data = datasets.CIFAR100(root=data_relative_path, train=False, download=True)

        return cls(
            train_subset,
            val_subset,
            test_data,
            64,
            100,
            2,
            (1, 32, 32),
            dataset_name,
            Task.CLASSIFICATION,
        )

    def train_data_loader(self, distributed: bool) -> DataLoader:
        generator = Generator().manual_seed(self.seed)

        cpu_count: int | None = os.cpu_count()

        assert cpu_count is not None

        dataset = self.train

        # during bt training if the last batch has 1 element, training breaks at last batch norm.
        # therefore, we drop the last batch
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            generator=generator,
            sampler=DistributedSampler(dataset) if distributed else None,
        )

    def validation_data_loader(self, distributed: bool) -> DataLoader:
        generator = Generator().manual_seed(self.seed)

        cpu_count: int | None = os.cpu_count()

        assert cpu_count is not None

        dataset = self.validation

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            generator=generator,
            sampler=DistributedSampler(dataset) if distributed else None,
        )

    def test_data_loader(self, distributed: bool) -> DataLoader:
        generator = Generator().manual_seed(self.seed)

        cpu_count: int | None = os.cpu_count()

        assert cpu_count is not None

        return DataLoader(
            self.test,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            generator=generator,
            sampler=DistributedSampler(self.test) if distributed else None,
        )
