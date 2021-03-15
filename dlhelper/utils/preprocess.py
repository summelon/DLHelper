import torch
from torchvision import transforms

from ..dataset import TorchDataset as Dataset


def trans_pipeline(is_train):
    if is_train:
        aug_list = [transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip()]
    else:
        aug_list = [transforms.Resize(256),
                    transforms.CenterCrop(224)]

    aug_list.append(transforms.ToTensor())
    aug_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(aug_list)


def prepare_loader(
        name: str,
        is_train: bool,
        batch_size: int = 128,
        num_workers: int = 5
        ):
    trans_func = trans_pipeline(is_train=is_train)
    dataset = Dataset(name, trans_func, is_train=is_train)
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=is_train)

    return dataloader
