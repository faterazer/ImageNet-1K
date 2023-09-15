import os
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, test_dir: str, transforms: Callable = None) -> None:
        self.data = sorted([os.path.join(test_dir, filename) for filename in os.listdir(test_dir)])
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tensor:
        image = Image.open(self.data[index]).convert("RGB")
        if self.transforms:
           image = self.transforms(image)
        return image
