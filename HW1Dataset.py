# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:01:15 2019

@author: Zonsor
"""
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class HW1TestDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.transforms = transforms
        self.root_dir = Path(root_dir)
        self.input = []
        self.filename = []
        for i, file_path in enumerate(self.root_dir.glob('*')):
            self.input.append(file_path)
            self.filename.append(file_path.name)

    def __getitem__(self, index):
        image = Image.open(self.input[index]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)

        return image, self.filename[index]

    def __len__(self):
        return len(self.input)
