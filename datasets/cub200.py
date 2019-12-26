import os

import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image
from collections import OrderedDict


class CUB200(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.annotations = self._read_annotations('train.txt' if train else 'test.txt')
        
    def _read_annotations(self, split):
        annotations = OrderedDict()
        for line in open(os.path.join(self.root, split)):
            image_id, label = line.split()
            annotations[image_id] = int(label)
        return [[image_id, label] for image_id, label in annotations.items()]

    def __getitem__(self, idx):
        image_id, target = self.annotations[idx]
        image_path = os.path.join(self.root, 'images', image_id)
        img = Image.open(image_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        target = torch.tensor(target, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.annotations)
