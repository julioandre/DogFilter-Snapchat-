from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import cv2
import argparse


class Fer2013Dataset(Dataset):
    """Face Emotion Recognition dataset.
    Utility for loading FER into PyTorch. Dataset curated by Pierre-Luc Carrier
    and Aaron Courville in 2013.
    Each sample is 1 x 1 x 48 x 48, and each label is a scalar.
    """

    def __init__(self, path: str):
        """
        Args:
            path: Path to `.np` file containing sample nxd and label nx1
        """
        with np.load(path) as data:
            self._samples = data['X']
            self._labels = data['Y']
        self._samples = self._samples.reshape((-1, 1, 48, 48))

        self.X = Variable(torch.from_numpy(self._samples)).float()
        self.Y = Variable(torch.from_numpy(self._labels)).float()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {'image': self._samples[idx], 'label': self._labels[idx]}

trainset = Fer2013Dataset('data/fer2013_train.npz')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = Fer2013Dataset('data/fer2013_test.npz')
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

if __name__ == '__main__':
    loader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=False)
    print(next(iter(loader)))