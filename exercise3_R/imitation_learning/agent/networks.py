import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

"""
Imitation learning network
"""

class CNN(nn.Module):

    def _update_size(self, dim, kernel_size, stride, padding=0.0, dilation=1.0):
        """
        Helper method to keep track of changing output dimensions between convolutions and Pooling layers
        returns the updated dimension "dim" (e.g. height or width)
        """
        return(int((dim + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1))
        # return int(np.floor((dim + 2*padding - (dilation*(kernel_size-1) + 1))/stride + 1))

    def __init__(self, history_length=1, n_classes=3):
        super(CNN, self).__init__()
        input_size = 96
        input_channels = history_length
        # Convolution Layer 1
        cnn1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=8, stride=4)
        bn1 = nn.BatchNorm2d(num_features=16)
        out_size = self._update_size(dim=input_size, kernel_size=8, stride=4)
        # Convolution Layer 2
        cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        bn2 = nn.BatchNorm2d(num_features=32)
        out_size = self._update_size(dim=out_size, kernel_size=4, stride=2)
        out_size = out_size * out_size * 32
        # FC layer
        fc = nn.Linear(in_features=out_size, out_features=256)
        # dropout = nn.Dropout(0.5)
        actv = nn.ReLU()
        # Output layer
        output = nn.Linear(in_features=256, out_features=n_classes)
        self.convolutions = nn.Sequential(
                                cnn1,
                                actv,
                                bn1,
                                cnn2,
                                actv,
                                bn2
                            )
        self.fcs = nn.Sequential(
                        fc,
                        actv,
                        # dropout,
                        output
                  )
        # TODO : define layers of a convolutional neural network

    def forward(self, x):
        # TODO: compute forward pass
        x = self.convolutions(x)
        x = x.view(x.shape[0], np.prod(x.shape[1:]))
        x = self.fcs(x)
        return x
