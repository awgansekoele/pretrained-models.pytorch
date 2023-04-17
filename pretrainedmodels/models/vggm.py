from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.legacy import nn as nnl
import torch.utils.model_zoo as model_zoo

__all__ = ['vggm']


class SpatialCrossMapLRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool2d(kernel_size=(local_size, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0))
        else:
            self.average = nn.AvgPool1d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class VGGM(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(3, 96, (7,), (2,)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool1d((3,), (2,), (0,), ceil_mode=True),
            nn.Conv1d(96, 256, (5,), (2,), (1,)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool1d((3,), (2,), (0,), ceil_mode=True),
            nn.Conv1d(256, 512, (3,), (1,), (1,)),
            nn.ReLU(),
            nn.Conv1d(512, 512, (3,), (1,), (1,)),
            nn.ReLU(),
            nn.Conv1d(512, 512, (3,), (1,), (1,)),
            nn.ReLU(),
            nn.MaxPool1d((3,), (2,), (0,), ceil_mode=True)
        )
        self.classif = nn.Sequential(
            nn.Linear(18432, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x


def vggm(num_classes=1000):
    return VGGM(num_classes=num_classes)
