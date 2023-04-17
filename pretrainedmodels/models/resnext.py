from __future__ import print_function, division, absolute_import
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

from pretrainedmodels.models.resnext_features import resnext101_32x4d_features
from pretrainedmodels.models.resnext_features import resnext101_64x4d_features

__all__ = ['ResNeXt101_32x4d', 'resnext101_32x4d',
           'ResNeXt101_64x4d', 'resnext101_64x4d']
class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def resnext101_32x4d(num_classes=1000):
   return ResNeXt101_32x4d(num_classes=num_classes)

def resnext101_64x4d(num_classes=1000):
    return ResNeXt101_64x4d(num_classes=num_classes)

if __name__ == "__main__":
    model = resnext101_32x4d()
    input = Variable(torch.randn(2, 2, 1024))
    output = model(input)
    print(output.size())

    model = resnext101_64x4d()
    input = Variable(torch.randn(2, 2, 1024))
    output = model(input)
    print(output.size())