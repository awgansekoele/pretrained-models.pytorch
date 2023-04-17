from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys

__all__ = ['InceptionV4', 'inceptionv4']

from torch.autograd import Variable


class BasicConv1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm1d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.conv = BasicConv1d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv1d(160, 64, kernel_size=1, stride=1),
            BasicConv1d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv1d(160, 64, kernel_size=1, stride=1),
            BasicConv1d(64, 64, kernel_size=(7,), stride=1, padding=3),
            BasicConv1d(64, 64, kernel_size=(7,), stride=1, padding=3),
            BasicConv1d(64, 96, kernel_size=(3,), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv1d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv1d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(384, 64, kernel_size=1, stride=1),
            BasicConv1d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(384, 64, kernel_size=1, stride=1),
            BasicConv1d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv1d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv1d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv1d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv1d(384, 192, kernel_size=1, stride=1),
            BasicConv1d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv1d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv1d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(1024, 192, kernel_size=1, stride=1),
            BasicConv1d(192, 224, kernel_size=7, stride=1, padding=3),
            BasicConv1d(224, 256, kernel_size=7, stride=1, padding=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(1024, 192, kernel_size=1, stride=1),
            BasicConv1d(192, 192, kernel_size=7, stride=1, padding=3),
            BasicConv1d(192, 224, kernel_size=7, stride=1, padding=3),
            BasicConv1d(224, 224, kernel_size=7, stride=1, padding=3),
            BasicConv1d(224, 256, kernel_size=7, stride=1, padding=3)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv1d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv1d(1024, 192, kernel_size=1, stride=1),
            BasicConv1d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv1d(1024, 256, kernel_size=1, stride=1),
            BasicConv1d(256, 256, kernel_size=7, stride=1, padding=3),
            BasicConv1d(256, 320, kernel_size=7, stride=1, padding=3),
            BasicConv1d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv1d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv1d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv1d(384, 256, kernel_size=3, stride=1, padding=1)
        self.branch1_1b = BasicConv1d(384, 256, kernel_size=3, stride=1, padding=1)

        self.branch2_0 = BasicConv1d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv1d(384, 448, kernel_size=3, stride=1, padding=1)
        self.branch2_2 = BasicConv1d(448, 512, kernel_size=3, stride=1, padding=1)
        self.branch2_3a = BasicConv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.branch2_3b = BasicConv1d(512, 256, kernel_size=3, stride=1, padding=1)

        self.branch3 = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv1d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        # Modules
        self.features = nn.Sequential(
            BasicConv1d(2, 32, kernel_size=3, stride=2),
            BasicConv1d(32, 32, kernel_size=3, stride=1),
            BasicConv1d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool1d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def inceptionv4(num_classes=1000):
    return InceptionV4(num_classes=num_classes)


'''
TEST
Run this code with:
```
cd $HOME/pretrained-models.pytorch
python -m pretrainedmodels.inceptionv4
```
'''
if __name__ == '__main__':

    model = inceptionv4(num_classes=1000)
    input = Variable(torch.randn(2, 2, 1024))

    output = model(input)
    print(output.size())
