
import numpy as np
import torch.nn as nn
#import torch.nn.functional as F
from collections import OrderedDict
#import sys
#sys.setrecursionlimit(20000)

class BasicConv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Maxpool2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Maxpool2D, self).__init__()

    def forward(self, x):
        x = nn.MaxPool2d(kernel_size=2, stride=2)
        return x


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = nn.MaxPool2d(self.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class TinyYoloNet(nn.Module):
    def __init__(self, num_output=1000):
        super(TinyYoloNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            BasicConv2D(3, 16, 3, 1, 1, bias=False),
            Maxpool2D(2, 2),

            # conv2
            BasicConv2D(16, 32, 3, 1, 1, bias=False),
            Maxpool2D(2, 2),

            # conv3
            BasicConv2D(32, 64, 3, 1, 1, bias=False),
            Maxpool2D(2, 2),

            # conv4
            BasicConv2D(64, 128, 3, 1, 1, bias=False),
            Maxpool2D(2, 2),

            # conv5
            BasicConv2D(128, 256, 3, 1, 1, bias=False),
            Maxpool2D(2, 2),

            # conv6
            BasicConv2D(256, 512, 3, 1, 1, bias=False),
            MaxPoolStride1(),

            # conv7
            BasicConv2D(512, 1024, 3, 1, 1, bias=False),

            # conv8
            BasicConv2D(1024, 1024, 3, 1, 1, bias=False)
        )

        # fc
        #self.fc = nn.Conv2d(1024, num_output, 1, 1, 0)
        self.fc = nn.Linear(1024, num_output)
        # load weights
        #self.load_state_dict()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x

    '''
    def load_weights(self, path='TRN-pytorch/pretrain/tiny-yolo-voc.weights'):
        # buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
        buf = np.fromfile(path, dtype=np.float32)
        start = 4

        start = load_conv_bn(buf, start, self.cnn[0], self.cnn[1])  # conv 1
        start = load_conv_bn(buf, start, self.cnn[4], self.cnn[5])  # conv 2
        start = load_conv_bn(buf, start, self.cnn[8], self.cnn[9])  # conv 3
        start = load_conv_bn(buf, start, self.cnn[12], self.cnn[13])  # conv 4
        start = load_conv_bn(buf, start, self.cnn[16], self.cnn[17])  # conv 5
        start = load_conv_bn(buf, start, self.cnn[20], self.cnn[21])  # conv 6

        start = load_conv_bn(buf, start, self.cnn[24], self.cnn[25])  # conv 7
        start = load_conv_bn(buf, start, self.cnn[27], self.cnn[28])  # conv 8

        start = load_conv(buf, start, self.fc)  # output
    '''
