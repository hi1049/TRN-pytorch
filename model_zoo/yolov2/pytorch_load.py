import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class yolov2(nn.Module):
    def __init__(self, num_classes=1001):
        super(yolov2, self).__init__()
        # self.seen = 0
        # self.num_classes = 20
        # self.anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
        # self.num_anchors = len(self.anchors)/2
        # num_output = (5+self.num_classes)*self.num_anchors
        self.width = 160
        self.height = 160

        # self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)
        self.features = nn.Sequential(OrderedDict([
            # conv1 + maxpool
            ('conv1', nn.Conv2d(3, 32, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(32)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # conv2 + maxpool
            ('conv2', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # conv3
            ('conv3', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(128)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),

            # conv4
            ('conv4', nn.Conv2d(128, 64, 1, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(64)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),

            # conv5 + maxpool
            ('conv5', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(128)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, 2)),

            # conv6
            ('conv6', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(256)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),

            # conv7
            ('conv7', nn.Conv2d(256, 128, 1, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8 + maxpool
            ('conv8', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(256)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),
            ('pool8', nn.MaxPool2d(2, 2)),

            # conv9
            ('conv9', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn9', nn.BatchNorm2d(512)),
            ('leaky9', nn.LeakyReLU(0.1, inplace=True)),

            # conv10
            ('conv10', nn.Conv2d(512, 256, 1, 1, 1, bias=False)),
            ('bn10', nn.BatchNorm2d(256)),
            ('leaky10', nn.LeakyReLU(0.1, inplace=True)),

            # conv11
            ('conv11', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn11', nn.BatchNorm2d(512)),
            ('leaky11', nn.LeakyReLU(0.1, inplace=True)),

            # conv12
            ('conv12', nn.Conv2d(512, 256, 1, 1, 1, bias=False)),
            ('bn12', nn.BatchNorm2d(256)),
            ('leaky12', nn.LeakyReLU(0.1, inplace=True)),

            # conv13 + maxpool
            ('conv13', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn13', nn.BatchNorm2d(512)),
            ('leaky13', nn.LeakyReLU(0.1, inplace=True)),
            ('pool13', nn.MaxPool2d(2, 2)),

            # conv14
            ('conv14', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn14', nn.BatchNorm2d(1024)),
            ('leaky14', nn.LeakyReLU(0.1, inplace=True)),

            # conv15
            ('conv15', nn.Conv2d(1024, 512, 1, 1, 1, bias=False)),
            ('bn15', nn.BatchNorm2d(512)),
            ('leaky15', nn.LeakyReLU(0.1, inplace=True)),

            # conv16
            ('conv16', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn16', nn.BatchNorm2d(1024)),
            ('leaky16', nn.LeakyReLU(0.1, inplace=True)),

            # conv17
            ('conv17', nn.Conv2d(1024, 512, 1, 1, 1, bias=False)),
            ('bn17', nn.BatchNorm2d(512)),
            ('leaky17', nn.LeakyReLU(0.1, inplace=True)),

            # conv18
            ('conv18', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn18', nn.BatchNorm2d(1024)),
            ('leaky18', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('fc', nn.Linear(1024, num_classes))
        ]))

    def forward(self, x):
        x = self.features(x)
        return x

    def print_network(self):
        print(self)

    def load_weights(self, path='TRN-pytorch/pretrain/tiny-yolo-voc.weights'):
        # buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
        buf = np.fromfile(path, dtype=np.float32)
        start = 4

        start = load_conv_bn(buf, start, self.features[0], self.features[1])    # conv 1
        start = load_conv_bn(buf, start, self.features[4], self.features[5])    # conv 2
        start = load_conv_bn(buf, start, self.features[8], self.features[9])    # conv 3
        start = load_conv_bn(buf, start, self.features[11], self.features[12])  # conv 4
        start = load_conv_bn(buf, start, self.features[14], self.features[15])  # conv 5
        start = load_conv_bn(buf, start, self.features[18], self.features[19])  # conv 6
        start = load_conv_bn(buf, start, self.features[21], self.features[22])  # conv 7

        start = load_conv_bn(buf, start, self.features[24], self.features[25])  # conv 8
        start = load_conv_bn(buf, start, self.features[28], self.features[29])  # conv 9

        start = load_conv_bn(buf, start, self.features[31], self.features[32])  # conv 10
        start = load_conv_bn(buf, start, self.features[34], self.features[35])  # conv 11
        start = load_conv_bn(buf, start, self.features[37], self.features[38])  # conv 12
        start = load_conv_bn(buf, start, self.features[40], self.features[41])  # conv 13
        start = load_conv_bn(buf, start, self.features[44], self.features[45])  # conv 14
        start = load_conv_bn(buf, start, self.features[47], self.features[48])  # conv 15
        start = load_conv_bn(buf, start, self.features[50], self.features[51])  # conv 16
        start = load_conv_bn(buf, start, self.features[53], self.features[54])  # conv 17
        start = load_conv_bn(buf, start, self.features[56], self.features[57])  # conv 18

        # start = load_conv(buf, start, self.features[30]) # output

