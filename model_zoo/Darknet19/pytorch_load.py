import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load_linear(buf, start, linear_model):
    num_w = linear_model.weight.numel()
    num_b = linear_model.bias.numel()
    linear_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    linear_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()

    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w 

    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    return start

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    #num_b = conv_model.bias.numel()
    #conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start



def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start

class BasicConv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, add_maxpool=False, add_bn=True, pool_stride=0, add_avgpool=False):
        super(BasicConv2D, self).__init__()
        self.add_maxpool = add_maxpool
        self.add_avgpool = add_avgpool
        self.add_bn = add_bn
        self.pool_stride = pool_stride

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false
        if self.add_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        if self.add_maxpool:
            if self.pool_stride == 2:
                self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.add_avgpool:
            self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        if self.add_bn:
            x = self.bn(x)
        x = self.relu(x)
        if self.add_maxpool:
            if self.pool_stride == 2:
                x = self.maxpool(x)
            elif self.pool_stride == 1:
                x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), kernel_size=2, stride=1)
        if self.add_avgpool:
            x = self.avgpool(x)
        return x


class Darknet19(nn.Module):
    def __init__(self, num_output=1000):
        super(Darknet19, self).__init__()
        self.features = nn.Sequential(
            # conv1  256>128
            BasicConv2D(3, 16, 3, 1, 1, bias=False, add_maxpool=True, pool_stride=2),

            # conv2  128>64
            BasicConv2D(16, 32, 3, 1, 1, bias=False, add_maxpool=True, pool_stride=2),

            # conv3  64>
            BasicConv2D(32, 64, 3, 1, 1, bias=False, add_maxpool=True, pool_stride=2),

            # conv4
            BasicConv2D(64, 128, 3, 1, 1, bias=False, add_maxpool=True, pool_stride=2),

            # conv5
            BasicConv2D(128, 256, 3, 1, 1, bias=False, add_maxpool=True, pool_stride=2),

            # conv6
            BasicConv2D(256, 512, 3, 1, 1, bias=False, add_maxpool=True, pool_stride=1),

            # conv7
            BasicConv2D(512, 1024, 3, 1, 1, bias=False),

            # conv8
            BasicConv2D(1024, 1024, 1, 1, 0, bias=False, add_bn=False, add_avgpool=True)
        )
        # fc
        # self.fc = nn.Conv2d(1024, num_output, 4, 1, 0, bias=False)
        self.fc = nn.Linear(1024, num_output)

        # load weights
        self.load_weights(path='pretrain/darknet19.weights')
        self.load_state_dict(self.state_dict())

    def forward(self, x):
        #print('x:', x.size())
        x = self.features(x)
        #for i in range(len(self.features)):
        #    print('x[%d]:'.format(i), x[i].size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #print('x_fc:', x.size())
        return x

    def load_weights(self, path='pretrain/darknet19.weights'):
        # buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
        buf = np.fromfile(path, dtype=np.float32)
        start = 4

        start = load_conv_bn(buf, start, self.features[0].conv, self.features[0].bn)  # conv 1
        start = load_conv_bn(buf, start, self.features[1].conv, self.features[1].bn)  # conv 2
        start = load_conv_bn(buf, start, self.features[2].conv, self.features[2].bn)  # conv 3
        start = load_conv_bn(buf, start, self.features[3].conv, self.features[3].bn)  # conv 4
        start = load_conv_bn(buf, start, self.features[4].conv, self.features[4].bn)  # conv 5
        start = load_conv_bn(buf, start, self.features[5].conv, self.features[5].bn)  # conv 6
        start = load_conv_bn(buf, start, self.features[6].conv, self.features[6].bn)  # conv 7
        start = load_conv(buf, start, self.features[7].conv)  # conv 8
        # start = load_linear(buf, start, self.fc)  # output

