# test the pre-trained model on a single video
# (working on it)
# Bolei Zhou and Alex Andonian
from torch.nn import functional as F

import argparse
import functools
import os
import re
import subprocess

import cv2
# imageio.plugins.ffmpeg.download()
import moviepy.editor as mpy
import numpy as np
from PIL import Image

from models import TSN
from transforms import *

# options
parser = argparse.ArgumentParser(description="test TRN on a single video")
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff', 'depth'], )
parser.add_argument('--rendered_output', type=str, default=None)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weight', type=str)
parser.add_argument('--video_length', type=int, default=14)
parser.add_argument('--cam', type=str, default='webcam') #webcam or realsense

args = parser.parse_args()

video_length = args.video_length
cam_mode = args.cam

# Get dataset categories.
categories_file = 'categories.txt'
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

# Load model.
net = TSN(num_class,
          args.test_segments,
          args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim, print_spec=False)

weights = args.weight
checkpoint = torch.load(weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net.cuda().eval()

# Initialize frame transforms.

transform = torchvision.transforms.Compose([
    GroupOverSample(net.input_size, net.scale_size),
    Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    GroupNormalize(net.input_mean, net.input_std),
])
capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()

    # Obtain video frames
    frames.append(frame)
    frames = frames[::int(np.ceil(len(frames) / float(video_length)))]
    video_count += 1
    cv2.imshow('frame', frame)

    if video_count >= video_length:
        # Make video prediction.
        data = transform(frames)
        input_var = torch.autograd.Variable(data.view(-1, 3, data.size(1), data.size(2)),
                                            volatile=True).unsqueeze(0).cuda()
        logits = net(input_var)
        h_x = torch.mean(F.softmax(logits, 1), dim=0).data
        probs, idx = h_x.sort(0, True)

        # Output the prediction.
        print('{:.3f} -> {}'.format(probs[0], categories[idx[0]]))
        video_count = 0
        frames = []
