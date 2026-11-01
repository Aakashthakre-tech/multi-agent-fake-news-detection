"""
forensic.py
===========
A compact frequency/forensic detector trained on noise residuals.

Motivation. The production model (`delpot/steganograph-ia-detector`) is a
Vision Transformer operating on the resized RGB image. It looks at learned
semantic and low-level texture cues. This model looks at a different signal
entirely: the residual left after subtracting a Gaussian blur from the
image, which leaves mostly sensor noise, resampling traces and generator
fingerprints. The two therefore have a real chance of making decorrelated
errors, which is the only thing that would justify running both.

Deliberately small (about 0.5M parameters) because the available hardware is
CPU-only. It trains in minutes rather than hours, and the noise signal does
not need a large backbone to be read.

This file defines the model and the residual computation. Training lives in
evaluation/train_forensic.py; the benchmark never imports it.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

RESIDUAL_SIZE = 64
BLUR_RADIUS = 0.6

# Normalisation for the residual. Residuals are small and near zero mean, so
# they are divided by this constant rather than standardised per image, which
# would destroy the absolute noise magnitude that carries the signal.
RESIDUAL_SCALE = 12.0


def compute_residual(image, size=RESIDUAL_SIZE):
    """
    Return a size x size x 3 float32 array of (image - gaussian blur).

    Works on the image as given; any resizing, compression or cropping
    augmentation must happen before this call so the residual reflects the
    degraded image the detector would actually receive.
    """
    image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)

    original = np.asarray(image, dtype=np.float32)

    blurred = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)),
        dtype=np.float32,
    )

    residual = original - blurred
    return residual / RESIDUAL_SCALE


def residual_to_tensor(residual):
    """HWC float32 numpy -> 1x3xHxW float32 torch tensor."""
    return torch.from_numpy(
        np.ascontiguousarray(residual.transpose(2, 0, 1))
    ).unsqueeze(0).float()


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ForensicCNN(nn.Module):
    """
    Small residual-CNN binary classifier.

    Four strided residual blocks take 3 x 64 x 64 down to a feature vector,
    which is pooled and classified. BatchNorm is used so the model is not
    dependent on the input scale staying fixed.
    """

    def __init__(self, num_blocks=4, base_channels=32):
        super().__init__()
        blocks = []
        in_ch = 3
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** i)
            blocks.append(ResidualBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = ForensicCNN()
    print("parameters: %d" % count_parameters(net))
    probe = torch.zeros(1, 3, RESIDUAL_SIZE, RESIDUAL_SIZE)
    print("output shape:", tuple(net(probe).shape))
