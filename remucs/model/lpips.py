# Taken from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# Implements the LPIPS class

from collections import namedtuple
import torch
from torch import nn, Tensor
import numpy as np
import torch.nn
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "./resources/ckpts/vgg_lpips.pth"


def spatial_average(in_tens: Tensor, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # Load pretrained vgg model from torchvision
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # Freeze vgg model
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Return output of vgg features
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

# Learned perceptual metric


class LPIPS(nn.Module):
    """ Learned Perceptual Image Patch Similarity (LPIPS) metric.

      forward:
       - in0, in1: Tensors of shape (N, C, H, W) (images should be in 0-1 range)
       - normalize: if True, will normalize the input to [-1, 1] range

       returns:
       - d: (N, ) Tensor of distances between the image pairs"""

    def __init__(self, means: list[float], stds: list[float], use_dropout=True):
        super(LPIPS, self).__init__()

        # Instantiate vgg model
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.net = vgg16(pretrained=True, requires_grad=False)

        # Add 1x1 convolutional Layers
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lins = nn.ModuleList(self.lins)

        # Load the weights of trained LPIPS model
        print('Loading model from: %s' % MODEL_PATH)
        self.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)

        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor(means))
        self.register_buffer('std', torch.tensor(stds))

    def forward(self, x0, x1, normalize=False):
        # Scale the inputs to -1 to +1 range if needed
        # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
        if normalize:
            x0 = 2 * x0 - 1
            x1 = 2 * x1 - 1

        # A quick hacky way to get around the VGG network requirements of having 3 channels
        # Do the input LPIPS 4 times to match the 4 channels of the input
        out = None
        for idx in [(0, 1, 2), (3, 0, 1), (2, 3, 0), (1, 2, 3)]:
            in0 = x0[:, idx]
            in1 = x1[:, idx]
            means = self.mean[torch.tensor(idx)][None, :, None, None]
            stds = self.std[torch.tensor(idx)][None, :, None, None]
            in0 = (in0 - means) / stds
            in1 = (in1 - means) / stds
            d = self.forward_single(in0, in1)

            if out is None:
                out = d
            else:
                out += d

        assert out is not None
        return out / 4

    def forward_single(self, in0, in1):
        # Get VGG outputs for image0 and image1
        outs0 = self.net.forward(in0)
        outs1 = self.net.forward(in1)
        feats0, feats1, diffs = {}, {}, {}

        # Compute Square of Difference for each layer output
        for kk in range(self.L):
            feats0[kk], feats1[kk] = torch.nn.functional.normalize(outs0[kk], dim=1), torch.nn.functional.normalize(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # 1x1 convolution followed by spatial average on the square differences
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        val = 0

        # Aggregate the results of each layer
        for l in range(self.L):
            val += res[l]

        return val


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


def load_lpips(mean: tuple[float, ...] = (0.1885, 0.1751, 0.1698, 0.0800), std: tuple[float, ...] = (0.1164, 0.1066, 0.1065, 0.0672), use_dropout: bool = True) -> LPIPS:
    """ Load the LPIPS model with the given means and stds. The default is calculated over the whole training + val set """
    return LPIPS(
        means=list(mean),
        stds=list(std),
        use_dropout=use_dropout
    )