import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
import numpy as np
from ..config import VAEConfig
from dataclasses import dataclass


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class NLayerDiscriminator(nn.Module):
    def __init__(self, nsources: int, ndf: int, n_layers: int, downsampling_factor: int):
        super().__init__()
        model = []

        model.append(nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(nsources, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        ))

        nf = ndf
        nf_prev = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model.append(nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            ))

        nf = min(nf * 2, 1024)
        model.append(nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        ))

        model.append(WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        ))

        self.model = nn.ModuleList(model)

    def forward(self, x):
        # Expect input shape: (B, source, L)
        for layer in self.model:
            x = layer(x)
        return x.squeeze(1)


class AudioDiscriminator(nn.Module):
    def __init__(self, naudios: int, ndiscriminators: int, nfilters: int, n_layers: int, downsampling_factor: int):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(ndiscriminators):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                naudios, nfilters, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, audio: torch.Tensor):
        results = []
        for key, disc in self.model.items():
            results.append(disc(audio))
            audio = self.downsample(audio)
        return results


class SpectrogramPatchModel(nn.Module):
    """This uses the idea of PatchGAN but changes the architecture to use Conv2d layers on each bar (8, 128, 512) patches

    Assumes input is of shape (B, 8, 512, 512), outputs a tensor of shape (B, 4)"""

    def __init__(self, nsources: int, nfft: int, ntimeframes: int, num_patches: int = 4):
        super(SpectrogramPatchModel, self).__init__()
        assert ntimeframes % num_patches == 0, "ntimeframes must be divisible by num_patches"

        # Define a simple CNN architecture for each patch
        self.conv1 = nn.Conv2d(nsources, 16, kernel_size=3, padding=1)
        self.pool11 = nn.AdaptiveMaxPool2d((nfft // 2, ntimeframes // num_patches))
        self.pool12 = nn.AdaptiveAvgPool2d((nfft // 2, ntimeframes // (num_patches * 2)))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool21 = nn.AdaptiveMaxPool2d((nfft // 4, ntimeframes // (num_patches * 2)))
        self.pool22 = nn.AdaptiveAvgPool2d((nfft // 4, ntimeframes // (num_patches * 4)))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool31 = nn.AdaptiveMaxPool2d((nfft // 8, ntimeframes // (num_patches * 4)))
        self.pool32 = nn.AdaptiveAvgPool2d((nfft // 8, ntimeframes // (num_patches * 8)))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Conv2d(128, 1, (nfft // 8, ntimeframes // (num_patches * 8)))  # Equivalent to FC layers over each channel
        self.nsources = nsources
        self.nfft = nfft
        self.ntimeframes = ntimeframes
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor):
        # x shape: (B, nsources, nfft, ntimeframes)
        batch_size = x.size(0)
        assert x.shape == (batch_size, self.nsources, self.nfft, self.ntimeframes), f"Input shape mismatch: {x.shape} vs {(batch_size, self.nsources, self.nfft, self.ntimeframes)}"
        # Splitting along the T axis into 4 patches
        x = x.unflatten(3, (self.num_patches, self.ntimeframes // self.num_patches)).permute(0, 3, 1, 2, 4)  # (B, npatch, nsources, nfft, ntimeframes // npatch)
        x = x.flatten(0, 1).contiguous()

        # Apply CNN
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool11(x)
        x = self.pool12(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool21(x)
        x = self.pool22(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool31(x)
        x = self.pool32(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.fc(x)
        x = x.view(batch_size, self.num_patches, -1).squeeze(-1)
        return x


@dataclass
class DiscriminatorOutput:
    audio_results: list[torch.Tensor]
    spectrogram_results: torch.Tensor


class Discriminator(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.audio_discriminator = AudioDiscriminator(
            config.nsources // 2, config.ndiscriminators, config.nfilters, config.naudio_disc_layers, config.audio_disc_downsampling_factor
        )
        self.spectrogram_discriminator = SpectrogramPatchModel(
            config.nsources, config.nfft, config.ntimeframes, config.nspec_disc_patches
        )
        self.config = config

    def forward(self, audio: torch.Tensor, spectrogram: torch.Tensor) -> DiscriminatorOutput:
        # Pass audio through the audio discriminator
        audio_results = self.audio_discriminator(audio)

        # Pass spectrogram through the spectrogram discriminator
        spectrogram_results = self.spectrogram_discriminator(spectrogram)

        return DiscriminatorOutput(
            audio_results=audio_results,
            spectrogram_results=spectrogram_results,
        )

    def __call__(self, audio: torch.Tensor, spectrogram: torch.Tensor) -> DiscriminatorOutput:
        # Exists purely for type annotation
        return super().__call__(audio, spectrogram)
