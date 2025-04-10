import torch
import torchaudio
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrogramMaker:
    """This config object decides the dimensions of the audio and the spectrograms that we will use

    Converts between stuff but also doubles as a config object
    """
    sample_rate: int
    target_features: int
    target_time_frames: int
    hop_length: int
    nfft: int
    target_nframes: int

    def mel(self, x: torch.Tensor):
        # x shape: (..., T)
        assert x.shape[-1] == self.target_nframes, f"Expected {self.target_nframes}, got {x.shape[-1]}"
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            n_mels=self.target_features,
            center=False,
            power=1,
        )

        xmel = mel(x.squeeze(1)).log1p()
        assert xmel.shape[-2] == self.target_features, f"Expected {self.target_features}, got {xmel.shape[-2]}"
        assert xmel.shape[-1] == self.target_time_frames, f"Expected {self.target_time_frames}, got {xmel.shape[-1]}"
        return xmel


def get_standard_config():
    return SpectrogramMaker(
        sample_rate=44100,
        target_features=256,
        target_time_frames=1536,
        hop_length=256,
        nfft=511,
        target_nframes=195456,
    )
