import os
from dataclasses import dataclass, asdict
from typing import List
import yaml
from .stft import STFT


@dataclass(frozen=True)
class VAEConfig:
    nbars: int = 4
    nsources: int = 8
    num_workers_ds: int = 0
    dataset_dir: str = "E:/audio-dataset-v3"
    output_dir: str = "E:/output"
    val_count: int = 100
    sample_rate: int = 44100
    ntimeframes: int = 768
    nfft: int = 1025

    z_channels: int = 4
    codebook_size: int = 1024
    nquantizers: int = 4
    down_channels: tuple[int, ...] = (32, 64, 128, 128)
    mid_channels: tuple[int, ...] = (128, 128)
    down_sample: tuple[int, ...] = (2, 2, 4)
    attn_down: tuple[bool, ...] = (False, False, False)
    norm_channels: int = 32
    num_heads: int = 4
    num_down_layers: int = 2
    num_mid_layers: int = 2
    num_up_layers: int = 2
    gradient_checkpointing: bool = True

    seed: int = 1943
    num_workers_dl: int = 0
    batch_size: int = 1
    ds_batch_size: int = 1
    disc_start: int = 3
    disc_spec_weight: float = 0.5
    disc_audio_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    codebook_weight: float = 1
    commitment_beta: float = 0.2
    perceptual_weight: int = 1
    steps: int = 100000
    autoencoder_lr: float = 0.00001
    autoencoder_acc_steps: int = 16
    ndiscriminators: int = 4
    nfilters: int = 1024
    naudio_disc_layers: int = 4
    audio_disc_downsampling_factor: int = 4
    nspec_disc_patches: int = 4
    save_steps: int = 512
    vqvae_autoencoder_ckpt_name: str = 'vqvae_autoencoder_ckpt.pth'
    run_name: str = "vqvae-training"
    disc_loss: str = "bce"
    val_steps: int = 512
    validate_at_step_1: bool = True

    @property
    def audio_length(self) -> int:
        return STFT(self.nfft, self.ntimeframes).l

    @staticmethod
    def load(file_path: str = "./resources/config/vae.yaml") -> 'VAEConfig':
        """Loads the VAEConfig from a YAML file. By default loads the one inside resources/config"""
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}

        default_dict = VAEConfig().asdict()
        extra_keys = set(yaml_data.keys()) - set(default_dict.keys())
        if extra_keys:
            raise ValueError(f"Unexpected keys in YAML configuration: {extra_keys}")

        extra_keys = set(default_dict.keys()) - set(yaml_data.keys())
        if extra_keys:
            raise ValueError(f"Missing keys in YAML configuration: {extra_keys}")

        merged_config = {**default_dict, **yaml_data}
        for k, v in merged_config.items():
            if isinstance(v, list) and isinstance(default_dict[k], tuple):
                merged_config[k] = tuple(v)
        return VAEConfig(**merged_config)

    def __post_init__(self):
        assert self.disc_loss in ("bce", "mse", "hinge")
        assert self.nspec_disc_patches > 0
        assert self.ndiscriminators == len(self.disc_audio_weights)

    def get_vae_save_path(self, step: int) -> str:
        path = os.path.join(self.output_dir, self.run_name, f"step-{step:06d}", self.vqvae_autoencoder_ckpt_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_disc_save_path(self, step: int) -> str:
        path = os.path.join(self.output_dir, self.run_name, f"step-{step:06d}", "disc_ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def asdict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d
