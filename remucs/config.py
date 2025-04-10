from dataclasses import dataclass, asdict
from typing import List
import yaml


@dataclass(frozen=True)
class VAEConfig:
    nbars: int = 4
    nsources: int = 8
    num_workers_ds: int = 0
    dataset_dir: str = "D:/audio-dataset-v3"
    output_dir: str = "D:/output"
    val_count: int = 100
    sample_rate: int = 48000
    ntimeframes: int = 768
    nfft: int = 1025

    z_channels: int = 4
    codebook_size: int = 1024
    nquantizers = 4
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

    kl_mean: bool = True
    seed: int = 1943
    num_workers_dl: int = 0
    autoencoder_batch_size: int = 1
    disc_start: int = 3
    disc_weight: float = 0.5
    disc_hidden: int = 128
    codebook_weight: float = 0.5
    reconstruction_weight: float = 1.0
    perceptual_weight: int = 1
    wasserstein_regularizer: float = 0.1
    gen_weight: int = 1
    spec_weight: int = 1
    epochs: int = 2
    autoencoder_lr: float = 0.00001
    autoencoder_acc_steps: int = 16
    save_steps: int = 512
    vqvae_autoencoder_ckpt_name: str = 'vqvae_autoencoder_ckpt.pth'
    run_name: str = "vqvae-training"
    disc_loss: str = "wgan"
    recon_loss: str = "both"
    turn_off_checking_steps: int = 128
    val_steps: int = 512

    @staticmethod
    def load(file_path: str = "./resources/config/vae.yaml") -> 'VAEConfig':
        """Loads the VAEConfig from a YAML file. By default loads the one inside resources/config"""
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}

        default_dict = asdict(VAEConfig())
        extra_keys = set(yaml_data.keys()) - set(default_dict.keys())
        if extra_keys:
            raise ValueError(f"Unexpected keys in YAML configuration: {extra_keys}")
        merged_config = {**default_dict, **yaml_data}
        for k, v in merged_config.items():
            if isinstance(v, list) and isinstance(default_dict[k], tuple):
                merged_config[k] = tuple(v)
        return VAEConfig(**merged_config)
