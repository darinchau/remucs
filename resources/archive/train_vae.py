# This script is used to train the VAE model with a discriminator for adversarial loss
# As of right now, due to time constraints, this code is held together by duct tape
# I promise I can do better :))
# Use the config file in resources/config/vqvae.yaml to set the parameters for training
from typing import List
from dataclasses import dataclass, replace, asdict
import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch import nn, Tensor
import torch.autograd as autograd
from torch.amp.autocast_mode import autocast
import wandb
import pickle
from accelerate import Accelerator
from remucs.model.vae import VAE, VAEConfig, VAEOutput
from remucs.model.vggish import Vggish
from remucs.preprocess import spectro, ispectro
from remucs.constants import TRAIN_SPLIT_PERCENTAGE, VALIDATION_SPLIT_PERCENTAGE
import torch.nn.functional as F
from AutoMasher.fyp import SongDataset, YouTubeURL
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class Config:
    nbars: int
    nsources: int
    num_workers_ds: int
    dataset_dir: str
    output_dir: str
    val_count: int
    sample_rate: int
    splice_size: int
    nchannels: int
    down_channels: List[int]
    mid_channels: List[int]
    down_sample: List[int]
    norm_channels: int
    num_heads: int
    num_down_layers: int
    num_mid_layers: int
    num_up_layers: int
    gradient_checkpointing: bool
    kl_mean: bool
    seed: int
    num_workers_dl: int
    autoencoder_batch_size: int
    disc_start: int
    disc_weight: float
    disc_hidden: int
    kl_weight: float
    perceptual_weight: float
    wasserstein_regularizer: float
    gen_weight: float
    spec_weight: float
    epochs: int
    autoencoder_lr: float
    autoencoder_acc_steps: int
    save_steps: int
    vqvae_autoencoder_ckpt_name: str
    run_name: str
    disc_loss: str
    recon_loss: str
    val_steps: int
    turn_off_checking_steps: int


def get_vae_config(config: Config) -> VAEConfig:
    return VAEConfig(
        down_channels=config.down_channels,
        mid_channels=config.mid_channels,
        down_sample=config.down_sample,
        norm_channels=config.norm_channels,
        num_heads=config.num_heads,
        num_down_layers=config.num_down_layers,
        num_mid_layers=config.num_mid_layers,
        num_up_layers=config.num_up_layers,
        gradient_checkpointing=config.gradient_checkpointing,
        nsources=config.nsources,
        nchannels=config.nchannels,
        kl_mean=config.kl_mean,
    )


def load_config_from_yaml(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    # Create an instance of the Config data class using parsed YAML data
    config = Config(**config)
    return config


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


class Discriminator(nn.Module):
    def __init__(self, hidden_size: int):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


class Inference:
    def __init__(self, model: VAE, config: Config, do_sanity_check: bool = True):
        self.model = model

        self.vggish = Vggish().to(device)
        self.config = config

        for p in self.vggish.model.parameters():
            p.requires_grad = False

        self.discriminator = Discriminator(config.disc_hidden).to(device)
        self.do_sanity_check = do_sanity_check

    @property
    def sr(self):
        return self.config.sample_rate

    def get_wgan_gp(self, batch_size: int, real_data: Tensor, fake_data: Tensor):
        alpha = torch.rand(batch_size, 1).expand(real_data.size()).to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.discriminator(interpolates)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.config.wasserstein_regularizer
        return gradient_penalty

    def generator_round(self, im: Tensor, target: Tensor):
        self.discriminator.eval()

        # im is (batch, source, channel, time)
        im = im.float().to(device)
        B, S, C, T = im.shape

        if self.do_sanity_check:
            assert target.shape == (B, C, T), f"Expected {(B, C, T)}, got {target.shape}"
            assert S == self.config.nsources, f"Expected {self.config.nsources}, got {S}"
            assert C == self.config.nchannels, f"Expected {self.config.nchannels}, got {C}"
            assert T == self.config.splice_size, f"Expected {self.config.splice_size}, got {T}"

        # Preprocess
        with torch.autocast("cuda"):
            output: VAEOutput = self.model(im, mean=None, logvar=None, z=None, in_spec=None, check=self.do_sanity_check)
        out = output.out
        kl_loss = output.kl_loss
        out_spec = output.out_spec
        assert out is not None
        assert kl_loss is not None
        assert out_spec is not None

        if self.do_sanity_check:
            # out.shape = (batch, channel, time)
            assert out.shape == (B, C, T), f"Expected {(B, C, T)}, got {out.shape}"

        target_spec = self.model._preprocess(target[:, None], check=False)[0]

        losses = {}
        t_features = self.vggish((target, self.sr)).flatten(-2, -1)
        s_features = self.vggish((out, self.sr)).flatten(-2, -1)
        with torch.autocast("cuda"):
            perceptual_loss = F.mse_loss(t_features, s_features)
            l2_loss = F.mse_loss(target, out)
            spec_recon_loss = F.mse_loss(target_spec, out_spec)

        dgz = self.discriminator(s_features)

        losses['recon_loss'] = l2_loss
        losses['spec_recon_loss'] = spec_recon_loss
        losses['perceptual_loss'] = perceptual_loss
        losses['kl_loss'] = kl_loss

        # Generator wants to minimize dgz
        if self.config.disc_loss == 'wgan':
            losses['gen_loss'] = torch.mean(dgz)
        else:
            losses['gen_loss'] = F.binary_cross_entropy_with_logits(dgz, torch.ones_like(dgz))

        self.discriminator.train()
        return out, losses

    def discriminator_round(self, im: Tensor, target: Tensor):
        self.model.eval()

        # im is (batch, source, channel, time)
        im = im.float().to(device)
        B, S, C, T = im.shape

        if self.do_sanity_check:
            assert target.shape == (B, C, T), f"Expected {(B, C, T)}, got {target.shape}"
            assert S == self.config.nsources, f"Expected {self.config.nsources}, got {S}"
            assert C == self.config.nchannels, f"Expected {self.config.nchannels}, got {C}"
            assert T == self.config.splice_size, f"Expected {self.config.splice_size}, got {T}"

        # Preprocess
        with torch.no_grad():
            with torch.autocast("cuda"):
                output: VAEOutput = self.model(im, mean=None, logvar=None, z=None, in_spec=None, check=self.do_sanity_check)
        out = output.out
        assert out is not None

        if self.do_sanity_check:
            # out.shape = (batch, channel, time)
            assert out.shape == (B, C, T), f"Expected {(B, C, T)}, got {out.shape}"

        losses = {}
        with torch.no_grad():
            t_features = self.vggish((target, self.sr)).flatten(-2, -1)
            s_features = self.vggish((out, self.sr)).flatten(-2, -1)

        dx = self.discriminator(t_features)
        dgz = self.discriminator(s_features)

        # Discriminator wants to maximize dx and minimize dgz
        if self.config.disc_loss == 'wgan':
            d_loss = torch.mean(dx) - torch.mean(dgz)
            gp = self.get_wgan_gp(B, t_features, s_features)
            d_loss += gp
        else:
            d_loss = F.binary_cross_entropy_with_logits(dx, torch.ones_like(dx)) + F.binary_cross_entropy_with_logits(dgz, torch.zeros_like(dgz))

        losses['disc_loss'] = d_loss

        self.model.train()
        return out, losses


class TrainDataset(Dataset):
    def __init__(self, sd: SongDataset, urls: list[YouTubeURL], splice_size: int):
        self.urls = urls
        self.sd = sd
        self.splice_size = splice_size

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx: int):
        url = self.urls[idx]
        audio = self.sd.get_audio(url)
        start_frame = random.randint(0, audio.nframes - self.splice_size)

        parts = self.sd.get_parts(url)

        # Stack in VDIBN convention
        x = [
            aud.slice_frames(start_frame, start_frame + self.splice_size).data
            for aud in (parts.vocals, parts.drums, parts.other, parts.bass, audio)
        ]
        audio = torch.stack(x)
        return audio


def _show_num_params(vae: nn.Module):
    numel = 0
    for p in vae.parameters():
        numel += p.numel()
    print('Total number of parameters: {}'.format(numel))


def save_model(inference: Inference, output_dir: str, step: int):
    model_save_path = os.path.join(output_dir, f"vqvae_{step}_{inference.config.vqvae_autoencoder_ckpt_name}")
    torch.save(inference.model.state_dict(), model_save_path)


def train(config: Config):
    vae_config = get_vae_config(config)
    vae = VAE(vae_config).to(device)
    set_seed(config.seed)

    _show_num_params(vae)

    sd = SongDataset(config.dataset_dir, load_on_the_fly=True)  # We don't need the data files anyway so use load_on_the_fly to skip loading them
    print('Dataset size: {}'.format(len(sd)))

    songs = sd.list_urls("audio")
    # Do train-val-test split
    random.shuffle(songs)
    train_urls = songs[:int(len(songs) * TRAIN_SPLIT_PERCENTAGE)]
    val_urls = songs[int(len(songs) * TRAIN_SPLIT_PERCENTAGE):int(len(songs) * (TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE))]

    print('Train size: {}'.format(len(train_urls)))
    print('Val size: {}'.format(len(val_urls)))

    train_ds = TrainDataset(sd, train_urls, config.splice_size)
    val_ds = TrainDataset(sd, val_urls, config.splice_size)

    train_dl = DataLoader(train_ds, batch_size=config.autoencoder_batch_size, shuffle=True, num_workers=config.num_workers_dl)
    val_dl = DataLoader(val_ds, batch_size=config.autoencoder_batch_size, shuffle=False, num_workers=config.num_workers_dl)

    # Create output directories
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    optimizer_g = Adam(vae.parameters(), lr=config.autoencoder_lr, betas=(0.5, 0.999))
    optimizer_d = Adam(vae.parameters(), lr=config.autoencoder_lr, betas=(0.5, 0.999))

    accelerator = Accelerator(mixed_precision="bf16")

    vae, optimizer_g, optimizer_d, train_dl, val_dl = accelerator.prepare(
        vae, optimizer_g, optimizer_d, train_dl, val_dl
    )

    inference = Inference(vae, config)

    wandb.init(
        # set the wandb project where this run will be logged
        project=config.run_name,
        config=asdict(config),
    )

    step_count = 0

    for epoch in range(config.epochs):
        optimizer_g.zero_grad()
        for im in tqdm(train_dl, desc=f'Epoch {epoch + 1}/{config.epochs}'):
            # Do some bookkeeping
            step_count += 1

            if step_count % config.save_steps == 0:
                save_model(inference, config.output_dir, step_count)

            t = time.time()
            if step_count >= config.turn_off_checking_steps:
                inference.do_sanity_check = False

            # Hard code to be one step of discriminator for every step of generator
            is_train_discriminator = step_count >= config.disc_start and step_count % 2 == 0
            if is_train_discriminator:
                _, loss = inference.discriminator_round(im[:, :-1], im[:, -1])
                d_loss = loss['disc_loss']
                accelerator.backward(d_loss)
                optimizer_d.step()

                wandb.log({
                    "Discriminator Loss": d_loss.item(),
                    "Time": time.time() - t
                }, step=step_count)
            else:
                _, loss = inference.generator_round(im[:, :-1], im[:, -1])
                g_loss = (
                    loss['recon_loss'] +
                    loss['kl_loss'] * config.kl_weight +
                    loss['perceptual_loss'] * config.perceptual_weight +
                    loss['gen_loss'] * config.gen_weight +
                    loss['spec_recon_loss'] * config.spec_weight
                )

                accelerator.backward(g_loss)
                optimizer_g.step()

                wandb.log({
                    "Reconstruction Loss": loss['recon_loss'].item(),
                    "Perceptual Loss": loss['perceptual_loss'].item(),
                    "KL Loss": loss['kl_loss'].item(),
                    "Generator Loss": loss['gen_loss'].item(),
                    "Spectrogram Reconstruction Loss": loss['spec_recon_loss'].item(),
                    "Time": time.time() - t
                }, step=step_count)

            if step_count % config.val_steps == 0:
                vae.eval()
                val_step_count = 0
                with torch.no_grad():
                    val_recon_losses = []
                    val_perceptual_losses = []
                    val_kl_losses = []
                    val_spec_losses = []
                    for val_im in tqdm(val_dl, desc='Validation...', total=config.val_count):
                        val_step_count += 1
                        _, val_loss = inference.generator_round(val_im[:, :-1], val_im[:, -1])
                        val_recon_losses.append(val_loss['recon_loss'].item())
                        val_perceptual_losses.append(val_loss['perceptual_loss'].item())
                        val_kl_losses.append(val_loss['kl_loss'].item())
                        val_spec_losses.append(val_loss['spec_recon_loss'].item())
                        if val_step_count >= config.val_count:
                            break

                wandb.log({
                    "Val Reconstruction Loss": np.mean(val_recon_losses),
                    "Val Perceptual Loss": np.mean(val_perceptual_losses),
                    "Val KL Loss": np.mean(val_kl_losses),
                    "Val Spectrogram Loss": np.mean(val_spec_losses)
                }, step=step_count)

                print(f"Validation complete: Reconstruction loss: {np.mean(val_recon_losses)}, Perceptual Loss: {np.mean(val_perceptual_losses)}, KL loss: {np.mean(val_kl_losses)}")
                vae.train()

    wandb.finish()
    print('Done Training...')

    save_model(inference, config.output_dir, step_count)


def main():
    parser = argparse.ArgumentParser(description='Train VAE model with discriminator')
    parser.add_argument('--config', type=str, help='Path to the config file', default='resources/config/vae.yaml')
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    train(config)


if __name__ == '__main__':
    main()

# TODO: Start from middle of the training
