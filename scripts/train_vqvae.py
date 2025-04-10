# This script is used to train the VQ-VAE model with a discriminator for adversarial loss
# Use the config file in resources/config/vqvae.yaml to set the parameters for training
# Adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/tools/train_vqvae.py
import yaml
import argparse
import torch
import random
import os
import wandb
import pickle
import numpy as np
import json
import torch.nn.functional as F
import time
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch import nn, Tensor
from torch.amp.autocast_mode import autocast
from accelerate import Accelerator
from math import isclose
from remucs.model.vae import RVQVAE as VAE, VAEOutput
from remucs.model.vggish import Vggish
from remucs.model.discriminator import Discriminator
from remucs.config import VAEConfig
from AutoMasher.fyp import SongDataset, YouTubeURL, Audio
from AutoMasher.fyp.audio import DemucsCollection
from remucs.stft import STFT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training splits
TRAIN_SPLIT_PERCENTAGE = 0.8
VALIDATION_SPLIT_PERCENTAGE = 0.1
TEST_SPLIT_PERCENTAGE = 0.1

assert isclose(TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE, 1.0)

KEY = "project-remucs-vqvae-split"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VQVAEDataset(torch.utils.data.Dataset):
    def __init__(self, config: VAEConfig, urls: list[YouTubeURL]):
        self.urls = urls
        self.config = config
        self.sd = SongDataset(config.dataset_dir, load_on_the_fly=True)
        self.slice_length = STFT(config.nfft, config.ntimeframes).l

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx: int):
        url = self.urls[idx]
        path = self.sd.get_path("parts", url)
        parts = DemucsCollection.load(path).map(
            lambda x: x.resample(self.config.sample_rate)
        )
        if parts.nframes < self.slice_length:
            return self.__getitem__(random.randint(0, len(self.urls) - 1))

        start = random.randint(0, max(0, parts.nframes - self.slice_length))
        end = start + self.slice_length

        parts = parts.map(
            lambda x: x.slice_frames(start, end).to_nchannels(1)
        ).data.squeeze(1)  # Shape: (nsource=4, L)

        return parts


def load_lpips(config: VAEConfig):
    class _PerceptualLossWrapper(nn.Module):
        def __init__(self, stride: int):
            super().__init__()
            self.lpips = Vggish()
            self.stride = stride

        def forward(self, pred_audio, targ_audio):
            pred_audio = pred_audio.unflatten(-1, (-1, self.stride)).mean(dim=-1)
            targ_audio = targ_audio.unflatten(-1, (-1, self.stride)).mean(dim=-1)
            pred = self.lpips((pred_audio, 16000))
            targ = self.lpips((targ_audio, 16000))
            return F.mse_loss(pred, targ)

    assert config.sample_rate % 16000 == 0, f"Sample rate must be an integer multiple of the VGGish sample rate (16000)"
    model = _PerceptualLossWrapper(config.sample_rate // 16000)
    model = model.eval().to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def maybe_make_split(sd: SongDataset):
    filepath = "project-remucs-data-split.json"
    sd.register(KEY, filepath)
    path = sd.get_path(KEY)
    if os.path.isfile(path):
        with open(path, 'r') as f:
            split = json.load(f)
        assert set(split.keys()) == {"train", "val", "test"}
        return split

    print("Creating dataset split...")
    audios = sd.list_urls("parts")

    random.shuffle(audios)

    train_count = int(len(audios) * TRAIN_SPLIT_PERCENTAGE)
    val_count = int(len(audios) * VALIDATION_SPLIT_PERCENTAGE)
    test_count = len(audios) - train_count - val_count

    split = {
        "train": audios[:train_count],
        "val": audios[train_count:train_count + val_count],
        "test": audios[train_count + val_count:]
    }

    with open(path, 'w') as f:
        json.dump(split, f)

    return split


def train(config_path: str, start_from_iter: int = 0):
    """Retrains the discriminator. If discriminator is None, a new discriminator is created based on the PatchGAN architecture."""

    config = VAEConfig.load(config_path)
    dataset_dir = config.dataset_dir

    sd = SongDataset(dataset_dir, load_on_the_fly=True)
    split = maybe_make_split(sd)
    set_seed(config.seed)

    # Create the model and dataset #
    model = VAE(config).to(device)
    print(f"Starting from iteration {start_from_iter}")

    numel = 0
    for p in model.parameters():
        numel += p.numel()
    print('Total number of parameters: {}'.format(numel))

    # Create the dataset
    im_dataset = VQVAEDataset(config, split['train'])
    val_dataset = VQVAEDataset(config, split['val'])

    print('Dataset size: {}'.format(len(im_dataset)))

    data_loader = DataLoader(
        im_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers_dl,
        shuffle=True
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers_dl,
        shuffle=False
    )

    os.makedirs(config.output_dir, exist_ok=True)

    reconstruction_loss = torch.nn.MSELoss()
    disc_loss = torch.nn.BCEWithLogitsLoss() if config.disc_loss == 'bce' else torch.nn.MSELoss()
    perceptual_loss = load_lpips(config)

    discriminator = Discriminator(config).to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=config.autoencoder_lr, betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=config.autoencoder_lr, betas=(0.5, 0.999))

    accelerator = Accelerator(mixed_precision="bf16")
    step_count = 0

    # Reload checkpoint
    if start_from_iter > 0:
        model_save_path = config.get_vae_save_path(start_from_iter)
        model_sd = torch.load(model_save_path)
        model.load_state_dict(model_sd)
        disc_save_path = config.get_disc_save_path(start_from_iter)
        disc_sd = torch.load(disc_save_path)
        discriminator.load_state_dict(disc_sd)
        step_count = start_from_iter

    model, optimizer_g, data_loader, optimizer_d = accelerator.prepare(
        model, optimizer_g, data_loader, optimizer_d
    )

    stft = STFT(config.nfft, config.ntimeframes)

    wandb.init(
        # set the wandb project where this run will be logged
        project=config.run_name,
        config=config.asdict()
    )

    for epoch_idx in range(config.epochs):
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for target_audio in tqdm(data_loader):
            step_count += 1

            assert isinstance(target_audio, torch.Tensor)
            assert target_audio.dim() == 3  # im shape: B, nsources/2=4, L
            assert target_audio.shape[1] == config.nsources / 2

            batch_size = target_audio.shape[0]
            target_audio = target_audio.float().to(device)
            target_spec = stft.forward(
                target_audio.float().flatten(0, 1)  # B*4, L
            )
            target_spec = torch.view_as_real(target_spec).permute(0, 3, 1, 2)  # B*4, 2, N, T
            target_spec = target_spec.unflatten(0, (batch_size, 4)).flatten(1, 2).float().to(device)  # B, 4*2, N, T

            # Fetch autoencoders output(reconstructions)
            with autocast('cuda'):
                model_output: VAEOutput = model(target_spec)

            pred_spec = model_output.output
            pred_audio = stft.inverse(
                torch.view_as_complex(pred_spec.unflatten(1, (4, 2)).flatten(0, 1).permute(0, 2, 3, 1)).contiguous()
            ).unflatten(0, (batch_size, 4))

            ######### Optimize Generator ##########
            # L2 Loss
            with autocast('cuda'):
                recon_loss = reconstruction_loss(pred_audio, target_audio) + reconstruction_loss(pred_spec, target_spec)

            recon_loss = recon_loss / config.autoencoder_acc_steps
            g_loss: torch.Tensor = (recon_loss +
                                    (config.codebook_weight * model_output.codebook_loss / config.autoencoder_acc_steps) +
                                    (config.commitment_beta * model_output.commitment_loss / config.autoencoder_acc_steps))

            # Adversarial loss only if disc_step_start steps passed
            if step_count > config.disc_start:
                disc_fake_preds: list[Tensor] = discriminator(pred_audio, pred_spec)
                disc_fake_loss = sum(
                    disc_loss(x, torch.zeros_like(x)) for x in disc_fake_preds
                )
                g_loss += config.disc_weight * disc_fake_loss / config.autoencoder_acc_steps

            # Perceptual Loss
            lpips_loss = torch.mean(perceptual_loss(pred_audio, target_audio))
            g_loss += config.perceptual_weight * lpips_loss / config.autoencoder_acc_steps

            accelerator.backward(g_loss)
            #####################################

            ######### Optimize Discriminator #######
            disc_losses = []
            if step_count > config.disc_start:
                disc_fake_pred: Tensor = discriminator(pred_audio, pred_spec)
                disc_real_pred: Tensor = discriminator(target_audio, target_spec)
                disc_fake_loss = disc_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = disc_loss(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = config.disc_weight * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / config.autoencoder_acc_steps
                accelerator.backward(disc_loss)
                if step_count % config.autoencoder_acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################

            if step_count % config.autoencoder_acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

            # Log losses
            wandb.log({
                "Reconstruction Loss": recon_loss.item(),
                "Perceptual Loss": lpips_loss.item(),
                "Codebook Loss": model_output.codebook_loss.item(),
                "Generator Loss": g_loss.item(),
                "Discriminator Loss": disc_losses[-1] if disc_losses else 0.0
            }, step=step_count)

            if step_count % config.save_steps == 0:
                model_save_path = VAEConfig.get_vae_save_path(config, step_count)
                disc_save_path = VAEConfig.get_disc_save_path(config, step_count)
                torch.save(model.state_dict(), model_save_path)
                torch.save(discriminator.state_dict(), disc_save_path)

            ########### Perform Validation #############
            val_count_ = 0
            if step_count % config.val_steps == 0:
                model.eval()
                with torch.no_grad():
                    val_recon_losses = []
                    val_perceptual_losses = []
                    val_codebook_losses = []
                    for val_im in tqdm(val_data_loader, f"Performing validation (step={step_count})", total=min(config.val_count, len(val_data_loader))):
                        val_count_ += 1
                        if val_count_ > config.val_count:
                            break

                        batch_size = target_audio.shape[0]
                        target_audio = target_audio.float().to(device)
                        target_spec = stft.forward(
                            target_audio.float().flatten(0, 1)  # B*4, L
                        )
                        target_spec = torch.view_as_real(target_spec).permute(0, 3, 1, 2)  # B*4, 2, N, T
                        target_spec = target_spec.unflatten(0, (batch_size, 4)).flatten(1, 2).float().to(device)  # B, 4*2, N, T

                        with autocast('cuda'):
                            model_output: VAEOutput = model(target_spec)

                        pred_spec = model_output.output
                        pred_audio = stft.inverse(
                            torch.view_as_complex(pred_spec.unflatten(1, (4, 2)).flatten(0, 1).permute(0, 2, 3, 1)).contiguous()
                        ).unflatten(0, (batch_size, 4))

                        with autocast('cuda'):
                            recon_loss = reconstruction_loss(pred_audio, target_audio) + reconstruction_loss(pred_spec, target_spec)

                        val_recon_loss = recon_loss.item()
                        val_recon_losses.append(val_recon_loss)

                        val_lpips_loss = torch.mean(perceptual_loss(pred_audio, target_audio)).item()
                        val_perceptual_losses.append(val_lpips_loss)

                        val_codebook_loss = model_output.codebook_loss.item()
                        val_codebook_losses.append(val_codebook_loss)

                wandb.log({
                    "Val Reconstruction Loss": np.mean(val_recon_losses),
                    "Val Perceptual Loss": np.mean(val_perceptual_losses),
                    "Val Codebook Loss": np.mean(val_codebook_losses)
                }, step=step_count)

                tqdm.write(f"Validation complete: Reconstruction loss: {np.mean(val_recon_losses)}, Perceptual Loss: {np.mean(val_perceptual_losses)}, Codebook loss: {np.mean(val_codebook_losses)}")
                model.train()

        # End of epoch. Clean up the gradients and losses and save the model
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        model_save_path = config.get_vae_save_path(step_count)
        disc_save_path = config.get_disc_save_path(step_count)
        torch.save(model.state_dict(), model_save_path)
        torch.save(discriminator.state_dict(), disc_save_path)

    wandb.finish()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path', default='resources/config/vqvae.yaml', type=str)
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='resources/models/vqvae')
    parser.add_argument('--start_iter', dest='start_iter', type=int, default=0)
    args = parser.parse_args()
    train(args.config_path, start_from_iter=args.start_iter)
