# This script is used to train the VQ-VAE model with a discriminator for adversarial loss
# Use the config file in resources/config/vqvae.yaml to set the parameters for training
# Adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/tools/train_vqvae.py
import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch import nn, Tensor
from torch.amp.autocast_mode import autocast
import wandb
import pickle
from accelerate import Accelerator
from remucs.model.vae import VQVAE, VQVAEConfig
from remucs.model.vae import SpectrogramPatchModel as Discriminator
from remucs.model.lpips import load_lpips
from remucs.spectrogram import load_dataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training splits
TRAIN_SPLIT_PERCENTAGE = 0.8
VALIDATION_SPLIT_PERCENTAGE = 0.1
TEST_SPLIT_PERCENTAGE = 0.1

assert isclose(TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE, 1.0)


def read_config(config_path: str):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)
    return config


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def train(config_path: str, base_dir: str, *, start_from_iter: int = 0,
          dataset_params=None, train_params=None, autoencoder_params=None):
    """Retrains the discriminator. If discriminator is None, a new discriminator is created based on the PatchGAN architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = read_config(config_path)
    if dataset_params is not None:
        config['dataset_params'].update(dataset_params)
    if train_params is not None:
        config['train_params'].update(train_params)
    if autoencoder_params is not None:
        config['autoencoder_params'].update(autoencoder_params)

    dataset_dir = config['dataset_params']['dataset_dir']

    # Check credentials first
    credentials_path = config['dataset_params']['credentials_path']
    if not os.path.exists(credentials_path):
        print(f"Credentials file not found at {credentials_path}")
        exit(1)

    # Check cache path
    cache_dir = config['dataset_params']['cache_dir']
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Check lookup tables
    for split in ['train', 'val']:
        lookup_table_path = config['dataset_params'][f"{split}_lookup_table_path"]
        if not os.path.exists(lookup_table_path):
            print(f"Lookup table not found at {lookup_table_path}")
            exit(1)

    vae_config = VQVAEConfig(**config['autoencoder_params'])

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    set_seed(train_config['seed'])

    # Create the model and dataset #
    model = VQVAE(im_channels=dataset_config['im_channels'], model_config=vae_config).to(device)
    print(f"Starting from iteration {start_from_iter}")

    numel = 0
    for p in model.parameters():
        numel += p.numel()
    print('Total number of parameters: {}'.format(numel))

    # Create the dataset
    im_dataset = load_dataset(
        lookup_table_path=dataset_config["train_lookup_table_path"],
        local_dataset_dir=dataset_dir,
        credentials_path=dataset_config["credentials_path"],
        bucket_name=dataset_config["bucket_name"],
        cache_dir=dataset_config["cache_dir"],
        nbars=dataset_config["nbars"],
        backup_dataset_first_n=dataset_config["backup_dataset_first_n_train"]
    )

    val_dataset = load_dataset(
        lookup_table_path=dataset_config["val_lookup_table_path"],
        local_dataset_dir=dataset_dir,
        credentials_path=dataset_config["credentials_path"],
        bucket_name=dataset_config["bucket_name"],
        cache_dir=dataset_config["cache_dir"],
        nbars=dataset_config["nbars"],
        backup_dataset_first_n=dataset_config["backup_dataset_first_n_val"]
    )

    print('Dataset size: {}'.format(len(im_dataset)))

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             num_workers=train_config['num_workers_dl'],
                             shuffle=False)  # Bad machine learning practice but saves so much on my cloud bill + the loaded dataset is essentially random

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=train_config['autoencoder_batch_size'],
                                 num_workers=train_config['num_workers_dl'],
                                 shuffle=False)

    val_count = dataset_config['val_count']

    # Create output directories
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    num_epochs = train_config['epochs']

    reconstruction_loss = torch.nn.MSELoss()
    disc_loss = torch.nn.BCEWithLogitsLoss() if train_config['disc_loss'] == 'bce' else torch.nn.MSELoss()
    perceptual_loss = load_lpips().eval().to(device)

    # Freeze perceptual loss parameters
    for param in perceptual_loss.parameters():
        param.requires_grad = False

    discriminator = Discriminator().to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    accelerator = Accelerator(mixed_precision="bf16")

    model, optimizer_g, data_loader, optimizer_d = accelerator.prepare(
        model, optimizer_g, data_loader, optimizer_d
    )

    disc_step_start = train_config['disc_start']
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']

    val_steps = train_config['val_steps']

    # Reload checkpoint
    if start_from_iter > 0:
        model_save_path = os.path.join(base_dir, f"vqvae_{start_from_iter}_{train_config['vqvae_autoencoder_ckpt_name']}")
        model_sd = torch.load(model_save_path)
        model.load_state_dict(model_sd)
        disc_save_path = os.path.join(base_dir, f"discriminator_{start_from_iter}_{train_config['vqvae_autoencoder_ckpt_name']}")
        disc_sd = torch.load(disc_save_path)
        discriminator.load_state_dict(disc_sd)
        step_count = start_from_iter

    wandb.init(
        # set the wandb project where this run will be logged
        project=train_config['run_name'],
        config=config
    )

    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)

            # im is (4, 4, 2, 512, 512) -> take the mean of two channels
            im = im.mean(dim=2)

            # Fetch autoencoders output(reconstructions)
            with autocast('cuda'):
                model_output = model(im)
            output, z, quantize_losses = model_output

            # Save the model
            if step_count % image_save_steps == 0:
                model_save_path = os.path.join(base_dir, f"vqvae_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
                disc_save_path = os.path.join(base_dir, f"discriminator_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
                torch.save(model.state_dict(), model_save_path)
                torch.save(discriminator.state_dict(), disc_save_path)

            ######### Optimize Generator ##########
            # L2 Loss
            with autocast('cuda'):
                recon_loss = reconstruction_loss(output, im)
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss: torch.Tensor = (recon_loss +
                                    (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                                    (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())

            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred: Tensor = discriminator(model_output[0])
                disc_fake_loss = disc_loss(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps

            # Perceptual Loss
            lpips_loss = torch.mean(perceptual_loss(output, im, normalize=True))
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps

            losses.append(g_loss.item())
            accelerator.backward(g_loss)
            #####################################

            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_loss(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))
                disc_real_loss = disc_loss(disc_real_pred, torch.ones(disc_real_pred.shape, device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                accelerator.backward(disc_loss)
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################

            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

            # Log losses
            wandb.log({
                "Reconstruction Loss": recon_losses[-1],
                "Perceptual Loss": perceptual_losses[-1],
                "Codebook Loss": codebook_losses[-1],
                "Generator Loss": gen_losses[-1] if gen_losses else 0,
                "Discriminator Loss": disc_losses[-1] if disc_losses else 0
            }, step=step_count)

            ########### Perform Validation #############
            val_count_ = 0
            if step_count % val_steps == 0:
                model.eval()
                with torch.no_grad():
                    val_recon_losses = []
                    val_perceptual_losses = []
                    val_codebook_losses = []
                    for val_im in tqdm(val_data_loader, f"Performing validation (step={step_count})", total=min(val_count, len(val_data_loader))):
                        val_count_ += 1
                        if val_count_ > val_count:
                            break
                        val_im = val_im.float().to(device).mean(dim=2)
                        val_model_output = model(val_im)
                        val_output, _, val_quantize_losses = val_model_output

                        val_recon_loss = reconstruction_loss(val_output, val_im)
                        val_recon_losses.append(val_recon_loss.item())

                        val_lpips_loss = torch.mean(perceptual_loss(val_output, val_im))
                        val_perceptual_losses.append(val_lpips_loss.item())

                        val_codebook_losses.append(val_quantize_losses['codebook_loss'].item())

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
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))

        model_save_path = os.path.join(base_dir, f"vqvae_epoch_{epoch_idx}_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
        disc_save_path = os.path.join(base_dir, f"discriminator_epoch_{epoch_idx}_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
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
    train(args.config_path, args.output_dir, start_from_iter=args.start_iter)