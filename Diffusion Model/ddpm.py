import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import GPT2Config

import datetime
import lightning as pl
import numpy as np
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
import os
import soundfile as sf

from openai_unet import UNetModel
from autoEncoder import AutoencoderKL
from cape.maeeg import MAEEG_pl
from cape.maeeg import Encoder
from audioldm2.latent_diffusion.modules.encoders.modules import SequenceGenAudioMAECond
from audioldm_train.utilities.data.eeg_dataset import EEGDataset

from config import config_dict
from lightning import Trainer
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor


import warnings

warnings.filterwarnings("ignore")


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = 16  # WIDTH // 8
LATENTS_HEIGHT = 256  # HEIGHT // 8

# from modules import *


class LDM(pl.LightningModule):
    def __init__(self, in_channels, t_steps, img_depth, num_tsteps):
        super().__init__()
        self.beta_start = 1e-4
        self.beta_end = 0.02

        self.num_tsteps = num_tsteps  # config["diffusion_steps"]

        self.t_steps = torch.arange(t_steps)
        self.unet = UNetModel(
            in_channels, channels=img_depth, channel_multipliers=(1, 2, 4, 8)
        )

        self.beta = np.array(
            [
                self.beta_start + (t / self.t_steps) * (self.beta_end - self.beta_start)
                for t in range(0, self.t_steps)
            ]
        )
        self.alpha = 1 - self.beta

        self.alphas_cumprod = np.cumprod(self.alpha)

    def forward(self, x, t, cond):
        return self.unet(x, t, cond)

    def get_loss(self, x, batch_idx):  # psuedo forward

        ts = torch.randint(0, self.t_steps, (x.shape[0]), device=self.device).long()

        noise = torch.randn_like(x, device=self.device)

        alphas_cumprod = self.alphas_cumprod[ts]
        alphas_cumprod = alphas_cumprod.to(device=self.device, dtype=x.dtype)

        alphas_cumprod = alphas_cumprod.flatten()
        while len(alphas_cumprod.shape) < len(x.shape):
            alphas_cumprod = alphas_cumprod.unsqueeze(-1)

        noisy_x = (alphas_cumprod**0.5) * x + ((1 - alphas_cumprod) ** 0.5) * noise

        pred_noise = self.forward(noisy_x, ts)

        loss = F.mse_loss(pred_noise, noise)  # reshape(-1,self.in_size)

        return loss

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
            else:
                z = 0
            # Get the predicted noise from the U-Net
            e_hat = self.forward(x, t.view(1).repeat(x.shape[0]))
            # Perform the denoising step to take the image from t to t-1
            pre_scale = 1 / torch.sqrt(self.alpha[t])
            e_scale = (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])
            post_sigma = torch.sqrt(self.beta[t]) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


class LatentDiffusion(nn.Module):
    def __init__(
        self,
        num_training_steps=1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):

        super().__init__()

        first_stage_config = {
            "base_learning_rate": 0.000008,
            "target": "audioldm2.latent_encoder.autoencoder.AutoencoderKL",
            "params": {
                "sampling_rate": 16000,
                "batchsize": 4,
                "monitor": "val/rec_loss",
                "image_key": "fbank",
                "subband": 1,
                "embed_dim": 8,
                "time_shuffle": 1,
                "lossconfig": {
                    "target": "audioldm2.latent_diffusion.modules.losses.LPIPSWithDiscriminator",
                    "params": {
                        "disc_start": 50001,
                        "kl_weight": 1000,
                        "disc_weight": 0.5,
                        "disc_in_channels": 1,
                    },
                },
                "ddconfig": {
                    "double_z": True,
                    "mel_bins": 64,
                    "z_channels": 8,
                    "resolution": 256,
                    "downsample_time": False,
                    "in_channels": 1,
                    "out_ch": 1,
                    "ch": 128,
                    "ch_mult": [1, 2, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0,
                },
            },
        }

        mcond_stage_config = {
            "crossattn_audiomae_generated": {
                "cond_stage_key": "all",
                "conditioning_key": "crossattn",
                "target": "audioldm2.latent_diffusion.modules.encoders.modules.SequenceGenAudioMAECond",
                "params": {
                    "always_output_audiomae_gt": False,
                    "learnable": True,
                    "use_gt_mae_output": True,
                    "use_gt_mae_prob": 1,
                    "base_learning_rate": 0.0002,
                    "sequence_gen_length": 512,
                    "use_warmup": True,
                    "sequence_input_key": ["film_clap_cond1", "crossattn_vits_phoneme"],
                    "sequence_input_embed_dim": [512, 192],
                    "batchsize": 16,
                    "cond_stage_config": {
                        "film_clap_cond1": {
                            "cond_stage_key": "text",
                            "conditioning_key": "film",
                            "target": "audioldm2.latent_diffusion.modules.encoders.modules.CLAPAudioEmbeddingClassifierFreev2",
                            "params": {
                                "sampling_rate": 48000,
                                "embed_mode": "text",
                                "amodel": "HTSAT-base",
                            },
                        },
                        "crossattn_vits_phoneme": {
                            "cond_stage_key": "phoneme_idx",
                            "conditioning_key": "crossattn",
                            "target": "audioldm2.latent_diffusion.modules.encoders.modules.PhonemeEncoder",
                            "params": {
                                "vocabs_size": 183,
                                "pad_token_id": 0,
                                "pad_length": 310,
                            },
                        },
                        "crossattn_audiomae_pooled": {
                            "cond_stage_key": "ta_kaldi_fbank",
                            "conditioning_key": "crossattn",
                            "target": "audioldm2.latent_diffusion.modules.encoders.modules.AudioMAEConditionCTPoolRand",
                            "params": {
                                "regularization": False,
                                "no_audiomae_mask": True,
                                "time_pooling_factors": [1],
                                "freq_pooling_factors": [1],
                                "eval_time_pooling": 1,
                                "eval_freq_pooling": 1,
                                "mask_ratio": 0,
                            },
                        },
                    },
                },
            }
        }

        unet_config = {
            "target": "audioldm2.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "context_dim": 768,  # [768, 1024],
                # "extra_film_condition_dim": 512,
                "in_channels": 8,
                "out_channels": 8,
                "model_channels": 128,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 3, 5],
                "num_head_channels": 32,
                "use_spatial_transformer": True,
                "transformer_depth": 1,
            },
        }

        self.first_stage_model = AutoencoderKL(**first_stage_config["params"])

        # self.cond_stage_models = nn.ModuleList([AudioMAE(), MAEEG_pl(), PhonemeEncoder()])

        self.cond_data_keys = [
            "stft",
            "eeg",
            "text",
        ]  # shld text be pre-processed to phonemes
        self.cond_dict = {"stft": 0, "eeg": 1, "text": 2}

        # no gpt for now

        self.unet = UNetModel(**unet_config["params"])

        # DDPM params:

        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_train_timesteps = num_training_steps

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, x: torch.FloatTensor, t: torch.FloatTensor):
        """
        Add noise to the input tensor.
        Args:
            x (torch.FloatTensor): Input tensor.
            t (torch.FloatTensor): Noise tensor.
        Returns:
            torch.FloatTensor: Output tensor.
        """

        noise = torch.randn_like(x, device=self.device)

        alphas_cumprod = self.alphas_cumprod[t]
        alphas_cumprod = alphas_cumprod.to(device=self.device, dtype=x.dtype)

        alphas_cumprod = alphas_cumprod.flatten()
        while len(alphas_cumprod.shape) < len(x.shape):
            alphas_cumprod = alphas_cumprod.unsqueeze(-1)

        noisy_x = (alphas_cumprod**0.5) * x + ((1 - alphas_cumprod) ** 0.5) * noise

        return noise, noisy_x

    def forward(self, x):
        x_cond = []
        for cond_model, cond_data_key in zip(
            self.cond_stage_models, self.cond_data_keys
        ):
            x_cond.append(cond_model(x[cond_data_key]))

        t = torch.randint(
            0, self.num_train_timesteps, (x.shape[0],), device=self.device
        ).long()

        noise, noisy_x = self.add_noise(x, t)

        cond = x_cond[0]  # pick between all three + uncondition
        pred_noise = self.unet(noisy_x, t, cond)
        loss = F.mse_loss(pred_noise, noise)  # reshape(-1,self.in_size)

        return loss

    def ddpm_step(self, t: int, latents, noise_pred_t):

        prev_t = t - 1
        alpha_prod_prev = self.alphas_cumprod[prev_t]
        alpha_prod = self.alphas_cumprod[t]
        alpha = self.alphas[t]
        beta = self.betas[t]

        mean_coeff_0 = beta * np.sqrt(alpha_prod_prev) / (1.0 - alpha_prod)
        mean_coeff_t = (1.0 - alpha_prod_prev) * np.sqrt(alpha) / (1 - alpha_prod)
        xt = latents
        # following ddpm eq 15
        x0 = (xt - (1 - alpha_prod) ** 0.5 * noise_pred_t) / alpha_prod**0.5
        # following ddpm eq 7
        mean = mean_coeff_0 * x0 + mean_coeff_t * xt
        variance = torch.clamp(
            beta * (1.0 - alpha_prod_prev) / (1.0 - alpha_prod), min=1e-20
        )

        # X = mu + sigma * N(0, 1)
        prev_xt = mean + (variance**0.5) * torch.randn_like(xt)

        return prev_xt

    def save_waveform(self, waveform, savepath, name="outwav"):
        for i in range(waveform.shape[0]):
            if type(name) is str:
                path = os.path.join(
                    savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                )
            elif type(name) is list:
                path = os.path.join(
                    savepath,
                    "%s.wav"
                    % (
                        os.path.basename(name[i])
                        if (not ".wav" in name[i])
                        else os.path.basename(name[i]).split(".")[0]
                    ),
                )
            else:
                raise NotImplementedError
            todo_waveform = waveform[i, 0]
            todo_waveform = (
                todo_waveform / np.max(np.abs(todo_waveform))
            ) * 0.8  # Normalize the energy of the generation output
            sf.write(path, todo_waveform, samplerate=self.sampling_rate)

    def mel_spectrogram_to_waveform(
        self, mel, savepath=".", bs=None, name="outwav", save=True
    ):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform

    # def sample(self,cond, batch_size, timesteps):

    def generate_sample(
        self,
        batchs,
        infer_steps=10,
        uncon=0,
        cond="eeg",
    ):
        waveform_save_path = "/mnt/nvme/node02/pranav/AE24/chitVachana/output"
        # cond_idx =self.cond_dict[cond]
        # cond_context=self.cond_stage_models[cond_idx](batchs[cond_idx])
        # uncond_context=self.cond_stage_models[cond_idx](uncond)

        # batch_size ,512

        B, _, dim = batchs[cond].shape
        cond_context = torch.randn((B, 512))
        latents_shape = (B, 8, LATENTS_HEIGHT, LATENTS_WIDTH)  # channels mean and var
        latents = torch.randn(latents_shape)  # generator=generator, device=device)

        self.set_inference_timesteps(infer_steps)
        ddpm_steps = self.timesteps

        for i, timestep in enumerate(tqdm(ddpm_steps)):
            # (1, 320)
            ts = torch.full((B,), timestep, dtype=torch.long)
            # model_input = torch.cat([latents,cond_context],dim=1)
            model_output = self.unet(
                latents, ts, y=cond_context, context_list=[], context_attn_mask_list=[]
            )
            latents = self.ddpm_step(timestep, latents, model_output)

        # z = 1.0 / self.scale_factor * z
        mel = self.first_stage_model.decode(latents)
        print("melshape:", mel.shape)
        waveform = self.mel_spectrogram_to_waveform(
            mel, savepath=waveform_save_path, bs=None, name="test.wav", save=False
        )

        # mel_out=self.calculate_mel_spectrogram(waveform,16000)
        return waveform
