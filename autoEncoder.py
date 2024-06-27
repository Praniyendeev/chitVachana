import torch
import os

import torch.nn.functional as F
import numpy as np

from torch import nn, einsum
from einops import rearrange, repeat
import soundfile as sf

# from audioldm2.latent_diffusion.modules.ema import *
import hifigan


def get_vocoder_config():
    return {
        "resblock": "1",
        "num_gpus": 6,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,
        "upsample_rates": [5, 4, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 8, 4, 4],
        "upsample_initial_channel": 1024,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "segment_size": 8192,
        "num_mels": 64,
        "num_freq": 1025,
        "n_fft": 1024,
        "hop_size": 160,
        "win_size": 1024,
        "sampling_rate": 16000,
        "fmin": 0,
        "fmax": 8000,
        "fmax_for_loss": None,
        "num_workers": 4,
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1,
        },
    }


def get_vocoder(config, device, mel_bins):
    name = "HiFi-GAN"
    speaker = ""
    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        if mel_bins == 64:
            config = get_vocoder_config()
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator_old(config)
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)

    return vocoder


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class UpsampleTimeStride4(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=5, stride=1, padding=2
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=(4.0, 2.0), mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # Do time downsampling here
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class DownsampleTimeStride4(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # Do time downsampling here
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=5, stride=(4, 2), padding=1
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=(4, 2), stride=(4, 2))
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # b,hw,c
        k = k.reshape(b, c, h * w).contiguous()  # b,c,hw
        w_ = torch.bmm(q, k).contiguous()  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w).contiguous()
        w_ = w_.permute(0, 2, 1).contiguous()  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(
            v, w_
        ).contiguous()  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w).contiguous()

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        downsample_time_stride4_levels=[],
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.downsample_time_stride4_levels = downsample_time_stride4_levels

        if len(self.downsample_time_stride4_levels) > 0:
            assert max(self.downsample_time_stride4_levels) < self.num_resolutions, (
                "The level to perform downsample 4 operation need to be smaller than the total resolution number %s"
                % str(self.num_resolutions)
            )

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level in self.downsample_time_stride4_levels:
                    down.downsample = DownsampleTimeStride4(block_in, resamp_with_conv)
                else:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        downsample_time_stride4_levels=[],
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.downsample_time_stride4_levels = downsample_time_stride4_levels

        if len(self.downsample_time_stride4_levels) > 0:
            assert max(self.downsample_time_stride4_levels) < self.num_resolutions, (
                "The level to perform downsample 4 operation need to be smaller than the total resolution number %s"
                % str(self.num_resolutions)
            )

        # compute in_ch_mult, block_in and curr_res at lowest res
        (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print(
        #     "Working with z of shape {} = {} dimensions.".format(
        #         self.z_shape, np.prod(self.z_shape)
        #     )
        # )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level - 1 in self.downsample_time_stride4_levels:
                    up.upsample = UpsampleTimeStride4(block_in, resamp_with_conv)
                else:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    # wavs = [wav for wav in wavs]

    # for i in range(len(mels)):
    #     if lengths is not None:
    #         wavs[i] = wavs[i][: lengths[i]]

    return wavs


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        batchsize=None,
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        sampling_rate=16000,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        image_key="fbank",
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
    ):
        super().__init__()
        self.automatic_optimization = False
        assert (
            "mel_bins" in ddconfig.keys()
        ), "mel_bins is not specified in the Autoencoder config"
        num_mel = ddconfig["mel_bins"]
        self.image_key = image_key
        self.sampling_rate = sampling_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.loss = None
        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if self.image_key == "fbank":
            self.vocoder = get_vocoder(None, "cpu", num_mel)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.learning_rate = float(base_learning_rate)
        # print("Initial learning rate %s" % self.learning_rate)

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.feature_cache = None
        self.flag_first_run = True
        self.train_step = 0

        self.logger_save_dir = None
        self.logger_exp_name = None

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # x = self.time_shuffle_operation(x)
        # x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # posterior = DiagonalGaussianDistribution(moments)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)

        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)

        return mean,std

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # bs, ch, shuffled_timesteps, fbins = dec.size()
        # dec = self.time_unshuffle_operation(dec, bs, int(ch*shuffled_timesteps), fbins)
        # dec = self.freq_merge_subband(dec)
        return dec

    def decode_to_waveform(self, dec):

        if self.image_key == "fbank":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = vocoder_infer(dec, self.vocoder)
        elif self.image_key == "stft":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = self.wave_decoder(dec)
        return wav_reconstruction

    def visualize_latent(self, input):
        import matplotlib.pyplot as plt

        # for i in range(10):
        #     zero_input = torch.zeros_like(input) - 11.59
        #     zero_input[:,:,i * 16: i * 16 + 16,:16] += 13.59

        #     posterior = self.encode(zero_input)
        #     latent = posterior.sample()
        #     avg_latent = torch.mean(latent, dim=1)[0]
        #     plt.imshow(avg_latent.cpu().detach().numpy().T)
        #     plt.savefig("%s.png" % i)
        #     plt.close()

        np.save("input.npy", input.cpu().detach().numpy())
        # zero_input = torch.zeros_like(input) - 11.59
        time_input = input.clone()
        time_input[:, :, :, :32] *= 0
        time_input[:, :, :, :32] -= 11.59

        np.save("time_input.npy", time_input.cpu().detach().numpy())

        posterior = self.encode(time_input)
        latent = posterior.sample()
        np.save("time_latent.npy", latent.cpu().detach().numpy())
        avg_latent = torch.mean(latent, dim=1)
        for i in range(avg_latent.size(0)):
            plt.imshow(avg_latent[i].cpu().detach().numpy().T)
            plt.savefig("freq_%s.png" % i)
            plt.close()

        freq_input = input.clone()
        freq_input[:, :, :512, :] *= 0
        freq_input[:, :, :512, :] -= 11.59

        np.save("freq_input.npy", freq_input.cpu().detach().numpy())

        posterior = self.encode(freq_input)
        latent = posterior.sample()
        np.save("freq_latent.npy", latent.cpu().detach().numpy())
        avg_latent = torch.mean(latent, dim=1)
        for i in range(avg_latent.size(0)):
            plt.imshow(avg_latent[i].cpu().detach().numpy().T)
            plt.savefig("time_%s.png" % i)
            plt.close()

    def get_input(self, batch):
        fname, text, label_indices, waveform, stft, fbank = (
            batch["fname"],
            batch["text"],
            batch["label_vector"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
        )
        # if(self.time_shuffle != 1):
        #     if(fbank.size(1) % self.time_shuffle != 0):
        #         pad_len = self.time_shuffle - (fbank.size(1) % self.time_shuffle)
        #         fbank = torch.nn.functional.pad(fbank, (0,0,0,pad_len))

        ret = {}

        ret["fbank"], ret["stft"], ret["fname"], ret["waveform"] = (
            fbank.unsqueeze(1),
            stft.unsqueeze(1),
            fname,
            waveform.unsqueeze(1),
        )

        return ret

    def save_wave(self, batch_wav, fname, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for wav, name in zip(batch_wav, fname):
            name = os.path.basename(name)

            sf.write(os.path.join(save_dir, name), wav, samplerate=self.sampling_rate)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # def get_log_dir(self):
    #     if self.logger_save_dir is None and self.logger_exp_name is None:
    #         return os.path.join(self.logger.save_dir, self.logger._project)
    #     else:
    #         return os.path.join(self.logger_save_dir, self.logger_exp_name)

    # def set_log_dir(self, save_dir, exp_name):
    #     self.logger_save_dir = save_dir
    #     self.logger_exp_name = exp_name
    # @torch.no_grad()
    # def log_images(self, batch, train=True, only_inputs=False, waveform=None, **kwargs):
    #     log = dict()
    #     x = batch.to(self.device)
    #     if not only_inputs:
    #         xrec, posterior = self(x)
    #         log["samples"] = self.decode(posterior.sample())
    #         log["reconstructions"] = xrec

    #     log["inputs"] = x
    #     wavs = self._log_img(log, train=train, index=0, waveform=waveform)
    #     return wavs

    # def _log_img(self, log, train=True, index=0, waveform=None):
    #     images_input = self.tensor2numpy(log["inputs"][index, 0]).T
    #     images_reconstruct = self.tensor2numpy(log["reconstructions"][index, 0]).T
    #     images_samples = self.tensor2numpy(log["samples"][index, 0]).T

    #     if train:
    #         name = "train"
    #     else:
    #         name = "val"

    #     if self.logger is not None:
    #         self.logger.log_image(
    #             "img_%s" % name,
    #             [images_input, images_reconstruct, images_samples],
    #             caption=["input", "reconstruct", "samples"],
    #         )

    #     inputs, reconstructions, samples = (
    #         log["inputs"],
    #         log["reconstructions"],
    #         log["samples"],
    #     )

    #     if self.image_key == "fbank":
    #         wav_original, wav_prediction = synth_one_sample(
    #             inputs[index],
    #             reconstructions[index],
    #             labels="validation",
    #             vocoder=self.vocoder,
    #         )
    #         wav_original, wav_samples = synth_one_sample(
    #             inputs[index], samples[index], labels="validation", vocoder=self.vocoder
    #         )
    #         wav_original, wav_samples, wav_prediction = (
    #             wav_original[0],
    #             wav_samples[0],
    #             wav_prediction[0],
    #         )
    #     elif self.image_key == "stft":
    #         wav_prediction = (
    #             self.decode_to_waveform(reconstructions)[index, 0]
    #             .cpu()
    #             .detach()
    #             .numpy()
    #         )
    #         wav_samples = (
    #             self.decode_to_waveform(samples)[index, 0].cpu().detach().numpy()
    #         )
    #         wav_original = waveform[index, 0].cpu().detach().numpy()

    #     if self.logger is not None:
    #         self.logger.experiment.log(
    #             {
    #                 "original_%s"
    #                 % name: wandb.Audio(
    #                     wav_original, caption="original", sample_rate=self.sampling_rate
    #                 ),
    #                 "reconstruct_%s"
    #                 % name: wandb.Audio(
    #                     wav_prediction,
    #                     caption="reconstruct",
    #                     sample_rate=self.sampling_rate,
    #                 ),
    #                 "samples_%s"
    #                 % name: wandb.Audio(
    #                     wav_samples, caption="samples", sample_rate=self.sampling_rate
    #                 ),
    #             }
    #         )

    #     return wav_original, wav_prediction, wav_samples

    # def tensor2numpy(self, tensor):
    #     return tensor.cpu().detach().numpy()

    # def to_rgb(self, x):
    #     assert self.image_key == "segmentation"
    #     if not hasattr(self, "colorize"):
    #         self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    #     x = F.conv2d(x, weight=self.colorize)
    #     x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
    #     return x
