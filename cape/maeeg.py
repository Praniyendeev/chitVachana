import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from timm.models.vision_transformer import Block
import numpy as np
import torch

import matplotlib.pyplot as plt


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0, f"embed_dim={embed_dim}"
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class DummyMod(nn.Module):
    """ """

    def __init__(self):
        super().__init__()

    def forward(self):
        return


class Patch2DEmbed(nn.Module):
    """Converts Patch EEGxTime to embed_dim"""

    def __init__(
        self,
        img_size=(64, 640),  # eeg x time (10s)
        in_chans=1,
        patch_size=(8, 16),
        embed_dim=512,
    ):

        super().__init__()

        self.num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0]
        )
        self.patch_hw = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):

        # B, C, H, W  = x.shape
        # B, C, H, W  => B, E, pH, pW => B, E, pH*pW => B, pH*pW, E (pH*pW==num_patches)

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# 512
class Encoder(nn.Module):
    def __init__(
        self,
        img_size=(64, 640),  # eeg x time (10s)
        in_chans=1,
        patch_size=(8, 16),
        enc_embed_dim=512,
        enc_depth=20,
        enc_num_heads=16,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        pos_trainable=False,
        pos_dim=1,
        mask_ratio=0.8,
        **kwargs,
    ):

        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = enc_embed_dim
        self.norm = norm_layer
        self.mask_ratio = mask_ratio

        self.patch_embed = Patch2DEmbed(img_size, in_chans, patch_size, enc_embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_embed_dim))
        self.enc_pos_embed = nn.Parameter(
            torch.zeros(pos_dim, self.num_patches + 1, enc_embed_dim),
            requires_grad=pos_trainable,
        )

        self.enc_blocks = nn.ModuleList(
            [
                Block(
                    enc_embed_dim,
                    enc_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(enc_depth)
            ]
        )
        self.enc_norm = norm_layer(enc_embed_dim)

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed_flexible(
            self.enc_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True
        )
        self.enc_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data  # CONV2D
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):  # patch

        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # sort noise for each sample
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio):
        if not mask_ratio:
            mask_ratio = self.mask_ratio
        x = self.patch_embed(x)
        x = x + self.enc_pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.enc_pos_embed[:, 0, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        if x.shape[1] > self.num_patches:
            x = x[:, : self.num_patches]
        return x, mask, ids_restore


class MAEEG(nn.Module):

    def __init__(
        self,
        img_size=(64, 640),  # eeg x time (10s)
        in_chans=1,
        patch_size=(8, 16),
        enc_embed_dim=512,
        enc_depth=20,
        enc_num_heads=16,
        dec_embed_dim=256,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        pos_trainable=False,
        pos_dim=1,
        **kwargs,
    ):

        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = enc_embed_dim
        self.norm = norm_layer

        self.encoder = Encoder(
            img_size=img_size,
            in_chans=in_chans,
            patch_size=patch_size,
            enc_embed_dim=enc_embed_dim,
            enc_depth=enc_depth,
            enc_num_heads=enc_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            pos_trainable=pos_trainable,
            pos_dim=pos_dim,
            **kwargs,
        )
        num_patches = self.encoder.num_patches

        self.num_patches = self.encoder.patch_embed.num_patches
        self.patch_size = self.encoder.patch_embed.patch_size
        self.patch_hw = self.encoder.patch_embed.patch_hw

        self.dec_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.msk_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(pos_dim, num_patches + 1, dec_embed_dim),
            requires_grad=pos_trainable,
        )
        self.dec_blocks = nn.ModuleList(
            [
                Block(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(dec_depth)
            ]
        )

        self.dec_norm = norm_layer(dec_embed_dim)
        self.dec_pred = nn.Linear(
            dec_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data  # CONV2D
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.encoder.initialize_weights()

        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(
            self.dec_pos_embed.shape[-1], self.patch_hw, cls_token=True
        )
        self.dec_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.msk_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        in_chans = self.in_chans
        p, q = self.patch_size
        h, w = self.patch_hw

        x = x.reshape(shape=(x.shape[0], in_chans, h, p, w, q))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(x.shape[0], h * w, p * q * in_chans))

        return x

    def unpatchify(self, x):
        in_chans = self.in_chans
        p, q = self.patch_size
        h, w = self.patch_hw

        assert (
            h * p == self.img_size[0] and w * q == self.img_size[1]
        ), "Image_size mismatch with patch_size"

        x = x.reshape(shape=(x.shape[0], h, w, p, q, in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(x.shape[0], in_chans, h * p, w * q))

        return x

    def random_masking(self, x, mask_ratio):  # patch

        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # sort noise for each sample
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_enc(self, x, mask_ratio):
        return self.encoder(x, mask_ratio)

    def forward_dec(self, x, ids_restore):

        x = self.dec_embed(x)

        mask_tokens = self.msk_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.dec_pos_embed

        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        x = self.dec_pred(x)

        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, preds, mask):

        target = self.patchify(imgs)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.8):
        imgs = imgs["eeg"][:, None, ...]
        emb_enc, mask, ids_restore = self.forward_enc(imgs, mask_ratio)
        preds = self.forward_dec(emb_enc, ids_restore)
        loss_reconstruction = self.forward_loss(imgs, preds, mask)

        return loss_reconstruction, preds, mask, loss_contrastive


import lightning as L


class MAEEG_pl(L.LightningModule):

    def __init__(
        self,
        img_size=(64, 640),  # eeg x time (10s)
        in_chans=1,
        patch_size=(8, 16),
        enc_embed_dim=512,
        enc_depth=20,
        enc_num_heads=16,
        dec_embed_dim=256,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        pos_trainable=False,
        pos_dim=1,
        mask_ratio=0.8,
        lr=2e-4,
        weight_decay=0.0,
        warmup_epochs=0,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = enc_embed_dim
        self.norm = norm_layer
        self.mask_ratio = mask_ratio
        self.encoder = Encoder(
            img_size=img_size,
            in_chans=in_chans,
            patch_size=patch_size,
            enc_embed_dim=enc_embed_dim,
            enc_depth=enc_depth,
            enc_num_heads=enc_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            pos_trainable=pos_trainable,
            pos_dim=pos_dim,
            mask_ratio=mask_ratio,
            **kwargs,
        )
        num_patches = self.encoder.num_patches

        self.num_patches = self.encoder.patch_embed.num_patches
        self.patch_size = self.encoder.patch_embed.patch_size
        self.patch_hw = self.encoder.patch_embed.patch_hw

        self.dec_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.msk_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(pos_dim, num_patches + 1, dec_embed_dim),
            requires_grad=pos_trainable,
        )
        self.dec_blocks = nn.ModuleList(
            [
                Block(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(dec_depth)
            ]
        )

        self.dec_norm = norm_layer(dec_embed_dim)
        self.dec_pred = nn.Linear(
            dec_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True
        )

        self.initialize_weights()

    def set_config(self, config):
        self.config = config

    def initialize_weights(self):

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data  # CONV2D
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.encoder.initialize_weights()

        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(
            self.dec_pos_embed.shape[-1], self.patch_hw, cls_token=True
        )
        self.dec_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.msk_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        in_chans = self.in_chans
        p, q = self.patch_size
        h, w = self.patch_hw

        x = x.reshape(shape=(x.shape[0], in_chans, h, p, w, q))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(x.shape[0], h * w, p * q * in_chans))

        return x

    def unpatchify(self, x):
        in_chans = self.in_chans
        p, q = self.patch_size
        h, w = self.patch_hw

        # assert (
        #     h * p == self.img_size[0] and w * q == self.img_size[1]
        # ), f"Image_size mismatch with patch_size {h=},{p=},{w=},{q=},{self.img_size=},{x.shape=}"

        x = x.reshape(shape=(x.shape[0], h, w, p, q, in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(x.shape[0], in_chans, h * p, w * q))

        return x

    def random_masking(self, x, mask_ratio):  # patch

        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # sort noise for each sample
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_enc(self, x, mask_ratio):
        return self.encoder(x, mask_ratio)

    def forward_dec(self, x, ids_restore):

        x = self.dec_embed(x)

        mask_tokens = self.msk_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.dec_pos_embed

        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        x = self.dec_pred(x)

        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, preds, mask):

        target = self.patchify(imgs)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.8):

        imgs = imgs["eeg"][:, None, ...]
        emb_enc, mask, ids_restore = self.forward_enc(imgs, mask_ratio)
        preds = self.forward_dec(emb_enc, ids_restore)
        loss_reconstruction = self.forward_loss(imgs, preds, mask)

        return loss_reconstruction, preds, mask

    def training_step(self, batch, batch_idx):

        loss, preds, masks = self(batch, self.mask_ratio)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        samples = batch["eeg"]

        b, c, l = samples.shape
        samples = samples.reshape((b * c, l)).to("cpu").numpy()
        preds = preds.reshape((b * c, l)).to("cpu").numpy()
        corrs = np.zeros((b * c))
        for i in range(b * c):
            corrs[i] = np.corrcoef(samples[i, :], preds[i, :])[0, 1]

        self.log(
            "train_corr",
            corrs.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # if self.current_epoch % 5 == 0 and batch_idx == 0:
        #     batch = batch['eeg'][:,None,...]
        #     plot_recon_figures(
        #         batch,
        #         self.unpatchify(preds),
        #         masks,
        #         self.patch_size,
        #         logger=self.logger,
        #         epoch= self.current_epoch,
        #     )

        return loss

    def validation_step(self, batch, batch_idx):

        loss, preds, masks = self(batch, self.mask_ratio)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        b, c, l = samples.shape
        samples = samples.reshape((b * c, l)).to("cpu").numpy()
        preds = preds.reshape((b * c, l)).to("cpu").numpy()
        corrs = np.zeros((b * c))
        for i in range(b * c):
            corrs[i] = np.corrcoef(samples[i, :], preds[i, :])[0, 1]

        self.log(
            "val_corr",
            corrs.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.current_epoch % 5 == 0 and batch_idx == 0:
            batch = batch["eeg"][:, None, ...]
            plot_recon_figures(
                batch,
                self.unpatchify(preds),
                masks,
                self.patch_size,
                logger=self.logger,
                epoch=self.current_epoch,
            )
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, masks = self(batch)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        b, c, l = samples.shape
        samples = samples.reshape((b * c, l)).to("cpu").numpy()
        preds = preds.reshape((b * c, l)).to("cpu").numpy()
        corrs = np.zeros((b * c))
        for i in range(b * c):
            corrs[i] = np.corrcoef(samples[i, :], preds[i, :])[0, 1]

        self.log(
            "test_corr",
            corrs.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.current_epoch % 5 == 0 and batch_idx == 0:
            plot_recon_figures(
                batch,
                self.unpatchify(preds),
                masks,
                self.patch_size,
                logger=self.logger,
            )
        return loss

    def configure_optimizers(self):  # New!
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        sch = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=0.0,
        )

        lr_scheduler = {
            "scheduler": sch,
            "name": "lr_cosine",
            "interval": "epoch",
            "frequency": 1,
        }

        return [opt], [lr_scheduler]


import io


def fig2numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return [im]


@torch.no_grad()
def plot_recon_figures(
    batch, preds, masks, patch_size, num_figures=5, logger=None, eeg_idx=0, epoch=0
):

    fig, axs = plt.subplots(num_figures, 3, figsize=(30, 15))
    fig.tight_layout()
    axs[0, 0].set_title("Ground-truth")
    axs[0, 1].set_title("Masked Ground-truth")
    axs[0, 2].set_title("Reconstruction")

    for i, ax in enumerate(axs):

        sample, pred, mask = batch[i], preds[i], masks[i]

        pred = pred.to("cpu").squeeze(0)[eeg_idx].numpy()
        sample = sample.to("cpu").squeeze(0)[eeg_idx].numpy()
        sample_with_mask = sample.reshape(-1, 40)

        mask = mask.to("cpu").numpy()

        cor = np.corrcoef([pred, sample])[0, 1]

        x_axis = np.arange(0, sample.shape[-1])

        # groundtruth
        ax[0].plot(x_axis, sample)

        # groundtruth with mask
        st = (eeg_idx // 8) * 40
        s = 0
        for x, m in zip(sample_with_mask, mask[st : st + 40]):
            if m == 0:
                ax[1].plot(x_axis[s : s + len(x)], x, color="#1f77b4")
            s += len(x)
        # pred

        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel("cor: %.4f" % cor, weight="bold")
        ax[2].yaxis.set_label_position("right")

        ax[0].set_xlim((-40, 680))
        ax[1].set_xlim((-40, 680))
        ax[2].set_xlim((-40, 680))

        ylim = (-5, 5)
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)
        ax[2].set_ylim(ylim)

    fig_name = "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M"))
    # if logger.save_dir:
    #     save_dir=logger.save_dir
    # else:
    save_dir = "/mnt/nvme/node02/pranav/AE24/chitVachana/cape/exps/"
    fig.savefig(os.path.join(save_dir, f"epoch_{epoch}_{fig_name}.png"))
    if logger:
        logger.log_image("reconst", fig2numpy(fig), step=epoch)

    plt.close(fig)


if __name__ == "__main__":

    model = MAEEG()
    print("awe")
