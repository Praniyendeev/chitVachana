import lightning as pl
from torch import nn
import torch
import numpy as np


class temp(pl.LightningModule):

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.initialize_weights()

    def set_config(self, config):
        self.config = config

    def initialize_weights(self):

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

    def forward(self, imgs, mask_ratio=0.8):
        loss_reconstruction, preds, mask = None, None, None

        return loss_reconstruction, preds, mask

    def training_step(self, batch, batch_idx):

        loss, preds, masks = self(batch)
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

        corrs = np.array([0])
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

        self.log(
            "val_corr",
            corrs.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
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
