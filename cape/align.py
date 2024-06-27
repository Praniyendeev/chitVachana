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

from maeeg import Encoder
from audioldm2.latent_diffusion.modules.encoders.modules import SequenceGenAudioMAECond
from audioldm_train.utilities.data.eeg_dataset import EEGDataset

from config import config_dict

from lightning import Trainer
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from info_nce import InfoNCE, info_nce

import warnings

warnings.filterwarnings("ignore")

# dataset_config = {
#     "preprocessing": {
#         "audio": {
#             "sampling_rate": 16000,
#             "max_wav_value": 32768.0,
#             "duration": 8,
#             "hop_length": 4,
#         },
#         "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
#         "mel": {"n_mel_channels": 64, "mel_fmin": 0, "mel_fmax": 8000},
#     }
# }

# dataset = EEGDataset(dataset_config, split="train")
# loader = DataLoader(
#     dataset,
#     batch_size=3,
#     num_workers=16,
#     pin_memory=True,
#     shuffle=True,
# )


default = {
    "img_size": (64, 640),  # eeg x time (10s)
    "in_chans": 1,
    "patch_size": (8, 10),
    "enc_embed_dim": 768,
    "enc_depth": 10,
    "enc_num_heads": 12,
    "mlp_ratio": 4,
    "norm_layer": nn.LayerNorm,
    "pos_trainable": False,
    "pos_dim": 1,
    "mask_ratio": 0.8,
}


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class eeg2latent(nn.Module):
    def __init__(self, device="cuda", freeze=True, pdrop=0.2, **kwargs):
        super().__init__()
        ckpt = "/mnt/nvme/node02/pranav/AE24/chitVachana/cape/checkpoint/maeeg_512_enc768_dec256_29-03-2024-02-17.ckpt"
        state_dict = torch.load(ckpt, map_location=device)

        self.encoder = Encoder(**kwargs)
        self.encoder.load_state_dict(state_dict, strict=False)
        self.encoder.requires_grad_(not freeze)

        # for param in self.encoder.parameters():
        #     param.requires_grad = not freeze

        self.freeze = freeze
        dim = kwargs["enc_embed_dim"]
        # self.projection = ProjectionHead(dim, dim, 0.5)
        print("pdrop=", pdrop)
        self.projection = GPT2Block(
            GPT2Config(
                n_embd=dim, resid_pdrop=pdrop, embd_pdrop=pdrop, attn_pdrop=pdrop
            ),
            layer_idx=0,
        )

    def forward(self, x):

        assert x.shape[1:] == torch.Size([1, 64, 640])
        # B, 1, 64, 640 -> B, 512, 786
        if self.freeze:
            with torch.no_grad():
                x, mask, ids_restore = self.encoder(x, 0)  # mask_ratio=0
        else:
            x, mask, ids_restore = self.encoder(x, 0)  # mask_ratio=0
        # B, 512, 786 -> B, 512, 786
        x = self.projection(x)
        return x


class alignEEG(pl.LightningModule):

    def __init__(
        self,
        maeeg_config=config_dict["maeeg_config"]["params"],
        seq_config=config_dict["seq_config"]["params"],
        lr=2e-4,
        weight_decay=0.0,
        warmup_epochs=0,
        freeze=True,
        weight=0.8,
        device="cuda",
        reload_path="",
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dev = device
        self.weight = weight
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.maeeg_config = maeeg_config
        self.seq_config = seq_config
        # self.init_seq_model(freeze)
        self.init_encoder(freeze)
        self.lossfn = InfoNCE(temperature=1).to(self.device)  # self.custom_lossfn
        if reload_path:
            state_dict = torch.load(reload_path, map_location=self.device)
            new_state_dict = {}
            if "state_dict" in state_dict.keys():
                for key, value in state_dict["state_dict"].items():
                    new_state_dict[key[8:]] = value
            self.eeg_enc.load_state_dict(new_state_dict)

        # self.audiolatents = np.memmap(
        #     "/mnt/nvme/node02/pranav/AE24/data/pytorch_save/test.npz",
        #     dtype="float32",
        #     mode="r",
        #     shape=(112800, 512, 768),
        # )

    def init_encoder(self, freeze=True):
        self.eeg_enc = eeg2latent(freeze=freeze, **self.maeeg_config)
        self.eeg_enc.to(self.device)

    def init_seq_model(self, freeze):
        checkpoint_path = hf_hub_download(
            repo_id="haoheliu/audioldm2-speech",
            filename="audioldm2-speech-gigaspeech.pth",
        )

        audioldm2_ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        seq_model = SequenceGenAudioMAECond(**self.seq_config)

        model_dict = seq_model.state_dict()
        for key in model_dict.keys():

            okey = key
            if "cond_stage_models.0." in key:
                okey = key.replace("cond_stage_models.0.", "cond_stage_models.1.")
            else:
                okey = key.replace("cond_stage_models.1.", "cond_stage_models.2.")

            if "input_sequence_embed_linear.0." in key:
                okey = okey.replace(
                    "input_sequence_embed_linear.0.", "input_sequence_embed_linear.1."
                )
            else:
                okey = okey.replace(
                    "input_sequence_embed_linear.1.", "input_sequence_embed_linear.2."
                )

            model_dict[key] = audioldm2_ckpt["state_dict"][
                "cond_stage_models.0." + okey
            ]

        seq_model.load_state_dict(model_dict)
        # if freeze:
        for param in seq_model.parameters():
            param.requires_grad = False

        del audioldm2_ckpt
        # self.seq_model.to(self.dev)
        self.audiomae = seq_model.cond_stage_models[1]
        # self.audiomae.to(self.dev)

    def cross_entropy(self, preds, targets, reduction="none"):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def custom_lossfn(self, Aemb, Bemb, temperature=1):
        # b, n, l = Aemb.shape
        # assert Aemb.shape == Bemb.shape, f"{Aemb.shape}  {Bemb.shape}"
        # Aemb = Aemb.reshape((b, n * l))  # B,512*768
        # Bemb = Bemb.reshape((b, n * l))
        # Calculating the Loss
        Aemb = Aemb / torch.norm(Aemb, dim=1)[:, None]
        Bemb = Bemb / torch.norm(Bemb, dim=1)[:, None]
        logits = (Aemb @ Bemb.T) / temperature
        Asimilarity = Aemb @ Aemb.T
        Bsimilarity = Bemb @ Bemb.T
        targets = F.softmax(
            (Asimilarity + Bsimilarity) / (2 * temperature), dim=-1
        )  # avoids arange(N) loss stuck at 4.09

        Aloss = self.cross_entropy(logits, targets, reduction="none")
        Bloss = self.cross_entropy(logits.T, targets, reduction="none")
        loss = (Aloss + Bloss) / 2.0  # shape: (batch_size)
        return loss.mean()

    def forward(self, batch, split="train"):

        # langLatents = {}
        # with torch.no_grad():
        #     langLatents["crossattn_audiomae_pooled"] = self.audiomae(
        #         batch["ta_kaldi_fbank"].contiguous()
        #     )

        audioLatent = batch["audio_latent"]
        # torch.from_numpy(self.audiolatents[batch["index"].cpu()].copy()).to(self.device)
        # textLatent = langLatents["crossattn_audiomae_generated"][0]
        # aud_path = (
        #     f"/mnt/nvme/node02/pranav/AE24/data/pytorch_save/{split}_audiomae.npy"
        # )
        # audioLatent = (
        #     torch.from_numpy(
        #         np.load(aud_path, mmap_mode="r")[batch["index"].cpu()].copy()
        #     )
        #     .type(torch.float32)
        #     .to(self.device)
        # )

        # fmt:off
        eegLatent = self.eeg_enc(batch["eeg"][:, None, :, :])[0]  # .contiguous())
        # fmt:on
        # batch["eeg_latent"] = eegLatent.to(self.device)
        b, n, d = eegLatent.shape
        eegLatent = eegLatent.reshape((b, n * d))
        audioLatent = audioLatent.reshape((b, n * d))

        gndLoss = self.lossfn(eegLatent, audioLatent)
        # expLoss = self.lossfn(eegLatent, textLatent)
        # loss = weight * gndLoss + (1 - weight) * expLoss

        return gndLoss

    def training_step(self, batch, batch_idx):

        loss = self(batch, "train")
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # if self.current_epoch % 5 == 0:
        #     samples = latents["eeg_latent"]
        #     b, c, l = samples.shape
        #     samples = samples.reshape((b * c, l)).to("cpu").detach().numpy()
        #     preds = latents["audiolatents"].reshape((b * c, l)).to("cpu").numpy()
        #     corrs = np.zeros((b * c))
        #     for i in range(b * c):
        #         corrs[i] = np.corrcoef(samples[i, :], preds[i, :])[0, 1]

        #     self.log(
        #         "train_corr",
        #         corrs.mean(),
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True,
        #         sync_dist=True,
        #     )

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self(batch, "val")
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # if self.current_epoch % 5 == 0:
        #     samples = latents["eeg_latent"]

        #     b, c, l = samples.shape
        #     samples = samples.reshape((b * c, l)).to("cpu").detach().numpy()
        #     preds = latents["audiolatents"].reshape((b * c, l)).to("cpu").numpy()
        #     corrs = np.zeros((b * c))
        #     for i in range(b * c):
        #         corrs[i] = np.corrcoef(samples[i, :], preds[i, :])[0, 1]

        #     self.log(
        #         "val_corr",
        #         corrs.mean(),
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True,
        #         sync_dist=True,
        #     )

        return loss

    def test_step(self, batch, batch_idx):
        loss = self(batch, "test")
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        samples = latents["eeg_latent"]

        b, c, l = samples.shape
        samples = samples.reshape((b * c, l)).to("cpu").detach().numpy()
        preds = latents["audiolatents"].reshape((b * c, l)).to("cpu").numpy()
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

        if self.warmup_epochs != 0:
            print("warmup::::::::::::::", self.warmup_epochs)
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
        else:
            return [opt]

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint["state_dict"] = self.eeg_enc.state_dict()


from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    # di.shape,waveform.shape,ids["phoneme_idx"].shape,len(raw_phones))
    try:
        batched = default_collate(batch)
    except Exception as e:
        print(
            "**************************************************************************"
        )
        # print(eeg.shape,mel.shape,ta_kal
        for k, v in batch.items():
            if isinstance(v, list):
                print(f"{k}=", len(v))
            elif isinstance(v, torch.Tensor):
                print(f"{k}=", v.shape)
        raise e


def main(
    maeeg_config,
    seq_config,
    dataset_config,
    run_name,
    # time_len,
    # hop_length,
    batch_size,
    num_workers,
    num_epochs,
    data_path,
    output_path,
    reload_path,
    normalize_data=False,
    resume=False,
    resume_wandb=None,
    subset="",
    **kwargs,
):
    if resume_wandb:
        wandb_logger = WandbLogger(
            project="maeeg_align", name=run_name, id=resume_wandb, resume="must"
        )
    else:
        wandb_logger = WandbLogger(project="maeeg_align", name=run_name)

    dataset_train_align = EEGDataset(
        dataset_config, split="train", normalize=normalize_data
    )

    dataset_val_align = EEGDataset(
        dataset_config, split="val", normalize=normalize_data
    )
    if subset:
        subset_idx = np.random.choice(
            len(dataset_train_align),
            int(len(dataset_train_align) * subset),  # subset should range from 0 to 1
            replace=False,
        )
        dataset_train_align = torch.utils.data.Subset(dataset_train_align, subset_idx)

        # dataset_val_align = torch.utils.data.Subset(
        #     dataset_val_align, np.arange(0, len(dataset_val_align), subset)
        # )

    print(f"Dataset size: {len(dataset_train_align)}")

    train_dataloader = DataLoader(
        dataset_train_align,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
    )
    val_dataloader = DataLoader(
        dataset_val_align,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path + "/checkpoint/align",
        monitor="val_loss",
        mode="min",
        filename=f"{run_name}"
        + "_{epoch}_{val_loss:.2f}",  # checkpointMM-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_epochs=5,
        save_top_k=2,
        auto_insert_metric_name=False,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    if resume:

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Reload:", reload_path)
        # reload_path="/mnt/nvme/node02/pranav/AE24/chitVachana/cape/checkpoint/maeeg_pretrain_99_18.95.ckpt"
        model = alignEEG.load_from_checkpoint(reload_path, reload_path="")
        model.lr = kwargs["lr"]
        model.weight_decay = kwargs["weight_decay"]
        model.warmup_epochs = kwargs["warmup_epochs"]
        model.freeze = kwargs["freeze"]
        model.init_encoder(kwargs["freeze"])

        model.save_hyperparameters()
        # loads hyperparameters as well

    else:
        model = alignEEG(
            maeeg_config["params"],
            seq_config["params"],
            reload_path=reload_path,
            **kwargs,
        )

    print("number of devices=", torch.cuda.device_count())
    scaler = MixedPrecision("16-mixed", "cuda", torch.cuda.amp.GradScaler())
    # precision =FSDPPrecision('16-mixed')#,torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler)

    now = "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M"))
    ckpt_name = f"{run_name}_{now}"
    ckpt_path = output_path + f"/checkpoint/{ckpt_name}.ckpt"

    # from lightning.pytorch.profilers import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    trainer = Trainer(
        plugins=scaler,
        num_nodes=1,
        logger=wandb_logger,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        num_sanity_val_steps=0,
        # profiler="simple",
        # resume_from_checkpoint=ckpt_paßß∂th,
    )

    # tuner = Tuner(trainer)
    # # Auto-scale batch size with binary search
    # tuner.scale_batch_size(model, mode="binsearch")

    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":
    config = config_dict
    for k, v in config_dict["trainer_maeeg_align_config"].items():
        config[k] = v
    # print(config)
    main(**config)
