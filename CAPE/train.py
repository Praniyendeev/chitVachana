import os, sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse
import time

import datetime

import wandb
import copy


from dataset import eeg_pretrain_dataset
from maeeg import MAEEG_pl

from config import Config_MBM_EEG, config_dict

from lightning import Trainer
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies import FSDPStrategy
from lightning.fabric.plugins.precision import FSDPPrecision
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor


def main(
    maeeg_config,
    run_name,
    time_len,
    hop_length,
    batch_size,
    num_workers,
    num_epochs,
    data_path,
    output_path,
    reload_path,
    lr=2.5e-4,
    weight_decay=0.05,
    warmup_epochs=10,
    normalize_data=False,
    resume_wandb=None,
    **kwargs,
):
    if resume_wandb:
        wandb_logger = WandbLogger(
            project="maeeg_pretrain", name=run_name, id=resume_wandb, resume="must"
        )
    else:
        wandb_logger = WandbLogger(project="maeeg_pretrain", name=run_name)

    dataset_pretrain = eeg_pretrain_dataset(
        path=data_path,
        frame_length=time_len,
        hop_length=hop_length,
        normalize=normalize_data,
    )

    dataset_val = eeg_pretrain_dataset(
        path=data_path,
        file_types=["val"],
        frame_length=time_len,
        hop_length=hop_length,
        normalize=normalize_data,
    )

    print(
        f"Dataset size: {len(dataset_pretrain)}\n Time len: {dataset_pretrain.data_len}"
    )

    train_dataloader = DataLoader(
        dataset_pretrain,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path + "/checkpoint/",
        monitor="val_loss",
        mode="min",
        filename=f"maeeg_{run_name}"
        + "_{epoch}_{val_loss:.2f}",  # checkpointMM-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_epochs=4,
        save_top_k=2,
        auto_insert_metric_name=False,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # create model

    model = MAEEG_pl(
        **maeeg_config["params"],
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        **kwargs,
    )

    if reload_path:
        # reload_path="/mnt/nvme/node02/pranav/AE24/chitVachana/cape/checkpoint/maeeg_pretrain_99_18.95.ckpt"
        model = MAEEG_pl.load_from_checkpoint(reload_path)

    print("number of devices=", torch.cuda.device_count())
    scaler = MixedPrecision("16-mixed", "cuda", torch.cuda.amp.GradScaler())
    # precision =FSDPPrecision('16-mixed')#,torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler)

    now = "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M"))
    ckpt_name = f"{run_name}_{now}"
    ckpt_path = output_path + f"/checkpoint/{ckpt_name}.ckpt"

    trainer = Trainer(
        plugins=scaler,
        num_nodes=1,
        logger=wandb_logger,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(),
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        # resume_from_checkpoint=ckpt_paßß∂th,
    )

    # tuner = Tuner(trainer)
    # # Auto-scale batch size with binary search
    # tuner.scale_batch_size(model, mode="binsearch")

    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":

    config = config_dict
    main(config_dict["maeeg_config"], **config_dict["trainer-config"])
