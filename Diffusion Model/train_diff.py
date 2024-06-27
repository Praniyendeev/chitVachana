import torch
import numpy as np
import os
import lightning as pl
from config import config_dict
from lightning import Trainer
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from audioldm_train.utilities.data.eeg_dataset import EEGDataset

from ddpm import LDM


def sample_speech(model: torch.nn.Module, train_dataset, output_dir) -> None:

    gen_samples = []
    sampled_steps = []

    x = torch.randn((expected_shape))

    sample_steps = torch.arange(model.t_range - 1, 0, -1)


def train_model(
    LDM_config,
    dataset_config,
    run_name,
    batch_size,
    num_workers,
    num_epochs,
    data_path,
    output_path,
    reload_path,
    normalize_data=False,
    resume_wandb=None,
    subset="",
    **kwargs,
) -> None:

    if resume_wandb:
        wandb_logger = WandbLogger(
            project="mindSpeak", name=run_name, id=resume_wandb, resume="must"
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

        dataset_train_align = torch.utils.data.Subset(
            dataset_train_align, np.arange(0, len(dataset_train_align), subset)
        )

        dataset_val_align = torch.utils.data.Subset(
            dataset_val_align, np.arange(0, len(dataset_val_align), subset)
        )

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
        dirpath=output_path + "/checkpoint",
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

    # create model

    if reload_path:
        model = LDM.load_from_checkpoint(ckpt, in_size, t_range, img_depth)
    else:
        model = LDM(in_size, t_range, img_depth)

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

    return


if __name__ == "__main__":
    config = get_config()
    model, train_ds, output_dir = train_model(config)
    sample_gif(model, train_ds, output_dir)
