import os
import numpy as np
from torch import nn

time_len = 640

config_dict = {
    "trainer_maeeg_pretrain_config": {
        "run_name": "maeeg_512_enc768_dec256",
        "lr": 1e-4,  # as the loss is small 2.5e-4
        "weight_decay": 0.07,
        "num_epochs": 200,
        "warmup_epochs": 10,
        "batch_size": 60,
        "num_workers": 2,
        "clip_grad": 0.8,
        "time_len": time_len,
        "hop_length": 128,  # 64
        "data_path": "/mnt/nvme/node02/pranav/AE24/data/split_data",
        "output_path": "/mnt/nvme/node02/pranav/AE24/chitVachana/cape",
        "reload_path": "",  # /mnt/nvme/node02/pranav/AE24/ch itVachana/cape/checkpoint/maeeg_Pretrain_m0.7_norm_59_0.55.ckpt",
        "resume_wandb": "",  #  yi7na7lp",
        "normalize_data": 2,  # 1 all 2 chan
        "seed": 2024,
    },
    "trainer_maeeg_align_config": {
        "run_name": "maeeg_afg_6",
        "lr": 1e-6,  # as the loss is small 2.5e-4
        "weight_decay": 0.5,
        "num_epochs": 100,
        "warmup_epochs": 0,
        "freeze": False,  # freezes maeeg pretrained layers
        "batch_size": 60,  # perfect max for 32GB
        "num_workers": 10,
        "clip_grad": 0.8,
        "data_path": "/mnt/nvme/node02/pranav/AE24/data/split_data",
        "output_path": "/mnt/nvme/node02/pranav/AE24/chitVachana/cape/aligne",
        "reload_path": "/mnt/nvme/node02/pranav/AE24/chitVachana/cape/aligne/checkpoint/align/maeeg_afg_5_24_5.48.ckpt",  # "/mnt/nvme/node02/pranav/AE24/chitVachana/cape/aligne/checkpoint/align/maeeg_align_freeze_15_4.84.ckpt",
        "resume_wandb": "",  #  [yi7na7lp",
        "normalize_data": 2,  # 1 all 2 chan
        "subset": 0.8,
        "seed": 2024,
        "weight": 0.7,  # how much audiomae how much text
        "resume": False,
    },
    "trainer_ldm_config": {
        "run_name": "ldm_finetune",
        "lr": 1e-3,  # as the loss is small 2.5e-4
        "weight_decay": 0.07,
        "num_epochs": 100,
        "warmup_epochs": 10,
        "freeze": True,  # freezes maeeg pretrained layers
        "batch_size": 120,  # perfect max for 32GB
        "num_workers": 8,
        "clip_grad": 0.8,
        "data_path": "/mnt/nvme/node02/pranav/AE24/data/split_data",
        "output_path": "/mnt/nvme/node02/pranav/AE24/chitVachana/ldm_models",
        "reload_path": "",
        "resume_wandb": "",  #  yi7na7lp",
        "normalize_data": 2,  # 1 all 2 chan
        "subset": 1,
        "seed": 2024,
        "weight": 0.8,  # how much audiomae how much text
    },
    "ldm_config": {
        "params": {"in_channels": 1, "t_steps": 1, "img_depth": 1, "num_tsteps": 1}
    },
    "dataset_config": {
        "preprocessing": {
            "audio": {
                "sampling_rate": 16000,
                "max_wav_value": 32768.0,
                "duration": 10,
                "hop_length": 4,
            },
            # stft is uesd in ddpm? should the filter_lengths be changed?
            "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
            "mel": {"n_mel_channels": 64, "mel_fmin": 0, "mel_fmax": 8000},
        },
    },
    "maeeg_config": {
        "params": {
            "img_size": (64, time_len),  # eeg x time (10s)
            "in_chans": 1,
            "patch_size": (8, 10),
            "enc_embed_dim": 768,
            "enc_depth": 10,
            "enc_num_heads": 12,
            "dec_embed_dim": 256,  # try 512
            "dec_depth": 6,
            "dec_num_heads": 8,
            "mlp_ratio": 4,
            "norm_layer": nn.LayerNorm,
            "pos_trainable": False,
            "pos_dim": 1,
            "pdrop": 0.7,
            "mask_ratio": 0,  # increase compute
        },
    },
    "seq_config": {
        "params": {
            "always_output_audiomae_gt": False,
            "learnable": True,
            "use_gt_mae_output": True,
            "use_gt_mae_prob": 1,
            "base_learning_rate": 0.0002,
            "sequence_gen_length": 512,
            "use_warmup": True,
            "sequence_input_key": ["crossattn_vits_phoneme"],
            "sequence_input_embed_dim": [192],
            "batchsize": 16,
            "cond_stage_config": {
                # "film_clap_cond1": {
                #     "cond_stage_key": "text",
                #     "conditioning_key": "film",
                #     "target": "audioldm2.latent_diffusion.modules.encoders.modules.CLAPAudioEmbeddingClassifierFreev2",
                #     "params": {
                #         "sampling_rate": 48000,
                #         "embed_mode": "text",
                #         "amodel": "HTSAT-base",
                #     },
                # },
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
        }
    },
}


class Config_MBM_EEG:
    # configs for fmri_pretrain.py
    def __init__(self):
        # --------------------------------------------
        # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.0
        self.weight_decay = 0.05
        self.num_epoch = 100
        self.warmup_epochs = 40
        self.batch_size = 192  # (192,31 GB)
        self.clip_grad = 0.8

        self.time_len = 640
        self.hop_length = 64
        self.max_epochs = 50
        # Model Parameters
        # self.mask_ratio = 0.6
        # self.patch_size = 4 #  1
        # self.embed_dim = 512 #256 # has to be a multiple of num_heads
        # self.decoder_embed_dim = 512 #128
        # self.depth = 24
        # self.num_heads = 16
        # self.decoder_num_heads = 16
        # self.mlp_ratio = 1.0

        # Project setting
        self.root_path = "/mnt/nvme/node02/pranav/AE24/chitVachana/cape/exps"
        self.output_path = "/mnt/nvme/node02/pranav/AE24/chitVachana/cape/exps/output"
        self.seed = 2024
        self.roi = "VC"
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1
