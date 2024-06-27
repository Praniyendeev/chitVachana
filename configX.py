from torch import nn


class Config_MAEEG:
    def __init__(self):

        self.img_size = ((64, 640),)  # eeg x time (10s)
        self.in_chans = (1,)
        self.patch_size = ((8, 16),)

        self.enc_embed_dim = (512,)
        self.enc_depth = (20,)
        self.enc_num_heads = (16,)

        self.mlp_ratio = (4,)
        self.norm_layer = (nn.LayerNorm,)

        self.pos_trainable = (False,)
        self.pos_dim = (1,)
