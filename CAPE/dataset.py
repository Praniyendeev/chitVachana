import numpy as np
import os
from torch.utils.data import Dataset
import torch

ch_mean,ch_std=(np.array([-5.8170e-03, -4.1821e-03,  7.7076e-04,  1.4204e-03,  6.0824e-04,
         -1.0782e-02, -2.0063e-03,  2.1018e-03,  1.5457e-03,  7.2137e-04,
          2.1337e-03,  1.9735e-03,  3.8127e-03,  2.2840e-03,  3.2022e-03,
          1.8450e-02,  1.4663e-03,  2.1602e-03,  1.9835e-03,  2.1182e-03,
         -1.2986e-02,  1.8628e-03,  1.6067e-03,  2.2432e-03, -4.8271e-02,
          2.0951e-03, -3.2702e-03, -2.8628e-03,  1.7240e-03,  5.8374e-03,
          2.0267e-03,  1.8051e-03, -1.1932e-02, -4.6033e-03, -3.4172e-03,
          6.8120e-05,  1.4299e-03,  1.8920e-03,  1.6683e-03,  1.3988e-03,
          1.1671e-03, -3.1471e-04,  2.4534e-03,  9.6888e-04,  1.6220e-03,
          2.0032e-03,  2.1148e-03,  2.0373e-03,  2.1102e-03,  1.3252e-03,
          1.2865e-03,  1.8334e-03,  2.6089e-03,  9.0179e-04,  1.8382e-03,
          1.8436e-03,  2.2808e-03,  1.9983e-03,  1.6189e-03, -1.8476e-03,
          2.1937e-03,  1.8645e-03,  1.7349e-03,  1.8910e-03]),
 np.array([ 5.6584,  6.8110,  4.8103,  4.7095,  5.1946,  8.9362,  6.8557,  5.9930,
          4.6838,  4.3577,  4.3504,  3.8786,  4.6396,  4.4478,  6.2938, 22.3406,
          4.6244,  4.7679,  4.1106,  4.8763,  8.2437,  5.0495,  5.5928,  6.6747,
         15.9443,  5.2880,  6.5989,  7.8452,  5.8939,  6.5213,  4.6768,  4.4128,
          4.8336,  5.6158,  6.6450,  4.8020,  4.5615,  4.8484,  4.7232,  4.9188,
          5.4910,  6.7138,  6.0080,  4.9908,  4.9232,  4.5763,  4.7114,  4.2873,
          4.4714,  4.2143,  4.7894,  5.7851, 11.7666,  4.6169,  4.3428,  4.1623,
          4.6776,  4.6606,  5.0907,  6.6647,  6.3601,  6.3310,  6.1148,  6.4376]))

mean,std=0,6.5272

class eeg_pretrain_dataset(Dataset):
    def __init__(
        self,
        path="/mnt/nvme/node02/pranav/AE24/data/split_data",
        file_types=["train"],
        frame_length=64,
        hop_length=30,
        normalize=None,

    ):
        
        super(eeg_pretrain_dataset, self).__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.normalize = normalize
        if normalize == 1:
            self.mean = mean 
            self.std = std 
        elif normalize == 2:
            self.mean = ch_mean[None,:] 
            self.std = ch_std[None,:] 


        file_types.append("eeg")
        self.input_paths = [
            os.path.join(path, file)
            for file in os.listdir(path)
            if all(typ in file for typ in file_types)
        ]
        assert len(self.input_paths) != 0, "No data found"
        # print(f"loaded {len(self.input_paths)} files with file types {file_types}")

        self.data_len = frame_length
        self.data_chan = 64

        self.index_map = {}
        global_index = 0
        for file_name in self.input_paths:
            data_shape = np.load(file_name, mmap_mode="r").shape
            num_samples = data_shape[0]

            num_windows = 1 + (num_samples - self.frame_length) // self.hop_length

            for window_offset in range(num_windows):
                self.index_map[global_index] = (file_name, window_offset)
                global_index += 1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_name, window_offset = self.index_map[idx]
        start = window_offset * self.hop_length
        end = start + self.frame_length
        sample = np.load(file_name, mmap_mode="r")[start:end]
        if self.normalize:
            try:
                sample=(sample-self.mean)/self.std
            except Exception as e:
                print(sample.shape,self.mean.shape,self.std.shape)
                raise e


        return {"eeg": torch.from_numpy(sample.T.copy()).float()}
