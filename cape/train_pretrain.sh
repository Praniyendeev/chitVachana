#!/bin/bash

#because of lightning
srun --nodes 1 --ntasks-per-node=2 -p node03 --gres=gpu:2 --pty python /mnt/nvme/node02/pranav/AE24/chitVachana/cape/train.py
#sbatch --begin=now+180minutes train_pretrain.sh
