# chitVachana EEG2Speech

## DIAGRAM

## Methodology
1. Train MAEEG
2. Contrastive Alignment of EEG-Audio Representations (testing - TBD)
3. Fine-tuning AudioLDM2 - TBD

## 1. Train MAEEG 

Inspired by [AudioMAE](https://github.com/facebookresearch/AudioMAE)

EEG is re-imagined as an image of (num_channels, num_time_samples). Then follows the AudioMAE methodology to learn representations.

### cue 2nd diagram RECOVERY?

##  2. Contrastive Alignment of EEG-Audio Representations

Ideally alignment would have taken place as encoders for both modalities are jointly trained.
However, due to restriction in terms of well-labelled data and computational resources. We use a different procedure:

1. Use pretrained encoders MAEEG and AudioMAE.
2. fine-tune the MAEEG with the contrastive objective using InfoNCE (similar to CLIP/CLAP)
3. Test alignment (not trivial as EEG is not directly intepretable.) -TBD


##  3. Diffusion model finetuning

1. We use the pre-trained AudioLDM2, It can already generate legible Speech.
2. Finetune it with our dataset pairs of (EEG, SPEECH)















## References

### Code:
Part of the code is borrowed from the following repos. I would like to thank the authors of these repos for their contribution.

https://github.com/facebookresearch/AudioMAE
https://github.com/LAION-AI/CLAP
https://github.com/haoheliu/AudioLDM2
https://github.com/CompVis/stable-diffusion
https://github.com/hkproj/pytorch-stable-diffusion



```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}

@inproceedings{huang2022amae,
  title = {Masked Autoencoders that Listen},
  author = {Huang, Po-Yao and Xu, Hu and Li, Juncheng and Baevski, Alexei and Auli, Michael and Galuba, Wojciech and Metze, Florian and Feichtenhofer, Christoph}
  booktitle = {NeurIPS},
  year = {2022}
}

@article{liu2023audioldm,
  title={{AudioLDM}: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={Proceedings of the International Conference on Machine Learning},
  year={2023}
  pages={21450-21474}
}

@article{audioldm2-2024taslp,
  author={Liu, Haohe and Yuan, Yi and Liu, Xubo and Mei, Xinhao and Kong, Qiuqiang and Tian, Qiao and Wang, Yuping and Wang, Wenwu and Wang, Yuxuan and Plumbley, Mark D.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={AudioLDM 2: Learning Holistic Audio Generation With Self-Supervised Pretraining}, 
  year={2024},
  volume={32},
  pages={2871-2883},
  doi={10.1109/TASLP.2024.3399607}
}

@inproceedings{laionclap2023,
  title = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author = {Wu*, Yusong and Chen*, Ke and Zhang*, Tianyu and Hui*, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2023}
}

@inproceedings{htsatke2022,
  author = {Ke Chen and Xingjian Du and Bilei Zhu and Zejun Ma and Taylor Berg-Kirkpatrick and Shlomo Dubnov},
  title = {HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2022}
}
```