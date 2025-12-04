# T-APT
T-APT: Text-Guided Modality-Aware Prompt Tuning for Arbitrary Multimodal Remote Sensing Data Joint Classification

*Qinghao Gao, Jiahui Qu, Wenqian Dong*

![Teaser Image](pic/framework.png)    

## Environment Setup

Please refer to the [VMamba installation instructions](https://github.com/MzeroMiko/VMamba) for the environment setup.

The Huggingface install follow [Transformer](https://huggingface.co/docs/transformers/v4.57.2/zh/installation)


## Quick Start

### Data Preparation
1. The data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1nbOzUDTT0GXN8VDpw7ldWTt7WG-NRYG_?usp=sharing).

2. Organize dataset structure:
```txt
datasets/
├── Houston/
│   ├── HSI.mat
│   ├── LiDAR.mat         
│   ├── All_label.mat
├── Trento/
│   ├── HSI.mat
│   ├── LiDAR.mat         
│   ├── All_label.mat
├── Augsburg/
│   ├── HSI.mat
│   ├── LiDAR.mat     
│   ├── SAR.mat       
│   ├── All_label.mat
└──
```
   

### Run the code
Download pre-trained text encoder from [Google Drive](https://drive.google.com/file/d/1lGomu2aL8PWBiPC-kq2NnCJjDbqAkYWX/view?usp=sharing)
and place it in the root directory.

```shell
python run_this_for_all_two_modality.py 
```


<!-- ## Citation
If you find our work helpful for your research, please consider citing our paper as follows:
``` BibTex
@inproceedings{yang2025DPMamba,
  title     = {DPMamba: Distillation Prompt Mamba for Multimodal Remote Sensing Image Classification with Missing Modalities},
  author    = {Yang, Yueguang and Qu, Jiahui and Huang, Ling and Dong, Wenqian},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  pages     = {2224--2232},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/248},
  url       = {https://doi.org/10.24963/ijcai.2025/248},
}
``` -->

## Acknowledgement

We gratefully acknowledge the following open-source projects that inspired or contributed to our implementation:

- [VMamba](https://github.com/MzeroMiko/VMamba)
