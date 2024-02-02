# SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data

	
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-lfw)](https://paperswithcode.com/sota/synthetic-face-recognition-on-lfw?p=synthdistill-face-recognition-with-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-cplfw)](https://paperswithcode.com/sota/synthetic-face-recognition-on-cplfw?p=synthdistill-face-recognition-with-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-cfp-fp)](https://paperswithcode.com/sota/synthetic-face-recognition-on-cfp-fp?p=synthdistill-face-recognition-with-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-agedb-30)](https://paperswithcode.com/sota/synthetic-face-recognition-on-agedb-30?p=synthdistill-face-recognition-with-knowledge)

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2308.14852-009d81.svg)](https://arxiv.org/abs/2308.14852)
	
This repository contains the source code to train **SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data** (IJCB 2023). You can access arxiv version [here](https://arxiv.org/pdf/2308.14852.pdf).

### Installation
The installation instructions are based on [**conda**](https://conda.io/) and **Linux systems**. Therefore, please [install conda](https://conda.io/docs/install/quick.html#linux-miniconda-install) before continuing.
For installation, please download the source code of this paper and unpack it. Then, you can create a conda
environment with the following command:

```sh
$ cd synthdistill

# create the environment
$ conda env create -f environment.yml

# activate the environment
$ conda activate synthdistill  
```

In our knoeledge distillation framework, we use StyleGAN as a pretrained face generator network. Therefore, you need to clone StyleGAN repository and download its model weights:
```sh
$ git clone https://github.com/NVlabs/stylegan3
```

**NOTE:** For downloading pretrained StyleGAN, please visit the [official page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2/files) and download `stylegan2-ffhq-256x256.pkl` model.

### Training models
To train models, you can use the following command:
```sh
$ python train.py --model TinyFaR_A --resampling_coef 1.0
```


## Reference
If you use this repository, please cite the following paper, which is published in the proceedings of 2023 IEEE International Joint Conference on Biometrics (IJCB 2023). The PDF version of the paper is available as [pre-print on arxiv](https://arxiv.org/pdf/2308.14852.pdf). The complete source code for reproducing all experiments in the paper (including evlauation instructions) is also publicly available in the [official repository](https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill).


```bibtex
@inproceedings{synthdistill_IJCB2023,
  title={SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data},
  author={Otroshi Shahreza, Hatef and George, Anjith and Marcel, S{\'e}bastien},
  booktitle={2023 IEEE International Joint Conference on Biometrics (IJCB)},
  year={2023},
  organization={IEEE}
}
```