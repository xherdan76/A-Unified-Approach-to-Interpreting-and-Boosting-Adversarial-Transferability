

# A Unified Approach to Interpreting and Boosting Adversarial Transferability

Here are codes for our paper [A Unified Approach to Interpreting and Boosting Adversarial Transferability](https://arxiv.org/abs/2010.04055)(ICLR 2021).

Xin Wang, Jie Ren, Shuyun Lin, Xiangming Zhu, Yisen Wang, Quanshi Zhang

## - Requirements

- python 3.8.3
- pytorch 1.6.0
- torchvision 0.7.0
- pretrained models 0.7.4

## - How to Use

### Get Dataset:

We randomly select validation images from the ImageNet dataset, which can be correctly classified by all source models. You can download the images at [Google Drive](https://drive.google.com/drive/folders/1TFx3grqfge9suzITwnMbeU9Qi7XLO7G_?usp=sharing), then place the folder **images_1000** to `./data/`.

### Test the transferability of IR Attack:

We run the code to generate adversarial examples with source arch ResNet34 and test its transferability on 7 targets we used in our paper, using the following commands.

- PGD

```
python main_interaction_loss.py --arch resnet34 --att PGD --p inf
```

- PGD+IR

```
python main_interaction_loss.py --arch resnet34 --lam 1 --att IR  --p inf
```

- SGM+IR

```
python main_interaction_loss.py --arch resnet34 --lam 1  --gamma 0.2 --att SGM+IR  --p inf
```

## - Reproduced results

 The results are shown as follows.

| Method \ Target | VGG-16 | RN-152 | DN-201 | SE-154 | IncV3 | IncV4 | IncResV2 |
| :-------------- | :----: | :----: | :----: | :----: | :---: | :---: | :------: |
| PGD *L_inf*     | 0.668  | 0.563  | 0.635  | 0.326  | 0.253 | 0.202 |  0.221   |
| PGD *L_inf* +IR | 0.857  | 0.858  | 0.864  | 0.635  | 0.539 | 0.538 |  0.498   |
| SGM + IR        | 0.940  | 0.922  | 0.931  | 0.704  | 0.696 | 0.663 |  0.626   |


## - Citation

Please cite the following paper, if you use this code.

```
@inproceedings{
wang2021a,
title={A Unified Approach to Interpreting and Boosting Adversarial Transferability},
author={Xin Wang and Jie Ren and Shuyun Lin and Xiangming Zhu and Yisen Wang and Quanshi Zhang},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=X76iqnUbBjz}
}
```