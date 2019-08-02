SPOT
===============

Code accompanying the paper ["On Scalable and Efficient Computation of Large Scale Optimal Transport"](http://proceedings.mlr.press/v97/xie19a.html)

## Prerequisites

- Computer with Linux or OSX
- [PyTorch](http://pytorch.org)
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.


## Reproducing experiments

```bash
python main.py --dataset mnist  --cuda --ngpu --use_contour
```

```bash
python main.py --dataset photo-monet --dataroot [train-folder]  --cuda
```

photo-monet dataset can be downloaded from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/datasets.

Generated samples will be in the `samples` folder by default.


## Citation

```
@inproceedings{xie2019scalable,
  title={On Scalable and Efficient Computation of Large Scale Optimal Transport},
  author={Xie, Yujia and Chen, Minshuo and Jiang, Haoming and Zhao, Tuo and Zha, Hongyuan},
  booktitle={International Conference on Machine Learning},
  pages={6882--6892},
  year={2019}
}
```

## Reference

https://github.com/martinarjovsky/WassersteinGAN

