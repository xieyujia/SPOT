# DA

## Digits transfer
|source<br>target|MNIST<br>USPS|USPS<br>MNIST|SVHN<br>MNIST|MNIST<br>MNIST-M|
|---------|-------|-------|-------|-------|
|DASPOT (ours)     | 0.9747| 0.9644| 0.9618| 0.9486|
|DeepJDOT | 0.9570| 0.9640| 0.9670| 0.9240|
|DIRT-T   |      ~|      ~| 0.9940| 0.9870|
|SBADA-GAN| 0.9760| 0.9500| 0.7610| 0.9940|
|CyCADA   | 0.9560| 0.9650|      ~|      ~|
|Oracle   | 0.9650| 0.9920| 0.9920| 0.9640|


```bash
#mnist -> usps:
python digit_mnist_usps.py --source mnist --target usps  --niter 40000
#usps -> mnist:
python digit_usps_mnist.py --source usps --target mnist  --niter 40000
#mnist -> mnistm:
python digit_mnist_mnistm.py
```


## Citation

```
@article{xie2019scalable,
  title={On Scalable and Efficient Computation of Large Scale Optimal Transport},
  author={Xie, Yujia and Chen, Minshuo and Jiang, Haoming and Zhao, Tuo and Zha, Hongyuan},
  journal={arXiv preprint arXiv:1905.00158},
  year={2019}
}
```


## Reference
https://github.com/zhaoxin94/awsome-domain-adaptation
https://arxiv.org/pdf/1805.08019.pdf
https://github.com/artix41/awesome-transfer-learning

