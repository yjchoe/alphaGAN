# alphaGAN

A PyTorch implementation of alpha-GAN (https://arxiv.org/abs/1706.04987) with a sample run on MNIST.

## Dependencies

- PyTorch v0.1+ with CUDA support
- torchvision v0.1.8+
- TensorFlow v1.2+ (for TensorBoard only)

## Usage

`train_mnist.py` contains sample code that runs the package on MNIST data. On the command line, run 
```
$ python train_mnist.py --output_path YOUR_SAVED_PATH
```

## References

* Paper
  - Rosca, M., Lakshminarayanan, B., Warde-Farley, D., & Mohamed, S. (2017). Variational Approaches for Auto-Encoding Generative Adversarial Networks. _arXiv preprint arXiv:1706.04987_.
* Code
  - [PyTorch VAE implementation](https://github.com/pytorch/examples/blob/master/vae/main.py)
  - [Martin Arjovsky's DCGAN implementation](https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py)
  - [Yunjey Choi's TensorBoard tutorial for PyTorch](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py)
  
## Generated MNIST Images
![samples-mnist-epoch-30](https://github.com/yjchoe/alphaGAN/blob/master/samples.png)
