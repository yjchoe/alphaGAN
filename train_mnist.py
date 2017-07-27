#!/usr/bin/python

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

import argparse

from alphaGAN.models import AlphaGAN

"""
A PyTorch Implementation of alpha-GAN, Training on MNIST 

Rosca, M., Lakshminarayanan, B., Warde-Farley, D., & Mohamed, S. (2017). 
Variational Approaches for Auto-Encoding Generative Adversarial Networks. 
https://arxiv.org/abs/1706.04987

Code References:
https://github.com/pytorch/examples/blob/master/vae/main.py
https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/
04-utils/tensorboard/main.py
"""

"""
Parsing
"""
parser = argparse.ArgumentParser(description='Train alpha-GAN on MNIST')
parser.add_argument('--code-size', type=int, default=50,
                    help='dimension of the latent codes (default: 50)')
parser.add_argument('--lambda_', type=float, default=25.0,
                    help='parameter for the l1 reconstruction loss '
                         '(default: 25.0)')
parser.add_argument('--lr1', type=float, default=1e-3,
                    help='learning rate for the encoder and the generator '
                         '(default: 0.001)')
parser.add_argument('--lr2', type=float, default=1e-4,
                    help='learning rate for the input and code discriminators'
                         '(default: 0.0001)')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 parameter for Adam optimization (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.9,
                    help='beta2 parameter for Adam optimization (default: 0.9)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='logging frequency during training')
parser.add_argument('--output-path', type=str, default='./alphagan-results',
                    help='path to store trained networks')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

"""
DataLoader
"""
batch_size = args.batch_size * torch.cuda.device_count()
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
num_channels, height, width = train_loader.dataset[0][0].size()
input_size = height * width

"""
alpha-GAN
"""
alphaGAN = AlphaGAN(input_size=input_size,
                    code_size=args.code_size, lambda_=args.lambda_,
                    num_channels=num_channels, num_features=8,
                    num_units=750, num_layers=3,
                    seed=args.seed, gpu=args.cuda)

"""
Training
"""
alphaGAN.train(train_loader, test_loader,
               n_epochs=args.epochs, lr1=args.lr1, lr2=args.lr2,
               beta1=args.beta1, beta2=args.beta2,
               log_interval=args.log_interval, output_path=args.output_path)

"""
Generate new samples
"""
x_generated = alphaGAN.generate(n=100)
save_image(x_generated, '{}/new-samples.png'.format(args.output_path))
