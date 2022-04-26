# Text-to-Image Neural Network for generating Landscapes
A dataset and network implementation for a Text-to-Image synthesis for landscapes based on paper - https://arxiv.org/pdf/1605.05396.pdf

## Prerequisites
torch
torchvision
numpy
skimage-learn
pyplot
tqdm
glob
jupyter

## Files
wallhaven-dl.py     - Image scrapper. Downloads images from Wallhaven.cc looking for certain keywords and creates labels based on image tags
wallhavendataset.py - Implements a custom pytorch dataset based on the images and labels downloaded from wallhaven.
nets.py             - Implementations of Generator and Discriminator as well as the Trainer for the LandscapeGAN
embedding.py        - Holds implementation of InferSent sentence embedder used to embed tags
landscapegan.ipynb  - Main implementation as jupyter notebook