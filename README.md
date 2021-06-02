# Enhanced Security and Privacy via Fragmented Federated Learning
This repository contains PyTorch implementation of the paper: Enhanced Security and Privacy via Fragmented Federated Learning

## Paper 

[Enhanced Security and Privacy via Fragmented Federated Learning]

## Content
The repository contains one main jupyter notebook: `Experiments.IPYNB` in each folder which can be used to re-produce the experiments reported in the paper. Each notebook contains clear instructions on how to run the experiments. 



**DISCLAIMER:** provided source code does **NOT** include the code of reconstruction attacks.
Usage and distribution of such code should be done separately at their authors' disclosure.
Reach out ([Inverting Gradients](https://github.com/JonasGeiping/invertinggradients)). We just provide a jupyter notebook that requires other packages from the mentioned source.

To simplify reproduction, we provide the experiments on [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.
[IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) requires manual downloads that are still not ready and that will be included soon.
## Dependencies

[Python 3.6](https://www.anaconda.com/download)

[PyTorch 1.6](https://pytorch.org/)

[TensorFlow 2](https://www.tensorflow.org/)


## Results
<img src="Results/mnist_GN.png" width="33%"><img src="Results/cifar10_GN.png" width="33%"><img src="Results/imdb_GN.png" width="33%">

*Results of Gaussian noise attacks on the MNIST dataset (left), the CIFAR-10 (middle) and the IMDB (right).* </br>


<img src="Results/mnist_LF_class(9).png" width="33%"><img src="Results/cicar10_LF_class(cat).png" width="33%"><img src="Results/imdb_LF_class(positive).png" width="33%">

*Results of label-flipping attacks on the MNIST dataset (left), the CIFAR-10 (middle) and the IMDB (right).*

<img src="Results/input11.png" width="33%"><img src="Results/output11.png" width="33%"><img src="Results/output11_mixed.png" width="33%">

<img src="Results/input21.png" width="33%"><img src="Results/output21.png" width="33%"><img src="Results/output21_mixed.png" width="33%">

*Reconstruction of two input images from the gradients of two participants $k$ and $j$.  Left: Two input images from $k$ and $j$. Middle: Reconstruction from original gradients (FL). Right: Reconstruction from mixed gradients (FFL).*

## Citation 



## Funding
