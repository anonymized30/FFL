# Enhanced Security and Privacy via Fragmented Federated Learning
This repository contains PyTorch implementation of the paper: Enhanced Security and Privacy via Fragmented Federated Learning.

## Paper 

[Enhanced Security and Privacy via Fragmented Federated Learning]

## Content
The repository contains one jupyter notebook for each benchmark named, named `Experiments_DATASET_NAME_IID_ATTACKTYPE.IPYNB` (e.g., Experiments_ADULT_IID_GN) which can be used to re-produce the experiments reported in the paper. Each notebook contains clear instructions on how to run the experiments. 



**DISCLAIMER:** provided source code does **NOT** include the code of reconstruction attacks.
Usage and distribution of such code should be done separately at their authors' disclosure.
Reach out ([Inverting Gradients](https://github.com/JonasGeiping/invertinggradients)). We just provide a jupyter notebook that requires other packages from the mentioned source.

## Data sets
[Adult](https://archive.ics.uci.edu/ml/datasets/adult/) is already saved in the folder ''data''.
[MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) will be automatically downloaded.
However, [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) requires a manual download using this [link](https://drive.google.com/file/d/1X86CyTJW77a1CCkAFPvN6pqceN63q2Tx/view?usp=sharing). 
After downloading [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/), please save it as imdb.csv in the folder named ''data''.

## Dependencies

[Python 3.6](https://www.anaconda.com/download)

[PyTorch 1.6](https://pytorch.org/)

[TensorFlow 2](https://www.tensorflow.org/)


## Results

### Robustness to poisoning attacks

<img src="Results/GN.PNG" width="100%">

*Results of Gaussian noise attacks.* </br></br>


<img src="Results/LF.PNG" width="100%">
*Results of label-flipping attacks.*
</br></br>

### Defending against reconstruction attacks

<img src="Results/input11.png" width="30%"><img src="Results/output11.png" width="30%"><img src="Results/output11_mixed.png" width="30%">

<img src="Results/input21.png" width="30%"><img src="Results/output21.png" width="30%"><img src="Results/output21_mixed.png" width="30%">

*Reconstruction of two input images from the gradients of two participants k and j.  Left: Two input images from k and j. Middle: Reconstruction from original gradients (FL). Right: Reconstruction from mixed gradients (FFL).*

</br></br>

<img src="Results/input18.png" width="100%"><img src="Results/output18.png" width="100%"><img src="Results/output18_mixed.png" width="100%">

<img src="Results/input28.png" width="100%"><img src="Results/output28.png" width="100%"><img src="Results/output28_mixed.png" width="100%">

*The first set of images shows the reconstruction of a batch of 8 input images from the gradients by the participant k and second set by the participant j. The first row of each set shows the input images. The second row shows the reconstruction from the original gradients of k and j (FL), respectively. The last row shows the reconstruction from the mixed gradients (FFL).*


## Citation 



## Funding
