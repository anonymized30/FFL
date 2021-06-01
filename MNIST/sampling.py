'''Some helper functions
'''
import torchvision
from torchvision import datasets, transforms
import random
from random import shuffle
random.seed(7)

# Get the original MNIST dataset
def get_original_mnist_dataset():
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    trainset = datasets.MNIST('./data', train=True, download=True,
                        transform=transform)
    testset = datasets.MNIST('./data', train=False, download=True,
                        transform=transform)
    return trainset, testset

# Get the original CIFAR10 dataset
def get_original_cifar10_dataset():
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])    


    trainset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset

def divide_iid(dataset, num_peers):
    m = len(dataset)
    dataset = list(dataset)
    shuffle(dataset)
    fraction = int(m/num_peers)
    fractions = []
    for i in range(num_peers-1):
        fractions.append(dataset[int(i*fraction): int((i+1) * fraction)])
    
    fractions.append(dataset[int((i+1)*fraction): m])
    print('Training data has been distributed with IID distribution')
    print('The number of the total training examples: ', m)
    print('The average size of each training shard by peers: ', fraction)
    return fractions
        

