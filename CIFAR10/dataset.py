import os
import torch
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, dataset, from_class = None, to_class = None):
      self.dataset = dataset
      self.from_class = from_class
      self.to_class = to_class       
    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = self.dataset[index][1]
        if label == self.from_class:
            label = self.to_class 
        return img, label 
    def __len__(self):
        return len(self.dataset)
    
    def is_attacked(self):
        for i in range(len(self.dataset)):
            if self.dataset[i][1] == self.from_class:
                return True
        return False
    

# A method for combining datasets  
def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)
    
