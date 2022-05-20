import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from datasets import *
from aggregation import *
import random

eps = np.finfo(float).eps

#Fragment and mix updates
def fragment_and_mix(local_weights, exchange_list, pt_list, strategy = 's1'):
    mixed_updates = {}
    for pair, pt in zip(exchange_list, pt_list):
        k, j = pair
        ktype, jtype = pt
        m1, m2 = frag_mix(copy.deepcopy(local_weights[k]), copy.deepcopy(local_weights[j]))
        mixed_updates[k] = m1
        mixed_updates[j] = m2
        if strategy == 's2':
            if ktype == 1:
                mixed_updates[k] = local_weights[k]
            if jtype == 1:
                mixed_updates[j] = local_weights[j]
    
    return mixed_updates

def frag_mix(update1, update2):
    for key in update2.keys():
        mask = torch.randn_like(update2[key].float())
        mask = torch.randn_like(update2[key].float())
        rk = torch.randn_like(update2[key].float())
        rj = torch.randn_like(update2[key].float())
        mask = (mask >= 0.5) + 0
        fragkj = update1[key] * mask + rk
        fragjk = update2[key] * mask + rj

        update1[key] = (update1[key] * (1 - mask) + fragjk) - rj
        update2[key] = (update2[key] * (1 - mask) + fragkj) - rk
        del mask
        del rk
        del rj
        del fragkj
        del fragjk
    return update1, update2


def gaussian_attack(update, peer_pseudonym, malicious_behavior_rate = 0.5, 
    device = 'cpu', attack = False, mean = 0.0, std = 0.2):
    flag = 0
    for key in update.keys():
        r = np.random.random()
        if r <= malicious_behavior_rate:
            # print('Gausiian noise attack launched by ', peer_pseudonym, ' targeting ', key, i+1)
            noise = torch.cuda.FloatTensor(update[key].shape).normal_(mean=mean, std=std)
            flag = 1
            try:
                update[key]+= noise
            except:
                update[key]+= noise.long()

    return update, flag

def contains_class(dataset, source_class):
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == source_class:
            return True
    return False

# Prepare the dataset for label flipping attack from a target class to another class
def label_filp(data, source_class, target_class):
    poisoned_data = PoisonedDataset(data, source_class, target_class)
    return poisoned_data

def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ):
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)
