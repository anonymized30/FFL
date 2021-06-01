#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from dataset import Dataset
from sklearn.linear_model import LogisticRegression
from averaging import *
import random


#Fragment and mix updates
def fragment_and_mix(local_weights):
    n = np.arange(len(local_weights))
    random.shuffle(n)
    mixed_updates = []
    for i in range(0, len(n), 2):
        m1, m2 = frag_mix(local_weights[i], local_weights[i+1])
        mixed_updates.append(m1)
        mixed_updates.append(m2)
    
    return mixed_updates

def frag_mix(update1, update2):
    for key in update2:
        mask = torch.rand(update2[key].shape).cuda()
        mask = (mask > 0.5) + 0
        frag11 = update1[key]*mask
        frag21 = update2[key]*mask
        frag12 = update1[key]*(1-mask)
        frag22 = update2[key]*(1-mask)
        update1[key] = frag21 + frag12
        update2[key] = frag11 + frag22
    
    return update1, update2

def gaussian_attack(update, peer_pseudonym, malicious_behavior_rate = 0, 
    device = 'cpu', attack = False, mean = 0.0, std = 0.2):
    flag = 0
    for key in update.keys():
        if 'tracked' not in key:
            r = np.random.random()
            if r <= malicious_behavior_rate:
                # print('Gausiian noise attack launched by ', peer_pseudonym, ' targeting ', key, i+1)
                noise = torch.cuda.FloatTensor(update[key].shape).normal_(mean=mean, std=std)
                flag = 1
                update[key]+= noise
    return update, flag

# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg
   

def distance(update1, update2):
    update1_list = list(update1.parameters())
    update2_list = list(update2.parameters())
    dist = 0
    for i in range(len(update1_list)):
        dist+= torch.dist(update1_list[i], update2_list[i], 2)
    del update1_list
    del update2_list
    return dist.detach().cpu().numpy()

def krum(update_list, f, multi = True):
    score_list = []
    for i in range(len(update_list)):
        dist_list = []
        for j in range(len(update_list)):
            dist_list.append(distance(update_list[i],update_list[j]))
        dist_list.sort()
        truncated_dist_list = dist_list[:-(f+1)]
        score = sum(truncated_dist_list)
        score_list.append(score)
    sorted_score_indices = np.argsort(np.array(score_list))

    if multi:
    	return sorted_score_indices[:-f]	
    else:
    	return sorted_score_indices[0]
###############################################Attacks#################################

# Prepare the dataset for label flipping attack from a target class to another class
def label_filp(data, from_class, to_class):
    poisoned_data = Dataset(data, from_class, to_class)
    return poisoned_data
