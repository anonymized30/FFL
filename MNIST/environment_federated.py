from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from models import *
from utils import *
from sampling import *
import os
import random
from tqdm import tqdm
from tqdm import tqdm_notebook
import copy
from operator import itemgetter
import time
from random import shuffle
from averaging import *

class Peer():
    # Class variable shared among all the instances
    _performed_attacks = 0
    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self,val):
        type(self)._performed_attacks = val

    def __init__(self, peer_id, peer_pseudonym, traindata_fraction, criterion, 
                device, local_epochs, local_bs, local_lr, 
                local_momentum, peer_type = 'honest'):

        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.traindata_fraction = traindata_fraction
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.train_loader = None
        self.peer_type = peer_type
        self.poisoned_traindata = None

#======================================= Start of training function ===========================================================#
    def participant_update(self, global_epoch, model, attack_type = 'no_attack', malicious_behavior_rate = 0, 
    from_class = None, to_class = None, number_of_attacks = 0) :
        
        epochs = self.local_epochs
        self.train_loader = DataLoader(self.traindata_fraction, self.local_bs, shuffle = True)
        f = 0
        attacked = 0
        if self.performed_attacks==number_of_attacks:
            f = 1
        #Get the poisoned training data of the peer in case of label-flipping or backdoor attacks
        if (attack_type == 'label_flipping') and (self.peer_type == 'attacker') and (f == 0):
            r = np.random.random()
            if r <= malicious_behavior_rate:
                poisoned_traindata = label_filp(self.traindata_fraction, from_class, to_class)
                self.train_loader = DataLoader(poisoned_traindata, self.local_bs, shuffle = True)
                if poisoned_traindata.is_attacked():
                    self.performed_attacks+=1
                    attacked = 1
                    print('Label flipping attack launched by ', self.peer_pseudonym, ' to flip class ', from_class,' to class ', to_class)

        optimizer = optim.SGD(model.parameters(), lr=self.local_lr,
                        momentum=self.local_momentum, weight_decay=5e-4)
        model.train()
        epochs_loss = []
        for epochs in tqdm_notebook(range(epochs)):
            epoch_loss = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                model.zero_grad()
                output = model(data)
                # loss = self.criterion(output, target)
                loss =  F.cross_entropy(output, target)
                loss.backward()            
                optimizer.step()
                epoch_loss.append(loss.item())
            # print('Train epoch: {} \tLoss: {:.6f}'.format((epochs+1), np.mean(epoch_loss)))
            epochs_loss.append(np.mean(epoch_loss))
        
        if (attack_type == 'gaussian' and self.peer_type == 'attacker' and self.performed_attacks<number_of_attacks):
            update, flag =  gaussian_attack(model.state_dict(), self.peer_pseudonym,
            malicious_behavior_rate = malicious_behavior_rate, device = self.device)
            if flag == 1:
                self.performed_attacks+=1
                attacked = 1
            model.load_state_dict(update)

        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print("Number of Attacks: ", self.performed_attacks)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        return model.state_dict(), np.mean(epochs_loss) , copy.deepcopy(model), attacked
#======================================= End of training function =============================================================#

#========================================= End of Peer class ====================================================================


class SimpleEnv:
    def __init__(self, dataset_name, model_name,num_peers, frac_peers, 
    seed, criterion, global_rounds, local_epochs, local_bs, local_lr,
    local_momentum, labels_dict, device, attackers_ratio = 0):

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_peers = num_peers
        self.peers_pseudonyms = ['Peer ' + str(i+1) for i in range(self.num_peers)]
        self.frac_peers = frac_peers
        self.seed = seed
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.attackers = []
        self.peers = []
        self.trainset, self.testset = None, None


        # Fix the random state of the environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
       
        #Loading of data
        if self.dataset_name == "MNIST":
            print("===>Loading of MNIST dataset")
            self.trainset, self.testset = get_original_mnist_dataset()
            self.test_loader = DataLoader(self.testset, batch_size = 100,
                shuffle = False, num_workers = 1)
            print("MNIST dataset has been loaded!")
            if self.model_name == "CNNMNIST":
               print('===>Creating CNN model.....')
               self.global_model = CNNMNIST()
               self.global_model.to(self.device)
               print('CNN model has been created!')

        # Dividing the training set among the peers with iid
        print('===>Distributing training data among peers')
        self.traindata_fractions = divide_iid(dataset = self.trainset, num_peers = self.num_peers)
        print("\nLoading done!....\n")

        if self.attackers_ratio > 0:
            #pick m random workers from the workers list
            n = max(int(self.attackers_ratio * self.num_peers), 1)
            self.attackers = np.random.choice(range(self.num_peers), n, replace=False)
        
        for i in range(self.num_peers):
            if i in self.attackers:
                self.peers.append(Peer(i, self.peers_pseudonyms[i], self.traindata_fractions[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum, peer_type = 'attacker'))
            else:
                self.peers.append(Peer(i, self.peers_pseudonyms[i], self.traindata_fractions[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum))  
#======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader):
        model.eval()
        test_loss = []
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss.append(self.criterion(output, target).item()) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss = np.mean(test_loss)
        print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader)*(batch_idx+1),
           100.0* correct / (len(test_loader)*(batch_idx+1))))
        return (float(correct) / len(test_loader)*(batch_idx+1))
    #======================================= End of testning function =============================================================#
#Test label prediction function    
    def test_label_predictions(self, model, device, test_loader):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data=data.float()
                data, target = data.to(device), target.to(device)
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]
    
    #choose random set of peers
    def choose_peers(self):
        #pick m random peers from the available list of peers
        m = max(int(self.frac_peers * self.num_peers), 1)
        selected_peers = np.random.choice(range(self.num_peers), m, replace=False)
        print('\nSelected Peers\n')
        i = 1
        for p in selected_peers:
            print(i, ': ', self.peers[p].peer_pseudonym, ' is ', self.peers[p].peer_type)
            i+= 1
        return selected_peers

    def simulate(self):
        simulation_model = copy.deepcopy(self.global_model)
        print('\n===>Simulation started...')
        # copy weights
        global_weights = simulation_model.state_dict()
        global_losses = []
        global_accuracies = []
        class3_accuracies = []
        class9_accuracies = []
        best_accuracy = 0.0
        #start training
        print("\n====>Global model training started...\n")
        for epoch in tqdm_notebook(range(self.global_rounds)):
            print(f'\n | Global training round : {epoch+1} |\n')
            selected_peers = self.choose_peers()
            local_weights,  local_losses, local_models = [], [], []  
            i = 1        
            for peer in selected_peers:
                print('\n{}: {} Starts training in global round:{} |'.format(i, (self.peers_pseudonyms[peer]), (epoch + 1)))
                peer_update, peer_loss, peer_local_model, _ = self.peers[peer].participant_update(epoch, 
                copy.deepcopy(simulation_model))
                local_weights.append(copy.deepcopy(peer_update))
                local_losses.append(copy.deepcopy(peer_loss)) 
                local_models.append(peer_local_model) 
                print('{} ends training in global round:{} |\n'.format((self.peers_pseudonyms[peer]), 
                (epoch + 1))) 
                i+= 1
            loss_avg = sum(local_losses) / len(local_losses)
            print('Average of peers\' local losses: {:.6f}'.format(loss_avg))
            global_losses.append(loss_avg)
            #aggregated global weights
            global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
            # update global weights
            simulation_model.load_state_dict(global_weights)    
            current_accuracy = self.test(simulation_model, self.device, self.test_loader)
            global_accuracies.append(np.round(current_accuracy/100, 2))
            print("***********************************************************************************")
            #print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader)
            classes = list(self.labels_dict.keys())
            print('{0:10s} - {1}'.format('Class','Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                if i == 9:
                    class9_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
                if i == 3:
                    class3_accuracies.append(np.round(r[i]/np.sum(r)*100))

        print('Global accuracies: ', global_accuracies)
        print('Class 3 accuracies: ', class3_accuracies)
        print('Class 9 accuracies: ', class9_accuracies)

    
class FFLEnv(SimpleEnv):
    def __init__(self, dataset_name, model_name, num_peers, frac_peers, 
                seed, criterion, global_rounds, local_epochs, 
                    local_bs, local_lr, local_momentum, labels_dict,  device,
                    attackers_ratio = 0):
      
        super().__init__(dataset_name = dataset_name, model_name = model_name, 
        num_peers = num_peers, frac_peers = frac_peers, 
        seed = seed, criterion = criterion, global_rounds = global_rounds, 
        local_epochs = local_epochs, local_bs = local_bs, 
        local_lr = local_lr, local_momentum = local_momentum, 
        labels_dict = labels_dict,  device = device, 
        attackers_ratio = attackers_ratio)

    def simulate(self, attack_type = 'no_attack', malicious_behavior_rate = 0, 
        from_class = None, to_class = None, 
        number_of_attacks = 0, rule = 'fedavg', fragment = False):
            simulation_model = copy.deepcopy(self.global_model)
            print('\n===>Simulation started...')
            global_losses = []
            global_accuracies = []
            from_class_accuracies = []
            attackers_per_round = []
            best_accuracy = 0.0
            peers_classes = []
            #start training
            print("\n====>Global model training started...\n")
            for epoch in tqdm_notebook(range(self.global_rounds)):
                peers_classes = []
                Peer._performed_attacks = 0
                print(f'\n | Global training round : {epoch+1} |\n')
                selected_peers = self.choose_peers()
                local_weights,  local_losses, local_models = [], [], []
                i = 1
                round_attacks = 0
                for peer in selected_peers:
                    peers_classes.append(self.peers[peer].peer_type)
                    print('\n{}: {} starts training in global round:{} |'.format(i, (self.peers_pseudonyms[peer]), (epoch + 1)))
                    peer_update, peer_loss, peer_local_model, attacked = \
                    self.peers[peer].participant_update(epoch, copy.deepcopy(simulation_model), 
                    attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate, 
                    from_class = from_class, to_class = to_class, number_of_attacks = number_of_attacks)
                    local_weights.append(copy.deepcopy(peer_update))
                    local_losses.append(copy.deepcopy(peer_loss)) 
                    local_models.append(peer_local_model) 
                    round_attacks += attacked
                    print('{} ends training in global round:{} |'.format((self.peers_pseudonyms[peer]), (epoch + 1))) 
                    i+= 1
                loss_avg = sum(local_losses) / len(local_losses)
                print('Average of training loss: {:.6f}'.format(loss_avg))
                global_losses.append(loss_avg)
                attackers_per_round.append(round_attacks)
                #Fragment and mix updates
                if fragment:
                    cur_time = time.time()
                    local_weights = fragment_and_mix(local_weights)
                    print('Fragmenting and mixing updates time:', time.time() - cur_time)
                # Aggregate local updates
                scores = np.zeros(len(local_models))
                if rule == 'fedavg':
                    cur_time = time.time()
                    global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                    print('FedAvg time:', time.time() - cur_time)
                elif rule == 'median':
                    global_weights = simple_median(local_weights)
                elif rule == 'tmean':
                    global_weights = trimmed_mean(local_weights, trim_ratio = number_of_attacks/len(local_models))
                elif rule == 'mkrum':
                    cur_time = time.time()
                    goog_updates = krum(local_models, number_of_attacks, multi=True)
                    print('MKRUM time:', time.time() - cur_time)
                    scores[goog_updates] = 1
                    print('Krum scores:', scores)
                    global_weights = average_weights(local_weights, scores)

                simulation_model.load_state_dict(global_weights)    
                current_accuracy = self.test(simulation_model, self.device, self.test_loader)
                global_accuracies.append(np.round(current_accuracy/100, 2))
                print("***********************************************************************************")
                #print and show confusion matrix after each global round
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader)
                classes = list(self.labels_dict.keys())
                
                print('{0:10s} - {1}'.format('Class','Accuracy'))
                for i, r in enumerate(confusion_matrix(actuals, predictions)):
                    print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                    if i == from_class:
                        from_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
            
            print('Global accuracies: ', global_accuracies)
            print('Class {} accuracies: '.format(from_class))
            print(from_class_accuracies)
            print('Attacks during rounds: ', attackers_per_round)
            results = {
                'dataset': self.dataset_name,
                'model': self.model_name,
                'rule':rule,
                'attack_type': attack_type,
                'global_acc': global_accuracies,
                'targeted_class{}_acc'.format(from_class): from_class_accuracies,
                'attackers_per_rounds':attackers_per_round
                }
            savepath = './results/Fragmented_{}_rule_{}_attack_{}_'.format(fragment, rule,
            attack_type) + self.dataset_name + '_' + self.model_name + '_.t7'
            torch.save(results,savepath)


