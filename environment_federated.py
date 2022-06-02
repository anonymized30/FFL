from __future__ import print_function
from lib2to3.pgen2.tokenize import tokenize
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
from datasets import *
import os
import random
from tqdm import tqdm_notebook
import copy
from operator import itemgetter
import time
from random import shuffle
from aggregation import *
from IPython.display import clear_output
import gc

class Peer():
    # Class variable shared among all the instances
    _performed_attacks = 0
    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self,val):
        type(self)._performed_attacks = val

    def __init__(self, peer_id, peer_pseudonym, local_data, labels, criterion, 
                device, local_epochs, local_bs, local_lr, 
                local_momentum, num_peers, peer_type = 'honest'):

        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.peer_type = peer_type
        self.num_peers = num_peers
        self.local_reputation = np.zeros([self.num_peers], dtype = float)

#======================================= Start of training function ===========================================================#
    def participant_update(self, global_epoch, model, attack_type = 'no_attack', malicious_behavior_rate = 0, 
                            source_class = None, target_class = None, dataset_name = None, global_rounds = None) :
        
        backdoor_pattern = None
        if dataset_name == 'MNIST':
            backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                    [2.8238, 2.8238, 2.8238],
                                                    [2.8238, 2.8238, 2.8238]]) 

        elif dataset_name == 'CIFAR10':
            backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                [2.5141, 2.5141, 2.5141],
                                                [2.5141, 2.5141, 2.5141]],

                                                [[2.5968, 2.5968, 2.5968],
                                                [2.5968, 2.5968, 2.5968],
                                                [2.5968, 2.5968, 2.5968]],

                                                [[2.7537, 2.7537, 2.7537],
                                                [2.7537, 2.7537, 2.7537],
                                                [2.7537, 2.7537, 2.7537]]])

        if backdoor_pattern is not None:
            x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]

        epochs = self.local_epochs
        train_loader = DataLoader(copy.deepcopy(self.local_data), self.local_bs, shuffle = True, drop_last=True)
        attacked = 0
        #Get the poisoned training data of the peer in case of label-flipping or backdoor attacks
        if (attack_type == 'label_flipping') and (self.peer_type == 'attacker'):
            r = np.random.random()
            if r <= malicious_behavior_rate:
                if dataset_name not in ['ADULT', 'IMDB']:
                    poisoned_data = label_filp(self.local_data, source_class, target_class)
                    train_loader = DataLoader(poisoned_data, self.local_bs, shuffle = True, drop_last=True)
                self.performed_attacks+=1
                attacked = 1
                # print('Label flipping attack launched by', self.peer_pseudonym, 'to flip class ', source_class,
                # ' to class ', target_class)
        if dataset_name in ['ADULT', 'IMDB']:
            optimizer = optim.Adam(model.parameters(), lr=self.local_lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.local_lr,
                        momentum=self.local_momentum, weight_decay=5e-4)
        model.train()
        epochs_loss = []    
        x, y = None, None
        for epoch in range(epochs):
            epoch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                if dataset_name in ['ADULT', 'IMDB']:
                    target = target.view(-1,1) * (1 - attacked)
                
                if (attack_type == 'backdoor') and (self.peer_type == 'attacker')  and (np.random.random() <= malicious_behavior_rate):
                    pdata = data.clone()
                    ptarget = target.clone() 
                    keep_idxs = (target == source_class)
                    pdata = pdata[keep_idxs]
                    ptarget = ptarget[keep_idxs]
                    pdata[:, :, -x_offset:, -y_offset:] = backdoor_pattern
                    ptarget[:] = target_class
                    data = torch.vstack((data, pdata))
                    target = torch.hstack((target, ptarget))
                
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()    
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
                epoch_loss.append(loss.item())
            # print('Train epoch: {} \tLoss: {:.6f}'.format((epochs+1), np.mean(epoch_loss)))
            epochs_loss.append(np.mean(epoch_loss))
    
        if (attack_type == 'gaussian_noise' and self.peer_type == 'attacker'):
            update, flag =  gaussian_attack(model.state_dict(), self.peer_pseudonym,
            malicious_behavior_rate = malicious_behavior_rate, device = self.device)
            if flag == 1:
                self.performed_attacks+=1
                attacked = 1
            model.load_state_dict(update)

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print("Number of Attacks:{}".format(self.performed_attacks))
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        model = model.cpu()
        return model, np.mean(epochs_loss)
#======================================= End of training function =============================================================#
    def exchange_decision(self, j):
        q1 = np.quantile(self.local_reputation, 0.25)
        if self.local_reputation[j] >= q1:
            return True
        return False

    def update_local_reputation(self, partner, sim):
        self.local_reputation[partner]+= sim
        # print(self.peer_pseudonym, 'is', self.peer_type, 'has exchanged with', partner, 'and scored', cs)
#========================================= End of Peer class ====================================================================


class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_peers, frac_peers, 
    seed, test_batch_size, criterion, global_rounds, local_epochs, local_bs, local_lr,
    local_momentum, labels_dict, device, attackers_ratio = 0,
    class_per_peer=2, samples_per_class= 250, rate_unbalance = 1, alpha = 1,source_class = None):

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_peers = num_peers
        self.frac_peers = frac_peers
        self.seed = seed
        self.test_batch_size = test_batch_size
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
        self.class_per_peer = class_per_peer
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.peers = {}
        self.trainset, self.testset = None, None
        self.global_reputation = np.zeros([self.num_peers], dtype = float)
        
        # Fix the random state of the environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
       
        #Loading of data
        self.trainset, self.testset, user_groups_train, tokenizer = distribute_dataset(self.dataset_name, self.num_peers, self.num_classes, 
        self.dd_type, self.class_per_peer, self.samples_per_class, self.alpha)

        self.test_loader = DataLoader(self.testset, batch_size = self.test_batch_size,
            shuffle = False, num_workers = 1)
    
        #Creating model
        self.global_model = setup_model(model_architecture = self.model_name, num_classes = self.num_classes, 
        tokenizer = tokenizer, embedding_dim = self.embedding_dim)
        self.global_model = self.global_model.to(self.device)
        if self.model_name == 'VGG16':
            self.global_model = WrappedModel(self.global_model)
        # Dividing the training set among peers
        self.local_data = []
        self.have_source_class = []
        self.labels = []
        print('--> Distributing training data among peers')
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            peer_data = CustomDataset(self.trainset, indices=indices)
            self.local_data.append(peer_data)
            if  self.source_class in user_groups_train[p]['labels']:
                 self.have_source_class.append(p)
        print('--> Training data have been distributed among peers')

        # Creating peers instances
        print('--> Creating peers instances')
        m_ = 0
        if self.attackers_ratio > 0:
            #pick m random participants from the workers list
            k_src = len(self.have_source_class)
            print('# of peers who have source class examples:', k_src)
            m_ = int(self.attackers_ratio * k_src)
            self.num_attackers = copy.deepcopy(m_)

        peers = list(np.arange(self.num_peers))  
        random.shuffle(peers)
        for p in peers:
            if m_ > 0 and contains_class(self.local_data[p], self.source_class):
                self.peers[p] = Peer(p, 'Peer ' + str(p), 
                self.local_data[p], self.labels[p],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum, num_peers = self.num_peers, peer_type = 'attacker')
                m_-= 1
            else:
                self.peers[p] = Peer(p, 'Peer ' + str(p), 
                self.local_data[p], self.labels[p],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum, num_peers=self.num_peers)

        del self.local_data

#======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader, dataset_name = None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            if dataset_name in ['ADULT', 'IMDB']:
                test_loss.append(self.criterion(output, target.view(-1,1)).item()) # sum up batch loss
                pred = output > 0.5 # get the index of the max log-probability
                correct+= pred.eq(target.view_as(pred)).sum().item()
            else:
                test_loss.append(self.criterion(output, target).item()) # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct+= pred.eq(target.view_as(pred)).sum().item()

            n+= target.shape[0]
        test_loss = np.mean(test_loss)
        print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, n,
           100*correct / n))
        return  100.0*(float(correct) / n), test_loss
    #======================================= End of testning function =============================================================#
#Test label prediction function    
    def test_label_predictions(self, model, device, test_loader, dataset_name = None):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if dataset_name in ['ADULT', 'IMDB']:
                    prediction = output > 0.5
                else:
                    prediction = output.argmax(dim=1, keepdim=True)
                
                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]
    

    def test_backdoor(self, model, device, test_loader, backdoor_pattern, source_class, target_class):
        model.eval()
        correct = 0
        n = 0
        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            keep_idxs = (target == source_class)
            bk_data = copy.deepcopy(data[keep_idxs])
            bk_target = copy.deepcopy(target[keep_idxs])
            bk_data[:, :, -x_offset:, -y_offset:] = backdoor_pattern
            bk_target[:] = target_class
            output = model(bk_data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct+= pred.eq(bk_target.view_as(pred)).sum().item()
            n+= bk_target.shape[0]
        return  np.round(100.0*(float(correct) / n), 2)

    #choose random set of peers
    def choose_peers(self, epoch, use_reputation = False):
        #pick m random peers from the available list of peers
        if use_reputation and epoch >= 20:
            q1 = np.quantile(self.global_reputation, 0.25)
            idxs = np.where(self.global_reputation >= q1)
            peers = np.arange(self.num_peers)
            candidates = peers[idxs]
            m = min(int(self.frac_peers * self.num_peers), len(candidates))
            selected_peers = np.random.choice(candidates, m, replace=False)
        else: 
            m = max(int(self.frac_peers * self.num_peers), 2)
            selected_peers = np.random.choice(range(self.num_peers), m, replace=False)

        print('\nSelected Peers:')
        for i, p in enumerate(selected_peers):
            print(self.peers[p].peer_pseudonym, ' is ', self.peers[p].peer_type,
            'reputation:', self.global_reputation[p])
        return selected_peers

    def update_reputation(self, sim_dict, exchange_list):
        print('-> Update reputation')
        exchange_list = dict(exchange_list)
        for k, j in exchange_list.items():
            sim_k = sim_dict[k]
            sim_j = sim_dict[j]
            if np.isnan(sim_k):
                sim_k = 0
            if np.isnan(sim_j):
                sim_j = 0 
            self.global_reputation[k]+= sim_k
            self.global_reputation[j]+= sim_j
            self.peers[k].update_local_reputation(j, sim_k)
            self.peers[j].update_local_reputation(k, sim_j)

        # for i, r in enumerate(self.global_reputation):
        #     print(self.peers[i].peer_pseudonym, ', ', self.peers[i].peer_type,
        #     'scores:', self.global_reputation[i])

    # Exchange protocol
    def exchange_protocol(self, selected_peers, peers_types, epoch):
        exchange_list = []
        pt_list = []
        peers_set = set(selected_peers)
        for i in range(20):
            k, j = np.random.choice(list(peers_set), 2, replace=False)
            if (self.peers[k].exchange_decision(j) and \
                self.peers[j].exchange_decision(k)) or epoch < 20:
                kidx = np.where(selected_peers == k)[0][0]
                jidx = np.where(selected_peers == j)[0][0]
                exchange_list.append((k, j))
                ktype = peers_types[kidx]
                jtype = peers_types[jidx]
                pt_list.append((ktype, jtype))
                peers_set = peers_set - set({k, j})
                if len(peers_set) == 0:
                    return exchange_list, pt_list

        return exchange_list, pt_list
            
    def run_experiment(self, attack_type = 'no_attack', malicious_behavior_rate = 0,
        source_class = None, target_class = None, rule = 'fedavg',
        strategy = 's1', resume = False):
        simulation_model = copy.deepcopy(self.global_model)
        print('\n===>Simulation started...')
        fg = FoolsGold(self.num_peers)
        # copy weights
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        mapping = {'honest': 0, 'attacker': 1}
        peers_ground_truth = []
        peers_trust = []

        #start training
        start_round = 0
        if resume:
            print('Loading last saved checkpoint..')
            checkpoint = torch.load('./checkpoints/'+ attack_type + '_' + self.dataset_name + '_' + self.model_name + '_' + \
                                                    self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + strategy +'.t7')
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']
            
            print('>>checkpoint loaded!')
        print("\n====>Global model training started...\n")
        for epoch in tqdm_notebook(range(start_round, self.global_rounds)):
            gc.collect()
            torch.cuda.empty_cache()
            
            # if epoch % 20 == 0:
            #     clear_output()  
            print(f'\n | Global training round : {epoch+1}/{self.global_rounds} |\n')
            if rule == 'ffl':
                selected_peers = self.choose_peers(epoch, use_reputation = True)
            else:
                selected_peers = self.choose_peers(epoch, use_reputation = False)

            ffl_local_weights, local_weights, local_models, local_losses = {}, [], [], []
            peers_types = []
            i = 1        
            Peer._performed_attacks = 0
            for peer in selected_peers:
                peers_types.append(mapping[self.peers[peer].peer_type])
                # print(i)
                # print('\n{}: {} Starts training in global round:{} |'.format(i, (self.peers_pseudonyms[peer]), (epoch + 1)))
                peer_local_model, peer_loss = self.peers[peer].participant_update(epoch, 
                copy.deepcopy(simulation_model),
                attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate, 
                source_class = source_class, target_class = target_class, 
                dataset_name = self.dataset_name, global_rounds = self.global_rounds)
                if rule == 'ffl':
                    ffl_local_weights[peer] = copy.deepcopy(peer_local_model).state_dict()
                elif rule == 'mkrum' or rule == 'foolsgold':
                    local_models.append(peer_local_model) 
                else:
                    local_weights.append(copy.deepcopy(peer_local_model).state_dict())
                local_losses.append(peer_loss) 
                
              
                # print('{} ends training in global round:{} |\n'.format((self.peers_pseudonyms[peer]), (epoch + 1))) 
                i+= 1
            loss_avg = sum(local_losses) / len(local_losses)
            peers_ground_truth.append(peers_types)
            print('Average of peers\' local losses: {:.6f}'.format(loss_avg))
            #aggregated global weights
            scores = np.zeros(len(selected_peers))
            # Expected malicious peers
            if rule == 'fedavg':
                cur_time = time.time()
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'median':
                    cur_time = time.time()
                    global_weights = simple_median(local_weights)
                    cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'rmedian':
                cur_time = time.time()
                global_weights = Repeated_Median_Shard(local_weights)
                cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'tmean':
                    cur_time = time.time()
                    global_weights = trimmed_mean(local_weights, trim_ratio = self.attackers_ratio)
                    cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'mkrum':
                f = int(self.attackers_ratio*len(local_models))+1
                cur_time = time.time()
                goog_updates = Krum(local_models, f = f, multi=True)
                scores[goog_updates] = 1
                global_weights = average_weights([model.state_dict() for model in local_models], scores)
                cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'foolsgold':
                cur_time = time.time()
                scores = fg.score_gradients(copy.deepcopy(simulation_model), 
                                            copy.deepcopy(local_models), 
                                            selected_peers)
                global_weights = average_weights([model.state_dict() for model in local_models], scores)
                t = time.time() - cur_time
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)
            elif rule == 'ffl':
                cur_time = time.time()
                exchange_list, pt_list = self.exchange_protocol(selected_peers, peers_types, epoch)
                mixed_updates = fragment_and_mix(ffl_local_weights, exchange_list, pt_list, strategy)
                t = time.time() - cur_time
                print('Mixing took', np.round(t, 4))
                cur_time = time.time()
                sim_dict = score_mixed_updates(mixed_updates, copy.deepcopy(simulation_model))
                self.update_reputation(sim_dict, exchange_list)
                models = []
                idxs = []
                for k, s in sim_dict.items():
                    models.append(mixed_updates[k])
                    idxs.append(k)
                if epoch < 20:
                    global_weights = simple_median(models)
                else:
                    q1 = np.quantile(self.global_reputation, 0.25)
                    v = self.global_reputation - q1
                    v = np.tanh(v)
                    v[v < 0] = 0
                    peers_trust.append(v)
                    v = v[idxs]
                    print(v)
                    global_weights = average_weights(models, v)
                t = time.time() - cur_time
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)
            
            else:
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                ##############################################################################################
            
            #plot updates
            # plot_parts(copy.deepcopy(simulation_model), copy.deepcopy(local_models), 
            #         peers_types = peers_types, source_class = source_class, 
            #         target_class = target_class, epoch=epoch)

            # update global weights
            g_model = copy.deepcopy(simulation_model)
            simulation_model.load_state_dict(global_weights)           
            if epoch >= self.global_rounds-10:
                last10_updates.append(global_weights) 

            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
         
            # print("***********************************************************************************")
            #print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())
            print('{0:10s} - {1}'.format('Class','Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                if i == source_class:
                    source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
            
            backdoor_asr = 0.0
            backdoor_pattern = None
            if attack_type == 'backdoor':
                if self.dataset_name == 'MNIST':
                    backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                            [2.8238, 2.8238, 2.8238],
                                                            [2.8238, 2.8238, 2.8238]]) 
                elif self.dataset_name == 'CIFAR10':
                    backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                        [2.5141, 2.5141, 2.5141],
                                                        [2.5141, 2.5141, 2.5141]],

                                                        [[2.5968, 2.5968, 2.5968],
                                                        [2.5968, 2.5968, 2.5968],
                                                        [2.5968, 2.5968, 2.5968]],

                                                        [[2.7537, 2.7537, 2.7537],
                                                        [2.7537, 2.7537, 2.7537],
                                                        [2.7537, 2.7537, 2.7537]]])

                backdoor_asr = self.test_backdoor(simulation_model, self.device, self.test_loader, 
                                backdoor_pattern, source_class, target_class)
            print('\nBackdoor ASR', backdoor_asr)
            
            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(),
                'global_model':g_model,
                'local_models':copy.deepcopy(local_models),
                'last10_updates':last10_updates,
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'peers_ground_truth':peers_ground_truth
                }
            savepath = './checkpoints/'+ attack_type + '_' + self.dataset_name + '_' + self.model_name + '_' + \
                self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + strategy + '.t7'
            torch.save(state,savepath)

            del local_models
            del local_weights
            del ffl_local_weights
            gc.collect()
            torch.cuda.empty_cache()

            if epoch == self.global_rounds-1:
                print('Last 10 updates results')
                global_weights = average_weights(last10_updates, 
                np.ones([len(last10_updates)]))
                simulation_model.load_state_dict(global_weights) 
                current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                global_accuracies.append(np.round(current_accuracy, 2))
                test_losses.append(np.round(test_loss, 4))
                print("***********************************************************************************")
                #print and show confusion matrix after each global round
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                classes = list(self.labels_dict.keys())
                print('{0:10s} - {1}'.format('Class','Accuracy'))
                lf_asr = 0.0
                for i, r in enumerate(confusion_matrix(actuals, predictions)):
                    print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                    if i == source_class:
                        source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
                        lf_asr = np.round(r[target_class]/np.sum(r)*100, 2)

                backdoor_asr = 0.0
                if attack_type == 'backdoor':
                    if self.dataset_name == 'MNIST':
                        backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                            [2.8238, 2.8238, 2.8238],
                                                            [2.8238, 2.8238, 2.8238]]) 
                    elif self.dataset_name == 'CIFAR10':
                        backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                            [2.5141, 2.5141, 2.5141],
                                                            [2.5141, 2.5141, 2.5141]],

                                                            [[2.5968, 2.5968, 2.5968],
                                                            [2.5968, 2.5968, 2.5968],
                                                            [2.5968, 2.5968, 2.5968]],

                                                            [[2.7537, 2.7537, 2.7537],
                                                            [2.7537, 2.7537, 2.7537],
                                                            [2.7537, 2.7537, 2.7537]]])

                    backdoor_asr = self.test_backdoor(simulation_model, self.device, self.test_loader, 
                                    backdoor_pattern, source_class, target_class)

        pt = []
        for i in range(self.num_peers):
            pt.append(self.peers[i].peer_type)
        state = {
                'state_dict': simulation_model.state_dict(),
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'lf_asr':lf_asr,
                'backdoor_asr': backdoor_asr,
                'avg_cpu_runtime':np.mean(cpu_runtimes),
                'peers_ground_truth':peers_ground_truth,
                'peers_types':pt,
                'trust':peers_trust
                }
        savepath = './results/'+ attack_type + '_' + self.dataset_name + '_' + self.model_name + '_' + \
                self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + strategy + '.t7'
        torch.save(state,savepath)    

        print('Global accuracies: ', global_accuracies)
        print('Class {} accuracies: '.format(source_class), source_class_accuracies)
        print('Test loss:', test_losses)
        print('Label-flipping attack succes rate:', lf_asr)
        print('Backdoor attack succes rate:', backdoor_asr)
        print('Average CPU aggregation runtime:', np.round(np.mean(cpu_runtimes), 3))
