from environment_federated import *


def run_exp(dataset_name, model_name,
    num_peers, frac_peers, seed, criterion, global_rounds, 
    local_epochs, local_bs, local_lr , local_momentum , labels_dict, device, 
    attackers_ratio, attack_type, malicious_behavior_rate, from_class, 
    to_class, number_of_attacks, rule, fragment = False):
    print('\n===> Starting experiment...')
    fflEnv = FFLEnv(dataset_name = dataset_name, model_name = model_name, num_peers = num_peers, 
    frac_peers = frac_peers, seed = seed, criterion = criterion, global_rounds = global_rounds, 
    local_epochs = local_epochs, local_bs = local_bs, local_lr = local_lr,
     local_momentum = local_momentum, 
    labels_dict = labels_dict, device = device, attackers_ratio = attackers_ratio)
    print('\n===> Start of a simulation')
    if fragment:
        print('Fragmented Federated Learning (FFL)')
    else:
        print('Standard Federated Learning (FL)')
    print('Aggregation rule:', rule)
    print('Attack Type:', attack_type)
    print('Attackers Ration:', attackers_ratio*100, '%')
    print('Malicious Behavior Rate:', malicious_behavior_rate*100, '%')
    fflEnv.simulate(attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate,
                    from_class = from_class, to_class = to_class,
                     number_of_attacks = number_of_attacks, rule=rule, fragment=fragment)
    print('\n===> End of Simulation.')
