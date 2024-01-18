import torch
import os
import numpy as np
from roles.client import Client
from models.nets import MLP_FMNIST, AlexNet_CIFAR

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def bipartite(cfg, ratio):
    idx = np.random.choice(cfg.num_clients, 1).item()

    print('idx = ' + str(idx))

    sim = [torch.cosine_similarity(torch.from_numpy(ratio[idx]), torch.from_numpy(ratio[i]), dim=0).item() for i in range(cfg.num_clients)]
    
    selector = np.random.choice(np.arange(cfg.num_clients), cfg.num_selector, replace=False, p=softmax(sim))

    be_selected = np.setdiff1d(range(cfg.num_clients), selector)

    print('Selector: ', selector)
    print('Be_selected: ', be_selected)

    return selector, be_selected

def log_and_save(cfg, record, t):
    # Log
    print('Round {:2d}:  loss  {:.6f},  acc mean  {:.6f},  acc var  {:.6f},  acc min  {:.6f},  acc max  {:.6f}'.format(t + 1, record['loss'][-1], record['acc_mean'][-1], record['acc_var'][-1], record['acc_min'][-1], record['acc_max'][-1]))
    with open(os.path.join(cfg.root_path, 'log', 'log_' + cfg.save_str + '.txt'), 'a+') as f:
        print('Round {:2d}:  loss  {:.6f},  acc mean  {:.6f},  acc var  {:.6f},  acc min  {:.6f},  acc max  {:.6f}'.format(t + 1, record['loss'][-1], record['acc_mean'][-1], record['acc_var'][-1], record['acc_min'][-1], record['acc_max'][-1]), file=f)
    
    # Save info
    if (t + 1) % 100 == 0:
        np.save(os.path.join(cfg.root_path, 'save', 'data', cfg.save_str + str(t + 1) + '.npy'), record)

def loss_and_acc(cfg, record, client, net):
    record['loss'].append(np.mean([client[i].avg_loss[-1] for i in range(cfg.num_clients)]))

    para_avg = {}
    for k in net.state_dict().keys():
        para_avg[k] = torch.zeros_like(net.state_dict()[k])
    for i in range(cfg.num_clients):
        w = torch.load(client[i].model_path, map_location=cfg.device)
        for k in net.state_dict().keys():
            para_avg[k] += w[k]
    for k in net.state_dict().keys():
        para_avg[k] = torch.div(para_avg[k], cfg.num_clients)

    acc = [client[i].test_model(net, para_avg) for i in range(cfg.num_clients)]
    record['acc_mean'].append(np.mean(acc))
    record['acc_var'].append(np.var(acc))
    record['acc_min'].append(np.min(acc))
    record['acc_max'].append(np.max(acc))

def calculate_weight(cfg, client, ratio):
    weight = np.zeros((cfg.num_clients, cfg.num_clients))

    if cfg.algorithm == 'D-PSGD' or cfg.algorithm == 'DFedAvg':
        for i in range(cfg.num_clients):
            weight[i][(i - 1 + cfg.num_clients) % cfg.num_clients] = weight[i][(i + 1) % cfg.num_clients] = (1 - cfg.Ring_self_weight) / 2
            weight[i][i] = cfg.Ring_self_weight
    
    elif cfg.algorithm == 'TBFL':
        selector, be_selected = bipartite(cfg, ratio)
        
        d = {i: [] for i in be_selected}

        for i in selector:
            selected_client = client[i].select(be_selected, cfg.TBFL_select_num)
            for j in selected_client:
                d[j].append(i)
        
        for i in be_selected:
            if len(d[i]) == 0:
                weight[i][i] = 1
            else:
                weight[i][i] = cfg.TBFL_self_weight
                for j in d[i]:
                    weight[i][j] = weight[j][i] = (1 - cfg.TBFL_self_weight) / len(d[i])

        for i in selector:
            weight[i][i] = 1 - np.sum(weight[i])
            assert weight[i][i] > 0
    
    elif cfg.algorithm == 'Gossip':
        d = {i: [] for i in range(cfg.num_clients)}

        for i in range(cfg.num_clients):
            selected_client = client[i].select(np.delete(np.arange(cfg.num_clients), i), cfg.Gossip_select_num)
            for j in selected_client:
                d[j].append(i)

        for i in range(cfg.num_clients):
            if len(d[i]) == 0:
                weight[i][i] = 1
            else:
                weight[i][i] = cfg.Gossip_self_weight
                for j in d[i]:
                    weight[i][j] = (1 - cfg.Gossip_self_weight) / len(d[i])
    
    else:
        exit('Error: no such algorithm...')

    return weight


def train(cfg, dict_users_train, train_set, dict_users_test, test_set, ratio):
    print('Start ' + cfg.algorithm + ' training...')
    
    net = AlexNet_CIFAR().to(cfg.device) if cfg.dataset == 'cifar' else MLP_FMNIST().to(cfg.device)
    
    client = [Client(cfg, train_set, dict_users_train[i], test_set, dict_users_test[i], i, net) for i in range(cfg.num_clients)]

    record = {'loss': [], 'acc_mean': [], 'acc_var': [], 'acc_max': [], 'acc_min': []}

    for t in range(cfg.iter):
        # Local train
        for i in range(cfg.num_clients):
            client[i].local_train(t, net)

        # Communication
        weight = calculate_weight(cfg, client, ratio)

        for i in range(cfg.num_clients):
            client[i].aggregate(weight[i], client, net)

        # Record
        loss_and_acc(cfg, record, client, net)

        # Log and save info
        log_and_save(cfg, record, t)
    
    print('Train ' + cfg.algorithm + ' finished...')
    return