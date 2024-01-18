import numpy as np
from fedlab.utils.dataset.partition import CIFAR10Partitioner

def noniid_train(data, cfg):
    if cfg.sample_mode == 'dirichlet':
        partition = CIFAR10Partitioner(targets=data.targets, 
                                       num_clients=cfg.num_clients, 
                                       balance=None, 
                                       partition='dirichlet', 
                                       dir_alpha=cfg.alpha, 
                                       seed=cfg.sample_seed)
    elif cfg.sample_mode == 'shards':
        partition = CIFAR10Partitioner(targets=data.targets, 
                                       num_clients=cfg.num_clients, 
                                       balance=None, 
                                       partition='shards',
                                       num_shards=cfg.num_clients * cfg.num_shards,
                                       seed=cfg.sample_seed)
    else:
        exit('Error: no such sample method...')
    
    dict_users_train = partition.client_dict
    client_data_num = np.array(partition.client_sample_count).reshape(-1)

    ratio = np.zeros((cfg.num_clients, 10))
    labels = np.array(data.targets)

    for i in range(cfg.num_clients):
        count = np.zeros(10)
        for j in dict_users_train[i]:
            count[labels[j]] += 1
        ratio[i] = count / client_data_num[i]

    return dict_users_train, ratio

def noniid_test(dataset, num_users, ratio, test_data_size):
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}

    labels = np.array(dataset.targets)

    # {label -> [idx]}
    bucket = [[] for _ in range(10)]
    for i in range(len(labels)):
        bucket[labels[i]].append(i)

    for i in range(num_users):
        total = test_data_size
        for j in range(9):
            num = int(ratio[i][j] * test_data_size)
            rand_idxs = np.random.choice(bucket[j], num, replace=False)
            dict_users_test[i] = np.concatenate((dict_users_test[i], rand_idxs), axis=0)
            total -= num
        rand_idxs = np.random.choice(bucket[9], total, replace=False)
        dict_users_test[i] = np.concatenate((dict_users_test[i], rand_idxs), axis=0)

    return dict_users_test

def split_dataset(cfg, train_set, test_set):
    print('Sample by ' + cfg.sample_mode + ' method...')

    dict_users_train, ratio = noniid_train(train_set, cfg)

    dict_users_test = noniid_test(test_set, cfg.num_clients, ratio, cfg.test_data_size)

    print('Split finished...')

    return dict_users_train, dict_users_test, ratio