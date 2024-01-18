import os

class FedConfig:
    def __init__(self, cfg=None):

        # path
        self.root_path = os.path.dirname(os.path.dirname(__file__))
        self.model_path = os.path.join(self.root_path, 'save', 'models')

        # general settings
        self.device = 'cuda:0'
        self.torch_seed = 419
        self.sample_seed = 2023

        # dataset setting
        self.dataset = 'cifar'
        # self.dataset = 'fmnist'
        self.sample_mode = 'dirichlet'
        # self.sample_mode = 'shards'
        self.num_shards = 4
        # self.num_shards = 2
        self.alpha = 0.4 # Dirichlet parameter


        self.iter = 400 # communication round
        self.num_clients = 20

        # TBFL setting
        self.num_selector = 10
        self.TBFL_select_num = 2
        self.TBFL_self_weight = 0.55

        # Gossip setting
        self.Gossip_select_num = 2
        self.Gossip_self_weight = 0.6

        # Ring setting
        self.Ring_self_weight = 0.6

        self.algorithm = 'TBFL'
        # self.algorithm = 'D-PSGD'
        # self.algorithm = 'DFedAvg'
        # self.algorithm = 'Gossip'

        # client settings
        self.train_batch_size = 64
        self.local_epoch = 4
        self.lr0 = 0.01
        self.decay = 0.997

        # test settings
        self.test_batch_size = 50
        self.test_data_size = 1000

        # save str
        if self.sample_mode == 'shards':
            self.save_str = self.sample_mode + str(self.num_shards) + '_' + self.dataset + '_' + self.algorithm
        else:
            self.save_str = self.sample_mode + str(self.alpha) + '_' + self.dataset + '_' + self.algorithm

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)

    def __str__(self):
        string = ""
        for (k, v) in self.__dict__.items():
            string += "{}:{}\n".format(k,v)

        return string