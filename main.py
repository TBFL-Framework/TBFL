from utils.configs import FedConfig
from utils.seed import set_seed
from utils.datasets import load_dataset
from utils.sample import split_dataset
from train.train import train

def main(cfg=None):
    
    if cfg is None:
        cfg = FedConfig()

    # Load dataset
    train_set, test_set = load_dataset(cfg)
    
    # Split dataset
    dict_users_train, dict_users_test, ratio = split_dataset(cfg, train_set, test_set)

    # Set seed
    set_seed(cfg)

    # Train
    train(cfg, dict_users_train, train_set, dict_users_test, test_set, ratio)

if __name__ == '__main__':
    main()