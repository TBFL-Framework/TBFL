import numpy as np
import os
from torch.utils.data import DataLoader
from utils.datasets import DatasetSplit

import torch.nn as nn
import torch

class Client:
    def __init__(self, cfg, train_set, train_idxs, test_set, test_idxs, idx, net):
        self.idx = idx
        self.cfg = cfg
        self.train_set = DataLoader(DatasetSplit(train_set, train_idxs), batch_size=cfg.train_batch_size, shuffle=True)
        self.test_set = DataLoader(DatasetSplit(test_set, test_idxs), batch_size=cfg.test_batch_size, shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        self.avg_loss = []
        self.model_path = os.path.join(cfg.model_path, str(idx) + '.pt')
        self.half_model_path = os.path.join(cfg.model_path, str(idx) + '_half.pt')
        self.gradient_path = os.path.join(cfg.model_path, str(idx) + '_grad.pt')
        torch.save(net.state_dict(), self.model_path)

    def local_train(self, t, net):
        net.load_state_dict(torch.load(self.model_path, map_location=self.cfg.device))
        
        net.train()

        lr = self.cfg.lr0 * (self.cfg.decay ** t)

        optimizer = torch.optim.SGD(net.parameters(), lr)

        train_loss = []

        for _ in range(self.cfg.local_epoch):
            for images, labels in self.train_set:
                images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
        
        torch.save(net.state_dict(), self.half_model_path)

        self.avg_loss.append(np.mean(train_loss))

        if self.cfg.algorithm == 'D-PSGD':
            pre_model = torch.load(self.model_path, map_location=self.cfg.device)
            gradient = {}
            for k in net.state_dict().keys():
                gradient[k] = net.state_dict()[k] - pre_model[k]
            torch.save(gradient, self.gradient_path)

    def test_model(self, net, w):
        net.load_state_dict(w)

        net.eval()
 
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_set:
                images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
                outputs = net(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total

        return acc
    
    def select(self, be_selected, select_num):
        return np.random.choice(be_selected, size=select_num, replace=False)

    def aggregate(self, weight, client, net):
        agg = {}
        for k in net.state_dict().keys():
            agg[k] = torch.zeros_like(net.state_dict()[k])

        for i in range(self.cfg.num_clients):
            if weight[i] != 0:
                if self.cfg.algorithm == 'D-PSGD':
                    w = torch.load(client[i].model_path, map_location=self.cfg.device)
                else:
                    w = torch.load(client[i].half_model_path, map_location=self.cfg.device)
                    
                for k in net.state_dict().keys():
                    agg[k] += weight[i] * w[k]

        if self.cfg.algorithm == 'D-PSGD':
            gradient = torch.load(self.gradient_path, map_location=self.cfg.device)
            for k in net.state_dict().keys():
                agg[k] += gradient[k]
        
        torch.save(agg, self.model_path)
