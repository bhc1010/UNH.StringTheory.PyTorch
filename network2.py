import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, netShape):
        super().__init__()

        self.netShape = netShape
        self.layers = []

        for i in range(len(self.netShape) - 1):
            self.layers.append(nn.Linear(self.netShape[i], self.netShape[i+1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X):
        for layer in self.layers:
            X = F.relu(layer(X))


        return F.log_softmax(X, dim=1)

    def train(self):
        pass
