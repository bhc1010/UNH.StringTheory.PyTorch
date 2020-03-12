import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

class Net(nn.Module):
    def __init__(self, netShape, imbedDim, device):
        super().__init__()

        self.netShape = netShape
        self.imbedDim = imbedDim
        self.device = device
        self.layers = []

        ## Define network layers - currently all layers are Linear
        for i in range(len(self.netShape) - 1):
            self.layers.append(nn.Linear(self.netShape[i], self.netShape[i+1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X):
        for layer in self.layers:
            X = F.relu(layer(X))

        return F.log_softmax(X, dim=1)

    def train(self, train_data, test_data, epochNum):
        totalTime = 0
        for lr, EPOCHS in zip([0.01, 0.001],[30,epochNum]):
            print(f"---------- Learning Rate: {lr} ----------")
            optimizer = optim.Adam(self.parameters(), lr=lr)
            ## Train the network over all the batches for EPOCHS number of epochs
            for epoch in range(EPOCHS):
                start_time = time.time()
                ## For each batch, optimize the network
                for i in range(len(train_data[0])):
                    ## Send data to current device (usually GPU for training)
                    X = train_data[0][i].to(self.device)
                    Y = train_data[1][i].to(self.device)
                    self.zero_grad()
                    output = self(X.view(-1,2*self.imbedDim))
                    loss = F.nll_loss(output, Y)
                    loss.backward()
                    optimizer.step()
                end_time = time.time()
                correct = 0
                total = 0
                ## For testing, do not train the network (no gradient)
                with torch.no_grad():
                    # Iterate through test data batch by batch and count number of correct outputs
                    for i in range(len(test_data[0])):
                        X = test_data[0][i].to(self.device)
                        Y = test_data[1][i].to(self.device)
                        output = self(X.view(-1,2*self.imbedDim))
                        for idx, j in enumerate(output):
                            if torch.argmax(j) == Y[idx]:
                                correct += 1
                            total += 1
                ## Readouts
                myLoss = np.round(torch.Tensor.cpu(loss.detach()).numpy(), 4)
                myAccuracy = round(correct*100 / total,2)
                deltaT = end_time - start_time
                totalTime += deltaT
                ## Print info about current epoch
                print(f"Epoch {epoch} / {epochNum} | Loss: {myLoss} | Accuracy: {myAccuracy} | Time: {deltaT}")
        ## Print total time to train network over all epochs
        print(f"Total time: {totalTime}")

    def test(self, test_data):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(test_data[0])):
                X = test_data[0][i].to(self.device)
                Y = test_data[1][i].to(self.device)
                output = self(X.view(-1,2*self.imbedDim))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == Y[idx]:
                        correct += 1
                    total += 1
            print("Accuracy: ", correct / total)
        return correct / total
