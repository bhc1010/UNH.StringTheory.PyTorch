import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

    def train(self, epochNum):
        count = 0
        for lr, EPOCHS in zip([0.001],[epochNum]):
            print("---------- Learning Rate: {} ----------".format(lr))
            optimizer = optim.Adam(net.parameters(), lr=lr)
            # EPOCHS = 500
            for epoch in range(EPOCHS):
                for X, Y in tqdm(zip(train_batches, train_labels), total=len(train_labels)):
                    X, Y = X.to(device), Y.to(device)
                    net.zero_grad()
                    output = net(X.view(-1,2*imbedDim))
                    loss = F.nll_loss(output, Y)
                    loss.backward()
                    optimizer.step()
                count += 1
                correct = 0
                total = 0
                with torch.no_grad():
                    for X, Y in zip(test_batches, test_labels):
                        X, Y = X.to(device), Y.to(device)
                        output = net(X.view(-1,2*imbedDim))
                        for idx, i in enumerate(output):
                            if torch.argmax(i) == Y[idx]:
                                correct += 1
                            total += 1
                print("Epoch {} / {} | Loss: {} | Accuracy: {}".format(count, epochNum, np.round(torch.Tensor.cpu(loss.detach()).numpy(), 4), round(correct*100 / total,2)))

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for X, Y in zip(test_batches, test_labels):
                X, Y = X.to(device), Y.to(device)
                output = net(X.view(-1,2*imbedDim))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == Y[idx]:
                        correct += 1
                    total += 1
            print("Accuracy: ", correct / total)
        return correct / total
