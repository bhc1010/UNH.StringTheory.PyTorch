import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import network2
import polytope
import dataLoader
import numpy as np

imbedDim = 5

poly0 = polytope.Polytope([(0.,1.),(-1.,-1.), (1.,0.)], 0, imbedDim)
poly1 = polytope.Polytope([(-1.,2.),(-1.,-1.),(2.,-1.)], 1, imbedDim)
poly2 = polytope.Polytope([(0.,1.),(-1.,0.),(-1.,-1.),(1.,0.)], 2, imbedDim)
poly3 = polytope.Polytope([(-1.,2.),(-1.,-1.),(1.,-1.),(1.,0.)], 3, imbedDim)
poly4 = polytope.Polytope([(0.,1.),(-1.,1.),(-1.,-1.),(1.,0.)], 4, imbedDim)
poly5 = polytope.Polytope([(-1.,2.),(-1.,-1.),(-1.,0.),(1.,0.)], 5, imbedDim)
poly6 = polytope.Polytope([(0.,1.),(-1.,0.),(-1.,-1.),(0.,-1.),(1.,0.)], 6, imbedDim)
poly7 = polytope.Polytope([(0.,1.),(-1.,1.),(-1.,-1.),(1.,-1.),(1.,0.)], 7, imbedDim)

# poly8 = polytope.Polytope([], 0, imbedDim)
# poly9 = polytope.Polytope([(-1.,2.),(-1.,-1.),(2.,-1.)], 1, imbedDim)
# poly10 = polytope.Polytope([(0.,1.),(-1.,0.),(-1.,-1.),(1.,0.)], 2, imbedDim)
# poly11 = polytope.Polytope([(-1.,2.),(-1.,-1.),(1.,-1.),(1.,0.)], 3, imbedDim)
# poly12 = polytope.Polytope([(0.,1.),(-1.,1.),(-1.,-1.),(1.,0.)], 4, imbedDim)
# poly13 = polytope.Polytope([(-1.,2.),(-1.,-1.),(-1.,0.),(1.,0.)], 5, imbedDim)
# poly14 = polytope.Polytope([(0.,1.),(-1.,0.),(-1.,-1.),(0.,-1.),(1.,0.)], 6, imbedDim)
# poly15 = polytope.Polytope([(0.,1.),(-1.,1.),(-1.,-1.),(1.,-1.),(1.,0.)], 7, imbedDim)

polytopes = [poly0, poly1, poly2, poly3]
# polytopes = [poly0, poly1]

for poly in polytopes:
    poly.generateData()
    for i in range(len(poly.translations)):
        print(len(poly.translations[i]))

## Note batch_size must divide evenly into len(train_data_raw)
train_batches, train_labels, test_batches, test_labels = dataLoader.loadData(polytopes, batch_size=12)

net = network2.Net([2 * imbedDim, 14, 8])

count = 0

for lr, EPOCHS in zip([0.001],[1000]):
    print("---------- Learning Rate: {} ----------".format(lr))
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # EPOCHS = 500
    for epoch in range(EPOCHS):
        for X, Y in zip(train_batches, train_labels):
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
                output = net(X.view(-1,2*imbedDim))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == Y[idx]:
                        correct += 1
                    total += 1
        print("Epoch {} / {} | Loss: {} | Accuracy: {}".format(count, 100, np.round(loss.detach().numpy(), 4), round(correct*100 / total,2)))

correct = 0
total = 0

with torch.no_grad():
    for X, Y in zip(test_batches, test_labels):
        output = net(X.view(-1,2*imbedDim))
        for idx, i in enumerate(output):
            if torch.argmax(i) == Y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", correct / total)

if (correct*100 / total) >= 80:
    torch.save(net.state_dict(),'model4.pt')


## Check to see which polytopes the model is getting incorrect most often
## Filling out smaller data sets with duplicates to normalize classes
## Try it on all 16 (with and without duplicates)
