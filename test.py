import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import polytope
import dataLoader
import numpy as np

imbedDim = 25


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

for poly in polytopes:
    poly.generateData()

## Note batch_size must divide evenly into len(train_data_raw)
train_batches, train_labels, test_batches, test_labels = dataLoader.loadData(polytopes, batch_size=12)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the gpu")
else:
    device = torch.device("cpu")
    print("Running on the cpu")
    
# device = torch.device("cpu")
net = Net([2 * imbedDim, 25,25,25, 8]).to(device)
net.train(10)
accuracy = net.test()

if accuracy * 100 >= 80:
    torch.save(net.state_dict(),'model4.pt')


## Check to see which polytopes the model is getting incorrect most often
## Filling out smaller data sets with duplicates to normalize classes
## Try it on all 16 (with and without duplicates)
