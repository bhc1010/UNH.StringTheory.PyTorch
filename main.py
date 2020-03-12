import torch
from polytope import Polytope
from dataLoader import loadData
from network2 import Net

imbedDim = 6

poly0 = Polytope([(0.,1.),(-1.,-1.), (1.,0.)], 0, imbedDim)
poly1 = Polytope([(-1.,2.),(-1.,-1.),(2.,-1.)], 1, imbedDim)
poly2 = Polytope([(0.,1.),(-1.,0.),(-1.,-1.),(1.,0.)], 2, imbedDim)
poly3 = Polytope([(-1.,2.),(-1.,-1.),(1.,-1.),(1.,0.)], 3, imbedDim)
poly4 = Polytope([(0.,1.),(-1.,1.),(-1.,-1.),(1.,0.)], 4, imbedDim)
poly5 = Polytope([(-1.,2.),(-1.,-1.),(-1.,0.),(1.,0.)], 5, imbedDim)
poly6 = Polytope([(0.,1.),(-1.,0.),(-1.,-1.),(0.,-1.),(1.,0.)], 6, imbedDim)
poly7 = Polytope([(0.,1.),(-1.,1.),(-1.,-1.),(1.,-1.),(1.,0.)], 7, imbedDim)

# poly8 = polytope.Polytope([], 0, imbedDim)
# poly9 = polytope.Polytope([(-1.,2.),(-1.,-1.),(2.,-1.)], 1, imbedDim)
# poly10 = polytope.Polytope([(0.,1.),(-1.,0.),(-1.,-1.),(1.,0.)], 2, imbedDim)
# poly11 = polytope.Polytope([(-1.,2.),(-1.,-1.),(1.,-1.),(1.,0.)], 3, imbedDim)
# poly12 = polytope.Polytope([(0.,1.),(-1.,1.),(-1.,-1.),(1.,0.)], 4, imbedDim)
# poly13 = polytope.Polytope([(-1.,2.),(-1.,-1.),(-1.,0.),(1.,0.)], 5, imbedDim)
# poly14 = polytope.Polytope([(0.,1.),(-1.,0.),(-1.,-1.),(0.,-1.),(1.,0.)], 6, imbedDim)
# poly15 = polytope.Polytope([(0.,1.),(-1.,1.),(-1.,-1.),(1.,-1.),(1.,0.)], 7, imbedDim)

polytopes = [poly0, poly1, poly2, poly3, poly4, poly5, poly6, poly7]

## Note batch_size must divide evenly into len(train_data_raw)
train_data, test_data = loadData(polytopes, batch_size=12)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# device = torch.device("cpu")
EPOCHS = 100

net = Net([2 * imbedDim, 10, 8], imbedDim, device).to(device)
net.train(train_data, test_data, EPOCHS)
accuracy = net.test(test_data)

if accuracy * 100 >= 80:
    torch.save(net.state_dict(),'model4.pt')


## Check to see which polytopes the model is getting incorrect most often
## Filling out smaller data sets with duplicates to normalize classes
## Try it on all 16 (with and without duplicates)
