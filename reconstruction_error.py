import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import importlib

import pickle
import os.path

if os.path.isfile('celeba.pickle'):
    print('loading dataset cached')
    dataset = pickle.load(open('celeba.pickle', 'rb'))
else:
    print('loading dataset new')
    dataset = datasets.ImageFolder(root='/sailhome/sharonz/celeba/',
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
                        )
    print('dumping datset to celeba.pickle')
    pickle.dump(dataset, open('celeba.pickle', 'wb'))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True, num_workers=int(2))

models_egan = importlib.import_module("models.models_egan_celeba")
_netG = models_egan._netG

dtype = torch.cuda.FloatTensor

k  = 128 # latent dimension
sigma = 0.01
lr = 0.1
l2_regularization = 0.001
num_iter = 500
l2loss = nn.L1Loss().cuda()
nz = 128
input = torch.FloatTensor(64, 3, 64, 64)

for i, data in enumerate(dataloader, 0):
    real_cpu, img_context = data
    image = real_cpu
    batch_size = real_cpu.size(0)
    input.resize_(real_cpu.size()).copy_(real_cpu)
    image = Variable(input)

    noise = torch.FloatTensor(64, nz, 1, 1)
    noise = noise.resize_(64, noise.size(1), noise.size(2), noise.size(3)).normal_(0, 1)
    opt_z = Variable(noise, requires_grad=True) # backprop on this
    optimizer = torch.optim.Adam([opt_z], lr=lr, weight_decay= l2_regularization)
    # load the generator
    G = _netG(nz, 3, 64) # netG is the network

    for j in range(num_iter):
       optimizer.zero_grad()
       loss = l2loss(G(opt_z), image)
       loss.backward()
       optimizer.step()
    min_loss = loss
    past_z = opt_z.clone()

    opt_z = Variable(torch.randn(10, 0).type(dtype)*sigma, requires_grad=True)
    optimizer = torch.optim.Adam([opt_z], lr=lr, weight_decay= l2_regularization)

    for j in range(num_iter):
       optimizer.zero_grad()
       loss = l2loss(G(opt_z), image)
       loss.backward()
       optimizer.step()

    if current_z < loss:
       opt_z = past_z

    fake = G(opt_z)
    vutils.save_image(fake.data,
        # '%s/celeba_reconstructed.png' % ('log'),
        'celeba_reconstructed.png',
        normalize=True)

    break



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# import torchvision.utils as vutils
# from torch.autograd import Variable
# import torch.utils.data
# import torch.backends.cudnn as cudnn

# import random
# import argparse

# # Sampled 100 Xs from CelebA dataset
# import pickle
# import os.path

# if os.path.isfile('celeba.pickle'):
#     print('loading dataset cached')
#     dataset = pickle.load(open('celeba.pickle', 'rb'))
# else:
#     print('loading dataset new')
#     dataset = datasets.ImageFolder(root='/sailhome/sharonz/celeba/',
#                             transform=transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ])
#                         )
#     print('dumping datset to celeba.pickle')
#     pickle.dump(dataset, open('celeba.pickle', 'wb'))

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
#                                          shuffle=True, num_workers=int(2))

# input = torch.FloatTensor(64, 3, 64, 64)
# for i, data in enumerate(dataloader, 0):
#     if i == 100:
#         break
#     real_cpu, _ = data

#     batch_size = real_cpu.size(0)

#     input.resize_(real_cpu.size()).copy_(real_cpu)
#     inputv = Variable(input)
#     if i == 0:
#         sampled_Xs = inputv
#     else:
#         sampled_Xs = torch.cat((sampled_Xs, inputv))
# print(sampled_Xs.size())

# # Generate initial z (noise vector)
# import importlib

# models_egan = importlib.import_module("models.models_egan_celeba")
# _netG = models_egan._netG

# nz = 128
# noise = torch.FloatTensor(64, nz, 1, 1)
# noise = noise.resize_(64, noise.size(1), noise.size(2), noise.size(3)).normal_(0, 1)
# noisev = Variable(noise) # backprop on this
# y_pred = noisev

# loss_fn = nn.MSELoss()
# G = _netG(nz, 3, 64)
# optimizer = torch.optim.SGD(G.parameters(), lr=0.1, momentum=0.9)
# for t in range(500):
#     # TODO: for each image in sampled_Xs, do SGD on it
#     output = G(noisev)

#     y = torch.zeros_like(output)
#     loss = loss_fn(output, y)
#     print(t, "Loss: ", loss.item()) #loss.data[0]

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()




# From tf approach
# layer_sizes = [50 ,200]
# hidden = input_
# prev_hidden_size = hidden.size()
# for i, hidden_size in enumerate(layer_sizes):
#     weights nn.Tensor([prev_hidden_size, hidden_size])
#     biases = nn.zeros([hidden_size])
#     hidden = nn.Relu(torch.mm(hidden, weights) + biases)
#     prev_hidden_size = hidden_size
    
# weights = nn.Tensor([prev_hidden_size, 784])
# biases = nn.zeros([784])
# logits = torch.add(torch.mm(hidden, weights), biases)
    
# prediction = nn.sigmoid(logits)
# loss = get_loss(input_, logits)
    
# return loss, prediction

# def get_loss(input_, logits):
#     assert(input_.get_shape() == logits.get_shape())
#     all_losses = F.binary_cross_entropy(logits, input_)
#     loss = nn.reduce_mean(all_losses)
#     return loss


