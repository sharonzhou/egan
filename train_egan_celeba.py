import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torchvision import datasets, transforms
from torchvision import transforms
from imagefolder import ImageFolder
#from vision_egan.torchvision import datasets, transforms
import torchvision.utils as vutils
#import vision_egan.torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn

import random
import argparse
import numpy as np
from os import path
import json

parser = argparse.ArgumentParser(description='train SNDCGAN model')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda', type=bool, default=True, help='is cuda enabled')
parser.add_argument('--gpu_ids', default=range(4), help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--gpunum', default=3, help='gpu num: e.g. 0')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--earlyterminate', type=bool, default=False, help='use early termination')
parser.add_argument('--presetlearningrate', type=bool, default=True, help='use preset learning rate')
#parser.add_argument('--numdiscriminators', type=int, default=10, help='number of discriminators in the gang')
#parser.add_argument('--discriminators', type=str, default='0123456789', help='list of enabled discriminators')
parser.add_argument('--discriminators', type=str, default='0123456789', help='list of enabled discriminators')
parser.add_argument('--n_dis', type=int, default=5, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=88, help='dimension of lantent noise')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--datadir', type=str, default='/sailhome/sharonz/celeba/full/', help='data directory')
parser.add_argument('--model', type=str, default='models_egan_celeba', help='training batch size')

opt = parser.parse_args()
print(opt)

import datetime

logdir = 'logs_' + datetime.date.strftime(datetime.datetime.now(), '%Y%m%d_%H:%M:%S')

import os

os.makedirs(logdir)

from copy_info_to_logdir import copy_info_to_logdir
copy_info_to_logdir(logdir)

import importlib

models_egan = importlib.import_module("models." + opt.model)
_netG = models_egan._netG
_netC = models_egan._netC
_netE = models_egan._netE
_netD_list = models_egan._netD_list

def generate_learning_rate():
  base = random.randrange(3, 6)
  return 10 ** (-base)

discriminator_indexes_enabled = [int(x) for x in opt.discriminators]
new_netD_list = []
for idx in discriminator_indexes_enabled:
  new_netD_list.append(_netD_list[idx])
_netD_list = new_netD_list

#nd = int(opt.numdiscriminators)
nd = len(_netD_list)
#nd = len(_netD_list)
lr_G = generate_learning_rate()
lr_E = generate_learning_rate()
lr_list = [generate_learning_rate() for x in range(nd)]
if opt.presetlearningrate:
  # start this set of learning rates we have found to work well with the 10 discriminators in models_egan_celeba and BCELoss
  lr_list = [0.001, 0.000002, 0.0002, 0.000001, 0.0002, 0.003, 0.0002, 0.00001, 0.0001, 0.00001]
  lr_G = 0.0002
  lr_E = 0.0002
  # end
  #lr_list = [0.00001] * 10
  #lr_list = [0.000005] * 10
  #lr_G = 0.00005
  #lr_E = 0.00005


hyperparameters = {
  'lr_list': lr_list,
  'lr_G': lr_G,
  'lr_E': lr_E,
  'nd': nd,
  'discriminators': opt.discriminators,
}

_netD_list = _netD_list[:nd]

from copy_info_to_logdir import copy_info_to_logdir, copy_hyperparameters_to_logdir
copy_info_to_logdir(logdir)
copy_hyperparameters_to_logdir(logdir, hyperparameters)

opts_file = open(path.join(logdir, 'opts.txt'), 'wt')
opts_file.write(str(opt))
opts_file.close()

loss_outfile = open(path.join(logdir, 'losses.jsonl'), 'wt')

import pickle
import os.path

if os.path.isfile('celeba.pickle'):
    print('loading dataset cached')
    dataset = pickle.load(open('celeba.pickle', 'rb'))
else:
    print('loading dataset new')
    dataset = ImageFolder(root=opt.datadir,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
                        )
    print('dumping datset to celeba.pickle')
    pickle.dump(dataset, open('celeba.pickle', 'wb'))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
                                         shuffle=False, num_workers=int(2))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    #gpu_id = random.choice(opt.gpu_ids)
    gpu_id = opt.gpu_ids[int(opt.gpunum)]
    torch.cuda.set_device(gpu_id)

cudnn.benchmark = True

from csv import DictReader
import sys

filename_to_context_vector = {}
list_of_context_vectors = []
context_feature_names = None

for line in DictReader(open(opt.datadir + 'attr.csv')):
    #print(line)
    filename = line['File_Name']
    context_vector = []
    context_feature_names_current = []
    for k,v in line.items(): # this is an ordereddict so k is in alphabetical order
        if k == 'File_Name':
            continue
        #print(k + ' is ' + v)
        if v == '-1':
            context_vector.append(0)
        elif v == '1':
            context_vector.append(1)
        else:
            print('invalid value in attr: ' + v)
            sys.exit()
        context_feature_names_current.append(k)
        #context_vector.append(v)
    if context_feature_names == None:
        context_feature_names = context_feature_names_current
    else:
        if context_feature_names != context_feature_names_current:
            print('invalid ordering of context feature names')
            sys.exit()
    #sys.exit()
    #context_vector = torch.from_numpy(np.array(context_vector)).float().cuda()
    filename_to_context_vector[filename] = context_vector
    list_of_context_vectors.append(context_vector)

print('context feature names are')
print(context_feature_names)
context_vector_length = len(context_feature_names)
print('context vector length is')
print(context_vector_length)

'''
def generate_fake_context_vector():
    output = []
    for x in range(0, context_vector_length):
        val = 0
        if random.random() < 0.5:
            val = 1
        output.append(val)
    return output
'''

def generate_fake_context_vector():
    return random.choice(list_of_context_vectors)

def generate_fake_context_tensor(batch_size):
    output = [generate_fake_context_vector() for x in range(batch_size)]
    return torch.from_numpy(np.array(output)).float().cuda()

print('sample fake context vector is')
print(generate_fake_context_vector())
print('sample fake context length is')
print(len(generate_fake_context_vector()))

def get_context_vector_from_filename(filename):
    return filename_to_context_vector[filename]


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

n_dis = opt.n_dis
nz = opt.nz

losses_list = ['W', 'BCE', 'W', 'BCE', 'W', 'BCE', 'W', 'BCE', 'W', 'BCE'][:nd]
#losses_list = ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'][:nd]
#losses_list = ['BCE', 'BCE', 'BCE', 'BCE', 'BCE', 'BCE', 'BCE', 'BCE', 'BCE', 'BCE'][:nd]
include_sigmoid_list = [(x == 'BCE') for x in losses_list]
print('include_sigmoid_list is')
print(include_sigmoid_list)

G = _netG(nz, 3, opt.batchsize, context_vector_length)
C = _netC(3, opt.batchsize, context_vector_length)
SND_list = [_netD_x(3, opt.batchsize, include_sigmoid) for _netD_x,include_sigmoid in zip(_netD_list, include_sigmoid_list)]
nd = len(SND_list)
E = _netE(3, opt.batchsize, nd, context_vector_length)
print(G)
print(E)
G.apply(weight_filler)
for SNDx in SND_list:
    SNDx.apply(weight_filler)
E.apply(weight_filler)

input = torch.FloatTensor(opt.batchsize, 3, 64, 64)
noise = torch.FloatTensor(opt.batchsize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchsize, nz, 1, 1).normal_(0, 1)
label_g = torch.FloatTensor(opt.batchsize)
label_real = torch.FloatTensor(opt.batchsize)
label_fake = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0

fixed_noise = Variable(fixed_noise)
# TODO: Wasserstein distance - look at Wasserstein distance GAN 
# (D wants to maximize the range of/distance btw real - fake)
# D wants to max this distance: D(image) - D(G(z))
# generator G wants to maximize D(G(z))
criterion = nn.BCELoss() 
def w_loss_func_G(D_fake):
  return -torch.mean(D_fake)
  #return torch.exp(torch.mean(D_fake))

def w_loss_func_D(D_real, D_fake):
  return -(torch.mean(D_real) - torch.mean(D_fake))
  #return torch.exp(torch.mean(D_real) - torch.mean(D_fake))

dtype = torch.FloatTensor

# TODO: hyperparam tuning here
alpha = 0.1
uniform = torch.ones((opt.batchsize, nd)).type(dtype) / nd

if opt.cuda:
    G.cuda()
    C.cuda()
    for SNDx in SND_list:
        SNDx.cuda()
    E.cuda()
    criterion.cuda()
    input, label_real, label_fake, label_g = input.cuda(), label_real.cuda(), label_fake.cuda(), label_g.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    dtype = torch.cuda.FloatTensor
    uniform = uniform.cuda()

optimizerG = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.9))
#optimizerG = optim.SGD(G.parameters(), lr=0.0002)
# TODO change back to nozero
#optimizerG = optim.Adam(G.parameters(), lr=0.0, betas=(0, 0.9))
optimizerSND_list = []
# TODO: hyperparam tuning here
#lr_list = [0.001, 0.000002, 0.0002, 0.000001, 0.0002, 0.003, 0.0002, 0.00001, 0.0001, 0.00001][:nd]
for [SNDx, lrx] in zip(SND_list, lr_list):
    optimizerSNDx = optim.Adam(SNDx.parameters(), lr=lrx, betas=(0, 0.9))
    optimizerSND_list.append(optimizerSNDx)
# TODO change back to nonzero
optimizerE = optim.Adam(E.parameters(), lr=lr_E, betas=(0, 0.9))

kl_div_fcn = nn.KLDivLoss().cuda()
l2_fcn = nn.MSELoss().cuda()


loss_D_history = []
loss_G_history = []
num_increases_tolerated = 20

def append_and_ensure_length(arr, item):
  arr.append(item)
  while len(arr) > num_increases_tolerated:
    arr.pop(0)

def is_increasing_too_much(arr):
  if len(arr) < num_increases_tolerated:
    return False
  prev = arr[0]
  for x in arr[1:]:
    if x < prev:
      return False
    prev = x
  print('is_increasing_too_much is true for')
  print(arr)
  return True

for epoch in range(200):
    for i, data in enumerate(dataloader, 0):
        step = epoch * len(dataloader) + i
        real_cpu, filenames = data
        img_context =  torch.from_numpy(np.array([get_context_vector_from_filename(x) for x in filenames])).float().cuda()

        batch_size = real_cpu.size(0)

        input.resize_(real_cpu.size()).copy_(real_cpu)
        label_real.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv_real = Variable(label_real)
        ############################
        # (1) Update D_i networks: maximize log(D_i(x)) + log(1 - D_i(G(z)))
        ###########################
        # train with real
        for SNDx in SND_list:
            SNDx.zero_grad()

        classes_predicted_real = C(inputv)
        loss_C_real = criterion(classes_predicted_real, img_context)
        loss_C_real.backward(retain_graph=True)

        #loss_C_real_clone = loss_C_real.clone()
    
        # train with fake
        fake_context_vector = generate_fake_context_tensor(batch_size)
        
        # moved
        noisev = Variable(noise)
        fake = G(noisev, fake_context_vector) # fake context vecot should be passed here
        labelv_fake = Variable(label_fake.fill_(fake_label))
        label_fake.resize_(batch_size).fill_(fake_label)
        labelv_fake = Variable(label_fake) #Variable(label_fake.fill_(fake_label))
        # moved

        noise.resize_(batch_size, noise.size(1), noise.size(2), noise.size(3)).normal_(0, 1)
        loss_Ds_real = torch.zeros((batch_size, nd)).type(dtype)
        loss_Ds_fake = torch.zeros((batch_size, nd)).type(dtype)
        for j, SNDx in enumerate(SND_list):
            if losses_list[j] == 'BCE':
                loss_Ds_real[:,j] = criterion(SNDx(inputv), labelv_real)
                loss_Ds_fake[:,j] = criterion(SNDx(fake.detach()), labelv_fake)
            else:
                loss_Ds_real[:,j] = torch.mean(nn.Softplus()(SNDx(inputv)))
                loss_Ds_fake[:,j] = torch.mean(nn.Softplus()(-SNDx(fake.detach())))

        # TODO: add context - see conditional GANs (w/ classifier)
        W_real = E(inputv, nd, img_context) # batchsize x nd

        # just for printing and logging
        Wmeans_real = torch.mean(W_real, dim=0)
        bestD_real = torch.argmax(Wmeans_real)

        kl_div = - alpha * torch.mean(torch.log(W_real))
        loss_E = nd * (torch.mean(torch.mul(W_real, loss_Ds_real.detach() ) ) + kl_div)
        loss_E.backward(retain_graph=True)
        optimizerE.step()

        #loss_D = nd * ( torch.mean(loss_Ds_real) + torch.mean(loss_Ds_fake) )
        #loss_D.backward(retain_graph=True)
        

        E_G_z1 = loss_E.clone()

        # for optimizerSNDx in optimizerSND_list:
        #     optimizerSNDx.step()

        # train with fake
        # noisev = Variable(noise)
        # fake = G(noisev, fake_context_vector) # fake context vecot should be passed here
        # labelv_fake = Variable(label.fill_(fake_label))
        
        classes_predicted_fake = C(fake)
        loss_C_fake = criterion(classes_predicted_fake, fake_context_vector)
        loss_C_fake.backward(retain_graph=True)

        # loss_Ds = torch.zeros((batch_size, nd)).type(dtype)
        # for j, SNDx in enumerate(SND_list):
        #     loss_Ds[:,j] = criterion(SNDx(fake.detach()), labelv_fake)
            #fakeD = SNDx(fake.detach())
            #realD = SNDx(inputv)
            #loss_Ds[:,j] = w_loss_func_D(realD, fakeD)

        #fake_context_vector = [generate_fake_context_vector() for x in range(batch_size)]
        fake_context_vector = generate_fake_context_tensor(batch_size)

        W_fake = E(fake, nd, fake_context_vector) # batchsize x nd
        loss_D_real = nd * (torch.mean(loss_Ds_real))
        #loss_D_real.backward(retain_graph=True)
        loss_D_fake = nd * (torch.mean(loss_Ds_fake))
        loss_D_both = loss_D_real + loss_D_fake
        print(loss_D_both, loss_D_real, loss_D_fake)
        loss_D_both.backward(retain_graph=True)

        E_G_z2 = loss_E.clone()
        D_G_z2 = loss_D_fake.clone()
        D_G_z1 = loss_D_real.clone()


        for optimizerSNDx in optimizerSND_list:
            optimizerSNDx.step()

        ############################
        # (2) Run E network: given X and context c, output weights W of length len(D_i)
        # (dist/pdf of action space)
        # multiply p_i * W to get final output o
        ###########################
        # img_context = img_context.float().cuda() # size 64
        # img_context.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1) # 64 x 1 x 1 x 1
        # img_context = img_context.expand(-1, inputv.size()[1], inputv.size()[2], inputv.size()[3]) # 64 x 3 x 64 x 64

        # W = E(inputv, nd, EPSILON) # batchsize x nd
        # W = E(inputv, img_context, nd, EPSILON) # batchsize x nd
        # W = torch.sum(W, dim=0) # nd
        # W = torch.div(W, W.sum()) # normalize weights (sum to 1)

        ############################
        # (3) Update G network: maximize log(D(G(z))*E(X,c))
        ###########################
        if step % n_dis == 0:
            G.zero_grad()
            label_g.resize_(batch_size).fill_(real_label)
            labelv_g = Variable(label_g)  # fake labels are real for generator cost
            #labelv_g = Variable(label_g.fill_(real_label))  # fake labels are real for generator cost

            loss_Ds_g = torch.zeros((batch_size, nd)).type(dtype)
            for j, SNDx in enumerate(SND_list):
                if losses_list[j] == 'BCE':
                    loss_Ds_g[:,j] = criterion(SNDx(fake), labelv_g)
                else:
                    loss_Ds_g[:,j] = torch.mean(nn.Softplus()(-SNDx(fake)))

            #W = E(fake, nd, fake_context_vector) # batchsize x nd
            #loss_G = nd * torch.mean(torch.mul(W, loss_Ds)) 
            Wmeans_fake = torch.mean(W_fake, dim=0)
            bestD_fake = torch.argmax(Wmeans_fake)
            print('bestD_fake is: ' + str(bestD_fake))
            #bestD = 0
            loss_G = torch.mean(loss_Ds_g[bestD_fake])
            #loss_G = w_loss_func_G()
            loss_G.backward(retain_graph=True)

            optimizerG.step()

        if i % 20 == 0:
            # print(W)
            message = '[' + str(epoch) + '/' + str(200) + '][' + str(i) + '/' + str(len(dataloader)) + ']'
            message += ' Loss_D: ' + ('{:.4f}'.format(torch.mean(loss_D_real)))
            message += ' Loss_G: ' + ('{:.4f}'.format(loss_G.data.cpu().numpy())) 
            message += ' E(G(z)): ' + ('{:.4f}'.format(E_G_z1.data.cpu().numpy())) + ' / ' + ('{:.4f}'.format(E_G_z2.data.cpu().numpy()))
            message += ' D(G(z)): ' + ('{:.4f}'.format(D_G_z1.data.cpu().numpy())) + ' / ' + ('{:.4f}'.format(D_G_z2.data.cpu().numpy()))
            #message += ' KL: ' + ('{:.4f}'.format(kl_div))
            message += ' ' + logdir
            print(message)

            #print('[%d/%d][%d/%d] Loss_D1: %.4f Loss_D2: %.4f Loss_D3: %.4f Loss_G: %.4f = Loss_log(D(G(z))*E(X,c)) E(G(z)): %.4f / %.4f' % (epoch, 200, i, len(dataloader),
            #         errD1.data[0], errD2.data[0], errD3.data[0], errG.data[0], E_G_z1, E_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % logdir,
                    normalize=True)
            fake = G(fixed_noise, fake_context_vector) # fake context vector should be passed here
            vutils.save_image(fake.data,
                    '%s/celeba_E_D10_batch_fake_samples_epoch_%03d.png' % (logdir, epoch),
                    normalize=True)


    # do checkpointing
    torch.save(G.state_dict(), logdir + '/celeba_netG_batch.pth')
    for ix in range(len(SND_list)):
        ip = str(ix + 1)
        SND_x = SND_list[ix]
        torch.save(SND_x.state_dict(), logdir + '/celeba_netD_batch' + ip + '.pth')
    torch.save(E.state_dict(), logdir + '/celeba_netE_batch.pth')


loss_outfile.close()
finished_file = open(path.join(logdir, 'finished.txt'), 'wt')
finished_file.write('finished')
finished_file.close()

print('finished')