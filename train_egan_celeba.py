import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torchvision import datasets, transforms
from vision_egan.torchvision import datasets, transforms
#import torchvision.utils as vutils
import vision_egan.torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn

import random
import argparse

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=range(3), help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_dis', type=int, default=5, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=128, help='dimension of lantent noise')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--datadir', type=str, default='/media/Seagate4TB/celeba/subset/', help='data directory')
parser.add_argument('--model', type=str, default='models_egan_celeba', help='training batch size')

opt = parser.parse_args()
print(opt)

import importlib

models_egan = importlib.import_module("models." + opt.model)
_netG = models_egan._netG
_netE = models_egan._netE
_netD_list = models_egan._netD_list

import pickle
import os.path

if os.path.isfile('celeba.pickle'):
    print('loading dataset cached')
    dataset = pickle.load(open('celeba.pickle', 'rb'))
else:
    print('loading dataset new')
    dataset = datasets.ImageFolder(root=opt.datadir,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
                        )
    print('dumping datset to celeba.pickle')
    pickle.dump(dataset, open('celeba.pickle', 'wb'))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
                                         shuffle=True, num_workers=int(2))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.cuda.set_device(opt.gpu_ids[0])

cudnn.benchmark = True

def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

n_dis = opt.n_dis
nz = opt.nz

G = _netG(nz, 3, 64)
SND_list = [_netD_x(3, 64) for _netD_x in _netD_list]
nd = len(SND_list)
E = _netE(3, 64, nd)
print(G)
print(E)
G.apply(weight_filler)
for SNDx in SND_list:
    SNDx.apply(weight_filler)
E.apply(weight_filler)

input = torch.FloatTensor(opt.batchsize, 3, 64, 64)
noise = torch.FloatTensor(opt.batchsize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchsize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0

fixed_noise = Variable(fixed_noise)
# TODO: Wasserstein distance - look at Wasserstein distance GAN 
# (D wants to maximize the range of/distance btw real - fake)
# D wants to max this distance: D(image) - D(G(z))
# generator G wants to maximize D(G(z))
criterion = nn.BCELoss() 

dtype = torch.FloatTensor

# TODO: hyperparam tuning here
alpha = 0.1
uniform = torch.ones((opt.batchsize, nd)).type(dtype) / nd

if opt.cuda:
    G.cuda()
    for SNDx in SND_list:
        SNDx.cuda()
    E.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    dtype = torch.cuda.FloatTensor
    uniform = uniform.cuda()

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0, 0.9))
optimizerSND_list = []
# TODO: hyperparam tuning here
lr_list = [0.001, 0.000002, 0.0002, 0.000001, 0.0002, 0.003, 0.0002, 0.00001, 0.0001, 0.00001]
lr_list = [0 for _ in range(10)]
for [SNDx, lrx] in zip(SND_list, lr_list):
    optimizerSNDx = optim.Adam(SNDx.parameters(), lr=lrx, betas=(0, 0.9))
    optimizerSND_list.append(optimizerSNDx)
optimizerE = optim.Adam(E.parameters(), lr=0.0002, betas=(0, 0.9))

kl_div_fcn = nn.KLDivLoss().cuda()
l2_fcn = nn.MSELoss().cuda()

EPSILON = 0.1
for epoch in range(200):
    for i, data in enumerate(dataloader, 0):
        step = epoch * len(dataloader) + i
        real_cpu, img_context = data

        batch_size = real_cpu.size(0)

        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)
        ############################
        # (1) Update D_i networks: maximize log(D_i(x)) + log(1 - D_i(G(z)))
        ###########################
        # train with real
        for SNDx in SND_list:
            SNDx.zero_grad()

        loss_Ds = torch.zeros((batch_size, nd)).type(dtype)
        for j, SNDx in enumerate(SND_list):
            loss_Ds[:,j] = criterion(SNDx(inputv), labelv)

        # TODO: add context - see conditional GANs (w/ classifier)
        W = E(inputv, nd, EPSILON) # batchsize x nd

        kl_div = - alpha * torch.mean(torch.log(W))
        loss_E = nd * (torch.mean(torch.mul(W, loss_Ds.detach() ) ) + kl_div)
        loss_E.backward()
        optimizerE.step()

        loss_D = nd * (torch.mean(loss_Ds))
        loss_D.backward(retain_graph=True)

        E_G_z1 = loss_E.clone()
        D_G_z1 = loss_D.clone()

        for optimizerSNDx in optimizerSND_list:
            optimizerSNDx.step()
        
        # train with fake
        noise.resize_(batch_size, noise.size(1), noise.size(2), noise.size(3)).normal_(0, 1)
        noisev = Variable(noise)
        fake = G(noisev)
        labelv = Variable(label.fill_(fake_label))
        
        loss_Ds = torch.zeros((batch_size, nd)).type(dtype)
        for j, SNDx in enumerate(SND_list):
            loss_Ds[:,j] = criterion(SNDx(fake.detach()), labelv)

        W = E(inputv, nd, EPSILON) # batchsize x nd

        kl_div = - alpha * torch.mean(torch.log(W))
        loss_D = nd * (torch.mean(loss_Ds))
        loss_D.backward(retain_graph=True)

        E_G_z2 = loss_E.clone()
        D_G_z2 = loss_D.clone()

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
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost

            loss_Ds = torch.zeros((batch_size, nd)).type(dtype)
            for j, SNDx in enumerate(SND_list):
                loss_Ds[:,j] = criterion(SNDx(fake), labelv)

            W = E(inputv, nd, EPSILON) # batchsize x nd

            loss_G = nd * torch.mean(torch.mul(W, loss_Ds)) 
            loss_G.backward(retain_graph=True)

            optimizerG.step()

        if i % 20 == 0:
            # print(W)
            message = '[' + str(epoch) + '/' + str(200) + '][' + str(i) + '/' + str(len(dataloader)) + ']'
            message += ' Loss_D: ' + ('{:.4f}'.format(torch.mean(loss_D)))
            message += ' Loss_G: ' + ('{:.4f}'.format(loss_G.data.cpu().numpy())) 
            message += ' E(G(z)): ' + ('{:.4f}'.format(E_G_z1.data.cpu().numpy())) + ' / ' + ('{:.4f}'.format(E_G_z2.data.cpu().numpy()))
            message += ' D(G(z)): ' + ('{:.4f}'.format(D_G_z1.data.cpu().numpy())) + ' / ' + ('{:.4f}'.format(D_G_z2.data.cpu().numpy()))
            message += ' KL: ' + ('{:.4f}'.format(kl_div))
            print(message)

            #print('[%d/%d][%d/%d] Loss_D1: %.4f Loss_D2: %.4f Loss_D3: %.4f Loss_G: %.4f = Loss_log(D(G(z))*E(X,c)) E(G(z)): %.4f / %.4f' % (epoch, 200, i, len(dataloader),
            #         errD1.data[0], errD2.data[0], errD3.data[0], errG.data[0], E_G_z1, E_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % 'v2_log',
                    normalize=True)
            fake = G(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/celeba_E_D10_batch_fake_samples_epoch_%03d.png' % ('v2_log', epoch),
                    normalize=True)


    # do checkpointing
    torch.save(G.state_dict(), 'v2_log/celeba_netG_batch_epoch_' + str(epoch) +'.pth')
    for ix in range(len(SND_list)):
        ip = str(ix + 1)
        SND_x = SND_list[ix]
        torch.save(SND_x.state_dict(), 'v2_log/celeba_netD_batch' + ip + '_epoch_' + str(epoch) + '.pth')
    torch.save(E.state_dict(), 'v2_log/celeba_netE_batch_epoch_' + str(epoch) + '.pth')
