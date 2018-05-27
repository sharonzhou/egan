import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn

import random
import argparse
from models.models_egan import _netG, _netD1, _netD2, _netD3
# from models.models_egan import _netG, _netD1, _netD2, _netD3, _netE

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=range(3), help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_dis', type=int, default=1, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=128, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')

opt = parser.parse_args()
print(opt)

# dataset = datasets.ImageFolder(root='/home/chao/zero/datasets/cfp-dataset/Data/Images',
#                            transform=transforms.Compose([
#                                transforms.Scale(32),
#                                transforms.CenterCrop(32),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ])
#                                       )

dataset = datasets.CIFAR10(root='dataset', download=True,
                           transform=transforms.Compose([
                               transforms.Scale(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

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
SND1 = _netD1(3, 64)
SND2 = _netD2(3, 64)
SND3 = _netD3(3, 64)
# E = _netE(3, 12480, 3)
print(G)
print(SND1)
print(SND2)
print(SND3)
# print(E)
G.apply(weight_filler)
SND1.apply(weight_filler)
SND2.apply(weight_filler)
SND3.apply(weight_filler)
# E.apply(weight_filler)

input = torch.FloatTensor(opt.batchsize, 3, 32, 32)
noise = torch.FloatTensor(opt.batchsize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchsize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0

fixed_noise = Variable(fixed_noise)
criterion = nn.BCELoss()

if opt.cuda:
    G.cuda()
    SND1.cuda()
    SND2.cuda()
    SND3.cuda()
    # E.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0, 0.9))
optimizerSND1 = optim.Adam(SND1.parameters(), lr=0.001, betas=(0, 0.9))
optimizerSND2 = optim.Adam(SND2.parameters(), lr=0.000002, betas=(0, 0.9))
optimizerSND3 = optim.Adam(SND3.parameters(), lr=0.0002, betas=(0, 0.9))
# optimizerE = optim.Adam(E.parameters(), lr=0.0002, betas=(0, 0.9))

for epoch in range(200):
    print("Epoch 1 starting ")
    for i, data in enumerate(dataloader, 0):
        step = epoch * len(dataloader) + i
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        SND1.zero_grad()
        SND2.zero_grad()
        SND3.zero_grad()
        # E.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)


        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)
        output1 = SND1(inputv)
        output2 = SND2(inputv)
        output3 = SND3(inputv)
        # print("individual outputs for real1", list(output1.size()))
        # print("individual outputs for real2", list(output2.size()))
        # print("individual outputs for real3", list(output3.size()))
        # print("together for real", list(torch.cat((output1, output2, output3)).size()))
        cat_output = torch.cat(output1, output2, output3)
        print("Does E give an output? for D training fake ", cat_output.size())
        # output = E(inputv, cat_output)

        errD1_real = criterion(output1, labelv)
        errD1_real.backward()
        errD2_real = criterion(output2, labelv)
        errD2_real.backward()
        errD3_real = criterion(output3, labelv)
        errD3_real.backward()
        # For E
        # errD_real = criterion(output, labelv)
        # errD_real.backward()

        # D_x = output.data.mean()
        D_x = output1.data.mean()

        # train with fake
        noise.resize_(batch_size, noise.size(1), noise.size(2), noise.size(3)).normal_(0, 1)
        noisev = Variable(noise)
        fake = G(noisev)
        labelv = Variable(label.fill_(fake_label))
        output1 = SND1(fake.detach())
        output2 = SND2(fake.detach())
        output3 = SND3(fake.detach())
        cat_output = torch.cat(output1, output2, output3)
        # output = E(fake.detach(), cat_output)
        print("Does E give an output? for D training real ", cat_output.size())

        errD1_fake = criterion(output1, labelv)
        errD1_fake.backward()
        errD2_fake = criterion(output2, labelv)
        errD2_fake.backward()
        errD3_fake = criterion(output3, labelv)
        errD3_fake.backward()
        # For E
        # errD_real = criterion(output, labelv)
        # errD_real.backward()

        D1_G_z1 = output1.data.mean()
        D2_G_z1 = output2.data.mean()
        D3_G_z1 = output3.data.mean()
        D_G_z1 = output.data.mean()
        errD1 = errD1_real + errD1_fake
        errD2 = errD2_real + errD2_fake
        errD3 = errD3_real + errD3_fake
        # errD = errD_real + errD_fake

        optimizerSND1.step()
        optimizerSND2.step()
        optimizerSND3.step()
        # optimizerE.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if step % n_dis == 0:
            G.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output1 = SND1(fake)
            output2 = SND2(fake)
            output3 = SND3(fake)
            cat_output = torch.cat(output1, output2, output3)
            # output = E(fake, cat_output)
            print("Does E give an output? for G training ", cat_output.size())

            # errG = criterion(output, labelv)
            errG = criterion(output1, labelv)
            errG.backward()
            D_G_z2 = output1.data.mean()
            # D_G_z2 = output.data.mean()
            optimizerG.step()
        if i % 20 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, 200, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2)) #TODO
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % 'log',
                    normalize=True)
            fake = G(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % ('log', epoch),
                    normalize=True)

    # do checkpointing
torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % ('log', epoch))
torch.save(SND1.state_dict(), '%s/netD1_epoch_%d.pth' % ('log', epoch)) 
torch.save(SND2.state_dict(), '%s/netD2_epoch_%d.pth' % ('log', epoch)) 
torch.save(SND3.state_dict(), '%s/netD3_epoch_%d.pth' % ('log', epoch)) 
# torch.save(E.state_dict(), '%s/netE_epoch_%d.pth' % ('log', epoch)) 




























