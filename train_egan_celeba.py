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
from models.models_egan_celeba import _netG, _netE, _netD_list

# TODO: 
# 1. dataset: Celeba
# 2. 10 D_i's

# 1. array-ify
# 2. adding context to E's input
# 3. changing Adam optim for D_i to SGD (try and see if better)

# 0. [reg GAN] Critic/G generates Z 
# 1. [reg GAN] D_i outputs p_i(fake)
# 2. [reg GAN] D_i gets better in regular GAN (but G does not update)
# 3. [independent of 1 + 2] given X and context, 
    # Actor outputs weights W of length len(D_i) 
    # (dist/pdf of action space)
# 4. multiply p_i * W to get final output o
# 5. use o to train G: if o is correct, penalize, else encourage = loss_G (==loss_critic)
# 6. loss_actor = -loss_G

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=range(3), help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_dis', type=int, default=1, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=128, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')

opt = parser.parse_args()
print(opt)

dataset = datasets.ImageFolder(root='/sailhome/sharonz/celeba/',
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
                        )

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
SND_list = [_netD_x(3,64) for _netD_x in _netD_list]
E = _netE(3, 64, 0, 3)
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
criterion = nn.BCELoss()

if opt.cuda:
    G.cuda()
    for SNDx in SND_list:
        SNDx.cuda()
    E.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0, 0.9))
optimizerSND_list = []
lr_list = [0.001, 0.000002, 0.0002]
for [SNDx, lrx] in zip(SND_list, lr_list):
    optimizerSNDx = optim.Adam(SNDx.parameters(), lr=lrx, betas=(0, 0.9))
    optimizerSND_list.append(optimizerSNDx)
optimizerE = optim.Adam(E.parameters(), lr=0.0002, betas=(0, 0.9))

DE_TRAIN_INTERVAL = 1
for epoch in range(200):
    # print("Epoch", epoch, "starting")
    for i, data in enumerate(dataloader, 0):
        step = epoch * len(dataloader) + i
        
        real_cpu, img_context = data
        print("input size, context size", real_cpu.size(), img_context.size())

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

        if i % DE_TRAIN_INTERVAL == 0:
            output_list = [SNDx(inputv) for SNDx in SND_list]

            errD_real_list = []
            for output_x in output_list:
                print("outputx size", output_x.size())
                print("labelv size", labelv.size())
                errD_real_x = criterion(output_x, labelv)
                errD_real_x.backward(retain_graph=True)
                errD_real_list.append(errD_real_x)

        # train with fake
        noise.resize_(batch_size, noise.size(1), noise.size(2), noise.size(3)).normal_(0, 1)
        noisev = Variable(noise)
        fake = G(noisev)
        labelv = Variable(label.fill_(fake_label))
        
        output_list = [SNDx(fake.detach()) for SNDx in SND_list]

        if i % DE_TRAIN_INTERVAL == 0:
            errD_fake_list = []
            errD_list = []
            for output_x in output_list:
                errD_fake_x = criterion(output_x, labelv)
                errD_fake_x.backward(retain_graph=True)
                errD_fake_list.append(errD_fake_x)
            for [errD_real_x, errD_fake_x] in zip(errD_real_list, errD_fake_list):
                errD_x = errD_real_x + errD_fake_x
                errD_list.append(errD_x)
        
        if i % DE_TRAIN_INTERVAL == 0:
            for optimizerSNDx in optimizerSND_list:
                optimizerSNDx.step()

        ############################
        # (2) Run E network: given X and context c, output weights W of length len(D_i)
        # (dist/pdf of action space)
        # multiply p_i * W to get final output o
        ###########################
        img_context = img_context.float().cuda() # size 64
        img_context.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1) # 64 x 1 x 1 x 1
        img_context = img_context.expand(-1, inputv.size()[1], inputv.size()[2], inputv.size()[3]) # 64 x 3 x 64 x 64

        contextual_input = torch.cat((inputv, img_context), -1)
        W = E(contextual_input) # 64 x 3 x 64 x 64
        W = torch.sum(W, dim=0) # size 3
        W = torch.div(W, W.sum()) # normalize weights (sum to 1)

        # Override W for debugging
        # W[0] = 0
        # W[1] = 0
        # W[2] = 1
        # print("W override ", W)

        output_weight_list = []
        for [output_x, W_x] in zip(output_list, W):
            output_weight_x = torch.mul(output_x, W_x)
            output_weight_list.append(output_weight_x)
        stacked = torch.stack(output_weight_list)
        E_G_z1 = torch.sum(stacked.mean(dim=1))
        ############################
        # (3) Update G network: maximize log(D(G(z))*E(X,c)) /////formerly: maximize log(D(G(z)))
        # (4) Update E network: minimize log(D(G(z))*E(X,c))
        ###########################
        if step % n_dis == 0:
            G.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output_list = [SNDx(fake) for SNDx in SND_list]
            output_weight_list = []
            for [output_x, W_x] in zip(output_list, W):
                output_weight_x = torch.mul(output_x, W_x)
                output_weight_list.append(output_weight_x)
            stacked = torch.stack(output_weight_list)
            E_G_z2 = torch.sum(stacked.mean(dim=1))

            errG_list = []
            for [output_x, W_x] in zip(output_list, W):
                errG_x = torch.mul(criterion(output_x, labelv), W_x)
                errG_list.append(errG_x)
            #errG = errG1 + errG2 + errG3
            errG = sum(errG_list)
            # print("errG1 ", errG1)
            # print("errG2 ", errG2)
            # print("errG3 ", errG3)
            # print("Total errG ", errG)
            errG.backward(retain_graph=True)

            # DG_E = output.data.mean()
            # DG_E = errG3

            optimizerG.step()

        # (4) Update E network: minimize log(D(G(z))*E(X,c))
        if i % DE_TRAIN_INTERVAL == 0:
            E.zero_grad()
            errE = -errG
            errE.backward(retain_graph=True)
            optimizerE.step()

        if i % 20 == 0:
            message = '[' + str(epoch) + '/' + str(200) + '][' + str(i) + '/' + str(len(dataloader)) + ']'
            for ix in range(len(errD_list)):
                errD_x = errD_list[ix]
                message += ' Loss_D' + str(ix + 1) + ': ' + ('{:.4f}'.format(errD_x.data[0]))
            message += ' Loss_G: ' + ('{:.4f}'.format(errG.data[0])) + ' = Loss_log(D(G(z))*E(X,c)) E(G(z)): ' + ('{:.4f}'.format(E_G_z1)) + ' / ' + ('{:.4f}'.format(E_G_z2))
            print(message)

            #print('[%d/%d][%d/%d] Loss_D1: %.4f Loss_D2: %.4f Loss_D3: %.4f Loss_G: %.4f = Loss_log(D(G(z))*E(X,c)) E(G(z)): %.4f / %.4f' % (epoch, 200, i, len(dataloader),
            #         errD1.data[0], errD2.data[0], errD3.data[0], errG.data[0], E_G_z1, E_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % 'log',
                    normalize=True)
            fake = G(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/arrcontext_E_fake_samples_epoch_%03d.png' % ('log', epoch),
                    normalize=True)


    # do checkpointing
torch.save(G.state_dict(), '%s/arrcontext_netG_epoch_%d.pth' % ('log', epoch))
for ix in range(len(SND_list)):
    ip = str(ix + 1)
    SND_x = SND_list[i]
    torch.save(SND_x.state_dict(), '%s/arrcontext_netD' + ip + '_epoch_%d.pth' % ('log', epoch))
#torch.save(SND1.state_dict(), '%s/netD1_epoch_%d.pth' % ('log', epoch)) 
#torch.save(SND2.state_dict(), '%s/netD2_epoch_%d.pth' % ('log', epoch)) 
#torch.save(SND3.state_dict(), '%s/netD3_epoch_%d.pth' % ('log', epoch)) 
torch.save(E.state_dict(), '%s/arrcontext_netE_epoch_%d.pth' % ('log', epoch)) 

