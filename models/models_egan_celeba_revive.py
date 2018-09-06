import torch.nn as nn
import torch
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
from src.snlayers.snconv2d import SNConv2d


class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        output = self.main(input)
        return output

'''
# Actor
class _netE(nn.Module):
    def __init__(self, nc, ndf, ncontext, ndiscriminators):
        super(_netE, self).__init__()

        self.main = nn.Sequential(
            # input is (nc * ncontext) x 32 x 32 TODO add ncontext
            #nn.Linear(32, ndiscriminators, bias=False),
            SNConv2d(nc, ndf, 7, 4, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, 3, 7, 4, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input) 
        return output.view(-1, 3).squeeze(1)
'''

'''
class _netD1(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD1, self).__init__()

        self.main = nn.Sequential(
            SNConv2d(nc, ndf, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, ndf, 6, 4, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # ndf x 16 x 16

            SNConv2d(ndf, ndf * 2, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 2, ndf * 2, 3, 3, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # (ndf * 2) x 4 x 4

            SNConv2d(ndf * 2, ndf * 4, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 4, 1, 1, 1, 0, bias=False),
            #nn.MaxPool3d([1,4,4]),
            nn.Sigmoid()
            
            # 1 x 1 x 1
        )
    def forward(self, input):
        output = self.main(input)
        print('output size iss')
        print(output.size())
        #return torch.tensor([1.0])
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD2(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD2, self).__init__()

        self.main = nn.Sequential(
            SNConv2d(nc, ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, ndf, 16, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # ndf x 30 x 30

            SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 2, ndf * 2, 16, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # (ndf * 2) x 9 x 9

            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 4, 1, 9, 1, 0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

class _netD3(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD3, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            SNConv2d(nc, ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 1 x 32
            SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf*2, ndf * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=True),

            # state size. (ndf*8) x 4 x 4
            SNConv2d(ndf * 8, ndf * 16, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1).squeeze(1)
        return output
'''

'''
class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        print('netG being called')
        output = self.main(input)
        print('netG output size')
        print(output.size())
        print('netG finished')
        return output
'''

# Actor
class _netE(nn.Module):
    def __init__(self, nc, ndf, ncontext, ndiscriminators):
        super(_netE, self).__init__()

        self.main = nn.Sequential(
            # input is (nc * ncontext) x 32 x 32 TODO add ncontext
            #nn.Linear(32, ndiscriminators, bias=False),
            SNConv2d(nc, ndf, 7, 4, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, 3, 7, 4, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input) 
        return output.view(-1, 3).squeeze(1)

class _netD1(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD1, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            SNConv2d(nc, ndf, 3, 4, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, ndf, 5, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, ndf, 3, 2, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD2(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD1, self).__init__()

        self.main = nn.Sequential(
            SNConv2d(nc, ndf, 5, 2, 2), 
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf, ndf * 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 2, ndf * 4, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 4, ndf * 8, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 8, 1, 4),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD3(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD1, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD3(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD1, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1).squeeze(1)
        return output

_netD_list = [_netD1]