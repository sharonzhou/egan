import torch.nn as nn
import torch
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
from src.snlayers.snconv2d import SNConv2d
import random

class _netC(nn.Module):
    def __init__(self, nc, ndf, num_classes):
        super(_netC, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SNConv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            SNConv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 4, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 16, ndf * 64, 4, 1, 0, bias=False),
            # state size. (ndf*16) x 4 x 4
            #SNConv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.linear = nn.Linear(1024*4, num_classes)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        #output = self.main(input)
        #output = output.view(-1, 1).squeeze(1)
        #return output
        #print('input shape')
        #print(input.shape)
        #print('befor running main')
        output = self.main(input)
        #print('output size is')
        #print(output.size())
        flattened = output.view(-1, 1024*4)
        #print('flattened output is')
        #print(flattened.size())
        output = self.linear(flattened)
        #print('after linear layer size is')
        #print(output.size())
        #print(output)
        #isreal = self.sigmoid(output)
        #print('after sigmoid layer size is')
        #print(output.size())
        #print(output)
        classes = self.softmax(output)
        return classes

class _netG(nn.Module):
    def __init__(self, nz, nc, ngf, context_vector_length):
        super(_netG, self).__init__()
        #self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + context_vector_length, ngf * 8, 4, 1, 0, bias=True),
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

    def forward(self, input, context_vector):
        #print('nz is')
        #print(self.nz)
        #print('input shape')
        #print(input.shape)
        #print('context_vector shape')
        #print(context_vector.shape)
        next_input = torch.cat((input.view(input.size(0), -1), context_vector.view(context_vector.size(0), -1)), -1)
        #next_input = torch.cat((input, context_vector), dim=-1)
        #print('after cat next_input')
        #print(next_input.shape)
        #  next_input.view(input.size(0) + context_vector.size(0), -1)
        next_input = next_input.unsqueeze(-1).unsqueeze(-1)
        #print('next_input shape')
        #print(next_input.shape)
        output = self.main(next_input)
        return output

# Actor
class _netE(nn.Module):
    def __init__(self, nc, nef, ndiscriminators, context_vector_size):
        super(_netE, self).__init__()

        self.main = nn.Sequential(
            # input is (nc * ncontext) x 32 x 32 TODO add ncontext
            #nn.Linear(32, ndiscriminators, bias=False),
            SNConv2d(nc, nef, 7, 4, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(nef, ndiscriminators, 7, 4, 1, bias=False),
            nn.Sigmoid()
        )

        self.image = nn.Sequential(
            SNConv2d(nc, nef, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(nef, nef * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(nef * 2, nef * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(nef * 4, ndiscriminators, 7, 4, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Softmax() # remove once context has been added
        )

        self.context = nn.Sequential(
            #nn.Linear(nc * nef * nef + ndiscriminators, nef * 8),
            nn.Linear(context_vector_size + ndiscriminators, nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nef * 8, ndiscriminators),
            nn.Softmax()
        )

    # def forward(self, input, context, ndiscriminators, eps):
    def forward(self, input, ndiscriminators, context_vector):
        # if random.random() < eps:
        #     output = torch.rand((64, ndiscriminators)).cuda()
        # else:
        #     output = self.image(input) 
        output = self.image(input).float().cuda()
        next_input = torch.cat((output.view(output.size(0), -1), context_vector.view(context_vector.size(0), -1)), -1)
        output = self.context(next_input)
        #output = output.squeeze(-1).squeeze(-1)
        return output
        # return output.view(-1, ndiscriminators).squeeze(1)

class _netD1(nn.Module):
    def __init__(self, nc, ndf, include_sigmoid):
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
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD2(nn.Module):
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD2, self).__init__()

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
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD3(nn.Module):
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD3, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SNConv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            SNConv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SNConv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD3(nn.Module):
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD3, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SNConv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            SNConv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 4, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            SNConv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD4(nn.Module):
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD4, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SNConv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            SNConv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 4, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            SNConv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD5(nn.Module):
    # AAE
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD5, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(64, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.MaxPool3d([3, 64, 1]),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD6(nn.Module):
    # BEGAN
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD6, self).__init__()

        # Upsampling
        self.down = nn.Sequential(
            SNConv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
        )
        # Fully-connected layers
        self.down_size = (64 // 2)
        down_dim = 64 * (64 // 2)**2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True)
        )
        # Upsampling
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            SNConv2d(64, 3, 3, 1, 1)
        )
        self.final = nn.Sequential(
            nn.MaxPool3d([3, 64, 64]),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        out = self.down(input)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        out = self.final(out)
        for extra_layer in self.extra_layers_to_run:
            out = extra_layer(out)
        out = out.view(-1, 1).squeeze(1)
        return out

class _netD7(nn.Module):
    # BGAN
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD7, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(64, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.MaxPool3d([3, 64, 1]),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

class _netD8(nn.Module):
    # BGAN
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD8, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(64, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.MaxPool3d([3, 64, 1]),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

def cyclegan_discriminator_block(in_filters, out_filters, normalize=True):
    """Returns downsampling layers of each discriminator block"""
    layers = [SNConv2d(in_filters, out_filters, 4, stride=2, padding=1)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

class _netD9(nn.Module):
    # CycleGAN
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD9, self).__init__()

        self.main = nn.Sequential(
            *cyclegan_discriminator_block(3, 64, normalize=False),
            *cyclegan_discriminator_block(64, 128),
            *cyclegan_discriminator_block(128, 256),
            *cyclegan_discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            SNConv2d(512, 1, 4, padding=1),
            nn.MaxPool3d([1, 4, 4]),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

def dcgan_discriminator_block(in_filters, out_filters, bn=True):
    block = [   SNConv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)]
    if bn:
        block.append(nn.BatchNorm2d(out_filters, 0.8))
    return block

class _netD10(nn.Module):
    # DCGAN
    def __init__(self, nc, ndf, include_sigmoid):
        super(_netD10, self).__init__()

        self.main = nn.Sequential(
            *dcgan_discriminator_block(3, 16, bn=False),
            *dcgan_discriminator_block(16, 32),
            *dcgan_discriminator_block(32, 64),
            *dcgan_discriminator_block(64, 128),
            nn.MaxPool3d([128, 4, 4]),
            #nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()
        self.extra_layers_to_run = []
        if include_sigmoid:
            self.extra_layers_to_run.append(self.sigmoid)
    def forward(self, input):
        output = self.main(input)
        for extra_layer in self.extra_layers_to_run:
            output = extra_layer(output)
        output = output.view(-1, 1).squeeze(1)
        return output

_netD_list = [_netD1, _netD2, _netD3, _netD4, _netD5, _netD6, _netD7, _netD8, _netD9, _netD10]
#_netD_list = [_netD1]