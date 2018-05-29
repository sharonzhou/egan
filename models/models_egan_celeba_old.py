# Possible models
    """ AAE: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae/aae.py
    """
    nn.Linear(opt.latent_dim, 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1),
    nn.Sigmoid()

    """ BEGAN 
    """
    # Upsampling
    self.down = nn.Sequential(
        nn.Conv2d(opt.channels, 64, 3, 2, 1),
        nn.ReLU(),
    )
    # Fully-connected layers
    self.down_size = (opt.img_size // 2)
    down_dim = 64 * (opt.img_size // 2)**2
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
        nn.Conv2d(64, opt.channels, 3, 1, 1)
    )

    """ BGAN 
    """
    nn.Linear(int(np.prod(img_shape)), 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1),
    nn.Sigmoid()

    """ BicycleGAN 
    """
    class MultiDiscriminator(nn.Module):
        def __init__(self, in_channels=3):
            super(MultiDiscriminator, self).__init__()

            def discriminator_block(in_filters, out_filters, normalize=True):
                """Returns downsampling layers of each discriminator block"""
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            # Extracts three discriminator models
            self.models = nn.ModuleList()
            for i in range(3):
                self.models.add_module('disc_%d' % i,
                    nn.Sequential(
                        *discriminator_block(in_channels, 64, normalize=False),
                        *discriminator_block(64, 128),
                        *discriminator_block(128, 256),
                        *discriminator_block(256, 512),
                        nn.Conv2d(512, 1, 3, padding=1)
                    )
                )

            self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

        def compute_loss(self, x, gt):
            """Computes the MSE between model output and scalar gt"""
            loss = sum([torch.mean((out - gt)**2) for out in self.forward(x)])
            return loss

        def forward(self, x):
            outputs = []
            for m in self.models:
                outputs.append(m(x))
                x = self.downsample(x)
            return outputs

    """ CCGAN
    """
    def discriminator_block(in_filters, out_filters, stride, normalize):
        """Returns layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    layers = []
    in_filters = channels
    for out_filters, stride, normalize in [ (64, 2, False),
                                            (128, 2, True),
                                            (256, 2, True),
                                            (512, 1, True)]:
        layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
        in_filters = out_filters

    layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

    self.model = nn.Sequential(*layers)


    """ CGAN
    """
    nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 512),
    nn.Dropout(0.4),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 512),
    nn.Dropout(0.4),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 1)

    """Context-encoder
    """
    def discriminator_block(in_filters, out_filters, stride, normalize):
    """Returns layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

    layers = []
    in_filters = channels
    for out_filters, stride, normalize in [ (64, 2, False),
                                            (128, 2, True),
                                            (256, 2, True),
                                            (512, 1, True)]:
        layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
        in_filters = out_filters

    layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

    self.model = nn.Sequential(*layers)

    """CycleGAN
    """
    def discriminator_block(in_filters, out_filters, normalize=True):
        """Returns downsampling layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    self.model = nn.Sequential(
        *discriminator_block(in_channels, 64, normalize=False),
        *discriminator_block(64, 128),
        *discriminator_block(128, 256),
        *discriminator_block(256, 512),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(512, 1, 4, padding=1)
    )

    """DCGAN
    """
    def discriminator_block(in_filters, out_filters, bn=True):
        block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    self.model = nn.Sequential(
        *discriminator_block(opt.channels, 16, bn=False),
        *discriminator_block(16, 32),
        *discriminator_block(32, 64),
        *discriminator_block(64, 128),
    )

    # The height and width of downsampled image
    ds_size = opt.img_size // 2**4
    self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1),
                                    nn.Sigmoid())


    """DiscoGAN
    """
    def discriminator_block(in_filters, out_filters, normalization=True):
        """Returns downsampling layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    self.model = nn.Sequential(
        *discriminator_block(in_channels, 64, normalization=False),
        *discriminator_block(64, 128),
        *discriminator_block(128, 256),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(256, 1, 4, padding=1)
    )

    """DraGAN
    """
    def discriminator_block(in_filters, out_filters, bn=True):
        block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    self.model = nn.Sequential(
        *discriminator_block(opt.channels, 16, bn=False),
        *discriminator_block(16, 32),
        *discriminator_block(32, 64),
        *discriminator_block(64, 128),
    )

    # The height and width of downsampled image
    ds_size = opt.img_size // 2**4
    self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1),
                                    nn.Sigmoid())

    """EBGAN
    """
    # Upsampling
    self.down = nn.Sequential(
        nn.Conv2d(opt.channels, 64, 3, 2, 1),
        nn.ReLU(),
    )
    # Fully-connected layers
    self.down_size = (opt.img_size // 2)
    down_dim = 64 * (opt.img_size // 2)**2

    self.embedding = nn.Linear(down_dim, 32)

    self.fc = nn.Sequential(
        nn.BatchNorm1d(32, 0.8),
        nn.ReLU(inplace=True),
        nn.Linear(32, down_dim),
        nn.BatchNorm1d(down_dim),
        nn.ReLU(inplace=True)
    )
    # Upsampling
    self.up = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(64, opt.channels, 3, 1, 1)
    )

    """InfoGAN
    """
    def discriminator_block(in_filters, out_filters, bn=True):
        """Returns layers of each discriminator block"""
        block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    self.conv_blocks = nn.Sequential(
        *discriminator_block(opt.channels, 16, bn=False),
        *discriminator_block(16, 32),
        *discriminator_block(32, 64),
        *discriminator_block(64, 128),
    )

    # The height and width of downsampled image
    ds_size = opt.img_size // 2**4

    # Output layers
    self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, 1))
    self.aux_layer = nn.Sequential(
        nn.Linear(128*ds_size**2, opt.n_classes),
        nn.Softmax()
    )
    self.latent_layer = nn.Sequential(nn.Linear(128*ds_size**2, opt.code_dim))

    """LSGAN
    """
    def discriminator_block(in_filters, out_filters, bn=True):
        block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    self.model = nn.Sequential(
        *discriminator_block(opt.channels, 16, bn=False),
        *discriminator_block(16, 32),
        *discriminator_block(32, 64),
        *discriminator_block(64, 128),
    )

    # The height and width of downsampled image
    ds_size = opt.img_size // 2**4
    self.adv_layer = nn.Linear(128*ds_size**2, 1)

    """Pix2Pix
    """
    def discriminator_block(in_filters, out_filters, normalization=True):
        """Returns downsampling layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    self.model = nn.Sequential(
        *discriminator_block(in_channels*2, 64, normalization=False),
        *discriminator_block(64, 128),
        *discriminator_block(128, 256),
        *discriminator_block(256, 512),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(512, 1, 4, padding=1, bias=False)
    )

    """Pixelda
    """
    def block(in_features, out_features, normalization=True):
        """Discriminator block"""
        layers = [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_features))
        return layers

    self.model = nn.Sequential(
        *block(opt.channels, 64, normalization=False),
        *block(64, 128),
        *block(128, 256),
        *block(256, 512),
        nn.Conv2d(512, 1, 3, 1, 1))

    """SRGAN
    """
    def discriminator_block(in_filters, out_filters, stride, normalize):
        """Returns layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    layers = []
    in_filters = in_channels
    for out_filters, stride, normalize in [ (64, 1, False),
                                            (64, 2, True),
                                            (128, 1, True),
                                            (128, 2, True),
                                            (256, 1, True),
                                            (256, 2, True),
                                            (512, 1, True),
                                            (512, 2, True),]:
        layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
        in_filters = out_filters

    # Output layer
    layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

    self.model = nn.Sequential(*layers)

    """StarGAN
    """
    channels, img_size, _ = img_shape

     def discriminator_block(in_filters, out_filters):
         """Returns downsampling layers of each discriminator block"""
         layers = [  nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                     nn.LeakyReLU(0.01)]
         return layers

     layers = discriminator_block(channels, 64)
     curr_dim = 64
     for _ in range(n_strided - 1):
         layers.extend(discriminator_block(curr_dim, curr_dim*2))
         curr_dim *= 2

     self.model = nn.Sequential(*layers)

     # Output 1: PatchGAN
     self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
     # Output 2: Class prediction
     kernel_size = img_size // 2**n_strided
     self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

     """WGAN
     """
     nn.Linear(int(np.prod(img_shape)), 512),
     nn.LeakyReLU(0.2, inplace=True),
     nn.Linear(512, 256),
     nn.LeakyReLU(0.2, inplace=True),
     nn.Linear(256, 1)

     


# import torch.nn as nn
# import torch
# from torch.nn.modules import conv, Linear
# import torch.nn.functional as F
# from src.snlayers.snconv2d import SNConv2d

# class _netG(nn.Module):
#     def __init__(self, nz, nc, ngf):
#         super(_netG, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=True),
#             nn.Tanh()
#             # state size. (nc) x 32 x 32
#         )

#     def forward(self, input):
#         output = self.main(input)
#         return output

# # Actor
# class _netE(nn.Module):
#     def __init__(self, nc, ndf):
#         super(_netE, self).__init__()

#         self.main = nn.Sequential(
#             SNConv2d(nc, ndf, 7, 4, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf, 3, 7, 4, 1, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, input):
#         output = self.main(input) 
#         return output.view(-1, 3).squeeze(1)

# class _netD1(nn.Module):
#     def __init__(self, nc, ndf):
#         super(_netD1, self).__init__()

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
    #     )
    # def forward(self, input):
    #     output = self.main(input)
    #     output = output.view(-1, 1).squeeze(1)
    #     return output

# class _netD2(nn.Module):
#     def __init__(self, nc, ndf):
#         super(_netD2, self).__init__()

#         self.main = nn.Sequential(
#             SNConv2d(nc, ndf, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf, ndf, 16, 2, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             # ndf x 30 x 30

#             SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf * 2, ndf * 2, 16, 2, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             # (ndf * 2) x 9 x 9

#             SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf * 4, 1, 9, 1, 0, bias=False),
#             nn.Sigmoid()
#             # 1 x 1 x 1
#         )
#     def forward(self, input):
#         output = self.main(input)
#         return output.view(-1, 1).squeeze(1)

# class _netD3(nn.Module):
#     def __init__(self, nc, ndf):
#         super(_netD3, self).__init__()

#         self.main = nn.Sequential(
#             # input is (nc) x 32 x 32
#             SNConv2d(nc, ndf, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf, ndf, 4, 2, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             # state size. (ndf) x 1 x 32
#             SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf*2, ndf * 2, 4, 2, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),

#             SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=True),

#             # state size. (ndf*8) x 4 x 4
#             SNConv2d(ndf * 8, ndf * 16, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             SNConv2d(ndf * 16, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, input):
#         output = self.main(input)
#         output = output.view(-1, 1).squeeze(1)
#         return output


# _netD_list = [_netD1]
