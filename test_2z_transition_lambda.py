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
import math

import importlib


models_egan = importlib.import_module("models.models_egan_celeba")
_netG = models_egan._netG

nz = 128
batch_size = 64

G = _netG(nz, 3, 64)
save_path = 'log/celeba_netG_batch_epoch_89.pth'
G.load_state_dict(torch.load(save_path))

noise1 = torch.FloatTensor(batch_size, nz, 1, 1)
noise2 = torch.FloatTensor(batch_size, nz, 1, 1)
noise1 = noise1.resize_(batch_size, noise1.size(1), noise1.size(2), noise1.size(3)).normal_(0, 1)
noise2 = noise2.resize_(batch_size, noise2.size(1), noise2.size(2), noise2.size(3)).normal_(0, 1)
noise1v = Variable(noise1)
noise2v = Variable(noise2)
print("Are the noise vectors equal? Should not be, so False", noise1v.equal(noise2v))
fake1 = G(noise1v)
fake2 = G(noise2v)

for lambd in range(20):
	if lambd == 0:
		fake = fake1
	elif lambd == 19:
		fake = fake2
	else:
		print(-math.log(lambd / 20.0), 1.0+math.log(lambd / 20.0))
		fake = -math.log(lambd / 20.0) * fake1 + (1.0+math.log(lambd / 20.0)) * fake2
	# output as png
	vutils.save_image(fake.data,
                    '%s/2z_transition%02d.png' % ('2z_transition', lambd),
                    normalize=True)