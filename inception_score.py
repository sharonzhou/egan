import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import random
from torchvision import transforms
from imagefolder import ImageFolder
from models.models_egan_celeba import _netG
from csv import DictReader


def inception_score(imgs, cuda=True, batchsize=32, resize=True, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batchsize -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batchsize > 0
    assert N >= batchsize

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batchsize=batchsize)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    #up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    up = nn.Upsample(size=(265, 265), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batchsize_i = batch.size()[0]

        preds[i*batchsize:i*batchsize + batchsize_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def load_model(model, filename):
        model.load_state_dict(torch.load(filename, map_location='cuda'))
        return model

if __name__ == '__main__':
    # class IgnoreLabelDataset(torch.utils.data.Dataset):
    #     def __init__(self, orig):
    #         self.orig = orig
    #
    #     def __getitem__(self, index):
    #         return self.orig[index][0]
    #
    #     def __len__(self):
    #         return len(self.orig)
    #
    # import torchvision.datasets as dset
    # import torchvision.transforms as transforms
    #
    # cifar = dset.CIFAR10(root='data/', download=True,
    #                          transform=transforms.Compose([
    #                              transforms.Scale(32),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                          ])
    # )
    #
    #IgnoreLabelDataset(cifar)
    batchsize = 64
    nz = 88
    noise = torch.FloatTensor(batchsize, nz, 1, 1)
    noisev = Variable(noise).cuda()
    filename_to_context_vector = {}
    list_of_context_vectors = []
    context_feature_names = None
    datadir = "/sailhome/sharonz/celeba/full/"
    for line in DictReader(open(datadir + 'attr.csv')):
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

    def generate_fake_context_vector():
        return random.choice(list_of_context_vectors)

    def generate_fake_context_tensor(batchsize):
        output = [generate_fake_context_vector() for x in range(batchsize)]
        return torch.from_numpy(np.array(output)).float().cuda()
    
    fake_context_vector = generate_fake_context_tensor(batchsize)
    context_vector_length = len(fake_context_vector[0])
    egan = _netG(nz, 3, batchsize, context_vector_length)
    egan = load_model(egan, "logs_20180930_17:00:29/celeba_netG_batch.pth")
    noise.resize_(batchsize, noise.size(1), noise.size(2), noise.size(3)).normal_(0, 1)
    fake_context_vector = fake_context_vector.cuda()
    print(noisev.dtype, fake_context_vector.dtype)
    gen_imgs = egan(noisev, fake_context_vector)
    gen_imgs = TensorDataset(gen_imgs.data, gen_imgs.data)

    print ("Calculating Inception Score...")
    print (inception_score(gen_imgs, cuda=True, batchsize=64, resize=True, splits=10))

    #image = ImageFolder("test_is")
    #print (inception_score(IgnoreLabelDataset(image), cuda=True, batchsize=32, resize=True, splits=10))
