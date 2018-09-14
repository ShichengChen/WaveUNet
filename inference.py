from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='PyTorch unet')
parser.add_argument('--checkpoint', type=str, default='pyramid',
                    help='name of checkpoint')
parser.add_argument('--test_number', type=int, default='1',
                    help='number of test songs')
args = parser.parse_args()

print('checkpoint:',args.checkpoint)
print('number of test songs:',args.test_number)

import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
#from torchvision import transforms


from readDataset.readtest import Testset
#from unet import Unet
#from modelStruct.pyramidnet import Unet
from modelStruct.unet import Unet

from transformData import mu_law_encode,mu_law_decode
# In[2]:

batchSize = 1
sampleSize = 16384*batchSize  # the length of the sample size
sample_rate = 16384
songnum=45
savemusic='vsCorpus/unet{}.wav'
#savemusic0='vsCorpus/nus10xtr{}.wav'
#savemusic1='vsCorpus/nus11xtr{}.wav'
resumefile = 'model/'+args.checkpoint  # name of checkpoint
continueTrain = False  # whether use checkpoint
sampleCnt=0
USEBOARD = False
quan=False



use_cuda = torch.cuda.is_available()  # whether have available GPU
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")

test_set = Testset(np.arange(args.test_number), 'ccmixter2/')
loadtest = data.DataLoader(test_set,batch_size=1,num_workers=10)
# In[6]:

#model = Unet(skipDim, quantization_channels, residualDim,device)
model = Unet()
#model = nn.DataParallel(model)
if(device == 'cuda'):model = model.cuda()
criterion = nn.MSELoss()
# in wavenet paper, they said crossentropyloss is far better than MSELoss
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6,betas=(0.9, 0.999))

iteration = 0
start_epoch=0
if continueTrain:  # if continueTrain, the program will find the checkpoints
    if os.path.isfile(resumefile):
        print("=> loading checkpoint '{}'".format(resumefile))
        checkpoint = torch.load(resumefile)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resumefile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resumefile))
        assert('exist checkpoint' == True)


# In[9]:


def test(epoch):  # testing data
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for iloader, xtrain in loadtest:
            iloader=iloader.item()
            print(iloader)
            listofpred0 = []
            for ind in range(0, xtrain.shape[-1] - sampleSize, sampleSize):
                output = model(xtrain[:, :,ind:ind + sampleSize].to(device))
                listofpred0.append(output.reshape(-1))
            ans0 = mu_law_decode(np.concatenate(listofpred0))
            if not os.path.exists('vsCorpus/'): os.makedirs('vsCorpus/')
            sf.write(savemusic.format(iloader), ans0, sample_rate)
            print('test stored done', np.round(time.time() - start_time))

test(epoch=0)