from __future__ import print_function

import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

from readxy import Dataset, Testset, RandomCrop, ToTensor
from unet import Unet
from tensorboardX import SummaryWriter

# In[2]:

batchSize = 10
sampleSize = 16384*batchSize  # the length of the sample size
sample_rate = 16384
songnum=45
savemusic='vsCorpus/nus1xtr{}.wav'
#savemusic0='vsCorpus/nus10xtr{}.wav'
#savemusic1='vsCorpus/nus11xtr{}.wav'
resumefile = 'model/instrument1'  # name of checkpoint
continueTrain = False  # whether use checkpoint
sampleCnt=0
USEBOARD = True

if(USEBOARD):writer = SummaryWriter()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use specific GPU

# In[4]:
from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
if(USEBOARD):writer = SummaryWriter(log_dir='../conditioned-wavenet/runs/'+str(current_time),comment="uwavenet")

use_cuda = torch.cuda.is_available()  # whether have available GPU
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
# device = 'cpu'
# torch.set_default_tensor_type('torch.cuda.FloatTensor') #set_default_tensor_type as cuda tensor


#training_set = Dataset(np.arange(45), 'ccmixter3/',pad=pad,transform=transform)
#validation_set = Testset(np.arange(45,50), 'ccmixter3/',pad=pad)
training_set = Dataset(np.arange(45), 'ccmixter3/',transform=None)
validation_set = Testset(np.arange(50), 'ccmixter3/')

worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
loadtr = data.DataLoader(training_set, batch_size=10,shuffle=True,num_workers=10,worker_init_fn=worker_init_fn)
loadval = data.DataLoader(validation_set,batch_size=1,num_workers=10)
# In[6]:

#model = Unet(skipDim, quantization_channels, residualDim,device)
model = Unet()
#model = nn.DataParallel(model)
model = model.cuda()
criterion = nn.MSELoss()
# in wavenet paper, they said crossentropyloss is far better than MSELoss
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
# use adam to train

maxloss=np.zeros(50)+100
# In[7]:
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


# In[9]:


def test(epoch):  # testing data
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for iloader, xtrain, ytrain in loadval:
            iloader=iloader.item()
            listofpred0 = []
            cnt,aveloss=0,0
            for ind in range(0, xtrain.shape[-1] - sampleSize, sampleSize):
                output = model(xtrain[:, :,ind:ind + sampleSize].to(device))
                listofpred0.append(output.reshape(-1))
                loss = criterion(output, (ytrain[:, :,ind:ind + sampleSize].to(device)))
                cnt+=1
                aveloss += float(loss)
            aveloss /= cnt
            print('loss for validation:{},num{},epoch{}'.format(aveloss, iloader,epoch))
            ans0 = np.concatenate(listofpred0)
            if not os.path.exists('vsCorpus/'): os.makedirs('vsCorpus/')
            sf.write(savemusic.format(iloader), ans0, sample_rate)
            print('test stored done', np.round(time.time() - start_time))



def train(epoch):  # training data, the audio except for last 15 seconds
    for iloader,xtrain, ytrain in loadtr:
        startx = 0
        idx = np.arange(startx, xtrain.shape[-1] - sampleSize, sampleSize//batchSize)
        np.random.shuffle(idx)
        #lens = 100
        #idx = idx[:lens]
        cnt, aveloss = 0, 0
        start_time = time.time()
        model.train()
        for i, ind in enumerate(idx):
            data = xtrain[:, :, ind:ind + sampleSize].to(device)
            target = ytrain[:, :,ind:ind + sampleSize].to(device)
            output = model(data)
            loss = criterion(output, target)
            aveloss+=float(loss)
            cnt+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global sampleCnt
            sampleCnt+=100
            if sampleCnt % 10000 == 0 and sampleCnt > 0:
                for param in optimizer.param_groups:
                    param['lr'] *= 0.98
        global iteration
        iteration += 1
        print('loss for train:{:.3f},epoch{},({:.3f} sec/step)'.format(
            aveloss / cnt, epoch,time.time() - start_time))
        if (USEBOARD):writer.add_scalar('waveunet loss', (aveloss / cnt), iteration)
    if epoch % 5 == 0:
        if not os.path.exists('model/'): os.makedirs('model/')
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'iteration': iteration}
        torch.save(state, resumefile)
        print('write finish')


# In[ ]:

print('training...')
for epoch in range(100000):
    train(epoch+start_epoch)
    #test(epoch + start_epoch)
    if (epoch+start_epoch) % 25 == 0 and epoch+start_epoch > 0: test(epoch+start_epoch)