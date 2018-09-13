import datetime

import h5py
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from transformData import mu_law_encode, mu_law_encode
import soundfile as sf

sampleSize = 16000
sample_rate = 16000  # the length of audio for one second
normalized = True

class Dataset(data.Dataset):
    def __init__(self, listx, rootx,pad, transform=None):
        self.rootx = rootx
        self.listx = listx
        self.pad=int(pad)
        #self.device=device
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        np.random.seed()
        namex = self.listx[index]

        y, _ = sf.read('piano/piano{}.wav'.format(namex))
        print('train piano{}.wav,train audio shape{},rate{}'.format(namex,y.shape,_))
        print('normalized:',normalized)
        factor1 = np.random.uniform(low=0.83, high=1.0)
        y = y*factor1

        if normalized:
            ymean = y.mean()
            ystd = y.std()
            y = (y - ymean) / ystd


        y = mu_law_encode(y)

        y = torch.from_numpy(y.reshape(-1)).type(torch.LongTensor)
        #y = F.pad(y, (self.pad, self.pad), mode='constant', value=127)

        return namex,y


class RandomCrop(object):
    def __init__(self, pad,output_size=sample_rate):
        self.output_size = output_size
        self.pad=pad

    def __call__(self, sample):
        #print('randomcrop',np.random.get_state()[1][0])
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        x, y = sample['x'], sample['y']
        shrink = 0
        #startx = np.random.randint(self.pad + shrink * sampleSize, x.shape[-1] - sampleSize - self.pad - shrink * sampleSize)
        #print(startx)
        #x = x[startx - pad:startx + sampleSize + pad]
        #y = y[startx:startx + sampleSize]

        l = np.random.uniform(0.25, 0.5)
        sp = np.random.uniform(0, 1 - l)
        step = np.random.uniform(-0.5, 0.5)
        ux = int(sp * sample_rate)
        lx = int(l * sample_rate)
        # x[ux:ux + lx] = librosa.effects.pitch_shift(x[ux:ux + lx], sample_rate, n_steps=step)

        return {'x': x, 'y': y}


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample['x'], sample['y']
        return {'x': torch.from_numpy(x.reshape(1, -1)).type(torch.float32),
                'y': torch.from_numpy(y.reshape(-1)).type(torch.LongTensor)}


class Testset(data.Dataset):
    def __init__(self, listx, rootx,pad,dilations1,device):
        self.rootx = rootx
        self.listx = listx
        self.pad = int(pad)
        self.device=device
        self.dilations1=dilations1
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]

        y, _ = sf.read('piano/piano{}.wav'.format(namex))
        #factor1 = np.random.uniform(low=0.83, high=1.0)
        #y = y*factor1

        if normalized:
            ymean = y.mean()
            ystd = y.std()
            y = (y - ymean) / ystd

        y = mu_law_encode(y)

        #y = torch.from_numpy(y.reshape(-1)).type(torch.LongTensor)
        print('test piano{}.wav,train audio shape{},rate{}'.format(namex, y.shape, _))
        y = torch.from_numpy(y.reshape(-1)[int(16000*1):]).type(torch.LongTensor)
        print("first second as seed")
        #y = torch.randint(0, 256, (100000,)).type(torch.LongTensor)
        #print("random init")
        #y = F.pad(y, (self.pad, self.pad), mode='constant', value=127)


        return namex,y