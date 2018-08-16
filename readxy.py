import datetime

import h5py
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from transformData import mu_law_encode

sampleSize = 16384 * 60
sample_rate = 16384 * 60


class Dataset(data.Dataset):
    def __init__(self, listx, rootx, transform=None):
        self.rootx = rootx
        self.listx = listx
        #self.device=device
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        #print('dataset',np.random.get_state()[1][0])
        np.random.seed()
        namex = self.listx[index]

        h5f = h5py.File(self.rootx + str(namex) + '.h5', 'r')
        x, y, z = h5f['x'][:], h5f['y'][:],h5f['z'][:]
        h5f.close()


        factor0 = np.random.uniform(low=0.83, high=1.0)
        factor1 = np.random.uniform(low=0.83, high=1.0)
        #print(factor0,factor1)
        z = z*factor0
        y = y*factor1
        x = (y + z)


        start = np.random.randint(0, x.shape[0] - sampleSize - 1, size=1)[0]
        #print(start)
        x = x[start:start + sampleSize]
        y = y[start:start + sampleSize]

        x = torch.from_numpy(x.reshape(1,-1)).type(torch.float32)
        y = torch.from_numpy(y.reshape(1,-1)).type(torch.float32)


        return namex,x, y


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
    def __init__(self, listx, rootx):
        self.rootx = rootx
        self.listx = listx
        #self.device=device

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]

        h5f = h5py.File('ccmixter3/' + str(namex) + '.h5', 'r')
        x, y = h5f['x'][:], h5f['y'][:]

        x = torch.from_numpy(x.reshape(1,-1)).type(torch.float32)
        y = torch.from_numpy(y.reshape(1,-1)).type(torch.float32)


        return namex,x,y