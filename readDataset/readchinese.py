import datetime

import h5py
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
import soundfile as sf
from transformData import mu_law_encode,quan_mu_law_encode

sampleSize = 16384 * 60
sample_rate = 16384 * 60


class Dataset(data.Dataset):
    def __init__(self, listx, rootx,quan=False, transform=None):
        self.rootx = rootx
        self.quan = quan
        print('quan'+str(quan))
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
        x, _ = sf.read(self.rootx + str(namex) + '/0.wav')
        assert (_ == 16000)
        y, _ = sf.read(self.rootx + str(namex) + '/1.wav')
        assert (_ == 16000)
        assert (x.shape == y.shape)


        x = mu_law_encode(x)
        y = mu_law_encode(y)

        if (x.shape[0] <= sampleSize):
            while (x.shape[0] <= sampleSize):
                x = np.concatenate((x, x))
                y = np.concatenate((y, y))
            assert (x.shape == y.shape)
            # print('xy', x.shape, y.shape, namex)

        start = np.random.randint(0, x.shape[0] - sampleSize + 1, size=1)[0]
        x = x[start:start + sampleSize]
        y = y[start:start + sampleSize]

        x = torch.from_numpy(x.reshape(1, -1)).type(torch.float32)
        y = torch.from_numpy(y.reshape(1, -1)).type(torch.float32)


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



class Valtset(data.Dataset):
    def __init__(self, listx, rootx,quan=False):
        self.rootx = rootx
        self.listx = listx
        self.quan = quan
        #self.device=device

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        np.random.seed()
        namex = self.listx[index]
        x, _ = sf.read(self.rootx + str(namex) + '/0.wav')
        assert (_ == 16000)
        y, _ = sf.read(self.rootx + str(namex) + '/1.wav')
        assert (_ == 16000)
        assert (x.shape == y.shape)

        x = mu_law_encode(x)
        y = mu_law_encode(y)

        if (x.shape[0] <= sampleSize):
            while (x.shape[0] <= sampleSize):
                x = np.concatenate((x, x))
                y = np.concatenate((y, y))
            assert (x.shape == y.shape)

        start = np.random.randint(0, x.shape[0] - sampleSize + 1, size=1)[0]
        x = x[start:start + sampleSize]
        y = y[start:start + sampleSize]

        x = torch.from_numpy(x.reshape(1, -1)).type(torch.float32)
        y = torch.from_numpy(y.reshape(1, -1)).type(torch.float32)


        return namex,x,y



class Testset(data.Dataset):
    def __init__(self, listx, rootx,quan=False):
        self.rootx = rootx
        self.listx = listx
        self.quan = quan
        #self.device=device

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]
        x, _ = sf.read(self.rootx + str(namex) + '/0.wav')
        assert (_ == 16000)
        y, _ = sf.read(self.rootx + str(namex) + '/1.wav')
        assert (_ == 16000)
        assert (x.shape == y.shape)


        x = mu_law_encode(x)
        y = mu_law_encode(y)

        x = torch.from_numpy(x.reshape(1, -1)).type(torch.float32)
        y = torch.from_numpy(y.reshape(1, -1)).type(torch.float32)


        return namex,x,y