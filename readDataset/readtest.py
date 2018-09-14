import datetime

import h5py
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
import soundfile as sf
from transformData import mu_law_encode,quan_mu_law_encode

sampleSize = 16384 * 10
sample_rate = 16384 * 10

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
        x, _ = sf.read(self.rootx + 'x/' + str(namex) + '.wav')
        assert (_ == 16000)


        x = mu_law_encode(x)

        x = torch.from_numpy(x.reshape(1, -1)).type(torch.float32)


        return namex,x