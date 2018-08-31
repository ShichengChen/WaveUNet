import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UnetD(nn.Module):
    def __init__(self,nefilters=24):
        super(UnetD, self).__init__()
        nlayers = 12
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 15
        merge_filter_size = 5
        self.upsampling = 'linear'
        self.output_type = 'difference'
        self.context = True
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0]*2]+[(i) * nefilters + (i - 1) * nefilters for i in range(nlayers,1,-1)]
        for i in range(self.num_layers):
            self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,dilation=2,padding=filter_size//2*2))
            self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))
        self.middle=nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2)
        self.out = nn.Conv1d(nefilters+1,1,1)
    def forward(self,x):
        encoder = list()
        input = x
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            encoder.append(x)
            x = x[:,:,::2]
        x = self.middle(x)
        x = F.leaky_relu(x, 0.1)
        #print(x.shape)
        for i in range(self.num_layers):
            #print(x.shape)
            #x = torch.unsqueeze(x, 2)
            #print(x.shape)
            x = F.upsample(x,scale_factor=2,mode='linear')
            #print(i,x.shape,encoder[self.num_layers - i - 1].shape)
            #x = torch.squeeze(x, 2)
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,input],dim=1)
        x = self.out(x)
        x = F.tanh(x)
        return x

