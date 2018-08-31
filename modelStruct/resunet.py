import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Unet(nn.Module):
    def __init__(self,nefilters=24):
        super(Unet, self).__init__()
        print('resunet')
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
        self.downsample=nn.ModuleList()
        self.upsample = nn.ModuleList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0]*2]+[(i) * nefilters + (i - 1) * nefilters for i in range(nlayers,1,-1)]
        batchchannel = [echannelout[-1]] + dchannelout[:-1]
        #print(batchchannel)
        for i in range(self.num_layers):
            self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=filter_size//2))
            self.downsample.append(nn.Conv1d(echannelout[i],echannelout[i],5,stride=2,padding=2))

            self.upsample.append(
                nn.ConvTranspose1d(batchchannel[i], batchchannel[i], 5, stride=2, padding=2, output_padding=1))
            self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))
            self.dbatch.append(nn.BatchNorm1d(batchchannel[i]))
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))
        self.middle=nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2)
        self.out = nn.Conv1d(nefilters+1,1,1)
    def forward(self,x):
        encoder = list()
        input = x
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i * 2](x)
            x = F.leaky_relu(x, 0.1)
            encoder.append(x)
            x = self.downsample[i](x)
            x = self.ebatch[i * 2 + 1](x)
            x = F.leaky_relu(x, 0.1)
        x = self.middle(x)
        x = F.leaky_relu(x, 0.1)
        for i in range(self.num_layers):
            x = self.upsample[i](x)
            x = self.dbatch[i * 2](x)
            x = F.leaky_relu(x, 0.1)
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)
            x = self.decoder[i](x)
            x = self.dbatch[i * 2 + 1](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,input],dim=1)
        x = self.out(x)
        x = F.tanh(x)
        return x

