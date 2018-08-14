import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        '''
        nlayers = 12
        self.num_layers = nlayers
        nefilters = 24
        filter_size = 15
        merge_filter_size = 5
        self.upsampling = 'linear'
        self.output_type = 'difference'
        self.context = True
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        echannel = [1] + [(i + 1) * self.num_initial_filters for i in range(nlayers)]
        dchannel = [nefilters*(nlayers+1)] + [(i + 1) * nefilters for i in range(nlayers,0,-1)]
        for i in range(self.num_layers):
            self.encoder.append(nn.Conv1d(echannel[i],echannel[i+1],filter_size,padding=filter_size//2))
            self.decoder.append(nn.Conv1d(dchannel[i],dchannel[i+1],merge_filter_size,padding=merge_filter_size//2))
        self.middle=nn.Conv1d(echannel[i],echannel[i+1],filter_size,padding=filter_size//2)
        self.out = nn.Conv1d(nlayers,1,1)
    def forward(self,x):
        encoder = list()
        x = input
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = F.leaky_relu(x,0.2)
            encoder.append(x)
            x = x[:,:,::2]
        x = self.middle(x)
        x = F.leaky_relu(x, 0.2)
        for i in range(self.num_layers):
            x = F.upsample(x,scale_factor=2,mode='bilinear')
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=2)
            x = self.decoder[i](x)
            x = F.leaky_relu(x,0.2)
        x = torch.cat([x,input],dim=2)
        x = self.out(x)
        x = F.tanh(x)
        return x

