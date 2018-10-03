import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BottleneckV2(nn.Module):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, in_channels,channels,ks, stride=1,upsample=False,downsample=False):
        super(BottleneckV2, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, channels//4, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels//4)
        self.conv2 = nn.Conv1d(channels//4,channels//4, ks, stride=stride, padding=ks//2,bias=False)
        self.bn3 = nn.BatchNorm1d(channels//4)
        self.conv3 = nn.Conv1d(channels//4,channels, 1, stride=1, bias=False)
        if downsample:self.downsample = nn.Conv1d(in_channels,channels, 1, stride, bias=False)
        else:self.downsample = None
        if upsample:self.upsample = nn.Conv1d(in_channels, channels, 1, stride, bias=False)
        else:self.upsample = None

    def forward(self,x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample:residual = self.downsample(x)
        if self.upsample:residual = self.upsample(x)

        x = self.conv1(x)

        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3(x)
        #print(x.shape,residual.shape)
        return x + residual

class BasicBlockV2(nn.Module):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, in_channels,channels,ks, stride=1,upsample=False,downsample=False):
        super(BasicBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        if downsample:self.downsample = nn.Conv1d(in_channels,channels, 1, stride, bias=False)
        else:self.downsample = None
        if upsample:self.upsample = nn.Conv1d(in_channels, channels, 1, stride, bias=False)
        else:self.upsample = None

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample:residual = self.downsample(x)
        if self.upsample:residual = self.upsample(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual

class Resv2Unetv2(nn.Module):
    def __init__(self,nefilters=24):
        super(Resv2Unetv2, self).__init__()
        print('resv2unetv2')
        nlayers = 12
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 9
        merge_filter_size = 5
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        #self.downsample=nn.ModuleList()
        #self.upsample = nn.ModuleList()
        #echannelin = [24, 24, 48, 72, 96, 124, 144, 168, 192, 216, 240, 264]
        #echannelout = [24, 48, 72, 96, 124, 144, 168, 192, 216, 240, 264, 288]
        echannelin = [24] + [(i + 1) * nefilters for i in range(nlayers - 1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelin = echannelout[::-1]
        dchannelout = echannelin[::-1]
        #dchannelin = [echannelin[i] + echannelout[i] for i in range(nlayers-1,-1,-1)]
        for i in range(self.num_layers):
            self.encoder.append(BasicBlockV2(echannelin[i],echannelout[i],filter_size,upsample=True))
            #self.downsample.append(BottleneckV2(echannelin[i],echannelout[i],merge_filter_size,stride=2))
            #self.upsample.append(BottleneckV2(dchannelin[i], dchannelin[i],merge_filter_size,upsample=True))
            self.decoder.append(BasicBlockV2(dchannelin[i], dchannelout[i],merge_filter_size,downsample=True))
        self.first = nn.Conv1d(1,24,filter_size,padding=filter_size//2)
        self.middle = BasicBlockV2(echannelout[-1],echannelout[-1],filter_size)
        self.outbatch = nn.BatchNorm1d(nefilters)
        self.out = nn.Conv1d(nefilters,1,1)
    def forward(self,x):
        encoder = list()
        x = self.first(x)
        input = x
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            encoder.append(x)
            x = x[:, :, ::2]
        x = self.middle(x)
        for i in range(self.num_layers):
            x = F.upsample(x,scale_factor=2,mode='linear')
            x = x+encoder[self.num_layers - i - 1]
            x = self.decoder[i](x)
        x = x + input
        x = self.outbatch(x)
        x = F.leaky_relu(x)
        x = self.out(x)
        x = F.tanh(x)
        return x

