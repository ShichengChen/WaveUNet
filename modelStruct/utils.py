import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckV2(nn.Module):
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SEBasicBlock(nn.Module):
    def __init__(self, in_channels,channels,ks, stride=1,upsample=False,downsample=False):
        super(SEBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        self.se = SELayer(channels, reduction=16)
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
        x = self.se(x)
        return x + residual