
from numpy import transpose
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# Baseline for TCN model from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# modified for non-causal and no dropout


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, causal):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.causal = causal

    def forward(self, x):
        if self.causal:
            return x[:, :, :-self.chomp_size].contiguous()
        else:
            return x[:, :, self.chomp_size:-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, causal):
        super(TemporalBlock, self).__init__()
        self.chomp_size = padding if causal else padding//2

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(self.chomp_size, causal)
        self.relu1 = nn.PReLU()

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(self.chomp_size, causal)
        self.relu2 = nn.PReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2,)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) # residual


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, causal=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, causal=causal)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Model

class Wiener_TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, causal):
        super(Wiener_TCN, self).__init__()
        self.tcn1 = TemporalConvNet(input_size, num_channels, kernel_size, causal)
        self.norm1 = nn.GroupNorm(input_size, output_size)

        self.tcn2 = TemporalConvNet(input_size, num_channels, kernel_size, causal)
        self.norm2 = nn.GroupNorm(input_size, output_size)

        self.tcn3 = TemporalConvNet(input_size, num_channels, kernel_size, causal)
        self.line3 = nn.Linear(input_size, output_size)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        output = self.tcn1(x.transpose(1, 2))
        output = self.norm1(output)
        output = self.tcn2(output)
        output = self.norm2(output)
        output = self.tcn3(output).transpose(1, 2)
        output = self.line3(output)
        return self.act3(output)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size):
        super(TCN, self).__init__()
        self.tcn1 = TemporalConvNet(input_size, num_channels, kernel_size)
        self.tcn2 = TemporalConvNet(input_size, num_channels, kernel_size)
        self.tcn3 = TemporalConvNet(input_size, num_channels, kernel_size)

    def forward(self, x):
        output = self.tcn1(x.transpose(1, 2))
        output = self.tcn2(output)
        return self.tcn3(output).transpose(1, 2)