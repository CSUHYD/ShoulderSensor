from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler, SequentialSampler
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import savgol_filter


class LSTM(nn.Module):
    def __init__(self, batch_size, inp_dim, mid_dim, num_layers, out_dim, seq_len):
        super(LSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.rnn = nn.LSTM(inp_dim, mid_dim, num_layers,
                           batch_first=True).to(self.device)
        # inp_dim 是LSTM输入张量的维度，我们已经根据我们的数据确定了这个值是3
        # mid_dim 是LSTM三个门 (gate) 的网络宽度，也是LSTM输出张量的维度
        # num_layers 是使用两个LSTM对数据进行预测，然后将他们的输出堆叠起来。
        self.reg = nn.Sequential(
            nn.Linear(mid_dim * seq_len, out_dim)
            #             nn.Linear(30, 50), nn.ReLU(),
            #             nn.Linear(50, 100), nn.ReLU(),
            #             nn.Linear(100, 50), nn.ReLU(),
            #             nn.Linear(50, out_dim)
        )  # regression
        self.fc = nn.Linear(mid_dim * seq_len, out_dim)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.out_dim).to(
            self.device)  # (num_layers,batch,output_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.out_dim).to(
            self.device)  # (num_layers,batch,output_size)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        output = output.contiguous().view(self.batch_size, -1)
        output = self.reg(output)

        return output