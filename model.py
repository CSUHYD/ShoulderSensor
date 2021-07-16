from __future__ import print_function, division
import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, batch_size, inp_dim, mid_dim, num_layers, out_dim, seq_len):
        super(LSTM, self).__init__()
        self.liner_hidden_1 = 512
        self.liner_hidden_2 = 256
        self.liner_hidden_3 = 128
        self.liner_hidden_4 = 64
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
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
            nn.Linear(mid_dim * seq_len, self.liner_hidden_1),
            nn.ReLU(),
            nn.Linear(self.liner_hidden_1, self.liner_hidden_2),
            nn.ReLU(),
            nn.Linear(self.liner_hidden_2, self.liner_hidden_3),
            nn.ReLU(),
            nn.Linear(self.liner_hidden_3, self.liner_hidden_4),
            nn.ReLU(),
            nn.Linear(self.liner_hidden_4, out_dim),
        )  # regression

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.out_dim).to(
            self.device)  # (num_layers,batch,output_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.out_dim).to(
            self.device)  # (num_layers,batch,output_size)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        output = output.contiguous().view(self.batch_size, -1)
        output = self.reg(output)

        return output
