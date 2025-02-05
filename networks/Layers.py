import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from networks.fft import FourierLayer, series_decomp_multi
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[:, pos:pos+x.size(1), :]
        return self.dropout(x)

class DownConvLayer(nn.Module):
    def __init__(self, c_in, kernel_size):
        super(DownConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=kernel_size,
                                  stride=kernel_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class UpConvLayer(nn.Module):
    def __init__(self, c_in, kernel_size):
        super(UpConvLayer, self).__init__()
        self.upConv = nn.ConvTranspose1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=kernel_size,
                                  stride=kernel_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.upConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""
    def __init__(self, d_model, kernel_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(kernel_size, list):
            self.conv_layers = nn.ModuleList([
                DownConvLayer(d_inner, kernel_size),
                DownConvLayer(d_inner, kernel_size),
                DownConvLayer(d_inner, kernel_size)
                ])
        else:
            self.conv_layers = []
            for i in range(len(kernel_size)):
                self.conv_layers.append(DownConvLayer(d_inner, kernel_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):

        temp_input = enc_input.permute(0, 2, 1)
        all_inputs = []
        all_inputs.append(temp_input.permute(0, 2, 1))
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input.permute(0, 2, 1))
        return all_inputs
    
class SeriesDecomposition(nn.Module):
    def __init__(self, seq_len, d_model):
        super(SeriesDecomposition, self).__init__()
        self.start_linear= nn.Linear(in_features = d_model, out_features = d_model)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])
    
    def forward(self, time_series): 
        _, trend = self.trend_model(time_series)
        seasonality, _, _ = self.seasonality_model(time_series)
        x_trans = time_series + seasonality + trend
        
        return seasonality, trend

class SeasonTrendFusion(nn.Module):
    def __init__(self, d_model, out_d):
        super(SeasonTrendFusion, self).__init__()
        self.d_model = d_model
        self.out_d = out_d
        self.weight_high = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.weight_low = nn.Parameter(torch.randn(self.d_model, self.d_model))  
        
        self.d_ff = 64
        
        self.feedforward = nn.Sequential(nn.Linear(self.d_model * 4, self.d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(0.1),
                                nn.Linear(self.d_ff, self.out_d, bias=True))
    
    def forward(self, x, x_sea, x_trend):
        x_sea_low = x_sea @ self.weight_low
        x_sea_high = x_sea @ self.weight_high
        x_trans = torch.concat([x, x_sea_low, x_sea_high, x_trend], dim=-1)
        
        x_trans = self.feedforward(x_trans)
        
        return x_trans
    
class MSFusion(nn.Module):
    def __init__(self, d_model, kernel_size, d_inner, seq_length):
        super(MSFusion, self).__init__()
        self.scale_num = len(kernel_size) + 1
        self.seq_length = seq_length
        self.up = Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.conv_layers = []
        for i in range(len(kernel_size)):
            self.conv_layers.append(UpConvLayer(d_inner, kernel_size[i]))
        self.conv_layers = nn.ModuleList(self.conv_layers)
        
        self.d_ff = 64
        self.feedforward = nn.Sequential(nn.Linear(d_model, self.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.d_ff, d_model // 2, bias=True))

    def forward(self, input_list):
        up_logit = input_list[-1]
        for i in range(len(input_list) - 1, 0, -1):
            logit = up_logit
            temp_input = self.up(logit).permute(0, 2, 1)
            cur_up_logit = self.conv_layers[i - 1](temp_input).permute(0, 2, 1)
            padding_num = self.seq_length[i - 1] - cur_up_logit.size(1)
            cur_up_logit = F.pad(cur_up_logit, (0, 0, padding_num, 0), 'constant', 0)
            up_logit = cur_up_logit + input_list[i - 1]
            
        fused_logits = self.feedforward(up_logit)
        return fused_logits
