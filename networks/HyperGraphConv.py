
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import  degree, softmax


class HypergraphConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft = nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def __forward__(self,
                    x,
                    hyperedge_index,
                    hyperedge_weight=None,
                    alpha=None):

        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)

        B[B == float("inf")] = 0


        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        return out



    def message(self, x_j, edge_index_i, norm, alpha):     
        out = norm[edge_index_i].view(-1, 1, 1) * x_j.transpose(0, 1)
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out 
        return out.transpose(0, 1)

    def forward(self, x, hyperedge_index, edge_features, hyperedge_weight=None):
        x = torch.matmul(x, self.weight) 
        x1 =  x.transpose(0,1)

        alpha = None

        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0]) 
        result_list = edge_features
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1]) 
        

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) 
        alpha = F.leaky_relu(alpha, self.negative_slope) 
        alpha = softmax(alpha, hyperedge_index[0], num_nodes = x1.size(0)) 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  
        D = degree(hyperedge_index[0], x1.size(0), x.dtype) 
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B[B == float("inf")] = 0


        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x = x, norm=B, alpha=alpha) 
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x = out, norm=D, alpha=alpha)
        out=out.transpose(0, 1)


        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)