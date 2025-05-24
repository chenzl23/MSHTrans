import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks.HyperGraphUp import HypergraphFusion


class Multi_Adaptive_Hypergraph(nn.Module):
    def __init__(self, args, device):
        super(Multi_Adaptive_Hypergraph, self).__init__()
        self.pool_size_list = args["pool_size_list"]
        self.seq_len = args["seq_len"] 
        self.inner_size = args["inner_size"]
        self.dim = args["d_model"]
        self.hyper_num = args["hyper_num"]
        self.alpha = 3
        self.k = args["k"]
        self.device = device
        self.embedding_hyperedge = nn.ModuleList()
        self.embedding_node = nn.ModuleList()
        self.linear_hyperedge = nn.ModuleList()
        self.linear_node = nn.ModuleList()

        for i in range(len(self.hyper_num)):
            self.embedding_hyperedge.append(nn.Embedding(self.hyper_num[i],self.dim).to(device))
            self.linear_hyperedge.append(nn.Linear(self.dim,self.dim))
            self.linear_node.append(nn.Linear(self.dim,self.dim))
            if i==0:
                self.embedding_node.append(nn.Embedding(self.seq_len,self.dim).to(device))
            else:
                product=math.prod(self.pool_size_list[:i])
                layer_size=math.floor(self.seq_len/product)
                self.embedding_node.append(nn.Embedding(int(layer_size),self.dim).to(device))

        self.dropout = nn.Dropout(p=0.1)
        self.adjs = self.init_H_indicies()
        self.update_rate = args["update_rate"]
        
        self.node_num = []
        self.node_num.append(self.seq_len)
        for i in range(len(self.pool_size_list)):
            layer_size = math.floor(self.node_num[i] / self.pool_size_list[i])
            self.node_num.append(layer_size)
        
    def init_H_indicies(self):
        node_num = []
        node_num.append(self.seq_len)
        for i in range(len(self.pool_size_list)):
            layer_size = math.floor(node_num[i] / self.pool_size_list[i])
            node_num.append(layer_size)


        H_list = []
        for i in range(len(self.hyper_num)):
            H_index_matrix = torch.zeros(node_num[i], self.hyper_num[i]).to(self.device)
            for col in range(self.hyper_num[i] - 1, -1, -1):
                start_index = max(0, node_num[i] - self.k - self.hyper_num[i] + col - 1)
                end_index = node_num[i] - self.hyper_num[i] + col 
                if start_index <= end_index:
                    H_index_matrix[start_index:end_index + 1, col] = 1
            H_index_matrix = F.softmax(H_index_matrix)
            H_list.append(H_index_matrix)
        return H_list
    
    def get_embeddings(self):
        
        node_embedding_list = []
        edge_embedding_list = []
        
        for i in range(len(self.hyper_num)):
            hypidxc = torch.arange(self.hyper_num[i]).to(self.device)

            nodeidx = torch.arange(self.node_num[i]).to(self.device)

            hyperen = self.embedding_hyperedge[i](hypidxc)
            nodeec = self.embedding_node[i](nodeidx)
            
            edge_embedding_list.append(hyperen)
            node_embedding_list.append(nodeec)
        
        return node_embedding_list, edge_embedding_list
    
    def H2index(self, adj):
        adj = torch.where(adj > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))

        retain_edge = (adj != 0).any(dim=0)
        adj = adj[:, retain_edge]
        matrix_array = torch.tensor(adj, dtype=torch.int)
        
        result_list = [list(torch.nonzero(matrix_array[:, col]).flatten().tolist()) for col in
                        range(matrix_array.shape[1])]
        

        node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0])

        count_list = list(torch.sum(adj, dim=0).tolist())
        
        hperedge_list = torch.cat([torch.full((count,), idx) for idx, count in enumerate(count_list, start=0)])

        hypergraph = torch.vstack((node_list, hperedge_list))
        
        return hypergraph, retain_edge
    
    def minmax_norm(self, x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        x = (x - min_val) / (max_val - min_val)
        
        return x

    def forward(self, train=True):

        hyperedge_all=[]

        H_list = []

        edge_retain_list = []
        
        node_embeddings, edge_embeddings = self.get_embeddings()
        
        for i in range(len(self.hyper_num)):
            if train:
                hyperen = edge_embeddings[i]
                nodeec = node_embeddings[i]

                a = torch.mm(nodeec, hyperen.transpose(1, 0))
                cur_adj = F.softmax(F.relu(self.alpha * a))
                
    
                mask = torch.zeros(nodeec.size(0), hyperen.size(0)).to(self.device)
                mask.fill_(float('0'))

                s1, t1 = cur_adj.topk(min(cur_adj.size(1), self.k), 1)
                mask.scatter_(1, t1, s1.fill_(1))
                cur_adj = cur_adj * mask

                adj = self.update_rate * cur_adj + (1 - self.update_rate) * self.adjs[i]
                adj = self.minmax_norm(adj)
                self.adjs[i] = adj
                
                H_list.append(adj)
            else:
                adj = self.adjs[i]
                H_list.append(adj)
            
            hypergraph, retain_edge = self.H2index(adj)
            
            edge_retain_list.append(retain_edge)
            hyperedge_all.append(hypergraph)
        
        fused_H, full_H_list = HypergraphFusion(H_list, self.pool_size_list, self.pool_size_list, self.seq_len, self.device)
        fused_hypergraph, fused_retain_edge = self.H2index(fused_H)

        a = hyperedge_all, H_list, edge_retain_list, fused_hypergraph, fused_retain_edge
        return a
