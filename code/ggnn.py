import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class GGNN(nn.Module):
    def __init__(self, input_dim, time_step,  in_matrix,out_matrix,keep_prob):
        super(GGNN, self).__init__()
        self.input_dim = input_dim
        self.time_step = time_step
        
# =============================================================================
#         self._in_matrix  = torch.nn.Parameter(torch.FloatTensor(in_matrix.shape), requires_grad=True)
#         self._out_matrix = torch.nn.Parameter(torch.FloatTensor(out_matrix.shape), requires_grad=True)
# =============================================================================
        self._in_matrix  = in_matrix # 52 * 52
        self._out_matrix = out_matrix # 52 * 52
        self.keep_prob = keep_prob
        self.drop_layer = nn.Dropout(p=self.keep_prob)
        self.fc_eq3_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq3_u = nn.Linear(input_dim, input_dim)
        self.fc_eq4_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq4_u = nn.Linear(input_dim, input_dim)
        self.fc_eq5_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq5_u = nn.Linear(input_dim, input_dim)

    def forward(self, input):
        # input batch * 52 * 2048
        batch_size = input.size()[0]
        input = input.view(-1, self.input_dim) # input (batch * 52) * 2048
        node_num = self._in_matrix.size()[0]
        batch_aog_nodes = input.view(batch_size, node_num, self.input_dim) # batch * 52 * 2048
        batch_in_matrix = self._in_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1) # 64 * 52 * 52 共现矩阵扩大到batch
        batch_out_matrix = self._out_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1)
        for t in range(self.time_step):
            # eq(2)
            # bmm三维张量乘法 batch 不变，剩下两维度相乘
            av = torch.cat((torch.bmm(batch_in_matrix, batch_aog_nodes), torch.bmm(batch_out_matrix, batch_aog_nodes)), 2) # batch * 52 * 2048 在这里聚合出入的节点信息。
            av = av.view(batch_size * node_num, -1) # （batch * 52） * 2048

            flatten_aog_nodes = batch_aog_nodes.view(batch_size * node_num, -1) # # （batch * 52） * 2048
            # handmade GRU
            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_aog_nodes))
            zv_d = self.drop_layer(zv)
            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(flatten_aog_nodes)) # V为上一个？
            rv_d = self.drop_layer(rv)
            #eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv_d * flatten_aog_nodes)) # 为什么是rvrv_d * flatten_aog_nodes不是flatten_aog_nodes
            hv_d = self.drop_layer(hv)
            
            flatten_aog_nodes = (1 - zv_d) * flatten_aog_nodes + zv_d * hv_d # 等于下一轮的h
            batch_aog_nodes = flatten_aog_nodes.view(batch_size, node_num, -1)
        return batch_aog_nodes,self._in_matrix


