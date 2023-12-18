import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj): # adj: 2708*2708 input: 2708*1433
        # support = torch.mm(input, self.weight) # 矩阵乘法
        # output = torch.mm(adj, support) #
        # if self.bias is not None:
        #     return output + self.bias
        # else:
        #     return output

        batch_size = input.size()[0]
        input = input.view(-1, self.in_features)# (batch * 52) * 2048
        node_num = adj.size()[0]
        batch_aog_nodes = input.view(batch_size, node_num, self.in_features)  # batch * 52 * 2048
        batch_in_matrix = adj.repeat(batch_size, 1).view(batch_size, node_num, -1)# 64 * 52 * 52 共现矩阵扩大到batch
        # batch_in_matrix = adj.repeat(1, batch_size)  # 52 * （52 * batch）  共现矩阵扩大到batch

        support = torch.mm(input, self.weight) # 矩阵乘法 (batch * 52) * output_num
        support = support.view(batch_size, node_num, -1)
        output = torch.bmm(batch_in_matrix, support) # batch * 52 * output_num
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# # ###############
#         # input batch * 52 * 2048
#         batch_size = input.size()[0]
#         input = input.view(-1, self.input_dim)  # input (batch * 52) * 2048
#         node_num = self._in_matrix.size()[0]
#         batch_aog_nodes = input.view(batch_size, node_num, self.input_dim)  # batch * 52 * 2048
#         batch_in_matrix = self._in_matrix.repeat(batch_size, 1).view(batch_size, node_num,
#                                                                      -1)  # 64 * 52 * 52 共现矩阵扩大到batch
#         batch_out_matrix = self._out_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1)
#         for t in range(self.time_step):
#             # eq(2)
#             # bmm三维张量乘法 batch 不变，剩下两维度相乘
#             av = torch.cat((torch.bmm(batch_in_matrix, batch_aog_nodes), torch.bmm(batch_out_matrix, batch_aog_nodes)),
#                            2)  # batch * 52 * 2048 在这里聚合出入的节点信息。
#             av = av.view(batch_size * node_num, -1)  # （batch * 52） * 2048
#
#             flatten_aog_nodes = batch_aog_nodes.view(batch_size * node_num, -1)  # # （batch * 52） * 2048
#             # handmade GRU
#             # eq(3)
#             zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_aog_nodes))
#             zv_d = self.drop_layer(zv)
#             # eq(4)
#             rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(flatten_aog_nodes))  # V为上一个？
#             rv_d = self.drop_layer(rv)
#             # eq(5)
#             hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(
#                 rv_d * flatten_aog_nodes))  # 为什么是rvrv_d * flatten_aog_nodes不是flatten_aog_nodes
#             hv_d = self.drop_layer(hv)
#
#             flatten_aog_nodes = (1 - zv_d) * flatten_aog_nodes + zv_d * hv_d  # 等于下一轮的h
#             batch_aog_nodes = flatten_aog_nodes.view(batch_size, node_num, -1)
