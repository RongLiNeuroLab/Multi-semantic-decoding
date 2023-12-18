import scipy.sparse as sp
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, average_precision_score, \
    f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

#from networks.resnet import resnet101
from semantic_fMRI import semantic_fMRI
#from semantic_image import semantic_image
from ggnn import GGNN
from element_wise_layer import Element_Wise_Layer
from gcn import GCN
from torch.nn.parameter import Parameter
class fSGL(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim = 300):
        super(fSGL, self).__init__()
        self.seq = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True, dropout=0.3)

        # ablation set 1 No Extraction
        # self.seq = nn.Identity()
        # ablation set 2 MLP
        # self.seq = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout(0.2),
        #                          nn.Linear(2048, 2048))
        # self.seq = nn.Sequential(nn.Dropout(0.3),nn.Linear(2048, 2048))
        # ablation set 3 LSTM
        # self.seq = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True, dropout=0.3)

        self.num_classes = num_classes
        #self.transform_1 = nn.Linear(4854,2048)
        #self.transform_2 = nn.Linear(14,9)      
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.keep_prob = 0.7
        self.drop_layer = nn.Dropout(p=self.keep_prob)
        
# =============================================================================
#         self.word_semantic_image = semantic_image(num_classes= self.num_classes,
#                                       image_feature_dim = self.image_feature_dim,
#                                       word_feature_dim=self.word_feature_dim)
# =============================================================================
        self.word_semantic_fMRI = semantic_fMRI(num_classes = self.num_classes,
                                      voxel_num = self.image_feature_dim,
                                      word_feature_dim = self.word_feature_dim,
                                      keep_prob = self.keep_prob)

        self.word_features = word_features
        self._word_features = self.load_features()
        self.adjacency_matrix = adjacency_matrix
        self._in_matrix, self._out_matrix = self.load_matrix()
        self.time_step = time_step
        
        self.graph_net = GGNN(input_dim=self.image_feature_dim,
                              time_step=self.time_step,
                              in_matrix=self._in_matrix,
                              out_matrix=self._out_matrix,
                              keep_prob = self.keep_prob)

        self.output_dim = output_dim

        # self.fc_output = nn.Linear(2*self.image_feature_dim, self.output_dim)
        # self.fc_output_ablation = nn.Linear(self.image_feature_dim, self.output_dim)
        # self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

        self.fc_output = nn.Linear(2 * self.image_feature_dim, int(self.output_dim / 2))
        self.fc_output_ablation = nn.Linear(self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, int(self.output_dim * 2))
        self.classifiers_ablation = Element_Wise_Layer(self.num_classes, int(self.output_dim))

    def forward(self, y,alpha,beta):
        batch_size = y.size()[0] # batch * 14 * 2048
        #img_feature_map = self.resnet_101(x)
        #img_feature_map = img_feature_map.view(img_feature_map.shape[0],img_feature_map.shape[1],-1)
# =============================================================================
#         img_feature_map = self.transform_2(img_feature_map)
#         img_feature_map = self.drop_layer(img_feature_map)
# =============================================================================
        #img_feature_map = self.transform(img_feature_map)
        #fMRI_feature_map  = torch.nn.init.xavier_uniform_(torch.zeros(batch_size,9,self.image_feature_dim), gain=1).cuda()
        fMRI_feature_map  = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map  = fMRI_feature_map[:,:,:]
        fMRI_feature_map,hidden = self.seq(fMRI_feature_map)  # batch * 14 * 2048, 1 * 64 * 2048 已经提取的特征和最后时间点的输出
        # fMRI_feature_map = self.seq(fMRI_feature_map)
        fMRI_feature_map = torch.transpose(fMRI_feature_map,1,2) # 64 * 2048 * 14

        semantic_net_input = fMRI_feature_map
        semantic_net_input = torch.transpose(semantic_net_input,1,2) # batch * 14 * 2048
        graph_net_input,coefficient = self.word_semantic_fMRI(batch_size,
                                     semantic_net_input,
                                     torch.tensor(self._word_features).cuda())
        #graph_net_input =   graph_net_input
        graph_net_feature,_in_matrix = self.graph_net(graph_net_input)

        # ## original
        output = torch.cat((graph_net_feature.view(batch_size*self.num_classes,-1), graph_net_input.view(-1, self.image_feature_dim)), 1)
        # output = self.fc_output(output)  # （batch * 52） * 2048
        #output = torch.tanh(output)
        # output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        output = output.contiguous().view(batch_size, self.num_classes, int(self.output_dim * 2))
        #print("=======",output.shape)
        result,voxel_weight = self.classifiers(output)

        # ## ablation1 only h0
        # # output = self.fc_output_ablation(graph_net_input.view(-1, self.image_feature_dim))
        # output = graph_net_input.view(-1, self.image_feature_dim).contiguous().view(batch_size, self.num_classes, int(self.output_dim))
        # result, voxel_weight = self.classifiers_ablation(output)

        ## ablation2 only ht
        # output = self.fc_output_ablation(graph_net_feature.view(batch_size*self.num_classes,-1))
        # output = graph_net_feature.view(batch_size * self.num_classes, -1).view(batch_size, self.num_classes, int(self.output_dim))
        # result, voxel_weight = self.classifiers_ablation(output)

        result = torch.sigmoid(result)

        return result,voxel_weight,coefficient

    def load_features(self):
        return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()

    def load_matrix(self):
        mat = np.load(self.adjacency_matrix)
        #self.weight = Parameter(torch.Tensor(in_features, out_features))
        #mat = np.identity(57)
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
        _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
        return _in_matrix, _out_matrix


class fSGCN(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim=300):
        super(fSGCN, self).__init__()
        self.seq = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True, dropout=0.3)
        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.keep_prob = 0.7
        self.drop_layer = nn.Dropout(p=self.keep_prob)

        # =============================================================================
        #         self.word_semantic_image = semantic_image(num_classes= self.num_classes,
        #                                       image_feature_dim = self.image_feature_dim,
        #                                       word_feature_dim=self.word_feature_dim)
        # =============================================================================
        self.word_semantic_fMRI = semantic_fMRI(num_classes=self.num_classes,
                                                voxel_num=self.image_feature_dim,
                                                word_feature_dim=self.word_feature_dim,
                                                keep_prob=self.keep_prob)

        self.word_features = word_features
        self._word_features = self.load_features()
        self.adjacency_matrix = adjacency_matrix
        self._adj_matrix = self.load_and_preprocess_matrix()
        self.time_step = time_step

        self.graph_net = GCN(nfeat=self.image_feature_dim,
                             nhid=self.image_feature_dim,
                             nclass=self.image_feature_dim,
                             adj_matrix=self._adj_matrix,
                             keep_prob=self.keep_prob)

        self.output_dim = output_dim
        self.fc_output = nn.Linear(2 * self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, y, alpha, beta):
        batch_size = y.size()[0]  # batch * 14 * 2048
        # img_feature_map = self.resnet_101(x)
        # img_feature_map = img_feature_map.view(img_feature_map.shape[0],img_feature_map.shape[1],-1)
        # =============================================================================
        #         img_feature_map = self.transform_2(img_feature_map)
        #         img_feature_map = self.drop_layer(img_feature_map)
        # =============================================================================
        # img_feature_map = self.transform(img_feature_map)
        # fMRI_feature_map  = torch.nn.init.xavier_uniform_(torch.zeros(batch_size,9,self.image_feature_dim), gain=1).cuda()
        fMRI_feature_map = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map = fMRI_feature_map[:, :, :]
        fMRI_feature_map, hidden = self.seq(fMRI_feature_map)  # batch * 14 * 2048, 1 * 64 * 2048 已经提取的特征和最后时间点的输出
        fMRI_feature_map = torch.transpose(fMRI_feature_map, 1, 2)  # 64 * 2048 * 14

        semantic_net_input = fMRI_feature_map
        semantic_net_input = torch.transpose(semantic_net_input, 1, 2)  # batch * 14 * 2048
        graph_net_input, coefficient = self.word_semantic_fMRI(batch_size,
                                                               semantic_net_input,
                                                               torch.tensor(self._word_features).cuda())# batch * 52 * 2048
        # graph_net_input =   graph_net_input
        graph_net_feature = self.graph_net(graph_net_input) # batch * 52 * 52
        output = graph_net_feature
        # output = torch.cat((graph_net_feature.view(batch_size * self.num_classes, -1),
        #                     graph_net_input.view(-1, self.image_feature_dim)), 1)
        # output = self.fc_output(output)  # （batch * 52） * 2048
        # output = torch.tanh(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        # print("=======",output.shape)
        result, voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)

        # result = torch.sigmoid(result)

        return result, voxel_weight, coefficient

    def load_features(self):
        return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()

    def load_and_preprocess_matrix(self):
        mat = np.load(self.adjacency_matrix)
        # 变为 D^(-1/2)*A*D^(-1/2)
        # for i in range(mat.shape[0]):
        #     mat[i][i] = 1
        # D = np.zeros((mat.shape[0], mat.shape[1]))
        rowsum = np.array(mat.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mat = r_mat_inv.dot(mat)
        _adj_matrix = mat.astype(np.float32)

        # mat_2 = np.load('F:\Data\multi-semantic-decoding\data\label/co_matrix.npy')
        # mat_2 = mat_2.astype(np.float32)

        # self.weight = Parameter(torch.Tensor(in_features, out_features))
        # mat = np.identity(57)

        _adj_matrix = Variable(torch.from_numpy(_adj_matrix), requires_grad=False).cuda()
        # _adj_matrix = Variable(torch.from_numpy(mat_2), requires_grad=False).cuda()
        # _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
        return _adj_matrix

class DNN(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim = 300):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(2048,2048)
        self.l2 = nn.Linear(2048,2048)

        self.num_classes = num_classes      
        self.image_feature_dim = image_feature_dim
        self.word_features = word_features
        self._word_features = self.load_features()
        self.adjacency_matrix = adjacency_matrix
        self._in_matrix, self._out_matrix = self.load_matrix()
        self.time_step = time_step
        self.output_dim = output_dim
        self.fc_output = nn.Linear(2048, 2048)
        self.classifiers = Element_Wise_Layer(self.num_classes, 2048)

    def forward(self, y,alpha,beta):
        batch_size = y.size()[0]
        fMRI_feature_map = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map = fMRI_feature_map[:,5,:]
        fMRI_feature_map = F.relu(self.l1(fMRI_feature_map))
        fMRI_feature_map = F.relu(self.l2(fMRI_feature_map))
        output = self.fc_output(fMRI_feature_map)
        output = output.contiguous().view(batch_size, 1, 2048).repeat(1,self.num_classes,1)
        result,voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)
        return result,voxel_weight,voxel_weight

    def load_features(self):
        return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()

    def load_matrix(self):
        mat = np.load(self.adjacency_matrix)
        #self.weight = Parameter(torch.Tensor(in_features, out_features))
        #mat = np.identity(57)
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
        _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
        return _in_matrix, _out_matrix

class Linear(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim = 300):
        super(Linear, self).__init__()
        self.l1 = nn.Linear(2048,2048)
        self.l2 = nn.Linear(2048,2048)

        self.num_classes = num_classes      
        self.image_feature_dim = image_feature_dim
        self.word_features = word_features
        self._word_features = self.load_features()
        self.adjacency_matrix = adjacency_matrix
        self.time_step = time_step
        self.output_dim = output_dim
        self.fc_output = nn.Linear(2048, 2048)
        self.classifiers = Element_Wise_Layer(self.num_classes, 2048)

    def forward(self, y,alpha,beta):
        batch_size = y.size()[0]
        fMRI_feature_map = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map = fMRI_feature_map[:,5,:]
        fMRI_feature_map = F.relu(self.l1(fMRI_feature_map))
        fMRI_feature_map = F.relu(self.l2(fMRI_feature_map))
        output = self.fc_output(fMRI_feature_map)
        output = torch.tanh(output)
        output = output.contiguous().view(batch_size, 1, 2048).repeat(1,self.num_classes,1)
        result,voxel_weight = self.classifiers(output)
        return result,voxel_weight,voxel_weight

    def load_features(self):
        return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()

# =============================================================================
#     def load_matrix(self):
#         coefficient = np.load('A:/SSGRL-master/linear_coefficient.npy')
#         intercept = np.load('A:/SSGRL-master/linear_intercept.npy')
#         #self.weight = Parameter(torch.Tensor(in_features, out_features))
#         #mat = np.identity(57)
#         _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
#         _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
#         _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
#         return _in_matrix, _out_matrix
# =============================================================================

class GRU(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim = 300):
        super(GRU, self).__init__()
        self.GRU = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True,dropout=0.3)
        self.num_classes       = num_classes      
        self.word_feature_dim  = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.keep_prob         = 0.7
        self.drop_layer        = nn.Dropout(p=self.keep_prob)
        self.word_features     = word_features
        self._word_features    = self.load_features()
        self.adjacency_matrix  = adjacency_matrix
        self._in_matrix, self._out_matrix = self.load_matrix()
        self.time_step         = time_step

        self.output_dim = output_dim
        self.fc_output = nn.Linear(self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, y,alpha,beta):
        #batch_size = x.size()[0]
        fMRI_feature_map  = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map,hidden = self.GRU(fMRI_feature_map)
        fMRI_feature_map = torch.transpose(fMRI_feature_map,1,2)
        output = torch.sum(fMRI_feature_map,2).view(-1, 1,self.image_feature_dim).repeat(1,self.num_classes,1)
        output = self.fc_output(output)
        #output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result,voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)
        #print(torch.sigmoid(result))
        return result,voxel_weight,voxel_weight

    def load_features(self):
        return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()

    def load_matrix(self):
        mat = np.load(self.adjacency_matrix)
        #self.weight = Parameter(torch.Tensor(in_features, out_features))
        #mat = np.identity(57)
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
        _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
        return _in_matrix, _out_matrix

class LSTM(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim = 300):
        super(LSTM, self).__init__()
        self.seq = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True,dropout=0.5)
        self.num_classes = num_classes      
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.keep_prob = 0.5
        self.drop_layer = nn.Dropout(p=self.keep_prob)
        self.word_features = word_features
        # self._word_features = self.load_features()
# =============================================================================
#         self.adjacency_matrix = adjacency_matrix
#         self._in_matrix, self._out_matrix = self.load_matrix()
#         self.time_step = time_step
# =============================================================================

        self.output_dim = output_dim
        self.fc_output = nn.Linear(self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, y, alpha, beta):
        batch_size = y.size()[0]
        fMRI_feature_map  = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map,hidden = self.seq(fMRI_feature_map)
        fMRI_feature_map = torch.transpose(fMRI_feature_map,1,2)
        output = torch.sum(fMRI_feature_map,2).view(-1, 1,self.image_feature_dim).repeat(1,self.num_classes,1)
        output = self.fc_output(output)
        # output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result,voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)
        return result,voxel_weight,voxel_weight

# =============================================================================
#     def load_features(self):
#         return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()
# 
#     def load_matrix(self):
#         mat = np.load(self.adjacency_matrix)
#         #self.weight = Parameter(torch.Tensor(in_features, out_features))
#         #mat = np.identity(57)
#         _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
#         _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
#         _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
#         return _in_matrix, _out_matrix
# =============================================================================
    
class Transformer(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim = 300):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.num_classes = num_classes      
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.keep_prob = 0.5
        self.drop_layer = nn.Dropout(p=self.keep_prob)
        self.word_features = word_features
# =============================================================================
#         self._word_features = self.load_features()
#         self.adjacency_matrix = adjacency_matrix
#         self._in_matrix, self._out_matrix = self.load_matrix()
#         self.time_step = time_step
# =============================================================================

        self.output_dim = output_dim
        self.fc_output = nn.Linear(self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, y,alpha,beta):
        batch_size = y.size()[0]
        fMRI_feature_map  = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map = self.transformer(fMRI_feature_map)
        fMRI_feature_map = torch.transpose(fMRI_feature_map,1,2)
        output = torch.sum(fMRI_feature_map,2).view(-1, 1,self.image_feature_dim).repeat(1,self.num_classes,1)
        output = self.fc_output(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result,voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)
        return result,voxel_weight,voxel_weight

# =============================================================================
#     def load_features(self):
#         return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()
# 
#     def load_matrix(self):
#         mat = np.load(self.adjacency_matrix)
#         #self.weight = Parameter(torch.Tensor(in_features, out_features))
#         #mat = np.identity(57)
#         _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
#         _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
#         _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
#         return _in_matrix, _out_matrix   
# =============================================================================

class RandomForest():
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim=300):
        super(RandomForest, self).__init__()
        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.keep_prob = 0.5
        self.drop_layer = nn.Dropout(p=self.keep_prob)
        self.word_features = word_features
        # =============================================================================
        #         self._word_features = self.load_features()
        #         self.adjacency_matrix = adjacency_matrix
        #         self._in_matrix, self._out_matrix = self.load_matrix()
        #         self.time_step = time_step
        # =============================================================================

        # self.output_dim = output_dim
        # self.fc_output = nn.Linear(self.image_feature_dim, self.output_dim)
        # self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

        self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        # 创建一个Pipeline对象包括特征选择器和分类器
        self.selectors = SelectKBest(f_classif, k=100)
        self.pipe = Pipeline([
            ('select', self.selectors),
            ('classify', self.estimator)
        ])
        self.multi_output_classifier = MultiOutputClassifier(self.pipe)

    def train(self, Xtrain, alpha, beta, Ytrain=None, Xval=None, Yval=None):
        if Xtrain.shape[-1] > 21:
            param_grid = {'classify__bootstrap': [True, False],
                          'classify__n_estimators': [i for i in range(1, 51)],
                          'classify__max_features': [2, 4, 6, 8],
                          'classify__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                          'classify__min_samples_leaf': [1, 2, 4],
                          'classify__min_samples_split': [2, 5, 10],
                          'select__k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        else:
            param_grid = {'classify__bootstrap': [True, False],
                          'classify__n_estimators': [i for i in range(1, 51)],
                          'classify__max_features': [2, 4, 6, 8],
                          'classify__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                          'classify__min_samples_leaf': [1, 2, 4],
                          'classify__min_samples_split': [2, 5, 10],
                          'select__k': [5, 10, 15, 21]}

        scoring = {"AUC": "roc_auc", "Accuracy": "accuracy"}


        self.multi_output_classifier.fit(Xtrain, Ytrain)

    def val(self, Xval, alpha, beta, Yval):
        # 在测试集上进行预测
        Yprob = self.multi_output_classifier.predict_proba(Xval) # 52 * 500 * 2
        Yprob = np.stack(Yprob)
        Yprob = Yprob[:, :, 1]
        Yprob = Yprob.transpose(1,0)
        Ypred = self.multi_output_classifier.predict(Xval)
        # 设置自定义阈值（例如，0.3）
        custom_threshold = 0.5
        # 根据自定义阈值生成二进制类别预测
        Ypred = (Yprob > custom_threshold).astype("int")
        # 评估模型性能
        # report = classification_report(Yval, Yprob)
        # print(report)

        # 初始化变量
        all_precision = []
        all_recall = []
        all_ap = []
        all_f1_scores = []
        # 计算每个类别的PR曲线和AP
        for i in range(len(Yval[0])):
            precision, recall, _ = precision_recall_curve(Yval[:, i], Yprob[:, i])
            ap = average_precision_score(Yval[:, i], Yprob[:, i])

            all_precision.append(precision)
            all_recall.append(recall)
            all_ap.append(ap)
            # print(f"Class {i} - AP: {ap}")

            # f1 = f1_score(Yval[:, i], Ypred[:, i])
            # all_f1_scores.append(f1)
        # 计算mAP
        mAP = np.mean(all_ap)
        print(f"\nMean Average Precision (mAP): {mAP}")


        gt_label = Yval
        num_target = np.sum(gt_label, axis=1, keepdims=True)
        threshold = 1 / (num_target + 1e-6)  # 阈值，准确率高于标签数分之一才计入。
        # threshold = 0.5
        # 因为对于完全预测准确的，每个标签的概率应该为 0.5
        cw_predict_result = Yprob > threshold  # 卡阈值
        cw_prediction = cw_predict_result.astype(np.int)
        cw_target = gt_label.astype(np.int)
        a = cw_prediction + cw_target
        true_p = a[:, :] > 1  # TP
        false_p = a[:, :] <= 1
        TP = np.sum(true_p, 0)  # 每类的TP
        T_total = np.sum(cw_target, 0)  # 每类的TP + FN 所有标签为正的都在这里
        P_total = np.sum(cw_prediction, 0)  # TP + FP 所有被预测为正的都在这里
        cw_re = TP / (T_total + 1e-10)  # TP/ (TP + FN)
        cw_pr = TP / (P_total + 1e-10)  # TP / (TP + FP)
        class_wise_F1 = 2 * cw_re * cw_pr / (cw_re + cw_pr + 1e-10)


        f1 = np.mean(class_wise_F1)
        print(f"\nMean F1 score (MacroF1): {f1}")

        class_wise_ap = all_ap
        class_wise_F1 = class_wise_F1
        # ave_recall = np.mean(all_recall)
        ave_recall = 0
        class_wise_recall = all_recall

        return  mAP,f1,class_wise_ap,class_wise_F1,ave_recall, class_wise_recall

        # 输出详细的分类报告
        # report = classification_report(y_test, y_pred, target_names=mlb.classes_)



        # # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        # rf_random = RandomizedSearchCV(estimator=pipe,
        #                                param_distributions=param_grid,
        #                                n_iter=int(100),
        #                                verbose=1,
        #                                random_state=66,
        #                                # n_jobs = 4,
        #                                scoring=scoring,
        #                                refit=False)
        # rf_random.fit(Xtrain, Ytrain)
        # cv_result = rf_random.cv_results_