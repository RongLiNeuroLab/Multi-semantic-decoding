import argparse
import os,sys
import shutil
import time,pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from utils.transforms import get_train_test_set
#from networks.resnet import resnet101
from utils.load_pretrain_model import load_pretrain_model
from utils.metrics import voc12_mAP
from models import fSGL,DNN,GRU,LSTM,Linear,fSGCN,Transformer

global best_epoch
global best_prec1
best_prec1 = np.zeros((5,6))
global sub_name
global ROI_name
sub_name = ["sub01","sub02","sub03","sub04","sub05"]
ROI_name = ["V1","V2","V3","LVC","OCC","HVC"]
global mAP_all
global F1_all
global HL_all
global num_test, num_train
mAP_all = np.zeros((5,6))
F1_all = np.zeros((5,6))
HL_all = np.zeros((5,6))



def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch multi label Training')
    parser.add_argument('-dataset',     default='VG', metavar='DATASET',help='path to train dataset')
    parser.add_argument('-train_data',  default='A:/SSGRL-master/data/Image/',metavar='DIR',help='path to train dataset')
    parser.add_argument('-test_data',   default='A:/SSGRL-master/data/Image/',metavar='DIR', help='path to test dataset')
    parser.add_argument('-trainlist',   default='A:/SSGRL-master/data/train_data_list.txt', metavar='DIR',help='path to train list')
    parser.add_argument('-testlist',    default='A:/SSGRL-master/data/test_data_list.txt',  metavar='DIR',help='path to test list')
    parser.add_argument('-fMRI_dir',    default='F:\Data\multi-semantic-decoding\data\multi_label\step8_data\\',                  metavar='DIR',help='path to train dataset')
    parser.add_argument('-ftrain_list', default='F:\Data\multi-semantic-decoding\data\multi_label\step8_data/train_data_list.txt',       metavar='DIR',help='path to train list')
    parser.add_argument('-ftest_list',  default='F:\Data\multi-semantic-decoding\data\multi_label\step8_data/test_data_list.txt',        metavar='DIR',help='path to test list')
    parser.add_argument('-train_label', default='F:\Data\multi-semantic-decoding\data\multi_label\step8_data/train_label_list.txt', type=str, metavar='PATH',help='path to train label (default: none)')
    parser.add_argument('-test_label',  default='F:\Data\multi-semantic-decoding\data\multi_label\step8_data/test_label_list.txt',  type=str, metavar='PATH',help='path to test  label (default: none)')
    parser.add_argument('-graph_file',  default='F:\Data\multi-semantic-decoding\data\label/co_matrix.npy', type=str, metavar='PATH',help='path to graph (default: none)')
    parser.add_argument('-symmetry_graph_file',  default='F:\Data\multi-semantic-decoding\data\label/appear_num.npy', type=str, metavar='PATH',help='path to graph (symmetry) (default: none)')
    parser.add_argument('-word_file',   default='F:\Data\multi-semantic-decoding\data\label/Glove_vec.npy', type=str, metavar='PATH',help='path to word feature')
    parser.add_argument('--resume',     default='F:\Data\multi-semantic-decoding\\new_code\code/', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    #parser.add_argument('-pm','--pretrain_model', default='', type=str, metavar='PATH',help='path to latest pretrained_model (default: none)')
    parser.add_argument('--print_freq', '-p',     default=10, type=int, metavar='N',help='number of print_freq (default: 100)')
    parser.add_argument('--num_classes', '-n', default=52, type=int, metavar='N',help='number of classes (default: 80)')
    parser.add_argument('--epochs',               default=10, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('--start-epoch',          default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
    parser.add_argument('--step_epoch',           default=31, type=int, metavar='N',help='decend the lr in epoch number')
    parser.add_argument('-b', '--batch-size',     default=64, type=int,metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate',default=0.0001, type=float,metavar='LR', help='initial learning rate,default 0.0001')
    parser.add_argument('--momentum',             default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--timestep',             default=2, type=int, metavar='M',help='timestep of GNN')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained', type=int,default=0,help='use pre-trained model')
    parser.add_argument('--crop_size',  dest='crop_size',default=224, type=int,help='crop size')
    parser.add_argument('--scale_size', dest = 'scale_size',default=256, type=int,help='the size of the rescale image')
    parser.add_argument('--evaluate',   dest='evaluate', default=False, action='store_true',help='evaluate model on validation set')
    parser.add_argument('--post',       dest='post', type=str,default='',help='postname of save model')
    parser.add_argument('-j', '--workers',        default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
    args = parser.parse_args()
    return args

def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg,content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")


def main():
    global best_prec1
    args = arg_parse()
    print_args(args)
    global best_epoch
    global sub_name
    global ROI_name
    global mAP_all
    global F1_all
    global HL_all
    global num_test, num_train
    # Create dataloader
    for fMRI_length in range(14,15):
        for sub in range(0,5):
            # for roi in range(3,6): # LVC OCC HVC
            for roi in range(4, 5): # 仅occ
                print("==> Creating dataloader...")
                train_data_dir  = args.train_data
                test_data_dir   = args.test_data
                train_list      = args.trainlist
                test_list       = args.testlist
                train_label     = args.train_label
                test_label      = args.test_label
                args.start_epoch= 0
                # soffix = 'test_fSGCN_3_low_threshold_best'
                soffix = 'experiement_name'
                # soffix = 'ablation_f-GSL_whole'
                # soffix = 'GRU_seed00_low_57_drop07_4096output_ADD'
                model_opti = 'fSGL'   #fSGL,DNN,RR,GRU,LSTM,Transformer,fSGCN
                fMRI_dir        = args.fMRI_dir+sub_name[sub]+'/'+ROI_name[roi]+'/'
                save_model_path = args.resume+'/model/'+soffix+'/time_point_{0:02d}/'.format(fMRI_length) + sub_name[sub]+'/'+ROI_name[roi]+'/'
                result_path = args.resume+'/result/'+ soffix + '/time_point_{0:02d}/'.format(fMRI_length) + sub_name[sub] + '/' + ROI_name[roi] + '/'
                #result_path     = 'A:/SSGRL-master/result'+soffix+'/time_point_{0:02d}/'.format(fMRI_length)+sub_name[sub]+'/'+ROI_name[roi]+'/'
# =============================================================================
#                 save_model_path = 'A:/SSGRL-master/model_5_without_graph/'  +sub_name[sub]+'/'+ROI_name[roi]+'/'
#                 result_path     = 'A:/SSGRL-master/result_5_without_graph/'+sub_name[sub]+'/'+ROI_name[roi]+'/'
# =============================================================================
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                if not os.path.exists(args.resume+'/result/'+soffix):
                    os.mkdir(args.resume+'/result/'+soffix)
                if not os.path.exists(args.resume+'/model/'+soffix):
                    os.mkdir(args.resume+'/model/'+soffix)
                    
                leng_dir = args.resume+'/model/'+soffix+'/time_point_14/'
                if not os.path.exists(leng_dir):
                    os.mkdir(leng_dir)
                file_dir = leng_dir+'/'+sub_name[sub]
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                img_dir = file_dir+'/'+ROI_name[roi]
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)    
                    
                leng_dir = args.resume+'/result/'+soffix+'/time_point_14/'
                if not os.path.exists(leng_dir):
                    os.mkdir(leng_dir) 
                file_dir = leng_dir+'/'+sub_name[sub]
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                img_dir = file_dir+'/'+ROI_name[roi]
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)     
                    
                fMRI_train_list = args.ftrain_list
                fMRI_test_list  = args.ftest_list
                train_loader, test_loader, num_train, num_test = get_train_test_set(fMRI_dir,fMRI_train_list,fMRI_test_list,train_label, test_label,fMRI_length,args)
            
                # load the network
                print("==> Loading the network ...")
            
# =============================================================================
#                 model = GRU(image_feature_dim=2048,
#                               output_dim=2048, time_step=2,
#                               adjacency_matrix=args.graph_file,
#                               word_features=args.word_file,
#                               num_classes=args.num_classes)
# =============================================================================
                model,learning_rate = model_option(model_opti,args) 
                
                if args.pretrained:
                    model = load_pretrain_model(model,args)
                model.cuda()
                
                criterion = nn.BCELoss(reduce=True, size_average=True).cuda()
# =============================================================================
#                 for p in model.resnet_101.parameters():
#                     p.requires_grad=False
#                 for p in model.resnet_101.layer4.parameters():
#                     p.requires_grad=False
#                 for p in model.word_semantic_fMRI.parameters():
#                     p.requires_grad=True
#                 for p in model.graph_net.parameters():
#                     p.requires_grad=True
#                 for p in model.seq.parameters():
#                     p.requires_grad=True
# =============================================================================
  
                #optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad,model.parameters()), lr=args.lr)

                # 查找模型的checkpoint
                if save_model_path:
                    flag = 0
                    for i in os.listdir(save_model_path):
                        fulldirct = os.path.join(save_model_path, i)
                        flag += 1
                        if flag == 2:
                            if os.path.isfile(fulldirct):
                                print("=> loading checkpoint '{} {}'".format(save_model_path, i))
                                checkpoint = torch.load(fulldirct)
                                args.start_epoch = checkpoint['epoch']
                                best_prec1[sub,roi] = checkpoint['best_mAP']
                                model.load_state_dict(checkpoint['state_dict'])
                                print("=> loaded checkpoint '{}' (mAP {})".format(save_model_path, checkpoint['best_mAP']))
                            else:
                                print("=> no checkpoint found at '{}'".format(save_model_path))
            
                cudnn.benchmark = True
                Train_LOSS = np.zeros((args.epochs*2))
                Test_LOSS = np.zeros((args.epochs*1))
                beta = 1

                # Evaluate the trained model
                if args.evaluate:
                    with torch.no_grad():
                        alpha = beta
                        criterion = nn.BCELoss(reduction='sum')
                        Test_LOSS = torch.tensor(0, dtype=torch.float32).cuda()
                        mAP, F1, hammingLoss, _, _ = validate(test_loader, model, criterion, 0, alpha, beta, args,
                                                        Test_LOSS, result_path, record=True)
                        if sub == 4 and roi == 5: # 最后一个被试的最后一个脑区做完结束
                            return
                        continue

                for epoch in range(args.start_epoch,args.epochs):
                    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad,model.parameters()), lr=learning_rate) # filter可以去掉没有grad的参数
                    if epoch <41 :
                        alpha = beta/40*(epoch)
                        alpha = 0.1
                    else:
                        alpha = beta
                    print('image:{:4f},fMRI:{:4f}'.format((beta-alpha),(1-beta+alpha)))
                    best_epoch = 0
                    train(train_loader, model, criterion, optimizer, epoch,alpha,beta, args,Train_LOSS,result_path) # 训练模型
            
                    # evaluate on validation set
                    with torch.no_grad():
                        mAP,F1,hammingLoss,ave_recall, class_wise_recall, class_wise_F1  = validate(test_loader, model, criterion, epoch,alpha,beta, args,Test_LOSS,result_path)
                        mAP_all[sub,roi] = mAP
                        F1_all[sub,roi] = F1
                        HL_all[sub,roi] = hammingLoss
                    # remember best prec@1 and save checkpoint
                    is_best = mAP > best_prec1[sub,roi]
                    if is_best:
                        best_epoch = epoch
                        np.save(args.resume+'/result/'+soffix+'/time_point_14/ave_recall.npy', ave_recall)
                        np.save(args.resume+'/result/'+soffix+'/time_point_14/class_wise_rescall.npy', class_wise_recall)
                        np.save(result_path + 'best_class_wise_F1', class_wise_F1)
                    best_prec1[sub,roi] = max(mAP, best_prec1[sub,roi])
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_mAP': mAP,
                    }, is_best,args,save_model_path)
            # =============================================================================
            #     np.save('result_path/Train_LOSS.npy',Train_LOSS)
            #     np.save('results/Test_LOSS.npy',Test_LOSS)
            # =============================================================================
        np.save(args.resume+'/result/'+soffix+'/time_point_14/mAP_all.npy',mAP_all)
        np.save(args.resume+'/result/'+soffix+'/time_point_14/F1_all.npy',F1_all)
        np.save(args.resume+'/result/'+soffix+'/time_point_14/HL_all.npy',HL_all)
        np.save(args.resume+'/result/'+soffix+'/time_point_14/best_mAP_all.npy',best_prec1)
        print('best_epoch: {}'.format(best_epoch))
        torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, epoch, alpha,beta,args,Train_LOSS,result_path):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
# =============================================================================
#     model.resnet_101.eval()
#     model.resnet_101.layer4.train()
# =============================================================================
    for i, (input_fMRI, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = torch.tensor(target).cuda()
        #input_var_image = torch.tensor(input_image).cuda()
        input_var_fMRI = torch.tensor(input_fMRI).cuda()
        # compute output
        #inp  = input_image.cuda().data.cpu().numpy()
        #np.save('input.npy',inp)
        t1  = time.time()
        output,voxel_weight,coefficient = model(input_var_fMRI,alpha,beta)
        target = target.float()
        target = target.cuda()  
        target_var = torch.autograd.Variable(target)
        #loss = distance
        loss = criterion(output, target_var)
        #print('::::',distance)
        #losses.update(loss.item(), input_image.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        los = loss.cuda().data.cpu().numpy()
        loss.backward()
        optimizer.step()
        output_cpu = output.cuda().data.cpu()
        target_var_cpu = target_var.cuda().data.cpu()
        acc,pre,b,hammingLoss= accuracy(output, target_var)
        if i % args.print_freq == 0:
            #print('out:',b)
            #print('tra:',target_var)
            #print("==========:",matrix)
            #print('Epoch:{},step:{:02d},recall:{:4f},pre:{:4f},Loss:{loss.val:.4f}'.format(epoch, i,acc,pre,loss=losses))
            print('Epoch:{},step:{:02d},recall:{:4f},pre:{:4f},hammingLoss:{:4f},Loss:{:.4f}'.format(epoch, i,sum(acc)/52,sum(pre)/52,hammingLoss,criterion(output, target_var)))
            #Train_LOSS[int(4*epoch+i/10)] = los
            #print('LOSS!!!!!!:',los)
            np.save(result_path+'Train_LOSS.npy',Train_LOSS)
            #print('prediction:',output)
            #print('target:',target_var)

def validate(val_loader, model, criterion, epoch,alpha,beta,args,Test_LOSS,result_path,record=True):
    global num_test
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    CE_criterion = nn.CrossEntropyLoss(reduction='sum')
    x=[]
    for i, (input_fMRI ,target) in enumerate(val_loader):
        target = torch.tensor(target).cuda()
        #input_var_image = torch.tensor(input_image).cuda()
        input_var_fMRI = torch.tensor(input_fMRI).cuda()
        output,voxel_weight,coefficient = model(input_var_fMRI,alpha,beta)
        target = target.float()
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        #loss = distance
        loss = criterion(output, target_var)
        CELoss = CE_criterion(output, target_var)
        if args.evaluate:
            Test_LOSS += CELoss
        else:
            Test_LOSS[epoch] += CELoss.item()
        #losses.update(loss.item(),input_image.size(0))
        mask = (target > 0).float() # ?
        los = loss.cuda().data.cpu().numpy()
        v = torch.cat((output, mask),1) # 网络输出和真实标签的结合
        x.append(v)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        output_cpu = output.cuda().data.cpu().numpy()
        coeffi = coefficient.cuda().data.cpu().numpy()
        target_var_cpu = target_var.cuda().data.cpu().numpy()
        acc,pre,b,hammingLoss= accuracy(output, target_var)

        if i % args.print_freq == 0:
            print('Test: step:{}'
                  'acc:{:4f},pre:{:4f},hammingLoss:{:4f}'
                  'Loss:{:.4f}'.format(
                   i, sum(acc)/52,sum(pre)/52, hammingLoss,criterion(output, target_var)))
            #Test_LOSS[epoch] = los
            # if record:
                # np.save(result_path+'Test_LOSS.npy',Test_LOSS)
# =============================================================================
#             print('prediction:',output)
#             print('target:',target_var)
# =============================================================================
    Test_LOSS[epoch] /= num_test
    Ave_Test_LOSS = Test_LOSS
    x = torch.cat(x,0) # 500 * （52 + 52）
    x = x.cpu().detach().numpy()
    acc = acc.cpu().detach().numpy() # 只有最后一个batch的recall？
    pre = pre.cpu().detach().numpy()
    voxel_weight = voxel_weight.cpu().detach().numpy()

    if record:
        # print('save')
        np.savetxt(result_path+'_score', x)
        mAP,aps,recall,precise,predict,pre_score,hammingLoss,F1, ave_recall, class_wise_recall, class_wise_F1 =voc12_mAP(result_path+'_score', args.num_classes)
        # np.save(result_path+'Test_LOSS.npy', Ave_Test_LOSS)
        if args.evaluate:
            np.save(result_path + 'Ave_Test_LOSS.npy', Ave_Test_LOSS.cpu().numpy())
        else:
            np.save(result_path + 'Ave_Test_LOSS.npy', Ave_Test_LOSS)
        np.save(result_path+'class_wise_F1', class_wise_F1)
        np.save(result_path+'recall.npy',acc)
        np.save(result_path+'recall_all.npy',recall)
        np.save(result_path+'repcise_all.npy',precise)
        np.save(result_path+'precise.npy',pre)
        np.save(result_path+'voxel_weight.npy',voxel_weight)
        np.save(result_path+'aps.npy',aps)
        np.save(result_path+'predict.npy',predict)
        np.save(result_path+'coefficient.npy',coeffi)
        np.save(result_path+'pre_score.npy',pre_score)
        np.save(result_path + 'ave_recall.npy', ave_recall)
        np.save(result_path + 'class_wise_rescall.npy', class_wise_recall)


        print(' * mAP {mAP:.3f}'.format(mAP=mAP))
        print(' * hammingLoss {hammingLoss:.3f}'.format(hammingLoss = hammingLoss))
        print(' * F1 {F1:.3f}'.format(F1 = F1))
    #print(top5.val)
    #print(predict)
# =============================================================================
#     print(recall)
#     print(precise)
# =============================================================================
    return mAP,F1,hammingLoss,ave_recall, class_wise_recall, class_wise_F1

def save_checkpoint(state, is_best, args,filename):
    file_name = filename+'checkpoint.pth.tar'
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, filename+'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




# =============================================================================
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
# 
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     print(pred)
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
# 
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
# =============================================================================

# =============================================================================
# def accuracy(output, target):
#     assert output.shape == target.shape, \
#         "shapes output: {}, and target:{} are not aligned. ".\
#             format(output.shape, target.shape)
#     output.sigmoid_()
#     return torch.round(output).eq(target).sum().cpu().numpy()/target.numel()
# =============================================================================
def accuracy(output, target):
# =============================================================================
#     a = nn.Sigmoid()
#     b = a(output)
# =============================================================================
    
    prediction = output[:,:] > 0.2
    prediction = prediction.float()
    a = prediction+target
    #print(prediction.shape,"*******")
    temp = sum((prediction != target)) # 1*52 取每类下FP和FN的数量

    #print(temp)
    true_p  = a[:,:] > 1 # 64*52 TP
    false_p = a[:,:] <= 1 # 64*52 FP+TN+FN
    FP      = torch.sum(false_p.float(),0) # 52   64个sample中每类下预测错的的总数
    miss_pairs = sum(temp)
    num_instance,num_class = target.shape
    #print(miss_pairs)
    hammingLoss = miss_pairs/(num_class*num_instance)
    TP      = torch.sum(true_p.float(),0)
    T_total = torch.sum(target,0)
    P_total = torch.sum(prediction,0)
    #print('tp: ',true_p)
    #print('true',true_p)
    recall  = torch.div(TP,T_total+0.0001) # +0.0001是避免除0？ 这里实际上为Accuracy？
    Precise = torch.div(TP,P_total+0.0001)
    re = sum(recall)/52
    pr = sum(Precise)/52
    F1 = 2*re*pr/(re+pr)
    #print('F1:   ',F1)
    return recall,Precise,prediction,float(miss_pairs/(num_class*num_instance))

def model_option(option,args):
    if option == 'fSGL':
        model = fSGL(image_feature_dim=2048,
        output_dim=2048, time_step=args.timestep,
        adjacency_matrix=args.graph_file,
        word_features=args.word_file,
        num_classes=args.num_classes)
        lr = args.lr
        
    if option == 'GRU':
        model = GRU(image_feature_dim=2048,
        output_dim=2048, time_step=args.timestep,
        adjacency_matrix=args.graph_file,
        word_features=args.word_file,
        num_classes=args.num_classes)
        lr = args.lr*0.1
        
    if option == 'DNN':
        model = DNN(image_feature_dim=2048,
        output_dim=2048, time_step=args.timestep,
        adjacency_matrix=args.graph_file,
        word_features=args.word_file,
        num_classes=args.num_classes)
        lr = args.lr*0.1
        
    if option == 'RR':
        model = Linear(image_feature_dim=2048,
        output_dim=2048, time_step=args.timestep,
        adjacency_matrix=args.graph_file,
        word_features=args.word_file,
        num_classes=args.num_classes)
        lr = args.lr*0.1
        
    if option == 'LSTM':
        model = LSTM(image_feature_dim=2048,
        output_dim=2048, time_step=args.timestep,
        adjacency_matrix=args.graph_file,
        word_features=args.word_file,
        num_classes=args.num_classes)
        lr = args.lr*0.1

    if option == 'Transformer':
        model = Transformer(image_feature_dim=2048,
        output_dim=2048, time_step=args.timestep,
        adjacency_matrix=args.graph_file,
        word_features=args.word_file,
        num_classes=args.num_classes)
        lr = args.lr*0.1

    if option == 'fSGCN':
        model = fSGCN(image_feature_dim=2048,
             output_dim=2048, time_step=args.timestep,
             adjacency_matrix=args.symmetry_graph_file,
             word_features=args.word_file,
             num_classes=args.num_classes)
        lr = args.lr

    return model,lr

if __name__=="__main__":
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    main()




