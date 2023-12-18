import argparse
import os

import numpy as np
from utils.transforms import get_train_test_set
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils.metrics import voc12_mAP

global best_prec1
best_prec1 = np.zeros((5,6))
global sub_name
global ROI_name
sub_name = ["s_bailin","s_huangwei","s_lianlian","s_xiangchen","s_zhengzifeng"]
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
    parser.add_argument('--num_classes', '-n', default=52 , type=int, metavar='N',help='number of classes (default: 80)')
    parser.add_argument('--epochs',               default=10, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('--start-epoch',          default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
    parser.add_argument('--step_epoch',           default=31, type=int, metavar='N',help='decend the lr in epoch number')
    parser.add_argument('-b', '--batch-size',     default=64, type=int,metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate',default=0.0001, type=float,metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum',             default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--timestep',             default=2, type=int, metavar='M',help='timestep of GNN')

    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained', type=int,default=0,help='use pre-trained model')
    parser.add_argument('--crop_size',  dest='crop_size',default=224, type=int,help='crop size')
    parser.add_argument('--scale_size', dest = 'scale_size',default=256, type=int,help='the size of the rescale image')
    parser.add_argument('--evaluate',   dest='evaluate', default=False, action='store_true',help='evaluate model on validation set')
    parser.add_argument('--post',       dest='post', type=str,default='',help='postname of save model')
    parser.add_argument('-j', '--workers',        default=0, type=int, metavar='N',help='number of data loading workers (default: 4)')
    args = parser.parse_args()
    return args

def main():
    global best_prec1
    args = arg_parse()
    global sub_name
    global ROI_name
    global mAP_all
    global F1_all
    global HL_all
    global num_test, num_train
    # Create dataloader
    for fMRI_length in range(14, 15):
        for sub in range(5):
            for roi in range(3, 6):
                print("==> Creating dataloader...")
                train_label = args.train_label
                test_label = args.test_label
                args.start_epoch = 0
                soffix = 'F1_baseline_low_threshold'

                fMRI_dir = args.fMRI_dir + sub_name[sub] + '/' + ROI_name[roi] + '/'
                result_path = args.resume + '/result_test/' + soffix + '/time_point_{0:02d}/'.format(fMRI_length) + \
                              sub_name[sub] + '/' + ROI_name[roi] + '/'


                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                if not os.path.exists(args.resume + '/result/' + soffix):
                    os.mkdir(args.resume + '/result/' + soffix)
                if not os.path.exists(args.resume + '/model/' + soffix):
                    os.mkdir(args.resume + '/model/' + soffix)

                leng_dir = args.resume + '/model/' + soffix + '/time_point_14/'
                if not os.path.exists(leng_dir):
                    os.mkdir(leng_dir)
                file_dir = leng_dir + '/' + sub_name[sub]
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                img_dir = file_dir + '/' + ROI_name[roi]
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)

                leng_dir = args.resume + '/result/' + soffix + '/time_point_14/'
                if not os.path.exists(leng_dir):
                    os.mkdir(leng_dir)
                file_dir = leng_dir + '/' + sub_name[sub]
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                img_dir = file_dir + '/' + ROI_name[roi]
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)

                fMRI_train_list = args.ftrain_list
                fMRI_test_list = args.ftest_list
                train_loader, test_loader, num_train, num_test = get_train_test_set(fMRI_dir, fMRI_train_list,
                                                                                    fMRI_test_list, train_label,
                                                                                    test_label, fMRI_length, args)
                cudnn.benchmark = True
                all_epoch = 5000
                cw_F1_all = np.zeros(52)

                for epoch in range(all_epoch):
                    if epoch % 100 == 0:
                        print('Epoch: {}'.format(epoch))
                    x = []
                    for i, (input_fMRI, target) in enumerate(test_loader):
                        target = torch.tensor(target)
                        target = target.float()
                        batch_size = target.shape[0]
                        target_var = torch.autograd.Variable(target)
                        random_output = np.random.rand(batch_size, 52)
                        random_output = torch.tensor(random_output, dtype=torch.float)

                        mask = (target > 0).float()  # ?
                        v = torch.cat((random_output, mask), 1)  # 网络输出和真实标签的结合
                        x.append(v)

                    x = torch.cat(x, 0)  # 500 * （52 + 52）
                    x = x.cpu().detach().numpy()
                    np.savetxt(result_path + '_score', x)
                    mAP, aps, recall, precise, predict, pre_score, hammingLoss, F1, ave_recall, class_wise_recall, class_wise_F1 = voc12_mAP(
                        result_path + '_score', args.num_classes)
                    # np.save(result_path+'Test_LOSS.npy', Ave_Test_LOSS)

                    cw_F1_all[:] += class_wise_F1
                cw_F1_all /= all_epoch
                np.save(result_path + 'class_wise_F1', cw_F1_all)
                print('Done')
                torch.cuda.empty_cache()
                return






if __name__ == '__main__':
    main()