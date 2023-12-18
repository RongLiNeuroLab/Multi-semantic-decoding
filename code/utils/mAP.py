# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 09:45:52 2021

@author: 1
"""
import numpy as np
import sys
def voc_ap(rec, prec,true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap    

def mAP(predict,target):#这里的输入是所有测试样本对应的预测标签和真实标签，每一行对应一个样本
    class_num = 154
    seg = np.concatenate((predict,target),axis=1)
    gt_label = seg[:,class_num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims = True)
    #threshold = 1 / (num_target+1e-6)
    #predict_result = seg[:,0:class_num] > threshold    
    sample_num = len(gt_label)
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []
    recall = []
    precise = []
    for class_id in range(class_num):
        confidence = seg[:,class_id]
        sorted_ind = np.argsort(-confidence)
        #sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]   
        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp) 
        if true_num == 0:
            rec = tp/1000000
        else:
            rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        recall += [rec]
        precise += [prec]
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]
    np.set_printoptions(precision=3, suppress=True)
    mAPvalue = np.mean(aps)
    return mAPvalue,aps


mAP_sum = 0
predict_label = np.zeros((180,154))
aps_sum = np.zeros(154)
#target = np.load('A:/SSGRL-master/target.npy')
predict_result = np.load('F:/LJY/LJY/TEST_SMT_DECLAB.npy')
target         = np.load('F:/LJY/LJY/TEST_SMT_LAB.npy')
epoch_num = 10000
predict_baseline = np.zeros((180,154))

#计算180次采样结果
for epoch in range(180):
    predict = predict_result[epoch*50,:,:]
    mAP_one,aps_one = mAP(predict,target)
    mAP_sum += mAP_one
    aps_sum += aps_one
    if epoch % 1 == 0:
        sys.stdout.write('\r{:05d}次epoch已完成！！！！'.format(epoch))
        sys.stdout.flush()
    predict_label[epoch,:] = aps_one
np.save('F:/LJY/LJY/predict.npy',predict_label)

#计算baseline
mAP_sum = 0
aps_sum = np.zeros(154)
for epoch in range(epoch_num):
    predict = np.random.rand(250,154)
    mAP_one,aps_one = mAP(predict,target)
    mAP_sum += mAP_one
    aps_sum += aps_one
    if epoch % 1 == 0:
        sys.stdout.write('\r{:05d}次epoch已完成！！！！'.format(epoch))
        sys.stdout.flush()
print(mAP_sum/epoch_num,aps_sum/epoch_num)
np.save('F:/LJY/LJY/aps_baseline.npy',aps_sum/epoch_num)
