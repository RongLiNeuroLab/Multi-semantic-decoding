import numpy as np


def voc12_mAP(imagessetfile, num):
    with open(imagessetfile, 'r') as f:
        lines = f.readlines()

# 1.计算全类别的F1
    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:, num:].astype(np.int32)
    threshold = 1 / 5
    predict_result = seg[:, 0:num] > threshold

    temp = sum((predict_result != gt_label))
    miss_pairs = sum(temp)

    hammingLoss = miss_pairs / (500 * 52)

    prediction = predict_result.astype(np.int)
    target = gt_label.astype(np.int)
    a = prediction + target
    true_p = a[:, :] > 1
    false_p = a[:, :] <= 1
    TP = np.sum(true_p, 0)
    T_total = np.sum(target, 0)
    P_total = np.sum(prediction, 0)
    recall = TP / (T_total + 0.0001)
    Precise = TP / (P_total + 0.0001)
    re = sum(recall) / 52
    pr = sum(Precise) / 52
    F1 = 2 * re * pr / (re + pr)
    class_wise_recall = recall

# 2.计算每个类别单独的F1, cw : class wise
    cw_seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    cw_gt_label = cw_seg[:, num:].astype(np.int32)

    cw_predict_result = seg[:, 0:num] > threshold  # 卡阈值
    cw_prediction = cw_predict_result.astype(np.int)
    cw_target = cw_gt_label.astype(np.int)
    a = cw_prediction + cw_target
    true_p = a[:, :] > 1  # TP
    false_p = a[:, :] <= 1
    TP = np.sum(true_p, 0)  # 每类的TP
    T_total = np.sum(cw_target, 0)  # 每类的TP + FN 所有标签为正的都在这里
    P_total = np.sum(cw_prediction, 0)  # TP + FP 所有被预测为正的都在这里
    cw_re = TP / (T_total + 1e-10)  # TP/ (TP + FN)
    cw_pr = TP / (P_total + 1e-10)  # TP / (TP + FP)
    class_wise_F1 = 2 * cw_re * cw_pr / (cw_re + cw_pr + 1e-10)


# 3.计算
    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []
    per_class_recall = []
    recall = []
    precise = []
    for class_id in range(class_num):
        confidence = seg[:, class_id]
        sorted_ind = np.argsort(-confidence)  # 从大到小排输出的概率，并返回出索引
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]  # 按上面从大到小的顺序输出label

        for i in range(sample_num):
            tp[i] = (sorted_label[i] > 0) #
            fp[i] = (sorted_label[i] <= 0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num) # TP/ (TP + FN)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) # TP / (TP + FP)
        recall += [rec]
        precise += [prec]
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]
        # print(class_id,' recall: ',rec, ' precise: ',prec)
    np.set_printoptions(precision=3, suppress=True)

    mAP = np.mean(aps)
    return mAP, aps, recall, precise, predict_result, seg[:, 0:num], hammingLoss, F1, re, class_wise_recall, class_wise_F1


def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.])) # 502
    mpre = np.concatenate(([0.], prec, [0.])) # 502
    for i in range(mpre.size - 1, 0, -1): #倒着走，从小到大
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# =============================================================================
# voc12_mAP('A:/SSGRL-master/result_5/s_huangwei/OCC/_score',57)
# =============================================================================
