#encoding:utf-8
import numpy as np

def rank5_accuracy(preds,labels):
    #初始化
    rank1 = 0
    rank5 = 0
    # 遍历数据集
    for (p,gt) in zip(preds,labels):
        # 通过降序对概率进行排序
        p = np.argsort(p)[::-1]
        # 检查真实标签是否落在top5中
        if gt in p[:5]:
            rank5 += 1
        # 检验真实标签是否等于top1
        if gt == p[0]:
            rank1 += 1
        # 计算准确度
        rank1 /= float(len(labels))
        rank5 /= float(len(labels))

        return rank1,rank5


