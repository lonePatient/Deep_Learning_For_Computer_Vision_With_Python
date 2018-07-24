# -*- coding: utf-8 -*-
# 加载所需模块
from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io import hdf5datasetwriter as HDFW
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# 获取训练数据
trainPaths  = list(paths.list_images(config.TRAIN_IMAGES))
# 提取对应标签
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
# one-hot 编码
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# 数据分割
split = train_test_split(trainPaths,trainLabels,
                        test_size = config.NUM_TEST_IMAGES,
                        stratify = trainLabels,
                        random_state = 42)
(trainPaths,testPaths,trainLabels,testLabels) = split

# 读取验证书籍，并映射对应标签
M = open(config.VAL_MAPPINGS).read().strip().split("\n")
M = [r.split("\t")[:2] for r in M]
valPaths = [os.path.sep.join([config.VAL_IMAGES,m[0]]) for m in M ]
valLabels = le.transform([m[1] for m in M  ])

# 将数据组成一个列表，方便使用
datasets = [
        ("train",trainPaths,trainLabels,config.TRAIN_HDF5),
        ("val",valPaths,valLabels,config.VAL_HDF5),
        ("test",testPaths,testLabels,config.TEST_HDF5)]

# 初始化RGB三个颜色通道
(R,G,B) = ([],[],[])

# 遍历数据元祖
for (dType,paths,labels,outputPath) in datasets:
    # 初始化HDF5写入
    print("[INFO] building {} ....".format(outputPath))
    writer = HDFW.HDF5DatasetWriter((len(paths),64,64,3),outputPath)
    
    # 初始化进度条
    widgets = ['Building Dataset: ',progressbar.Percentage()," ",
               progressbar.Bar()," ",progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval = len(paths),
                                   widgets = widgets).start()
    # 遍历图像路径
    for (i,(path,label))  in enumerate(zip(paths,labels)):
        # 从磁盘中读取数据
        image = cv2.imread(path)
        
        #计算均值
        if dType == "train":
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        # 将图像跟标签写入HDF5中
        writer.add([image],[label])
        pbar.update(i)
    
    # 关闭数据库
    pbar.finish()
    writer.close()
    
#保存均值文件
print("[INFO] serializing means...")
D = {"R":np.mean(R),"G":np.mean(G),"B":np.mean(B)}
f = open(config.DATASET_MEAN,'w')
f.write(json.dumps(D))
f.close()
    

    