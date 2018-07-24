# -*- coding: utf-8 -*-
from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:

    def __init__(self,dbPath,batchSize,preprocessors = None,aug = None,binarize=True,classes=2):
        # 保存参数列表
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        # hdf5数据集
        self.db = h5py.File(dbPath)
        self.numImages = self.db['labels'].shape[0]
    
    def generator(self,passes=np.inf):
        epochs = 0
        # 默认是无限循环遍历
        while epochs < passes:
            # 遍历数据
            for i in np.arange(0,self.numImages,self.batchSize):
                # 从hdf5中提取数据集
                images = self.db['images'][i: i+self.batchSize]
                labels = self.db['labels'][i: i+self.batchSize]
                # one-hot编码
                if self.binarize:
                    labels = np_utils.to_categorical(labels,self.classes)
                # 预处理
                if self.preprocessors is not None:
                    proImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        proImages.append(image)
                    images = np.array(proImages)
                if self.aug is not None:
                    (images,labels) = next(self.aug.flow(images,
                        labels,batch_size = self.batchSize))
                # 返回
                yield (images,labels)
            epochs += 1
    def close(self):
        # 关闭db
        self.db.close()

