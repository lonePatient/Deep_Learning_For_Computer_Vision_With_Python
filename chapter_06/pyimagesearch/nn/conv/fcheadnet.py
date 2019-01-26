#encoding:utf-8
# 加载所需要的模块
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class FCHeadNet:
    @staticmethod
    def build(baseModel,classes,D):
        # 初始化top部分
        headModel = baseModel.output
        headModel = Flatten(name='flatten')(headModel)
        headModel = Dense(D,activation='relu')(headModel)
        headModel = Dropout(0.5)(headModel)
        # 增加一个softmaxc层
        headModel = Dense(classes,activation='softmax')(headModel)
        return headModel