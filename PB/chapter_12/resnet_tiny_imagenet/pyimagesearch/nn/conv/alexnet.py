# -*- coding: utf-8 -*-
# 加载所需模块

from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.regularizers import l2
from keras import backend as K

class AlexNet:
    @staticmethod
    def build(width,height,depth,classes,reg=0.0002):
        # 初始化序列模型
        model = Sequential()
        inputShape = (height,width,depth)
        chanDim = -1
        # 主要识别keras的后端是thensorflow还是theano[目前默认都是thensorflow]
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
            
        # Block # 1 first CONV => RELU => POOL layer set
        model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=inputShape,
                         padding='same',kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (3,3),strides=(2,2)))
        model.add(Dropout(0.25))
            
        # Block #2: second CONV => RELU => POOL layer set
        model.add(Conv2D(256,(5,5),padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(Dropout(0.25))

        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(384,(3,3),padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(384,(3,3),padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256,(3,3),padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(Dropout(0.25))
        
        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096,kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))
        
        # Block #5: second set of FC => RELU layers
        model.add(Dense(4096,kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))
        
        # softmax 分类器
        model.add(Dense(classes,kernel_regularizer=l2(reg)))
        model.add(Activation('softmax'))
        
        #返回模型结构
        return  model
        