# -*- coding: utf-8 -*-
# 加载所需模块
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras import backend as K

class MiniGoogLeNet:
    @staticmethod
    def conv_module(x,K,kX,kY,stride,chanDim,padding='same'):
        # define a CONV => BN => RELU pattern
        x = Conv2D(K,(kX,kY),strides = stride,padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation('relu')(x)
        return x
    
    @staticmethod
    def inception_module(x,numK1x1,numK3x3,chanDim):
        # 拼接两个CONV层
        conv_1x1 = MiniGoogLeNet.conv_module(x,numK1x1,1,1,(1,1),chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x,numK3x3,3,3,(1,1),chanDim)
        x = concatenate([conv_1x1,conv_3x3],axis = chanDim)
        return x
    
    @staticmethod
    def downsample_module(x,K,chanDim):
        # 定义CONV和POOL，并拼接
        conv_3x3 = MiniGoogLeNet.conv_module(x,K,3,3,(2,2),chanDim,padding='valid')
        pool = MaxPooling2D((3,3),strides=(2,2))(x)
        x = concatenate([conv_3x3,pool],axis = chanDim)
        return x
    
    @staticmethod
    def build(width,height,depth,classes):
        inputShape =(height,width,depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
        # 输入层和第一个CONV层
        inputs= Input(shape = inputShape)
        x = MiniGoogLeNet.conv_module(inputs,96,3,3,(1,1),chanDim)
        
        # 两个Inception和一个downsample层
        x = MiniGoogLeNet.inception_module(x,32,32,chanDim)
        x = MiniGoogLeNet.inception_module(x,32,48,chanDim)
        x = MiniGoogLeNet.downsample_module(x,80,chanDim)
        
        # 四个Inception和一个downsample层
        x = MiniGoogLeNet.inception_module(x,112,48,chanDim)
        x = MiniGoogLeNet.inception_module(x,96,64,chanDim)
        x = MiniGoogLeNet.inception_module(x,80,80,chanDim)
        x = MiniGoogLeNet.inception_module(x,48,96,chanDim)
        x = MiniGoogLeNet.downsample_module(x,96,chanDim)
        
        # 两个Inception和global POOL ，dropout
        x = MiniGoogLeNet.inception_module(x,176,160,chanDim)
        x = MiniGoogLeNet.inception_module(x,176,160,chanDim)
        x = AveragePooling2D((7,7))(x)
        x = Dropout(0.5)(x)
        
        # softmax分类器
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation('softmax')(x)
        
        # 建立模型
        model = Model(inputs,x,name='googlenet')
        return model
    
    