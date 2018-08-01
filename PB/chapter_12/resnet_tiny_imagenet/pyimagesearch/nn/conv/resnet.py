#encoding:utf-8
# 加载所需模块
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet:
    @staticmethod
    def residual_module(data,K,stride,chanDim,red = False,reg = 0.0001,bnEps=2e-5,bnMom=0.9):
        # resnet模块中最短的分支
        # 恒等映射
        shortcut = data
        # ResNet第一个block是1x1卷积
        bn1 = BatchNormalization(axis = chanDim,epsilon=bnEps,momentum = bnMom)(data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(int(K*0.25),(1,1),use_bias = False,kernel_regularizer=l2(reg))(act1)
        # ResNet第二个block是3x3卷积
        bn2 = BatchNormalization(axis = chanDim,epsilon=bnEps,momentum=bnMom)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(K*0.25),(3,3),strides = stride,padding='same',use_bias=False,kernel_regularizer=l2(reg))(act2)
        # ResNet第三个block是1x1卷积
        bn3 = BatchNormalization(axis = chanDim,epsilon=bnEps,momentum=bnMom)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(K,(1,1),use_bias=False,kernel_regularizer=l2(reg))(act3)
        #如果我们想搞降低feature map的个数，可以使用1x1卷积
        if red:
            shortcut = Conv2D(K,(1,1),strides=stride,use_bias=False,kernel_regularizer=l2(reg))(act1)
        # 两个分支输出相加
        x = add([conv3,shortcut])
        return x

    @staticmethod
    def build(width,height,depth,classes,stage,filters,reg = 0.0001,bnEps=2e-5,bnMom=0.9,dataset='cifar'):
        # 初始化输入shape
        inputShape = (height,width,depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
        # 输入层
        inputs = Input(shape=inputShape)
        # 对输入进行BN
        x = BatchNormalization(axis = chanDim,epsilon = bnEps,momentum = bnMom)(inputs)
        # 使用数据类型
        if dataset == 'cifar':
            # 单个卷积层
            x = Conv2D(filters[0],(3,3),use_bias = False,padding='same',kernel_regularizer = l2(reg))(x)
        elif dataset =='tiny_imagenet':
            # CONV => BN => ACT => POOL
            # 减少feature map的个数
            x = Conv2D(filters[0],(5,5),use_bias = False,padding='same',kernel_regularizer=l2(reg))(x)
            x = BatchNormalization(axis = chanDim,epsilon = bnEps,momentum=bnMom)(x)
            x = Activation('relu')(x)
            x = ZeroPadding2D((1,1))(x)
            x = MaxPooling2D((3,3),strides = (2,2))(x)
        # 遍历阶段个数
        for i in range(0,len(stage)):
            # 初始化步长
            # 减少feature map的个数
            stride = (1,1) if i==0 else (2,2)
            x = ResNet.residual_module(x,filters[i+1],stride,chanDim,red=True,bnEps = bnEps,bnMom=bnMom)
            # 每阶段的层个数
            for j in range(0,stage[i] -1):
                x = ResNet.residual_module(x,filters[i+1],(1,1),chanDim,bnEps=bnEps,bnMom=bnMom)
        # BN => ACT => POOL
        x = BatchNormalization(axis=chanDim,epsilon = bnEps,momentum = bnMom)(x)
        x =Activation('relu')(x)
        x = AveragePooling2D((8,8))(x)
        # 分类器
        x = Flatten()(x)
        x = Dense(classes,kernel_regularizer=l2(reg))(x)
        x = Activation('softmax')(x)
        # 创建模型
        model = Model(inputs,x,name='resnet')
        return model



