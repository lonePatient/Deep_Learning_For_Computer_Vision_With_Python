#encoding:utf-8
from keras.models import Sequential
from keras.layers.core import Activation,Flatten,Dense,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import  BatchNormalization
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential()
        inputShape = (height,width,depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1

        model.add(Conv2D(32,(3,3),padding='same',input_shape=inputShape,name='block1_conv1'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),padding='same',input_shape=inputShape,name='block1_conv2'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block1_pool'))
        model.add(Dropout(0.25))

        model.add(Conv2D(64,(3,3),padding='same',input_shape=inputShape,name='block2_conv1'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),padding='same',input_shape=inputShape,name='block2_conv2'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block2_pool'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
