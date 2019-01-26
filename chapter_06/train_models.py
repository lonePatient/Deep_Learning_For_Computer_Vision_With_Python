#encoding:utf-8
# 设置图像的背景
import matplotlib
matplotlib.use('agg')

# 加载模块
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet as MVN
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,help='path to output directory')
ap.add_argument('-m','--models',required=True,help='path to output models directory')
ap.add_argument('-n','--num_models',type = int,default=5,help='# of models to train')
args = vars(ap.parse_args())

# 加载数据集，并划分为train和test数据
# 图像数据预处理：归一化
((trainX,trainY),(testX,testY)) = cifar10.load_data()
trainX  = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0

# 标签进行编码化处理

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# 初始化标签名陈
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]


# 初始化数据增强模块
aug = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,
                         height_shift_range=0.1,horizontal_flip=True,
                         fill_mode='nearest')

# 遍历模型训练个数
for i in np.arange(0,args['num_models']):
    # 初始化优化器和模型
    print("[INFO] training model {}/{}".format(i+1,args['num_models']))
    opt = SGD(lr = 0.01,decay=0.01/ 40,momentum=0.9,
              nesterov=True)
    model = MVN.MiniVGGNet.build(width=32,height=32,depth=3,
                                 classes = 10)
    model.compile(loss = 'categorical_crossentropy',optimizer=opt,metrics = ['accuracy'])

    # 训练网络
    H = model.fit_generator(aug.flow(trainX,trainY,batch_size=64),
                            validation_data=(testX,testY),epochs=40,
                            steps_per_epoch=len(trainX) // 64,verbose = 1)
    # 将模型保存到磁盘中
    p = [args['models'],"model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    # 评估模型
    predictions = model.predict(testX,batch_size=64)
    report = classification_report(testY.argmax(axis =1),
                                   predictions.argmax(axis =1),target_names=labelNames)
    # 将模型结果保存到文件中
    p = [args['output'],'model_{}.text'.format(i)]
    with open(os.path.sep.join(p),'w') as fw:
        fw.write(report)

    # loss函数可视化
    p = [args['output'],'model_{}.png'.format(i)]
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0,40),H.history['loss'],
             label = 'train_loss')
    plt.plot(np.arange(0,40),H.history['val_loss'],
             label = 'val_loss')
    plt.plot(np.arange(0,40),H.history['acc'],
             label = 'train-acc')
    plt.plot(np.arange(0,40),H.history['val_acc'],
             label = 'val-acc')
    plt.title("Training Loss and Accuracy for model {}".format(i))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()




