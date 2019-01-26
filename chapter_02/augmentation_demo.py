#encoding:utf-8
# 加载所需要的模块
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

# 构造参数解析和解析参数
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help = 'path to the input image')
ap.add_argument('-o','--ouput',required=True,help ='path to ouput directory to store augmentation examples')
ap.add_argument('-p','--prefix',type=str,default='image',help='output fielname prefix')
args = vars(ap.parse_args())

# 加载图像，并转化为numpy的array
print('[INFO] loading example image...')
image = load_img(args['image'])
image = img_to_array(image)
#增加一个维度
image = np.expand_dims(image,axis = 0) #在0位置增加数据，主要是batch size

aug = ImageDataGenerator(
    rotation_range=30, # 旋转角度
    width_shift_range=0.1,#水平平移幅度
    height_shift_range= 0.1,#上下平移幅度
    shear_range=0.2,# 逆时针方向的剪切变黄角度
    zoom_range=0.2,#随机缩放的角度
    horizontal_flip=True,#水平翻转
    fill_mode='nearest'#变换超出边界的处理
)
# 初始化目前为止的图片产生数量
total = 0

print("[INFO] generating images...")
imageGen = aug.flow(image,batch_size=1,save_to_dir=args['output'],save_prefix=args['prefix'],save_format='jpg')
for image in imageGen:
    total += 1
    if total == 10:
        break

# python augmentation_demo.py --image jemma.png --output output