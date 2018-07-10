#encoding:utf-8
# 加载所需要模块
from keras.applications import VGG16
import argparse

# 命令行参数设置
ap = argparse.ArgumentParser()
ap.add_argument('-i','--include_top',type = int,default=1,help='whether or not to include top of CNN')
args = vars(ap.parse_args())

# 加载VGG16模型
print('[INFO] loading network...')
model = VGG16(weights='imagenet',include_top=args['include_top']>0)
print("[INFO] showing layers...")

# 遍历VGG16所有层结构
for (i,layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i,layer.__class__.__name__))