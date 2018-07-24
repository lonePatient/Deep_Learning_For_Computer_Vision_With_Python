# -*- coding: utf-8 -*-
from sklearn.feature_extraction.image import extract_patches_2d

class PatchPreprocessor:
    def __init__(self,width,height):
        # 目标图像的宽和高
        self.width = width
        self.height = height
        
    def preprocess(self,image):
        # 随机裁剪出目标大小图像
        return extract_patches_2d(image,(self.height,self.width),
                                  max_patches = 1)[0]