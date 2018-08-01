# -*- coding: utf-8 -*-
import cv2
class MeanPreprocessor:
    def __init__(self,rMean,gMean,bMean):
        # 三个颜色通道的平均值
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        
    def preprocess(self,image):
        # cv2分割得到的是BGR，而不是RGB
        (B,G,R) = cv2.split(image.astype("float32"))
        # 减去对应通道的均值
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        # 
        return cv2.merge([B,G,R])
        

