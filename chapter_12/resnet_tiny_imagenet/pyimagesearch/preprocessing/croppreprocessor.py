# -*- coding: utf-8 -*-

import numpy as np
import cv2

class CropPreprocessor:
    def __init__(self,width,height,horiz = True,inter=cv2.INTER_AREA):
        # 保存目标参数
        self.width  = width
        self.height = height
        self.horiz  = horiz
        self.inter  = inter
        
    def preprocess(self,image):
        crops = []
        # 原始图像的高跟宽
        (h,w) = image.shape[:2]
        #四个角
        coords = [
                [0,0,self.width,self.height],
                [w - self.width,0,w,self.height],
                [w - self.width,h - self.height,w,h],
                [0,h - self.height,self.width,h]
                ]
        # 计算中心区域
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW,dH,w - dW,h - dH])
        
        for (startX,startY,endX,endY) in coords:
            # 裁剪
            crop = image[startY:endY,startX:endX]
            # 由于裁剪过程，可能会造成大小相差1左右，所以进行插值
            crop = cv2.resize(crop,(self.width,self.height),
                              interpolation = self.inter)
            crops.append(crop)
        if self.horiz:
            # 水平翻转
            mirrors = [cv2.flip(x,1) for x in crops]
            crops.extend(mirrors)
        return np.array(crops)
        
