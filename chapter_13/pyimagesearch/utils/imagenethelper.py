#encoding:utf-8
import numpy as np
import progressbar
import os

class ImageNetHelper(object):
    def __init__(self,config):
        # 配置
        self.config = config
        # 标签映射
        self.labelMappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlacklist()

    def _pbar(self,maxval,name):
        widgets = [name, progressbar.Percentage(), ' ',
                   progressbar.Bar(), ' ', progressbar.ETA()]

        pbar = progressbar.ProgressBar(maxval=maxval,
                                       widgets=widgets).start()
        return pbar

    def buildClassLabels(self):
        # 文件名映射类标签
        # n02110185 3 Siberian_husky
        rows = open(self.config.WORD_IDS).read().strip().split('\n')
        labelMappings = {}

        for row in rows:
            (wordId,label,hrLabel) = row.split(" ")

            labelMappings[wordId] = int(label) - 1
        return labelMappings

    def buildBlacklist(self):
        # 验证集
        rows = open(self.config.VAL_BLACKLIST).read()
        rows = set(rows.strip().split("\n"))
        return rows

    def buildTrainingSet(self):
        # 训练数据集
        # n01440764/n01440764_12131 189
        rows = open(self.config.TRAIN_LIST).read().strip()
        rows = rows.split('\n')
        paths = []
        labels = []
        probar = self._pbar(name='building training set: ',maxval=len(rows))
        for i,row in enumerate(rows):
            (partialPath,imageNum) = row.strip().split(" ")
            # 原始图像数据路径
            path = os.path.sep.join([self.config.IMAGES_PATH,
                                     'train','{}.JPEG'.format(partialPath)])
            # wordId
            wordId = partialPath.split("/")[0]
            label = self.labelMappings[wordId]

            paths.append(path)
            labels.append(label)
            probar.update(i)
        probar.finish()
        return (np.array(paths),np.array(labels))

    def buildValidationSet(self):

        paths = []
        labels = []
        #验证数据
        # ILSVRC2012_val_00000001 1
        valFilenames = open(self.config.VAL_LIST).read()
        valFilenames = valFilenames.strip().split('\n')

        # 验证集对应的标签
        # 490
        valLabels = open(self.config.VAL_LABELS).read()
        valLabels = valLabels.strip().split("\n")
        probar = self._pbar(name='building validation set: ',maxval=len(valFilenames))
        for i,(row,label) in enumerate(zip(valFilenames,valLabels)):
            (partialPath,imageNum)  =  row.strip().split(" ")

            if imageNum  in self.valBlacklist:
                continue
            #val数据集真实图片数据
            path = os.path.sep.join([self.config.IMAGES_PATH,'val',
                                     "{}.JPEG".format(partialPath)])
            paths.append(path)
            labels.append(int(label) - 1)
            probar.update(i)
        probar.finish()
        return (np.array(paths),np.array(labels))


