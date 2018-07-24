# -*- coding: utf-8 -*-
'''
from keras.callbacks import ModelCheckpoint
import os

class EpochCheckpoint(ModelCheckpoint):
    def __init__(self,filepath,every,startAt=0):
        super(EpochCheckpoint,self).__init__(filepath)
        self.every = every
        self.startAt = startAt

    def on_epoch_end(self,epoch,logs = {}):
        filepath = os.path.join(self.filepath,"epoch_{epoch:02d}.hdf5")
        if (epoch+self.startAt+1) % self.every == 0:
            filepath = filepath.format(epoch = epoch +1+self.startAt)
            self.model.save(filepath,overwrite = True)
'''
from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt=0):
		# call the parent constructor
		super(Callback, self).__init__()

		# store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, logs={}):
		# check to see if the model should be serialized to disk
		if (self.intEpoch + 1) % self.every == 0:
			path = os.path.sep.join([self.outputPath, "epoch_{}.hdf5".format(self.intEpoch + 1)])
			self.model.save(path, overwrite=True)

		# increment the internal epoch counter
		self.intEpoch += 1
