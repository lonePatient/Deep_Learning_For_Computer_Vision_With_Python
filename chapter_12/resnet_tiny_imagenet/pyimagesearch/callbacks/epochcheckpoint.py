#encoding:utf-8
from keras.callbacks import ModelCheckpoint
import os

class EpochCheckpoint(ModelCheckpoint):
    def __init__(self, filepath,every, startAt=0):
        super(EpochCheckpoint, self).__init__(filepath)
        self.every = every
        self.startAt = startAt

    def on_epoch_end(self,epoch=24,logs= {}):
        filepath = os.path.join(self.filepath,"epoch_{epoch:02d}.hdf5")
        if (epoch+self.startAt+1) % self.every == 0:
            filepath = filepath.format(epoch=epoch + self.startAt + 1)
            self.model.save(filepath,overwrite = True)

