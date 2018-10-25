#encoding:utf-8
from os import path

# 定义Imagenet数据集路径
BASE_PATH = 'ILSVRC'

# 基于base path 定义原始图像和工具路径
IMAGES_PATH = path.sep.join([BASE_PATH,'Data/CLS-LOC'])
IMAGE_SETS_PATH = path.sep.join([BASE_PATH,'ImageSets/CLS-LOC/'])
DEVKIT_PATH = path.sep.join([BASE_PATH,'devkit/data'])

# 定义WordNet IDs文件路径
WORD_IDS = path.sep.join([DEVKIT_PATH,'map_clsloc.txt'])

# 定义training文件路径
TRAIN_LIST = path.sep.join([IMAGE_SETS_PATH,'train_cls.txt'])

# 定义验证集数据路径以及对应的标签文件路径
VAL_LIST = path.sep.join([IMAGE_SETS_PATH,'val.txt'])
VAL_LABELS = path.sep.join([DEVKIT_PATH,'ILSVRC2015_clsloc_validation_ground_truth.txt'])
# 定义val blacklisted 文件路径
VAL_BLACKLIST = path.sep.join([DEVKIT_PATH,'ILSVRC2015_clsloc_validation_blacklist.txt'])

# 定义类别个数
# 定义我们需要从train数据集中划分一个子集作为test的大小
NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# 定义tfrecord文件的输出路径
TF_OUTPUT = 'imagenet'
TRAIN_TFRECORD = path.sep.join([TF_OUTPUT,'tfrecords/train.tfrecords'])
VAL_TFRECORD  = path.sep.join([TF_OUTPUT,'tfrecords/val.tfrecords'])
TEST_TFRECORD = path.sep.join([TF_OUTPUT,'tfrecords/test.tfrecords'])

# 定义均值文件路径
DATASET_MEAN = 'outputs/imagenet_mean.json'

# 定义batch大小
BATCH_SIZE = 128
