#-*-coding:utf-8-*-

import os

from configs.mscoco.mbv2_config import config as mb2_config
from configs.mscoco.shufflenet_5x5_config import config as shufflenet_config
from configs.mscoco.resnet18_config import config as resnet_config
##### the config for different task
config=resnet_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config.TRAIN.num_gpu = 1




