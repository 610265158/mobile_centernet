#-*-coding:utf-8-*-

import os

from configs.mscoco.mbv2_config import config as mb2_config

##### the config for different task
config=mb2_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config.TRAIN.num_gpu = 1




