#-*-coding:utf-8-*-

import os

from configs.mscoco.mbnet_config import config as mb_config


##### the config for different task
config=mb_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config.TRAIN.num_gpu = 1





seed=42
from lib.core.utils.torch_utils import seed_everything
seed_everything(seed)



