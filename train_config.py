#-*-coding:utf-8-*-

import os

from configs.mscoco.mbv2_config import config as mb2_config
from configs.mscoco.shufflenet_5x5_config import config as shufflenet_config
from configs.mscoco.resnet18_config import config as resnet_config
##### the config for different task
config=mb2_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config.TRAIN.num_gpu = 1





seed=42
from lib.core.utils.torch_utils import seed_everything
seed_everything(seed)



