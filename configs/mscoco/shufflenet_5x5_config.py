#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu
config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 5                      ### process_num for data provider
config.TRAIN.prefetch_size = 4                  ### prefect Q size for data provider
config.TRAIN.test_interval=1
config.TRAIN.num_gpu = 1                         ##match with   os.environ["CUDA_VISIBLE_DEVICES"]
config.TRAIN.batch_size = 32                    ###A big batch size may achieve a better result, but the memory is a problem
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 50                      ###just keep training , evaluation shoule be take care by yourself,
                                               ### generally 10,0000 iters is enough

config.TRAIN.train_set_size=117266            ###widerface train size
config.TRAIN.val_set_size=5000             ###widerface val size


config.TRAIN.lr_decay='cos'
config.TRAIN.init_lr=0.001
config.TRAIN.warmup_step=1000
config.TRAIN.opt='Adamw'
config.TRAIN.weight_decay_factor = 1.e-5                  ##l2 regular
config.TRAIN.vis=False                                    ##check data flag

if config.TRAIN.vis:
    config.TRAIN.mix_precision=False
else:
    config.TRAIN.mix_precision = True

config.TRAIN.norm='BN'    ##'GN' OR 'BN'
config.TRAIN.lock_basenet_bn=False
config.TRAIN.frozen_stages=-1   ##no freeze
config.TRAIN.gradient_clip=False
config.TRAIN.SWA=-1

config.TRAIN.ema=False
config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.num_category=80                                  ###face 1  voc 20 coco 80
config.DATA.num_class = config.DATA.num_category


config.DATA.hin = 512  # input size
config.DATA.win = 512
config.DATA.channel = 3
config.DATA.max_size=[config.DATA.hin,config.DATA.win]  ##h,w
config.DATA.cover_obj=8                          ###cover the small objs

config.DATA.mutiscale=False                #if muti scale set False  then config.DATA.MAX_SIZE will be the inputsize
config.DATA.scales=(320,640)
config.DATA.use_int8_data=True
config.DATA.use_int8_enlarge=255.           ### use uint8 for heatmap generate for less memory acc, to speed up
config.DATA.max_objs=128
config.DATA.cracy_crop=0.0
config.DATA.alpha=0.54
config.DATA.beta=0.54
##mobilenetv3 as basemodel
config.MODEL = edict()
config.MODEL.net_structure='ShuffleNetV2'

config.MODEL.model_path = './model/'  # save directory
config.MODEL.pretrained_model=None
config.MODEL.task='mscoco'
config.MODEL.min_overlap=0.7
config.MODEL.max_box= 100



##model params
config.MODEL.global_stride=4
config.MODEL.backbone_feature_dims=[24,116,232,464]   ##c2,c3,c4,c5
config.MODEL.head_dims=[128,192,256]                ## c2,c3,c4

if config.MODEL.global_stride==8:
    config.MODEL.backbone_feature_dims = [ 116,232,464,480]  ##c3,c4,c5,c6

config.MODEL.freeze_bn=False




