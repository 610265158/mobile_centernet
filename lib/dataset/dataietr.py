


import os
import random
import cv2
import numpy as np
import traceback

from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator
from tensorpack.dataflow import BatchData, PrefetchDataZMQ,RepeatedData


from lib.dataset.centernet_data_sampler import get_affine_transform,affine_transform
from lib.dataset.ttf_net_data_sampler import CenternetDatasampler


from lib.dataset.augmentor.augmentation import Random_scale_withbbox,\
                                                Random_flip,\
                                                baidu_aug,\
                                                dsfd_aug,\
                                                Fill_img,\
                                                Rotate_with_box,\
                                                produce_heatmaps_with_bbox,\
                                                box_in_img
from lib.dataset.augmentor.data_aug.bbox_util import *
from lib.dataset.augmentor.data_aug.data_aug import *
from lib.dataset.augmentor.visual_augmentation import ColorDistort,pixel_jitter

from lib.dataset.centernet_data_sampler import produce_heatmaps_with_bbox_official,affine_transform
from train_config import config as cfg


import math
import albumentations as A

class data_info():
    def __init__(self,img_root,txt):
        self.txt_file=txt
        self.root_path = img_root
        self.metas=[]


        self.read_txt()

    def read_txt(self):
        with open(self.txt_file) as _f:
            txt_lines=_f.readlines()
        txt_lines.sort()
        for line in txt_lines:
            line=line.rstrip()

            _img_path=line.rsplit('| ',1)[0]
            _label=line.rsplit('| ',1)[-1]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label
            self.metas.append([current_img_path,current_img_label])

            ###some change can be made here
        logger.info('the dataset contains %d images'%(len(txt_lines)))
        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas



class AlaskaDataIter():

    def __init__(self, img_root_path='', ann_file=None, training_flag=True, shuffle=True):

        self.color_augmentor = ColorDistort()

        self.training_flag = training_flag

        self.lst = self.parse_file(img_root_path, ann_file)

        self.shuffle = shuffle

        self.train_trans = A.Compose([
                                      A.RandomBrightnessContrast(p=0.75, brightness_limit=0.1, contrast_limit=0.2),

                                      A.CLAHE(clip_limit=4.0, p=0.7),
                                      A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                                           val_shift_limit=10, p=0.5),

                                      A.OneOf([
                                          A.MotionBlur(blur_limit=5),
                                          A.MedianBlur(blur_limit=5),
                                          A.GaussianBlur(blur_limit=5),
                                          A.GaussNoise(var_limit=(5.0, 30.0)),
                                      ], p=0.7)
                                      ])

        self.target_producer = CenternetDatasampler()


    def __getitem__(self, item):

        return self._map_func(self.lst[item], self.training_flag)

    def __len__(self):
        return len(self.lst)

    def parse_file(self,im_root_path,ann_file):
        '''
        :return: [fname,lbel]     type:list
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()

        return all_samples

    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        try:
            fname, annos = dp
            image = cv2.imread(fname, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels = annos.split(' ')
            boxes = []


            for label in labels:
                bbox = np.array(label.split(','), dtype=np.float)
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])

            boxes = np.array(boxes, dtype=np.float)

            img=image

            if is_training:





                ###random crop and flip
                height, width = img.shape[0], img.shape[1]
                c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
                if 0:
                    input_h = (height | self.opt.pad) + 1
                    input_w = (width | self.opt.pad) + 1
                    s = np.array([input_w, input_h], dtype=np.float32)
                else:
                    s = max(img.shape[0], img.shape[1]) * 1.0
                    input_h, input_w = cfg.DATA.hin, cfg.DATA.win

                flipped = False
                if 1:
                    if 1:
                        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                        w_border = self._get_border(128, img.shape[1])
                        h_border = self._get_border(128, img.shape[0])
                        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

                    if np.random.random() < 0.5:
                        flipped = True
                        img = img[:, ::-1, :]
                        c[0] = width - c[0] - 1

                trans_output = get_affine_transform(c, s, 0, [input_w, input_h])

                inp = cv2.warpAffine(img, trans_output,
                                     (input_w, input_h),
                                     flags=cv2.INTER_LINEAR)

                boxes_ = boxes[:, :4]
                klass_ = boxes[:, 4:5]

                boxes_refine = []
                for k in range(boxes_.shape[0]):
                    bbox = boxes_[k]

                    cls_id = klass_[k]
                    if flipped:
                        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                    bbox[:2] = affine_transform(bbox[:2], trans_output)
                    bbox[2:] = affine_transform(bbox[2:], trans_output)
                    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, input_w - 1)
                    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, input_h - 1)

                    boxes_refine.append(bbox)

                boxes_refine = np.array(boxes_refine)
                image = inp.astype(np.uint8)

                # angle=random.choice([0,90,180,270])
                # image,boxes_refine=Rotate_with_box(image,angle,boxes_refine)

                boxes = np.concatenate([boxes_refine, klass_], axis=1)

                ####random crop and flip
                #### pixel level aug

                image=self.train_trans(image=image)['image']




                boxes_cleaned=[]
                for kk in range(boxes.shape[0]):
                    box=boxes[kk]

                    if not ((box[3] - box[1]) < cfg.DATA.cover_obj or (box[2] - box[0]) < cfg.DATA.cover_obj):
                        boxes_cleaned.append(box)
                boxes=np.array(boxes_cleaned)
                ####

            else:
                boxes_ = boxes[:, 0:4]
                klass_ = boxes[:, 4:]
                image, shift_x, shift_y = Fill_img(image, target_width=cfg.DATA.win, target_height=cfg.DATA.hin)
                boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
                h, w, _ = image.shape
                boxes_[:, 0] /= w
                boxes_[:, 1] /= h
                boxes_[:, 2] /= w
                boxes_[:, 3] /= h
                image = image.astype(np.uint8)
                image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

                boxes_[:, 0] *= cfg.DATA.win
                boxes_[:, 1] *= cfg.DATA.hin
                boxes_[:, 2] *= cfg.DATA.win
                boxes_[:, 3] *= cfg.DATA.hin
                image = image.astype(np.uint8)
                boxes = np.concatenate([boxes_, klass_], axis=1)

            if boxes.shape[0] == 0 or np.sum(image) == 0:
                boxes_ = np.array([[0, 0, -1, -1]])
                klass_ = np.array([0])
            else:
                boxes_ = np.array(boxes[:, 0:4], dtype=np.float32)
                klass_ = np.array(boxes[:, 4], dtype=np.int64)


        except:
            logger.warn('there is an err with %s' % fname)
            traceback.print_exc()
            image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.uint8)
            boxes_ = np.array([[0, 0, -1, -1]])
            klass_ = np.array([0])

        heatmap, wh_map, weight = self.target_producer.ttfnet_centernet_datasampler(image, boxes_, klass_)
        image=np.transpose(image,axes=[2,0,1])
        return image, heatmap, wh_map, weight

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i



