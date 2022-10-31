


import os
import random
import cv2
import numpy as np
import traceback

from lib.core.utils.logger import logger



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

    def __init__(self, ann_file=None, training_flag=True, shuffle=True):

        self.color_augmentor = ColorDistort()

        self.training_flag = training_flag

        self.lst = self.parse_file('', ann_file)

        self.shuffle = shuffle

        self.train_trans =  self.train_trans=A.Compose([

            A.RandomResizedCrop(height=cfg.DATA.hin ,
                                width=cfg.DATA.win ,
                                scale=(0.8, 1.0),
                                interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1,
                               scale_limit=0.5,
                               rotate_limit=45,
                               p=0.5,
                               border_mode=cv2.BORDER_CONSTANT,
                               mask_value=0),
            A.HueSaturationValue(p=0.3),
            A.RandomBrightnessContrast(p=0.3),

            A.OneOf([A.GaussNoise(),
                     A.ISONoise()],
                    p=0.2),

            # A.OneOf([A.GaussianBlur(),
            #         A.MotionBlur()],
            #        p=0.2),



            # A.ToGray(p=0.2),
            # A.JpegCompression(quality_lower=60,
            #                  quality_upper=99,
            #                  p=0.2),


        ],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'])
        )

        self.target_trans = A.Compose([


            # A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.ElasticTransform(mask_value=0,
                                   value=0,
                                   border_mode=cv2.BORDER_CONSTANT),
                A.GridDistortion(mask_value=0,
                                 value=0,
                                 border_mode=cv2.BORDER_CONSTANT),
                A.OpticalDistortion(mask_value=0,
                                    value=0,
                                    border_mode=cv2.BORDER_CONSTANT)],

                p=0.5),


        ]
        )

        self.target_producer = CenternetDatasampler()
        self.logos = []

        self.get_logs()

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


    def get_logs(self):
        for item in ["logos/1.jpg","logos/2.jpg"]:
            log_img=cv2.imread(item)

            self.logos.append(log_img)

    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        try:
            fname, annos = dp
            image = cv2.imread(fname, cv2.IMREAD_COLOR)


            if 1:

                ### do sync first
                boxes_ = []
                klass_ = []

                image_limit_h,image_limit_w,_=image.shape

                for i in range(1):
                    logo_image = self.logos[random.randint(0, 1)]

                    logo_size_h = random.randint(32, image_limit_h)
                    logo_size_w = int(random.uniform(0.5, 1.5) * logo_size_h)

                    logo_size_w = np.clip(logo_size_w, 32, image_limit_w)

                    logo_image = cv2.resize(logo_image, dsize=[logo_size_w, logo_size_h])

                    h_logo, w_logo, _ = logo_image.shape

                    start_y = random.randint(0, image_limit_h - h_logo)
                    start_x = random.randint(0, image_limit_w - w_logo)

                    image[start_y:start_y + h_logo, start_x:start_x + w_logo] = logo_image

                    end_x = np.clip(start_x + w_logo,start_x,image_limit_w)
                    end_y = np.clip(start_y + h_logo, start_y, image_limit_h)



                    boxes_.append([start_x, start_y,end_x, end_y])
                    klass_.append(0)



                transformed = self.train_trans(image=image,bboxes=boxes_,class_labels=klass_)
                image = transformed['image']
                boxes_ = transformed['bboxes']
                klass_ = transformed['class_labels']
                #### augmentation the target

            boxes_ = np.array(boxes_)
            klass_ = np.array(klass_)

        except:
            logger.warn('there is an err with %s' % fname)
            traceback.print_exc()
            image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.uint8)
            boxes_ = np.array([[0, 0, -1, -1]])
            klass_ = np.array([0])

        heatmap, wh_map, weight = self.target_producer.ttfnet_centernet_datasampler(image, boxes_, klass_)
        image=np.transpose(image,axes=[2,0,1])
        print(heatmap.shape)
        image=image/255.
        return image, heatmap, wh_map, weight

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i



