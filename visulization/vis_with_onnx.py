# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
import sys
sys.path.append('.')
from train_config import config as cfg

import coremltools
import cv2
import numpy as np
import os
import PIL.Image
from visulization.coco_id_map import coco_map
from train_config import config as cfg
import onnxruntime
import onnx
class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        detections = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return detections


def preprocess( image, target_height, target_width, label=None):
    ###sometimes use in objs detects
    h, w, c = image.shape

    bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype)

    scale_y = target_height / h
    scale_x = target_width / w

    scale = min(scale_x, scale_y)

    image = cv2.resize(image, None, fx=scale, fy=scale)

    h_, w_, _ = image.shape

    dx = (target_width - w_) // 2
    dy = (target_height - h_) // 2
    bimage[dy:h_ + dy, dx:w_ + dx, :] = image

    return bimage, scale, scale, dx, dy

def inference(model_path,img_dir,thres=0.3):
    """ inference mobilenet_v1 using a specific picture """
    onnx_model =ONNXModel(model_path)


    img_list=os.listdir(img_dir)
    for pic in img_list:
        image = cv2.imread(os.path.join(img_dir,pic))
        #cv2 read as bgr format #change to rgb format
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image,_,_,_,_ = preprocess(image,target_height=cfg.DATA.hin,target_width=cfg.DATA.win)

        image_show=image.copy()

        image = image.astype(np.uint8)

        image=np.expand_dims(image,0)
        image=np.transpose(image,axes=[0,3,1,2]).astype(np.float32)


        onnx_outputs = onnx_model.forward(image)

        boxes=onnx_outputs

        boxes=boxes[0][0]
        print(boxes.shape)
        for i in range(len(boxes)):
            bbox = boxes[i]

            print(bbox[0])
            if bbox[4]>thres:

                cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])), (0,0,255), 4)

                str_draw = '%s:%.2f' % (coco_map[int(bbox[5])%80][1], bbox[4])
                cv2.putText(image_show, str_draw, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 0, 255), 2)

        cv2.imshow('coreml result',image_show)
        cv2.waitKey(0)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model', type=str, default='./centernet.onnx', help='the mnn model ', required=False)
    parser.add_argument('--imgDir', type=str, default='../pubdata/mscoco/val2017', help='the image dir to detect')
    parser.add_argument('--thres', type=float, default=0.3, help='the thres for detect')
    args = parser.parse_args()

    data_dir = args.imgDir
    model_path=args.onnx_model
    thres=args.thres
    inference(model_path,data_dir,thres)
