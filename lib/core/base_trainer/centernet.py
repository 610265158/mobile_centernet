import sys

sys.path.append('.')
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from train_config import config as cfg

import timm


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                    bias=bias),
                                    nn.BatchNorm2d(in_channels))
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()

        # self.mean_tensor=torch.from_numpy(cfg.DATA.PIXEL_MEAN ).float().cuda()
        # self.std_val_tensor = torch.from_numpy(cfg.DATA.PIXEL_STD).float().cuda()
        # self.model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
        self.model = timm.create_model('mobilenetv2_110d', pretrained=True, features_only=True)
        # self.model = timm.create_model('hrnet_w32', pretrained=True)

    def forward(self, inputs):
        # do preprocess

        inputs = inputs / 255.
        # Convolution layers
        fms = self.model(inputs)

        return fms[1:]


class ComplexUpsample(nn.Module):
    def __init__(self, input_dim=128, outpt_dim=128):
        super().__init__()

        self.conv1 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU()
                                   )

        self.conv2 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU()
                                   )

    def forward(self, inputs):
        # do preprocess

        x = self.conv1(inputs)
        y = self.conv2(inputs)

        z = x + y

        z = nn.functional.upsample(z, scale_factor=2, )

        return z

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

class CenterNetHead(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.conv2 = nn.Sequential(SeparableConv2d(24, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )

        self.conv3 = nn.Sequential(SeparableConv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        self.upsample3 = ComplexUpsample(128, 64)

        self.conv4 = nn.Sequential(SeparableConv2d(104, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        self.upsample4 = ComplexUpsample(128, 64)

        self.conv5 = nn.Sequential(SeparableConv2d(252, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        self.upsample5 = ComplexUpsample(352, 64)

        self.cls =nn.Conv2d(128, 80, kernel_size=3, stride=1, padding=1, bias=True)
        self.wh =nn.Conv2d(128, 4, kernel_size=3, stride=1, padding=1, bias=True)

        normal_init(self.cls,0,0.01,-2.19)
        normal_init(self.wh, 0, 0.01, 0)



    def forward(self, inputs):
        ##/24,32,104,352
        c2, c3, c4, c5 = inputs

        c5_upsample = self.upsample5(c5)
        c4 = self.conv4(c4)
        p4 = torch.cat([c4, c5_upsample], dim=1)

        c4_upsample = self.upsample4(p4)
        c3 = self.conv3(c3)
        p3 = torch.cat([c3, c4_upsample], dim=1)

        c3_upsample = self.upsample3(p3)
        c2 = self.conv2(c2)
        p2 = torch.cat([c2, c3_upsample], dim=1)

        cls = self.cls(p2)
        wh = self.wh(p2)
        return cls, wh


class CenterNet(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.backbone = Net()
        self.head = CenterNetHead()

    def forward(self, inputs):
        ##/24,32,104,352
        fms = self.backbone(inputs)

        # for ff in fms:
        #     print(ff.size())
        cls, wh = self.head(fms)

        return cls,wh*16

        # if self.training:
        #     pass
        # else:
        #     detections = self.decode(cls, wh, 4)
        #     return detections

    def decode(self, heatmap, wh, stride, K=100):
        def nms(heat, kernel=3):
            ##fast

            score, clses = torch.max(heat, dim=1)

            scores = torch.nn.functional.sigmoid(score)

            hmax = torch.nn.functional.max_pool2d(scores, kernel, 1, padding=1)
            keep = scores = hmax
            return scores * keep, clses

        batch, cat, H, W = heatmap.size()

        score_map, label_map = nms(heatmap)

        ### decode the box
        shifts_x = torch.range(0, (W - 1) * stride + 1, stride,
                               dtype=torch.int32)

        shifts_y = torch.range(0, (H - 1) * stride + 1, stride,
                               dtype=torch.int32)

        x_range, y_range = torch.meshgrid(shifts_x, shifts_y)

        base_loc = torch.stack((x_range, y_range, x_range, y_range), axis=0)  # (h, w，4)

        base_loc = torch.unsqueeze(base_loc, dim=0)

        wh = wh * torch.from_numpy(np.array([1, 1, -1, -1]).reshape([1, 4, 1, 1]))
        pred_boxes = base_loc - wh

        # pred_boxes = tf.concat((base_loc[:, :, :, 0:1] - wh[:, :, :, 0:1],
        #                         base_loc[:, :, :, 1:2] - wh[:, :, :, 1:2],
        #                         base_loc[:, :, :, 0:1] + wh[:, :, :, 2:3],
        #                         base_loc[:, :, :, 1:2] + wh[:, :, :, 3:4]), axis=3)

        ###get the topk bboxes
        score_map = torch.reshape(score_map, shape=[batch, -1])
        # topk_scores, topk_inds = tf.nn.top_k(score_map, k=K)
        # # # [b,k]

        pred_boxes = torch.reshape(pred_boxes, shape=[batch, 4, -1])
        # pred_boxes = tf.batch_gather(pred_boxes, topk_inds)

        label_map = torch.reshape(label_map, shape=[batch, -1])
        # label_map = tf.batch_gather(label_map, topk_inds)

        score_map = torch.unsqueeze(score_map, 1)
        label_map = torch.unsqueeze(label_map, 1)

        pred_boxes = pred_boxes.float()
        label_map = label_map.float()

        detections = torch.cat([pred_boxes, score_map, label_map], dim=1)

        print(detections.size())
        return detections


if __name__ == '__main__':
    import torch
    import torchvision

    dummy_input = torch.randn(1, 3, 512, 512, device='cpu')
    model = CenterNet()

    ### load your weights
    model.eval()
    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.

    torch.onnx.export(model, dummy_input, "classifier.onnx")

