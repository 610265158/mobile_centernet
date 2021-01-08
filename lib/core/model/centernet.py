import sys

sys.path.append('.')
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from train_config import config as cfg

import timm

def get_nearestup_weight(cin):
    filt=torch.Tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])[None, None, ...]
    weight = np.zeros((cin, cin, 4, 4),
                      dtype=np.float64)
    weight[range(cin), range(cin), :, :] = filt
    return torch.from_numpy(weight).float()

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
        self.model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        # self.model = timm.create_model('hrnet_w32', pretrained=True)

    def forward(self, inputs):
        # do preprocess

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

        self.conv2 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU()
                                   )

    def forward(self, inputs):
        # do preprocess

        x = self.conv1(inputs)
        y = self.conv2(inputs)

        z = x + y


        n,c,h,w=z.size()



        z = nn.functional.interpolate(z, scale_factor=2,mode='nearest' )

        return z

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

class CenterNetHead(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.conv2 = nn.Sequential(SeparableConv2d(24, 64, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )

        self.conv3 = nn.Sequential(SeparableConv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        self.upsample3 = ComplexUpsample(128, 64)

        self.conv4 = nn.Sequential(SeparableConv2d(96, 64, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        self.upsample4 = ComplexUpsample(128, 64)

        self.conv5 = nn.Sequential(SeparableConv2d(320, 64, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        self.upsample5 = ComplexUpsample(320, 64)

        self.cls =SeparableConv2d(128, 80, kernel_size=3, stride=1, padding=1, bias=True)
        self.wh =SeparableConv2d(128, 4, kernel_size=3, stride=1, padding=1, bias=True)

        normal_init(self.cls.pointwise, 0, 0.01,-2.19)
        normal_init(self.wh.pointwise, 0, 0.01, 0)



    def forward(self, inputs):
        ##/24,32,96,320
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
    def __init__(self, inference=False,coreml=False ):
        super().__init__()

        self.backbone = Net()
        self.head = CenterNetHead()
        self.inference=inference

        self.coreml_=coreml

    def forward(self, inputs):
        ##/24,32,96,320
        fms = self.backbone(inputs)

        # for ff in fms:
        #     print(ff.size())
        cls, wh = self.head(fms)
        if not self.inference:
            return cls,wh*16
        else:
            detections = self.decode(cls, wh*16, 4)
            return detections

    def decode(self, heatmap, wh, stride, K=100):
        def nms(heat, kernel=3):
            ##fast

            heat=heat.permute([0,2,3,1])
            heat, clses = torch.max(heat, dim=3,keepdim=True)

            heat = heat.permute([0, 3,1,2])
            scores = torch.sigmoid(heat)

            hmax = nn.MaxPool2d(kernel,1,padding=1)(scores)
            keep = (scores == hmax).float()
            return scores * keep, clses
        def get_bboxes(wh):

            ### decode the box
            shifts_x = torch.arange(0, (W - 1) * stride + 1, stride,
                                   dtype=torch.int32)

            shifts_y = torch.arange(0, (H - 1) * stride + 1, stride,
                                   dtype=torch.int32)

            x_range, y_range = torch.meshgrid(shifts_x, shifts_y)

            base_loc = torch.stack((x_range, y_range, x_range, y_range), axis=0)  # (h, w，4)

            base_loc = torch.unsqueeze(base_loc, dim=0)

            wh = wh * torch.tensor([1, 1, -1, -1],requires_grad=False).reshape([1, 4, 1, 1])
            pred_boxes = base_loc - wh

            return pred_boxes

        batch, cat, H, W = heatmap.size()


        score_map, label_map = nms(heatmap)
        pred_boxes=get_bboxes(wh)


        score_map = torch.reshape(score_map, shape=[batch, -1])

        top_score,top_index=torch.topk(score_map,k=K)

        top_score = torch.unsqueeze(top_score, 2)


        if self.coreml_:
            pred_boxes = torch.reshape(pred_boxes, shape=[batch, 4, -1])
            pred_boxes = pred_boxes.permute([0, 2, 1])
            top_index_bboxes=torch.stack([top_index,top_index,top_index,top_index],dim=2)


            pred_boxes = torch.gather(pred_boxes,dim=1,index=top_index_bboxes)

            label_map = torch.reshape(label_map, shape=[batch, -1])
            label_map = torch.gather(label_map,dim=1,index=top_index)
            label_map = torch.unsqueeze(label_map, 2)

            pred_boxes = pred_boxes.float()
            label_map = label_map.float()


            detections = torch.cat([pred_boxes, top_score, label_map], dim=2)

        else:
            pred_boxes = torch.reshape(pred_boxes, shape=[batch, 4, -1])
            pred_boxes=pred_boxes.permute([0,2,1])

            pred_boxes = pred_boxes[:,top_index[0],:]

            label_map = torch.reshape(label_map, shape=[batch, -1])
            label_map = label_map[:,top_index[0]]
            label_map = torch.unsqueeze(label_map, 2)


            pred_boxes = pred_boxes.float()
            label_map = label_map.float()

            detections = torch.cat([ pred_boxes,top_score, label_map], dim=2)

        return detections


if __name__ == '__main__':
    import torch
    import torchvision

    dummy_input = torch.randn(1, 3, 512, 512, device='cpu')
    model = CenterNet(inference=True)

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

    torch.onnx.export(model, dummy_input, "classifier.onnx",opset_version=11)

    import onnx
    from onnxsim import simplify

    # load your predefined ONNX model
    model = onnx.load("centernet.onnx")

    # convert model
    model_simp, check = simplify(model)
    f = model_simp.SerializeToString()
    file = open("centernet.onnx", "wb")
    file.write(f)
    assert check, "Simplified ONNX model could not be validated"