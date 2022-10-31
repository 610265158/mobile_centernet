import math
from functools import partial

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md

from segmentation_models_pytorch.base.initialization import initialize_decoder,initialize_head

import segmentation_models_pytorch

from scipy import ndimage


import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from lib.core.loss.centernet_loss import CenterNetLoss
from lib.core.base_trainer.mobileone import MobileOneBlock
from lib.core.model.utils import normal_init

bn_momentum=0.1
class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=0, bias=False,
                 channel_multiplier=1., pw_kernel_size=1):
        super(SeparableConv2d, self).__init__()


        self.conv_dw = nn.Conv2d(
            int(in_channels*channel_multiplier), int(in_channels*channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, groups=int(in_channels*channel_multiplier))

        self.conv_pw = nn.Conv2d(
            int(in_channels*channel_multiplier), out_channels, pw_kernel_size, padding=0, bias=bias)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):

        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


class RepConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding=0,
                 bias=False,
                 channel_multiplier=1.,
                 pw_kernel_size=1,
                 inference_mode=False,
                 num_conv_branches=4,
                 use_se=False):
        super(RepConv2d, self).__init__()


        self.conv_dw = MobileOneBlock(in_channels=in_channels,
                                         out_channels=in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=kernel_size//2,
                                         groups=in_channels,
                                         inference_mode=inference_mode,
                                         use_se=False,
                                         num_conv_branches=num_conv_branches)
        self.conv_pw = MobileOneBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=inference_mode,
                                         use_se=False,
                                         num_conv_branches=num_conv_branches)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):

        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x

class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(

            nn.Conv2d(channels, channels, self.kernel_size,
                      stride=1, padding=kernel_size//2, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pool=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels,momentum=bn_momentum),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)

        return F.interpolate(x, size=size, mode='nearest')


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 128

        rate1, rate2, rate3 = tuple(atrous_rates)

        self.fm_conx1=nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
            nn.ReLU())

        self.fm_convx3_rate2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=2, bias=False,dilation=rate1),
            nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
            nn.ReLU())

        self.fm_convx3_rate4=nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=4, bias=False,dilation=rate2),
            nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
            nn.ReLU())

        self.fm_convx3_rate8=nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=8, bias=False,dilation=rate3),
            nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
            nn.ReLU())

        self.fm_pool=ASPPPooling(in_channels=in_channels,out_channels=out_channels//4)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels//4*5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels,momentum=bn_momentum),
            nn.ReLU())

    def forward(self, x):

        fm1=self.fm_conx1(x)
        fm2=self.fm_convx3_rate2(x)
        fm4=self.fm_convx3_rate4(x)
        fm8=self.fm_convx3_rate8(x)
        fm_pool=self.fm_pool(x)

        res = torch.cat([fm1,fm2,fm4,fm8,fm_pool], dim=1)

        return self.project(res)



class FeatureFuc(nn.Module):
    def __init__(self, inchannels=80,inference_mode=False):
        super(FeatureFuc, self).__init__()



        self.block1 = RepConv2d(in_channels=inchannels,
                                out_channels=256,
                                kernel_size=5,
                                inference_mode=inference_mode)
        self.block2 = RepConv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=5,
                                inference_mode=inference_mode)
        self.block3 = RepConv2d(in_channels=256,
                                out_channels=inchannels,
                                kernel_size=5,
                                inference_mode=inference_mode)
        initialize_decoder(self.block1)
        initialize_decoder(self.block2)
        initialize_decoder(self.block3)
    def forward(self, x):

        y1=self.block1(x)

        y2=self.block2(y1)
        y3 = self.block3(y2)

        return y3



class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_separable_conv=True,
            use_attention=False,
            use_second_conv=False,
            kernel_size=5,
            inference_mode=True
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            RepConv2d(in_channels=in_channels+skip_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      inference_mode=inference_mode,),
        )

        if use_second_conv:
            self.conv2 = nn.Sequential(
                RepConv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          inference_mode=inference_mode),

               )
        else:
            self.conv2 = nn.Identity()

        if use_attention:
            self.attention2 = SCSEModule(in_channels=out_channels)
        else:
            self.attention2 =nn.Identity()


        ###init the weights
        initialize_decoder(self.conv1)
        initialize_decoder(self.conv2)
    def forward(self, x, skip=None):

        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is None:
            return x

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            # x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            inference_mode=True

    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = nn.Identity()

        # combine decoder keyword arguments

        blocks = []

        kernel_size=[5,3,3,3,3]

        for n,(in_ch, skip_ch, out_ch) in enumerate(zip(in_channels, skip_channels, out_channels)):



            if n==0:
                use_attention=False
                use_separable_conv=True
                use_second_conv=True

            elif n==1  :
                use_attention=False
                use_separable_conv=True
                use_second_conv=False


            else:
                use_attention=False
                use_separable_conv=True
                use_second_conv=False

            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch, \
                                       use_separable_conv=use_separable_conv, \
                                       use_attention=use_attention,
                                       use_second_conv=use_second_conv,
                                       kernel_size=kernel_size[n],
                                       inference_mode=inference_mode))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        all_fms=[]


        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            all_fms.append(x)

        return all_fms



class CenterNet(nn.Module):
    def __init__(self,encoder_channels=[3, 16,24, 40 ,80],inference_mode=False):
        super(CenterNet, self).__init__()

        # self.extra_feature = FeatureFuc(encoder_channels[-1],inference_mode=inference_mode)
        self.aspp = ASPP(encoder_channels[-1], [2, 4, 8])

        encoder_channels[-1]=128

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[128, 128],

            n_blocks=2,
            inference_mode=inference_mode
        )

        self.cls = SeparableConv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.wh = SeparableConv2d(128, 4, kernel_size=3, stride=1, padding=1, bias=True)

        normal_init(self.cls.conv_pw, 0, 0.01, -2.19)
        normal_init(self.wh.conv_pw, 0, 0.01, 0)


        initialize_decoder(self.decoder)
        # initialize_head(self.hm)
        # initialize_head(self.wh)
    def forward(self, features):

        # extra_feature = self.extra_feature(features[-1])
        # features[-1] = extra_feature
        features[-1] = self.aspp(features[-1])

        #
        decoder_output = self.decoder(*features)

        pre_cls = self.cls(decoder_output[-1])

        pre_wh= self.wh(decoder_output[-1])


        return pre_cls, pre_wh*16, decoder_output

class Net(nn.Module):
    def __init__(self,inference_mode=False):
        super(Net, self).__init__()

        self.encoder = timm.create_model(model_name='tf_mobilenetv3_large_minimal_100',
                                         pretrained=True,
                                         features_only=True,
                                         out_indices=[0,1,2,4],
                                         bn_momentum=bn_momentum,
                                         bn_eps=1e-3,
                                         in_chans=3,
                                         output_stride=16,
                                         )

        # self.encoder.blocks[4]=nn.Identity()
        # self.encoder.blocks[5]=nn.Identity()
        self.encoder.blocks[6]=nn.Identity()

        # print(self.encoder)
        # self.encoder=MobileNetV2Backbone(in_channels=3)
        self.encoder.out_channels=[3, 16,24, 40 ,160]
        # self.encoder.out_channels=[3,16, 24, 32, 88, 720]

        self.centernet=CenterNet(self.encoder.out_channels,inference_mode=inference_mode)
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features=self.encoder(x)

        features=[x]+features
        pre_hm, pre_wh, decoder_output = self.centernet(features)
        return pre_hm, pre_wh, decoder_output


class TeacherNet(nn.Module):
    def __init__(self,):
        super(TeacherNet, self).__init__()

        self.encoder = timm.create_model(model_name='tf_efficientnet_b5_ns',
                                         pretrained=True,
                                         features_only=True,
                                         out_indices=[0,1,2,3],
                                         bn_momentum=bn_momentum,
                                         bn_eps=1e-3,
                                         in_chans=3,
                                         )
        # print(self.encoder)
        # print(self.encoder.blocks[6])

        # self.encoder.blocks[6]=torch.nn.Identity()
        # self.encoder=MobileNetV2Backbone(in_channels=3)
        self.encoder.out_channels=[3, 24 , 40, 64,176]
        # self.encoder.out_channels=[3,16, 24, 32, 88, 720]

        self.centernet = CenterNet(self.encoder.out_channels,inference_mode=False)


    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features=self.encoder(x)

        features=[x]+features
        pre_hm, pre_wh, ecoder_output = self.centernet(features)

        return  pre_hm, pre_wh, ecoder_output


class COTRAIN(nn.Module):
    def __init__(self, inference=False):
        super(COTRAIN, self).__init__()

        self.inference = inference

        from lib.core.model.centernet import CenterNet
        # self.student = CenterNet()
        #
        # state_dict = torch.load('centernet_mobilenetv2_stride4.pth', map_location='cpu')
        # self.student.load_state_dict(state_dict, strict=False)
        #
        self.student=Net()

        # self.teacher = TeacherNet()

        self.MSELoss = nn.MSELoss()

        self.act = nn.Sigmoid()

        self.criterion = CenterNetLoss()
    def reparameterize(self):
        from lib.core.base_trainer.mobileone import reparameterize_model
        self.student = reparameterize_model(self.student)

    def distill_loss(self, student_pres, teacher_pres):

        num_level = len(student_pres)
        loss = 0
        for i in range(num_level):
            loss += self.MSELoss(student_pres[i], teacher_pres[i])

        return loss / num_level

    def loss(self, pre_cls, pred_hw, hm_target, wh_target, weights):

        cls_loss, wh_loss = self.criterion([pre_cls, pred_hw], [hm_target, wh_target, weights])

        # print(cls_loss,wh_loss)
        current_loss = cls_loss + wh_loss

        return current_loss

    def forward(self, x, hm_target=None, wh_target=None, weights=None):

        student_pre_cls, student_pre_wh,student_decoder_output = self.student(x)

        # teacher_pre_cls, teacher_pre_hw, teacher_decoder_output=self.teacher(x)

        if self.inference:
            ##do decode
            detections = self.decode(student_pre_cls,student_pre_wh , 4)
            return detections

            # return alpha

        # teacher_pre_cls, teacher_pre_hw, teacher_decoder_output = self.teacher(x)
        # #
        # distill_loss = self.distill_loss(student_decoder_output, teacher_decoder_output)

        student_loss = self.loss( student_pre_cls, student_pre_wh, hm_target, wh_target, weights)

        # teacher_loss = self.loss( teacher_pre_cls, teacher_pre_hw, hm_target, wh_target, weights)

        return student_loss, student_loss, student_loss,None
    def decode(self, heatmap, wh, stride, K=10):
        def nms(heat, kernel=3):
            ##fast



            heat = heat.permute([0, 2, 3, 1])
            heat, clses = torch.max(heat, dim=3)

            heat = heat.unsqueeze(1)
            scores = torch.sigmoid(heat)

            hmax = nn.MaxPool2d(kernel, 1, padding=1)(scores)
            keep = (scores == hmax).float()

            return scores * keep, clses
        def get_bboxes(wh):

            ### decode the box
            shifts_x = torch.arange(0, (W - 1) * stride + 1, stride,
                                    dtype=torch.int32)

            shifts_y = torch.arange(0, (H - 1) * stride + 1, stride,
                                    dtype=torch.int32)

            y_range, x_range = torch.meshgrid(shifts_y, shifts_x)
            base_loc = torch.stack((x_range, y_range, x_range, y_range), axis=0)  # (h, w茂录艗4)

            base_loc = torch.unsqueeze(base_loc, dim=0).to(torch.device("cuda" if torch.cuda.is_available() else 'cpu'))


            wh = wh * torch.tensor([1, 1, -1, -1],requires_grad=False,
                                   device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')).reshape([1, 4, 1, 1])
            pred_boxes = base_loc - wh

            return pred_boxes

        batch, cat, H, W = heatmap.size()


        score_map, label_map = nms(heatmap)
        pred_boxes=get_bboxes(wh)


        score_map = torch.reshape(score_map, shape=[batch, -1])

        top_score,top_index=torch.topk(score_map,k=K)

        top_score = torch.unsqueeze(top_score, 2)


        # if self.coreml_:
        if 0:
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

    dummy_x = torch.randn(1, 3, 320, 320, device='cpu')

    dummy_mate = torch.randn(1, 1, 320, 320, device='cpu')

    dummy_trimap = torch.randn(1, 1, 320, 320, device='cpu')
    model = COTRAIN()

    losses = model.train_forward(dummy_x, dummy_mate, dummy_trimap)

    print(losses)
