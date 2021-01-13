import torch
import torch.nn as nn

from lib.core.model.utils import SeparableConv2d




class ComplexUpsample(nn.Module):
    def __init__(self, input_dim=128, outpt_dim=128):
        super().__init__()

        self.conv1 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv2 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, inputs):
        # do preprocess

        x = self.conv1(inputs)
        y = self.conv2(inputs)

        z = x + y

        z = nn.functional.interpolate(z, scale_factor=2,mode='bilinear' )

        return z

class Fpn(nn.Module):
    def __init__(self,input_dims=[24,32,96,320],head_dims=[128,128,128] ):
        super().__init__()





        self.latlayer2=nn.Sequential(SeparableConv2d(input_dims[0],head_dims[0]//2,kernel_size=5,padding=2),
                                      nn.BatchNorm2d(head_dims[0]//2),
                                      nn.ReLU(inplace=True))


        self.latlayer3=nn.Sequential(SeparableConv2d(input_dims[1],head_dims[1]//2,kernel_size=5,padding=2),
                                      nn.BatchNorm2d(head_dims[1]//2),
                                      nn.ReLU(inplace=True))

        self.latlayer4 = nn.Sequential(SeparableConv2d(input_dims[2], head_dims[2] // 2,kernel_size=5,padding=2),
                                       nn.BatchNorm2d(head_dims[2] // 2),
                                       nn.ReLU(inplace=True))



        self.upsample3=ComplexUpsample(head_dims[1],head_dims[0]//2)

        self.upsample4 =ComplexUpsample(head_dims[2],head_dims[1]//2)

        self.upsample5 = ComplexUpsample(input_dims[3],head_dims[2]//2)




    def forward(self, inputs):
        ##/24,32,96,320
        c2, c3, c4, c5 = inputs

        c4_lat = self.latlayer4(c4)
        c3_lat = self.latlayer3(c3)
        c2_lat = self.latlayer2(c2)


        upsample_c5=self.upsample5(c5)

        p4=torch.cat([c4_lat,upsample_c5],dim=1)


        upsample_p4=self.upsample4(p4)

        p3=torch.cat([c3_lat,upsample_p4],dim=1)

        upsample_p3 = self.upsample3(p3)

        p2 = torch.cat([c2_lat, upsample_p3],dim=1)


        return p2
