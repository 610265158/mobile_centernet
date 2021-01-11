#-*-coding:utf-8-*-

import numpy as np
import torch
import torch.nn as nn
import math
from train_config import config as cfg





class CenterNetLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        pass



    def forward(self,predicts,targets,base_step=cfg.MODEL.global_stride):

        pred_hm, pred_wh=predicts


        ## nchw to nhwc
        pred_hm=pred_hm.permute([0,2,3,1])

        hm_target, wh_target,weights_=targets

        hm_loss = self.focal_loss(
            pred_hm,
            hm_target
        )


        Batch,H, W,c = pred_hm.size()

        weights_=weights_.permute([0,3,1,2])
        mask = torch.reshape(weights_,shape=(-1, H, W))
        avg_factor = torch.sum(mask) + 1e-4

        ### decode the box
        shifts_x = torch.range(0, (W - 1) * base_step + 1, base_step,
                               dtype=torch.int32)

        shifts_y = torch.range(0, (H - 1) * base_step + 1, base_step,
                               dtype=torch.int32)

        x_range, y_range = torch.meshgrid(shifts_x, shifts_y)

        base_loc = torch.stack((x_range, y_range, x_range, y_range), axis=0)  # (h, w锛�4)

        base_loc = torch.unsqueeze(base_loc, dim=0).to(self.device)

        pred_wh = pred_wh * torch.from_numpy(np.array([1, 1, -1, -1]).reshape([1, 4, 1, 1])).to(self.device)

        pred_boxes = base_loc - pred_wh

        pred_boxes=pred_boxes.permute([0,2,3,1])

        ### nchw to nhwc
        boxes = wh_target

        wh_loss =self.ciou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor)


        return hm_loss, wh_loss*5


    def focal_loss(self,pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            pred (batch,h,w,c)
            gt_regr (batch,h,w,c)
        '''

        pos_inds = gt.eq(1).float()
        neg_inds = 1.0 - pos_inds
        neg_weights = torch.pow(1.0 - gt, 4.0)


        pred=torch.nn.functional.sigmoid(pred)
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos_inds
        neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * neg_inds

        num_pos = torch.sum(pos_inds)
        pos_loss = torch.sum(pos_loss)
        neg_loss = torch.sum(neg_loss)

        normalizer = torch.clamp( num_pos,1.)
        loss = - (pos_loss + neg_loss) / normalizer

        return loss

    def ciou_loss(self,
                  pred,
                  target,
                  weight,
                  avg_factor=None,
                  eps=1e-7):
        """GIoU loss.
        Computing the GIoU loss between a set of predicted bboxes and target bboxes.
        """
        """GIoU loss.
            Computing the GIoU loss between a set of predicted bboxes and target bboxes.
            """


        pos_mask = weight > 0
        weight = weight[pos_mask].float()
        if avg_factor is None:
            avg_factor = torch.sum(pos_mask) + 1e-6
        pred = torch.reshape(pred[pos_mask], (-1, 4)).float()
        target = torch.reshape(target[pos_mask], (-1, 4)).float()

        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose area
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        cw = enclose_wh[:, 0]
        ch = enclose_wh[:, 1]

        c2 = cw ** 2 + ch ** 2 + eps

        b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
        b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
        b2_x1, b2_y1 = target[:, 0], target[:, 1]
        b2_x2, b2_y2 = target[:, 2], target[:, 3]

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
        rho2 = left + right

        factor = 4 / math.pi ** 2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        # CIoU
        cious = ious - (rho2 / c2 + v ** 2 / (1 - ious + v))
        cious = 1 - cious


        cious = torch.where(torch.isnan(cious), torch.full_like(cious, 0), cious)

        avg_factor=torch.clamp(avg_factor,1)
        return torch.sum(cious * weight) / avg_factor






