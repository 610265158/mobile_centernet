#-*-coding:utf-8-*-

import numpy as np
import torch
import torch.nn as nn
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

        base_loc = torch.stack((x_range, y_range, x_range, y_range), axis=0)  # (h, w，4)

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
                  avg_factor=None):
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
        bboxes1 = torch.reshape(pred[pos_mask], (-1, 4)).float()
        bboxes2 = torch.reshape(target[pos_mask], (-1, 4)).float()

        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
        wh = torch.clamp((rb - lt + 1), 0)  # [rows, 2]
        # enclose_x1y1 = tf.minimum(bboxes1[:, :2], bboxes2[:, :2])
        # enclose_x2y2 = tf.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
        # enclose_wh =  tf.maximum((enclose_x2y2 - enclose_x1y1 + 1),0)

        overlap = wh[:, 0] * wh[:, 1]

        ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (ap + ag - overlap)


        # cal outer boxes
        outer_left_up = torch.min(bboxes1[:, :2], bboxes2[:, :2])
        outer_right_down = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        outer = torch.clamp(outer_right_down - outer_left_up, 0.0)
        outer_diagonal_line = (outer[:, 0])**2 + (outer[:, 1])**2

        boxes1_center = (bboxes1[:, :2] + bboxes1[:, 2:] + 1) * 0.5
        boxes2_center = (bboxes2[:, :2] + bboxes2[:, 2:] + 1) * 0.5
        center_dis = (boxes1_center[:, 0] - boxes2_center[:, 0])**2+ \
                     (boxes1_center[:, 1] - boxes2_center[:, 1])**2

        boxes1_size = torch.clamp(bboxes1[:, 2:] - bboxes1[:, :2], 0.0)
        boxes2_size = torch.clamp(bboxes2[:, 2:] - bboxes2[:, :2], 0.0)

        v = (4.0 / (np.pi ** 2)) * \
            (torch.atan(boxes2_size[:, 0] / (boxes2_size[:, 1] + 0.00001)) -
                      torch.atan(boxes1_size[:, 0] / (boxes1_size[:, 1] + 0.00001)))**2

        S = (ious> 0.5).float()
        alpha = S * v / (1 - ious + v)

        cious = ious - (center_dis / outer_diagonal_line) - alpha * v

        cious = 1 - cious

        cious=torch.where(torch.isnan(cious), torch.full_like(cious, 0), cious)
        return torch.sum(cious * weight) / avg_factor







