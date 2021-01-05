#-*-coding:utf-8-*-

import numpy as np
import torch
import torch.nn as nn
from train_config import config as cfg

from lib.core.model.loss.iouloss import *




def loss(predicts,targets,base_step=cfg.MODEL.global_stride):

    pred_hm, pred_wh=predicts
    pred_hm=torch.nn.functional.sigmoid(pred_hm)

    ## nchw to nhwc
    pred_hm=torch.transpose(pred_hm,[0,2,3,1])

    hm_target, wh_target,weights_=targets

    hm_loss = focal_loss(
        pred_hm,
        hm_target
    )




    Batch,H, W,c = pred_hm.size()

    weights_=torch.transpose(weights_,perm=[0,3,1,2])
    mask = torch.reshape(weights_,shape=(-1, H, W))
    avg_factor = torch.sum(mask) + 1e-4

    ### decode the box
    shifts_x = torch.range(0, (W - 1) * base_step + 1, base_step,
                           dtype=torch.int32)

    shifts_y = torch.range(0, (H - 1) * base_step + 1, base_step,
                           dtype=torch.int32)

    x_range, y_range = torch.meshgrid(shifts_x, shifts_y)

    base_loc = torch.stack((x_range, y_range, x_range, y_range), axis=0)  # (h, w，4)

    base_loc = torch.unsqueeze(base_loc, dim=0)

    pred_wh = pred_wh * torch.from_numpy(np.array([1, 1, -1, -1]).reshape([1, 4, 1, 1]))
    pred_boxes = base_loc - pred_wh
    pred_boxes=torch.transpose(pred_boxes,[0,3,1,2])
    # (batch, h, w, 4)
    boxes = torch.transpose(wh_target,[0,3,1,2])#.permute(0, 2, 3, 1)

    wh_loss = ciou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor)

    return hm_loss, wh_loss*5


def focal_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch,h,w,c)
        gt_regr (batch,h,w,c)
    '''
    pos_inds =  (gt==1.0).float()
    neg_inds = 1.0 - pos_inds
    neg_weights = torch.square(1.0 - gt)*torch.square(1.0 - gt)

    pred = torch.clamp()(pred, 1e-6, 1.0 - 1e-6)
    pos_loss = torch.log(pred) * (1.0 - pred)**2  * pos_inds
    neg_loss = torch.log(1.0 - pred) * (pred, 2.0)**2 * neg_weights * neg_inds

    num_pos = torch.sum(pos_inds)
    pos_loss = torch.sum(pos_loss)
    neg_loss = torch.sum(neg_loss)

    normalizer = torch.maximum(1., num_pos)
    loss = - (pos_loss + neg_loss) / normalizer

    return loss

def ciou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask) + 1e-6
    bboxes1 = torch.reshape(pred[pos_mask],(-1, 4))
    bboxes2 = torch.reshape(target[pos_mask],(-1, 4))


    lt = torch.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = torch.maximum((rb - lt + 1),0)  # [rows, 2]
    # enclose_x1y1 = tf.minimum(bboxes1[:, :2], bboxes2[:, :2])
    # enclose_x2y2 = tf.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    # enclose_wh =  tf.maximum((enclose_x2y2 - enclose_x1y1 + 1),0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)




    # cal outer boxes
    outer_left_up = torch.minimum(bboxes1[:, :2], bboxes2[:, :2])
    outer_right_down = torch.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    outer = torch.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = torch.square(outer[:, 0]) + torch.square(outer[:, 1])


    boxes1_center = (bboxes1[:, :2] + bboxes1[:, 2:]+ 1) * 0.5
    boxes2_center = (bboxes2[:, :2] + bboxes2[:, 2:]+ 1) * 0.5
    center_dis = torch.square(boxes1_center[:, 0] - boxes2_center[:, 0]) + \
                 torch.square(boxes1_center[:, 1] - boxes2_center[:, 1])





    boxes1_size = torch.maximum(bboxes1[:,2:]-bboxes1[:,:2],0.0)
    boxes2_size = torch.maximum(bboxes2[:, 2:] - bboxes2[:, :2], 0.0)

    v = (4.0 / (np.pi**2)) * \
        torch.square(torch.atan(boxes2_size[:, 0] / (boxes2_size[:, 1]+0.00001)) -
                    torch.atan(boxes1_size[:, 0] / (boxes1_size[:, 1]+0.00001)))

    S = (ious > 0.5).float()
    alpha = S * v / (1 - ious + v)

    cious = ious - (center_dis / outer_diagonal_line)-alpha * v

    cious = 1-cious

    return torch.sum(cious * weight) / avg_factor







