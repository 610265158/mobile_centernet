import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import numpy as np
import torch.nn as nn
from train_config import config as cfg

import torch
import torch.nn as nn


from lib.helper.logger import logger
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ACCMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0,1])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_true = np.argmax(y_true,1)
        y_pred = torch.sigmoid(y_pred).data.cpu().numpy()
        y_pred = np.argmax(y_pred,1)

        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))


    @property
    def avg(self):
        right=(self.y_pred==self.y_true).astype(np.float)
        return np.sum(right)/self.y_true.shape[0]

