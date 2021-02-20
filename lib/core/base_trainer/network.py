#-*-coding:utf-8-*-

import time
import os
import cv2

#from lib.dataset.dataietr import DataIter
from torch.utils.data import DataLoader

from lib.core.model.centernet import CenterNet
from lib.core.loss.centernet_loss import CenterNetLoss
from lib.core.utils.torch_utils import EMA
from lib.dataset.dataietr import AlaskaDataIter
from lib.core.base_trainer.metric import *
import torch

from torchcontrib.optim import SWA

import torch.nn as nn

if cfg.TRAIN.mix_precision:
    from apex import amp

class Train(object):
  """Train class.
  """

  def __init__(self,):




    trainds = AlaskaDataIter(cfg.DATA.root_path, cfg.DATA.train_txt_path, training_flag=True)
    self.train_ds = DataLoader(trainds,
                          cfg.TRAIN.batch_size,
                          num_workers=cfg.TRAIN.process_num,
                          shuffle=True)

    valds = AlaskaDataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, training_flag=False)
    self.val_ds = DataLoader(valds,
                         cfg.TRAIN.batch_size,
                         num_workers=cfg.TRAIN.process_num,
                         shuffle=False)



    self.init_lr=cfg.TRAIN.init_lr
    self.warup_step=cfg.TRAIN.warmup_step
    self.epochs = cfg.TRAIN.epoch
    self.batch_size = cfg.TRAIN.batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor

    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    self.model = CenterNet().to(self.device)

    self.load_weight()



    if 'Adamw' in cfg.TRAIN.opt:

      self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                         lr=self.init_lr,eps=1.e-5,
                                         weight_decay=self.l2_regularization)
    else:
      self.optimizer = torch.optim.SGD(self.model.parameters(),
                                       lr=self.init_lr,
                                       momentum=0.9,
                                       weight_decay=self.l2_regularization)

    if cfg.TRAIN.SWA>0:
        ##use swa
        self.optimizer = SWA(self.optimizer)

    if cfg.TRAIN.mix_precision:
        self.model, self.optimizer = amp.initialize( self.model, self.optimizer, opt_level="O1")


    self.model=nn.DataParallel(self.model)

    self.ema = EMA(self.model, 0.999)

    self.ema.register()
    ###control vars
    self.iter_num=0


    # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='max', patience=3,verbose=True)
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, self.epochs,eta_min=1.e-6)

    self.criterion = CenterNetLoss().to(self.device)


  def custom_loop(self):
    """Custom training and testing loop.
    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def train_epoch(epoch_num):

      summary_loss_cls = AverageMeter()
      summary_loss_wh= AverageMeter()
      self.model.train()

      if cfg.MODEL.freeze_bn:
          for m in self.model.modules():
              if isinstance(m, nn.BatchNorm2d):
                  m.eval()
                  if cfg.MODEL.freeze_bn_affine:
                      m.weight.requires_grad = False
                      m.bias.requires_grad = False
      for image,hm_target, wh_target,weights in self.train_ds:

        if epoch_num<10:
            ###excute warm up in the first epoch
            if self.warup_step>0:
                if self.iter_num < self.warup_step:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.iter_num / float(self.warup_step) * self.init_lr
                        lr = param_group['lr']

                    logger.info('warm up with learning rate: [%f]' % (lr))

        start=time.time()


        if cfg.TRAIN.vis:
            for i in range(image.shape[0]):



                img = image[i].numpy()
                img = np.transpose(img,axes=[1,2,0])
                hm = hm_target[i].numpy()
                wh = wh_target[i].numpy()

                if cfg.DATA.use_int8_data:
                    hm = hm[:, :, 0].astype(np.uint8)
                    wh = wh[:, :, 0]
                else:
                    hm = hm[:, :, 0].astype(np.float32)
                    wh = wh[:, :, 0].astype(np.float32)

                cv2.namedWindow('s_hm', 0)
                cv2.imshow('s_hm', hm)
                cv2.namedWindow('s_wh', 0)
                cv2.imshow('s_wh', wh + 1)
                cv2.namedWindow('img', 0)
                cv2.imshow('img', img)
                cv2.waitKey(0)
        else:
            data = torch.from_numpy(image).to(self.device).float()


            if cfg.DATA.use_int8_data:
                hm_target = hm_target.to(self.device).float()/cfg.DATA.use_int8_enlarge
            else:
                hm_target = hm_target.to(self.device).float()
            wh_target = wh_target.to(self.device).float()
            weights = weights.to(self.device).float()

            batch_size = data.shape[0]

            cls,wh = self.model(data)

            cls_loss,wh_loss= self.criterion([cls,wh],[hm_target, wh_target,weights])

            current_loss=cls_loss+wh_loss
            summary_loss_cls.update(cls_loss.detach().item(), batch_size)
            summary_loss_wh.update(wh_loss.detach().item(), batch_size)
            self.optimizer.zero_grad()

            if cfg.TRAIN.mix_precision:
                with amp.scale_loss(current_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                current_loss.backward()

            self.optimizer.step()
            if cfg.TRAIN.ema:
                self.ema.update()
            self.iter_num+=1
            time_cost_per_batch=time.time()-start

            images_per_sec=cfg.TRAIN.batch_size*cfg.TRAIN.num_gpu/time_cost_per_batch


            if self.iter_num%cfg.TRAIN.log_interval==0:

                log_message = '[TRAIN], '\
                              'Epoch %d Step %d, ' \
                              'summary_loss: %.6f, ' \
                              'cls_loss: %.6f, '\
                              'wh_loss: %.6f, ' \
                              'time: %.6f, '\
                              'speed %d images/persec'% (
                                  epoch_num,
                                  self.iter_num,
                                  summary_loss_cls.avg+summary_loss_wh.avg,
                                  summary_loss_cls.avg ,
                                  summary_loss_wh.avg,
                                  time.time() - start,
                                  images_per_sec)
                logger.info(log_message)



        if cfg.TRAIN.SWA>0 and epoch_num>=cfg.TRAIN.SWA:
            self.optimizer.update_swa()

      return summary_loss_cls,summary_loss_wh

    def test_epoch(epoch_num):
        summary_loss_cls = AverageMeter()
        summary_loss_wh = AverageMeter()

        self.model.eval()
        t = time.time()
        with torch.no_grad():
            for step,(image,hm_target, wh_target,weights) in enumerate(self.train_ds):


                data = image.to(self.device).float()


                if cfg.DATA.use_int8_data:
                    hm_target = hm_target.to(self.device).float() / cfg.DATA.use_int8_enlarge
                else:
                    hm_target = hm_target.to(self.device).float()

                wh_target = wh_target.to(self.device).float()
                weights = weights.to(self.device).float()
                batch_size = data.shape[0]


                with torch.no_grad():
                    cls, wh = self.model(data)

                cls_loss,wh_loss = self.criterion([cls,wh], [hm_target, wh_target, weights])

                summary_loss_cls.update(cls_loss.detach().item(), batch_size)
                summary_loss_wh.update(wh_loss.detach().item(), batch_size)

                if step % cfg.TRAIN.log_interval == 0:

                    log_message =   '[VAL], '\
                                    'Epoch %d Step %d, ' \
                                    'summary_loss: %.6f, ' \
                                    'cls_loss: %.6f, '\
                                    'wh_loss: %.6f, ' \
                                    'time: %.6f' % (epoch_num,
                                                    step,
                                                    summary_loss_cls.avg+summary_loss_wh.avg,
                                                    summary_loss_cls.avg,
                                                    summary_loss_wh.avg,
                                                    time.time() - t)

                    logger.info(log_message)


        return summary_loss_cls,summary_loss_wh

    for epoch in range(self.epochs):

      for param_group in self.optimizer.param_groups:
        lr=param_group['lr']
      logger.info('learning rate: [%f]' %(lr))
      t=time.time()

      summary_loss_cls,summary_loss_wh = train_epoch(epoch)

      train_epoch_log_message = '[centernet], '\
                                '[RESULT]: Train. Epoch: %d,' \
                                ' summary_loss: %.5f,' \
                                ' cls_loss: %.6f, ' \
                                ' wh_loss: %.6f, ' \
                                ' time:%.5f' % (epoch,
                                                summary_loss_cls.avg+summary_loss_wh.avg,
                                                summary_loss_cls.avg,
                                                summary_loss_wh.avg,
                                                (time.time() - t))
      logger.info(train_epoch_log_message)

      if cfg.TRAIN.SWA > 0 and epoch >=cfg.TRAIN.SWA:

          ###switch to avg model
          self.optimizer.swap_swa_sgd()

      ##switch eam weighta
      if cfg.TRAIN.ema:
        self.ema.apply_shadow()

      if epoch%cfg.TRAIN.test_interval==0:

          summary_loss_cls,summary_loss_wh = test_epoch(epoch)

          val_epoch_log_message = '[centernet], '\
                                  '[RESULT]: VAL. Epoch: %d,' \
                                  ' summary_loss: %.5f,' \
                                  ' cls_loss: %.6f, ' \
                                  ' wh_loss: %.6f, ' \
                                  ' time:%.5f' % (epoch,
                                                  summary_loss_cls.avg+summary_loss_wh.avg,
                                                  summary_loss_cls.avg,
                                                  summary_loss_wh.avg,
                                                  (time.time() - t))
          logger.info(val_epoch_log_message)

      self.scheduler.step()
      # self.scheduler.step(final_scores.avg)

      #### save model
      if not os.access(cfg.MODEL.model_path, os.F_OK):
          os.mkdir(cfg.MODEL.model_path)


      #### save the model every end of epoch
      current_model_saved_name='./model/centernet_epoch_%d_val_loss%.6f.pth'%(epoch,summary_loss_cls.avg+summary_loss_wh.avg)

      logger.info('A model saved to %s' % current_model_saved_name)
      torch.save(self.model.module.state_dict(),current_model_saved_name)

      ####switch back
      if cfg.TRAIN.ema:
        self.ema.restore()


      if cfg.TRAIN.SWA > 0 and epoch > cfg.TRAIN.SWA:
          ###switch back to plain model to train next epoch
          self.optimizer.swap_swa_sgd()




  def load_weight(self):
      if cfg.MODEL.pretrained_model is not None:
          state_dict=torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
          self.model.load_state_dict(state_dict,strict=False)



