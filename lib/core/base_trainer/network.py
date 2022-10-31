#-*-coding:utf-8-*-
import numpy as np
import sklearn.metrics
import cv2
import time
import os

from torch.utils.data import DataLoader, DistributedSampler

from lib.core.utils.torch_utils import EMA
from lib.dataset.dataietr import AlaskaDataIter
from train_config import config as cfg
#from lib.dataset.dataietr import DataIter

import sklearn.metrics
from lib.core.utils.logger import logger

# from lib.core.base_trainer.model import Net,COTRAIN

import random

from lib.core.base_trainer.metric import *
import torch
import torch.nn.functional as F
from lib.core.base_trainer.model import COTRAIN


if not cfg.TRAIN.vis:
    torch.distributed.init_process_group(backend="nccl",)

class Train(object):
    """Train class.
    """

    def __init__(self,):
        if cfg.TRAIN.vis:
            self.ddp=False
        else:
            self.ddp=True

        train_df=cfg.DATA.train_txt_path
        val_df=cfg.DATA.val_txt_path

        if self.ddp:

            self.train_generator = AlaskaDataIter(train_df, training_flag=True, shuffle=False)


            self.train_sampler=DistributedSampler(self.train_generator,
                                                  shuffle=True)
            self.train_ds = DataLoader(self.train_generator,
                                       cfg.TRAIN.batch_size,
                                       num_workers=cfg.TRAIN.process_num,
                                       sampler=self.train_sampler)

            self.val_generator = AlaskaDataIter(val_df, training_flag=False, shuffle=False)


            self.val_sampler=DistributedSampler(self.val_generator,
                                                shuffle=False)

            self.val_ds = DataLoader(self.val_generator,
                                     cfg.TRAIN.validatiojn_batch_size,
                                     num_workers=cfg.TRAIN.process_num,
                                     sampler=self.val_sampler)

            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)

        else:
            self.train_generator = AlaskaDataIter(train_df, training_flag=True, shuffle=False)

            self.train_ds = DataLoader(self.train_generator,
                                       cfg.TRAIN.batch_size,
                                       num_workers=cfg.TRAIN.process_num,shuffle=True)

            self.val_generator = AlaskaDataIter(val_df, training_flag=False, shuffle=False)

            self.val_ds = DataLoader(self.val_generator,
                                     cfg.TRAIN.validatiojn_batch_size,
                                     num_workers=cfg.TRAIN.process_num,shuffle=False)

            self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.fold = 0
        self.fp16=cfg.TRAIN.mix_precision

        self.init_lr = cfg.TRAIN.init_lr
        self.warup_step = cfg.TRAIN.warmup_step
        self.epochs = cfg.TRAIN.epoch
        self.batch_size = cfg.TRAIN.batch_size
        self.l2_regularization = cfg.TRAIN.weight_decay_factor

        self.early_stop = cfg.TRAIN.early_stop

        self.accumulation_step = cfg.TRAIN.accumulation_batch_size // cfg.TRAIN.batch_size

        self.gradient_clip = cfg.TRAIN.gradient_clip

        self.save_dir=cfg.MODEL.model_path
        #### make the device


        self.model = COTRAIN().to(self.device)

        self.load_weight()
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if 'Adamw' in cfg.TRAIN.opt:

            # self.optimizer = self.build_opt()
            self.optimizer=torch.optim.AdamW(self.model.parameters(),
                                             lr=self.init_lr,
                                             weight_decay=self.l2_regularization)

        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.init_lr,
                                             momentum=0.9,
                                             weight_decay=self.l2_regularization)

        if cfg.TRAIN.SWA > 0:
            ##use swa
            self.optimizer = SWA(self.optimizer)


        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[local_rank],
                                                                   output_device=local_rank,
                                                                   static_graph=True,
                                                                   find_unused_parameters=True

                                                                   )
        else:
            self.model=torch.nn.DataParallel(self.model)

        if cfg.TRAIN.vis:
            self.vis()

        self.ema = EMA(self.model, 0.97)

        # self.ema.register()
        ###control vars
        self.iter_num = 0

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='max', patience=5,
        #                                                             min_lr=1e-6,factor=0.5,verbose=True)

        # if cfg.TRAIN.lr_scheduler=='cos':
        if 1:
            logger.info('lr_scheduler.CosineAnnealingLR')
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        self.epochs,
                                                                        eta_min=1.e-7)
        else:
            logger.info('lr_scheduler.ReduceLROnPlateau')
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='max',
                                                                        patience=5,
                                                                        min_lr=1e-7,
                                                                        factor=cfg.TRAIN.lr_scheduler_factor,
                                                                        verbose=True)

        self.scaler = torch.cuda.amp.GradScaler()




    def build_opt(self):




        # def build_optimizer( model):
        #     """
        #     Build optimizer, set weight decay of normalization to 0 by default.
        #     """
        #     skip = {}
        #     skip_keywords = {}
        #     if hasattr(model, 'no_weight_decay'):
        #         skip = model.no_weight_decay()
        #     if hasattr(model, 'no_weight_decay_keywords'):
        #         skip_keywords = model.no_weight_decay_keywords()
        #     parameters = set_weight_decay(model, skip, skip_keywords)
        #
        #     # opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
        #     # optimizer = None
        #
        #     optimizer = torch.optim.AdamW(parameters,
        #                                   lr=self.init_lr,
        #                                   weight_decay=self.l2_regularization)
        #
        #     return optimizer
        #
        #
        # def set_weight_decay(model, skip_list=(), skip_keywords=()):
        #     has_decay = []
        #     no_decay = []
        #
        #     for name, param in model.named_parameters():
        #         if not param.requires_grad:
        #             continue  # frozen weights
        #         if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
        #                 check_keywords_in_name(name, skip_keywords):
        #             no_decay.append(param)
        #             # print(f"{name} has no weight decay")
        #         else:
        #             has_decay.append(param)
        #     return [{'params': has_decay},
        #             {'params': no_decay, 'weight_decay': 0.}]
        #
        #
        # def check_keywords_in_name(name, keywords=()):
        #     isin = False
        #     for keyword in keywords:
        #         if keyword in name:
        #             isin = True
        #     return isin
        #
        #
        #
        # opt=build_optimizer(self.model)

        from mmcv.runner import build_optimizer

        optimizer_cfg = dict( type='AdamW', lr=self.init_lr,
                              weight_decay=self.l2_regularization,
                              betas=(0.9, 0.999),
                              # weight_decay=0.01,
                              paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                              'relative_position_bias_table': dict(decay_mult=0.),
                                                              'norm': dict(decay_mult=0.)}))
        opt = build_optimizer(self.model, optimizer_cfg)



        return opt

    def criterion(self,y_pred, y_true):

        return 0.5*self.BCELoss(y_pred, y_true) + 0.5*self.DiceLoss(y_pred, y_true)

    def dice_coef(self,y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
        y_true = y_true.to(torch.float32)

        y_pred = (y_pred>thr).to(torch.float32)

        inter = (y_true*y_pred).sum(dim=dim)
        den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
        dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))

        return dice

    def iou_coef(self,y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred>thr).to(torch.float32)
        inter = (y_true*y_pred).sum(dim=dim)
        union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
        iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
        return iou

    def metric(self,targets,preds ):

        mse=torch.mean((preds-targets)**2)

        mad=torch.mean(torch.abs(preds-targets))
        return mad,mse

    def custom_loop(self):
        """Custom training and testing loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          test_dist_dataset: Testing dataset created using strategy.
          strategy: Distribution strategy.
        Returns:
          train_loss, train_accuracy, test_loss, test_accuracy
        """

        def distributed_train_epoch(epoch_num):

            self.train_sampler.set_epoch(epoch_num)

            summary_loss = AverageMeter()

            summary_studen_loss = AverageMeter()
            summary_teacher_loss = AverageMeter()
            summary_distill_loss = AverageMeter()

            self.model.train()

            if cfg.MODEL.freeze_bn:
                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        if cfg.MODEL.freeze_bn_affine:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

            for k,(image,hm_target, wh_target,weights) in enumerate(self.train_ds):

                if epoch_num<10:
                    ###excute warm up in the first epoch
                    if self.warup_step>0:
                        if self.iter_num < self.warup_step:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.iter_num / float(self.warup_step) * self.init_lr
                                lr = param_group['lr']

                            logger.info('warm up with learning rate: [%f]' % (lr))

                start=time.time()


                ## prepare data
                data = image.to(self.device).float()

                hm_target = hm_target.to(self.device).float()
                wh_target = wh_target.to(self.device).float()
                weights = weights.to(self.device).float()

                batch_size = data.shape[0]
                ###

                with torch.cuda.amp.autocast(enabled=self.fp16):

                    student_loss, teacher_loss, distill_loss,mate = self.model(data,hm_target, wh_target,weights)

                    # calculate the final loss, backward the loss, and update the model
                    current_loss =  student_loss+ teacher_loss+ distill_loss

                    if torch.isnan(current_loss):
                        print('there is a nan loss ')

                summary_loss.update(current_loss.detach().item(), batch_size)
                summary_studen_loss.update(student_loss.detach().item(), batch_size)
                summary_teacher_loss.update(teacher_loss.detach().item(), batch_size)
                summary_distill_loss.update(distill_loss.detach().item(), batch_size)

                self.scaler.scale(current_loss).backward()

                if ((self.iter_num + 1) % self.accumulation_step) == 0:
                    if self.gradient_clip>0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip, norm_type=2)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if cfg.TRAIN.ema:
                    self.ema.update()
                self.iter_num+=1
                time_cost_per_batch=time.time()-start

                images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch


                if self.iter_num%cfg.TRAIN.log_interval==0:


                    log_message = '[fold %d], ' \
                                  'Train Step %d, ' \
                                  '[%d/%d ] ' \
                                  'summary_loss: %.6f, ' \
                                  'summary_student_loss: %.6f, ' \
                                  'summary_teacher_loss: %.6f, ' \
                                  'summary_distill_loss: %.6f, ' \
                                  'time: %.6f, ' \
                                  'speed %d images/persec'% (
                                      self.fold,
                                      self.iter_num,
                                      k,
                                      len(self.train_ds),
                                      summary_loss.avg,
                                      summary_studen_loss.avg,
                                      summary_teacher_loss.avg,
                                      summary_distill_loss.avg,
                                      time.time() - start,
                                      images_per_sec)
                    logger.info(log_message)


            if cfg.TRAIN.SWA>0 and epoch_num>=cfg.TRAIN.SWA:
                self.optimizer.update_swa()

            return summary_loss

        def distributed_test_epoch(epoch_num):
            summary_student_loss = AverageMeter()
            summary_teacher_loss= AverageMeter()



            self.model.eval()
            t = time.time()

            with torch.no_grad():
                for step,(image,hm_target, wh_target,weights) in enumerate(self.val_ds):

                    data = image.to(self.device).float()


                    hm_target = hm_target.to(self.device).float()

                    wh_target = wh_target.to(self.device).float()
                    weights = weights.to(self.device).float()
                    batch_size = data.shape[0]

                    with torch.no_grad():
                        student_loss, teacher_loss, distill_loss,mate = self.model(data,hm_target, wh_target,weights)





                    if self.ddp:
                            torch.distributed.all_reduce(student_loss.div_(torch.distributed.get_world_size()))
                            torch.distributed.all_reduce(teacher_loss.div_(torch.distributed.get_world_size()))

                    summary_student_loss.update(student_loss.detach().item(), batch_size)
                    summary_teacher_loss.update(teacher_loss.detach().item(), batch_size)



                    # summary_dice.update(val_dice.detach().item(), batch_size)
                    # summary_mad.update(val_mad.detach().item(), batch_size)
                    # summary_mse.update(val_mse.detach().item(), batch_size)

                if step % cfg.TRAIN.log_interval == 0:

                    log_message = '[fold %d], ' \
                                  'Val Step %d, ' \
                                  'summary_student_loss: %.6f, ' \
                                  'summary_teacher_loss: %.6f, ' \
                                  'time: %.6f' % (
                                      self.fold,step,
                                      summary_student_loss.avg,
                                      summary_teacher_loss.avg,
                                      time.time() - t)

                    logger.info(log_message)



            return summary_student_loss,summary_teacher_loss




        best_roc_auc=0.
        not_improvement=0
        for epoch in range(self.epochs):

            for param_group in self.optimizer.param_groups:
                lr=param_group['lr']
            logger.info('learning rate: [%f]' %(lr))
            t=time.time()


            summary_loss = distributed_train_epoch(epoch)
            train_epoch_log_message = '[fold %d], ' \
                                      '[RESULT]: TRAIN. Epoch: %d,' \
                                      ' summary_loss: %.5f,' \
                                      ' time:%.5f' % (
                                          self.fold,
                                          epoch,
                                          summary_loss.avg,
                                          (time.time() - t))
            logger.info(train_epoch_log_message)

            if cfg.TRAIN.SWA > 0 and epoch >=cfg.TRAIN.SWA:

                ###switch to avg model
                self.optimizer.swap_swa_sgd()

            ##switch eam weighta
            if cfg.TRAIN.ema:
                self.ema.apply_shadow()

            if epoch%cfg.TRAIN.test_interval==0 and epoch>0 or epoch%10==0:

                summary_student_loss,summary_teacher_loss= distributed_test_epoch(epoch)

                val_epoch_log_message = '[fold %d], ' \
                                        '[RESULT]: VAL. Epoch: %d,' \
                                        ' summary_student_loss: %.5f,' \
                                        ' summary_teacher_loss: %.5f,' \
                                        ' time:%.5f' % (
                                            self.fold,
                                            epoch,
                                            summary_student_loss.avg,
                                            summary_teacher_loss.avg,
                                            (time.time() - t))

                logger.info(val_epoch_log_message)

                self.scheduler.step()
                # self.scheduler.step(acc_score.avg)

                #### save model
                if not os.access(cfg.MODEL.model_path, os.F_OK):
                    os.mkdir(cfg.MODEL.model_path)
                ###save the best auc model

                #### save the model every end of epoch
                #### save the model every end of epoch
                current_model_saved_name='%s/fold%d_epoch_%d_val_loss_%.6f.pth'%(cfg.MODEL.model_path,
                                                                                 self.fold,
                                                                                 epoch,
                                                                                 summary_student_loss.avg)
                logger.info('A model saved to %s' % current_model_saved_name)
                #### save the model every end of epoch
                if  self.ddp and torch.distributed.get_rank() == 0 :
                    torch.save(self.model.module.state_dict(),current_model_saved_name)
                else:
                    torch.save(self.model.module.state_dict(),current_model_saved_name)

            else:
                self.scheduler.step()
            ####switch back
            if cfg.TRAIN.ema:
                self.ema.restore()

            # save_checkpoint({
            #           'state_dict': self.model.state_dict(),
            #           },iters=epoch,tag=current_model_saved_name)

            if cfg.TRAIN.SWA > 0 and epoch > cfg.TRAIN.SWA:
                ###switch back to plain model to train next epoch
                self.optimizer.swap_swa_sgd()

            # if cur_roc_auc_score>best_roc_auc:
            #     best_roc_auc=cur_roc_auc_score
            #     logger.info(' best metric score update as %.6f' % (best_roc_auc))
            # else:
            #     not_improvement+=1

            if not_improvement>=self.early_stop:
                logger.info(' best metric score not improvement for %d, break'%(self.early_stop))
                break



    def load_weight(self):
        if cfg.MODEL.pretrained_model is not None:
            state_dict=torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
            self.model.load_state_dict(state_dict,strict=False)


    def vis(self):
        print('it is here')

        print('show it, here')


        # state_dict=torch.load('/Users/liangzi/Downloads/fold1_epoch_29_val_loss_0.049467_val_dice_0.964158.pth', map_location=self.device)
        # self.model.load_state_dict(state_dict,strict=False)
        self.model.eval()
        for step,(images,hm_target, wh_target,weights) in enumerate(self.train_ds):

            print(images.shape)
            print(hm_target.shape)
            batch_size = hm_target.shape[0]
            # output = self.model(data)
            #
            # predict = nn.Sigmoid()(output).cpu().detach().numpy()

            for i in range(images.shape[0]):

                example_image=np.array(images[i]*255,dtype=np.uint8)
                example_image=np.transpose(example_image,[1,2,0])
                example_hm=np.array(hm_target[i,:,:,0]*255)
                example_hm=example_hm.astype(np.uint8)
                # example_hm=np.transpose(example_hm,[1,2,0])


                _h, _w, _ = example_image.shape



                cv2.imshow('model_example',example_image)
                cv2.imshow('example_hm',example_hm)




                cv2.waitKey(0)
