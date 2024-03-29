import os

import torch
import copy

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from .centerpoint import CenterPoint


class CenterPoint_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.centerpoint = CenterPoint(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.centerpoint_ema = CenterPoint(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.centerpoint_ema.parameters():
            param.detach_()
        self.add_module('centerpoint', self.centerpoint)
        self.add_module('centerpoint_ema', self.centerpoint_ema)

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE

    def forward(self, batch_dict):
        if self.training:
            mask = batch_dict['mask'].view(-1)

            labeled_mask = torch.nonzero(mask).squeeze(1).long()
            unlabeled_mask = torch.nonzero(1-mask).squeeze(1).long()
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = copy.deepcopy(batch_dict[k])
                else:
                    batch_dict_ema[k] = batch_dict[k]

            with torch.no_grad():
                # self.centerpoint_ema.eval()  # Important! must be in train mode
                for cur_module in self.centerpoint_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts, recall_dicts = self.centerpoint_ema.post_processing(batch_dict_ema)
                pseudo_boxes = []
                pseudo_scores = []
                pseudo_labels = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                max_pseudo_box_num = 0
                for ind in unlabeled_mask:
                    pseudo_score = pred_dicts[ind]['pred_scores']
                    pseudo_box = pred_dicts[ind]['pred_boxes']
                    pseudo_label = pred_dicts[ind]['pred_labels']
                    # pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']

                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                        continue

                    conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                        0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1))

                    valid_inds = pseudo_score > conf_thresh.squeeze()

                    # valid_inds = valid_inds * (pseudo_sem_score > self.sem_thresh[0])

                    # pseudo_sem_score = pseudo_sem_score[valid_inds]
                    pseudo_box = pseudo_box[valid_inds]
                    pseudo_label = pseudo_label[valid_inds]

                    #  if len(valid_inds) > max_box_num:
                       #  _, inds = torch.sort(pseudo_score, descending=True)
                       #  inds = inds[:max_box_num]
                       #  pseudo_box = pseudo_box[inds]
                       #  pseudo_label = pseudo_label[inds]

                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]
                    # pseudo_scores.append(pseudo_score)
                    # pseudo_labels.append(pseudo_label)

                max_box_num = batch_dict['gt_boxes'].shape[1]

                # assert max_box_num >= max_pseudo_box_num
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_mask, ...]

                if max_box_num >= max_pseudo_box_num:
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        diff = max_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        batch_dict['gt_boxes'][unlabeled_mask[i]] = pseudo_box
                else:
                    ori_boxes = batch_dict['gt_boxes']
                    new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                            device=ori_boxes.device)
                    for i, inds in enumerate(labeled_mask):
                        diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                        new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                        new_boxes[inds] = new_box
                    for i, pseudo_box in enumerate(pseudo_boxes):

                        diff = max_pseudo_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        new_boxes[unlabeled_mask[i]] = pseudo_box
                    batch_dict['gt_boxes'] = new_boxes

                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_x_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_x'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_y_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_y'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_rotation_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['rot_angle'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_scaling_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['scale'][unlabeled_mask, ...]
                )

                pseudo_ious = []
                pseudo_accs = []
                pseudo_fgs = []
                #  for i, ind in enumerate(unlabeled_mask):
                    #  'statistics'
                    #  anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                        #  batch_dict['gt_boxes'][ind, ...][:, 0:7],
                        #  ori_unlabeled_boxes[i, :, 0:7])
                    #  cls_pseudo = batch_dict['gt_boxes'][ind, ...][:, 7]
                    #  unzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
                    #  cls_pseudo = cls_pseudo[unzero_inds]

            ones = torch.ones((1), device=unlabeled_mask.device)
            sem_score_fg = ones
            sem_score_bg = ones
            pseudo_ious.append(ones)
            pseudo_accs.append(ones)
            pseudo_fgs.append(ones)

            for cur_module in self.centerpoint.module_list:
                batch_dict = cur_module(batch_dict)

            disp_dict = {}
            # loss_rpn_cls, loss_rpn_box, tb_dict = self.centerpoint.dense_head.get_loss(scalar=False)
            loss_rpn_cls, loss_rpn_box, tb_dict = self.centerpoint.dense_head.get_ssl_loss(scalar=False)
           
            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum() + loss_rpn_cls[unlabeled_mask, ...].sum() * self.unlabeled_weight

            loss_rpn_box = loss_rpn_box[labeled_mask, ...].sum() + loss_rpn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight

            loss = loss_rpn_cls + loss_rpn_box
            tb_dict_ = {}
            hm_loss_labeled = 0
            hm_loss_unlabeled = 0
            loc_loss_labeled = 0
            loc_loss_unlabeled = 0
            for key in tb_dict.keys():
                if 'hm' in key:
                    hm_loss_labeled += tb_dict[key][labeled_mask, ...].sum()
                    hm_loss_unlabeled += tb_dict[key][unlabeled_mask, ...].sum()
                elif 'loc' in key:
                    loc_loss_labeled += tb_dict[key][labeled_mask, ...].sum()
                    loc_loss_unlabeled += tb_dict[key][unlabeled_mask, ...].sum()
                elif 'rpn' in key:
                    rpn_loss_labeled = tb_dict[key][labeled_mask, ...].sum()
                    rpn_loss_unlabeled = tb_dict[key][unlabeled_mask, ...].sum()
            tb_dict_["hm_loss_labeled"] = hm_loss_labeled
            tb_dict_["hm_loss_unlabeled"] = hm_loss_unlabeled
            tb_dict_["loc_loss_labeled"] = loc_loss_labeled
            tb_dict_["loc_loss_unlabeled"] = loc_loss_labeled
            tb_dict_["rpn_loss_labeled"] = rpn_loss_labeled
            tb_dict_["rpn_loss_unlabeled"] = rpn_loss_unlabeled

            tb_dict_['pseudo_ious'] = torch.cat(pseudo_ious, dim=0).mean()
            tb_dict_['pseudo_accs'] = torch.cat(pseudo_accs, dim=0).mean()
            tb_dict_['sem_score_fg'] = sem_score_fg.mean()
            tb_dict_['sem_score_bg'] = sem_score_bg.mean()


            tb_dict_['max_box_num'] = max_box_num
            tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.centerpoint.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.centerpoint.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        alpha = 0.999
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.centerpoint_ema.parameters(), self.centerpoint.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'centerpoint.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'centerpoint_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
