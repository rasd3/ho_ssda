# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

from torch import batch_norm
import torch.nn as nn
import torch 
# from .roi_head_template import RoIHeadTemplate
from .roi_head_template_centerpoint_pointpillar import RoIHeadTemplate_CenterPoint_PointPillar
from ..model_utils.model_nms_utils import class_agnostic_nms

# from det3d.core import box_torch_ops

# from ..registry import ROI_HEAD

# @ROI_HEAD.register_module
class RoIHeadDynamicPillarV2(RoIHeadTemplate_CenterPoint_PointPillar):
    def __init__(self, input_channels, model_cfg, num_class=1, code_size=7, add_box_param=False, test_cfg=None):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg 
        self.code_size = code_size
        self.add_box_param = add_box_param

        pre_channel = model_cfg.get('PRE_CHANNEL', 1920)

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False), #ori
                #nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False), #fc
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=code_size,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')
        self.batch_size = None

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
    
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def reorder_first_stage_pred_and_feature(self, batch_dict):
        batch_size = batch_dict['batch_size']
        box_length = batch_dict['rois'].shape[2] 
        features = batch_dict['roi_features']
        feature_vector_length = features[0].shape[-1] #sum([feat[0].shape[-1] for feat in features])
        NMS_POST_MAXSIZE= batch_dict['rois'].shape[1]

        rois = batch_dict['rois'].new_zeros((batch_size, 
            NMS_POST_MAXSIZE, box_length 
        ))
        roi_scores = batch_dict['roi_scores'].new_zeros((batch_size,
            NMS_POST_MAXSIZE
        ))
        roi_labels = batch_dict['roi_labels'].new_zeros((batch_size,
            NMS_POST_MAXSIZE), dtype=torch.long
        )
        roi_features = features[0].new_zeros((batch_size, 
            NMS_POST_MAXSIZE, feature_vector_length 
        ))

        for i in range(batch_size):
            num_obj = features[i].shape[0]
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = batch_dict['rois'][i]

            if box_length == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds[:num_obj]
            roi_labels[i, :num_obj] = batch_dict['roi_labels'][i, :num_obj] #+ 1
            roi_scores[i, :num_obj] = batch_dict['roi_scores'][i, :num_obj]
            roi_features[i, :num_obj] = features[i] #torch.cat([feat for feat in features[i]], dim=-1)

        batch_dict['rois'] = rois 
        batch_dict['roi_labels'] = roi_labels 
        batch_dict['roi_scores'] = roi_scores  
        batch_dict['roi_features'] = roi_features

        batch_dict['has_class_labels']= True 

        return batch_dict 

    def forward(self, batch_dict, training=True, disable_gt_roi_when_pseudo_labeling=False):
        """
        :param input_data: input dict
        :return:
        """
        batch_dict = self.reorder_first_stage_pred_and_feature(batch_dict)
        self.batch_size = batch_dict['batch_size']
        # targets_dict = self.proposal_layer(
        #     batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training and not disable_gt_roi_when_pseudo_labeling else 'TEST']
        # )

        if training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_features'] = targets_dict['roi_features']
            batch_dict['roi_scores'] = targets_dict['roi_scores']
            #batch_dict['roi_ious'] = targets_dict['gt_iou_of_rois']

        # RoI aware pooling
        if self.add_box_param:
            batch_dict['roi_features'] = torch.cat([batch_dict['roi_features'], batch_dict['rois'], batch_dict['roi_scores'].unsqueeze(-1)], dim=-1)

        pooled_features = batch_dict['roi_features'].reshape(-1, 1,
            batch_dict['roi_features'].shape[-1]).contiguous()  # (BxN, 1, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous() # (BxN, C, 1)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1)) #ori
        #shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1)) #fc
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2) #ori
        #rcnn_cls = self.cls_layers(shared_features).contiguous().squeeze(dim=1).view(-1, 1)  # (B, 1 or 2) #fc
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C) #ori
        #rcnn_reg = self.reg_layers(shared_features).contiguous().squeeze(dim=1)  # (B, C)  #fc

        if not training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # batch_dict['batch_cls_preds_roi'] = batch_cls_preds
            # batch_dict['batch_box_preds_roi'] = batch_box_preds
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        if self.training:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        
        return batch_dict        
