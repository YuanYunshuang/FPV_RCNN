from torch import nn
import torch
from models import *
# import numpy as np
# from torch.nn import Sequential
# from models.utils import xavier_init
# from matplotlib import pyplot as plt
# from vlib.image import draw_box_plt


def _build_deconv_block(in_channels, out_channels):
    return [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()]


def _build_conv_block(in_channels, out_channels, stride=1):
    return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()]


class FusionPVRCNN(nn.Module):
    """
    This model is based on CIA-SSD. Each point cloud will be forwarded once to obtain
    logits features, which are then fused to refine the object detection.
    """
    def __init__(self, mcfg, cfg, dcfg):
        super(FusionPVRCNN, self).__init__()
        self.mcfg = mcfg
        self.test_cfg = cfg.TEST
        self.pc_range = dcfg.pc_range

        self.vfe = MeanVFE(dcfg.n_point_features)
        self.spconv_block = VoxelBackBone8x(mcfg.SPCONV,
                                            input_channels=dcfg.n_point_features,
                                            grid_size=dcfg.grid_size)
        self.map_to_bev = HeightCompression(mcfg.MAP2BEV)
        self.ssfa = SSFA(mcfg.SSFA)
        self.head = MultiGroupHead(mcfg.HEAD, dcfg.pc_range)

        self.vsa = VoxelSetAbstraction(mcfg.VSA, dcfg.voxel_size, dcfg.pc_range, num_bev_features=128,
                                       num_rawpoint_features=3)
        self.semseg = SemSeg(**mcfg.SEMSEG)
        search_range = dcfg.gps_noise_std[[0, 1, -1]] * dcfg.gps_noise_max_ratio[[0, 1, -1]]
        search_range[:2] *= 2
        self.matcher = MatcherV3(mcfg.MATCHER, dcfg.pc_range, has_noise=dcfg.add_gps_noise,
                                 search_range=search_range)
        self.roi_head = RoIHead(mcfg.ROI_HEAD, self.head.box_coder)

        # self.set_trainable_parameters(mcfg.params_train)

    def set_trainable_parameters(self, block_names):
        for param in self.named_parameters():
            m = getattr(self, param[0].split('.')[0])
            if m.__class__.__name__ not in block_names:
                param[1].requires_grad = False

    def forward(self, batch_dict):
        n_coop = len(batch_dict['ids']) - 1
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        logits_feature = self.ssfa(batch_dict['spatial_features'])
        preds = self.head(logits_feature)
        batch_dict["preds_egos"] = preds

        if n_coop > 0:
            batch_dict, num_total_dets = self.get_preds(batch_dict)

            if num_total_dets > 0:
                batch_dict = self.vsa(batch_dict)
                batch_dict = self.semseg(batch_dict)
                batch_dict = self.matcher(batch_dict)
                batch_dict = self.roi_head(batch_dict)

        return batch_dict

    def shift_and_filter_preds(self, batch_dict):
        batch_dict['det_boxes'] = []
        batch_dict['det_scores'] = []
        batch_dict['det_boxes_ego_coords'] = []
        batch_dets = self.post_processing_ego(batch_dict, self.test_cfg, det_all=True)
        shifts_coop_to_ego = batch_dict['translations']- batch_dict['translations'][:1]
        total_dets = 0
        for b, dets in enumerate(batch_dets):
            pred_boxes = dets['box_lidar']
            pred_boxes_ego_coords = pred_boxes.clone().detach()
            pred_boxes_ego_coords[:, :2] = pred_boxes_ego_coords[:, :2] + shifts_coop_to_ego[b][None, :2]
            if torch.isnan(pred_boxes_ego_coords).any():
                print('debug')
            # mask pred. boxes that are in the detection range
            in_range_mask = torch.norm(pred_boxes_ego_coords[:, :2], dim=-1) < self.pc_range[3]
            batch_dict['det_boxes'].append(pred_boxes[in_range_mask])
            batch_dict['det_scores'].append(dets['scores'][in_range_mask])
            batch_dict['det_boxes_ego_coords'].append(pred_boxes_ego_coords[in_range_mask])
            total_dets += in_range_mask.sum()
        return batch_dict, total_dets

    def get_preds(self, batch_dict):
        batch_dict['det_boxes'] = []
        batch_dict['det_scores'] = []
        batch_dets = self.post_processing_ego(batch_dict, self.test_cfg, det_all=True)
        total_dets = 0
        for b, dets in enumerate(batch_dets):
            pred_boxes = dets['box_lidar'].clone().detach()
            batch_dict['det_boxes'].append(pred_boxes)
            batch_dict['det_scores'].append(dets['scores'])
            total_dets += len(pred_boxes)
        return batch_dict, total_dets

    def _make_model_input(self, batch_dict, batch_idx):
        data_dict = {}
        for k, v in batch_dict.items():
            if isinstance(v, list):
                data_dict[k] = v[batch_idx]
            elif k in ["anchors", "labels", "reg_targets", "reg_weights"]:
                data_dict[k] = v[batch_idx].unsqueeze(dim=0)
            else:
                data_dict[k] = v
        data_dict['batch_size'] = len(data_dict['cloud_sizes'])
        return data_dict

    def loss(self, batch_dict):
        batch_dict['preds_dict'] = batch_dict['preds_egos']
        loss_ego = self.head.loss(batch_dict)
        loss_kpts, accs = self.semseg.loss(batch_dict)

        # supervise fusion detetion only for ego vehicle
        if len(batch_dict['ids']) > 1 and 'boxes_fused' in batch_dict:
            loss_fuse = self.roi_head.get_loss(batch_dict)
            loss = {
                'loss': loss_ego['loss'] + loss_fuse['loss'] + loss_kpts,
                'loss_ego': loss_ego['loss'],
                'loss_kpts': loss_kpts,
                'loss_fuse': loss_fuse['loss'],
                'loss_fuse_cls': loss_fuse['loss_cls'],
                'loss_fuse_iou': loss_fuse['loss_iou'],
                'loss_fuse_reg': loss_fuse['loss_reg'],
            }

        else:
            loss = {
                'loss': loss_ego['loss'] + loss_kpts,
                'loss_kpts': loss_kpts,
                'loss_ego': loss_ego['loss']
            }
        loss.update(accs)

        return loss

    def post_processing(self, batch_dict, test_cfg):
        if 'rcnn_reg' in batch_dict.keys():
            detections = self.roi_head.get_detections(batch_dict)
        elif 'preds_final' in batch_dict.keys():
            detections = self.post_processing_final(batch_dict, test_cfg)
        else:
            detections = self.post_processing_ego(batch_dict, test_cfg)
        return detections

    def post_processing_final(self, batch_dict, test_cfg):
        preds_dict = batch_dict['preds_final']
        anchors = batch_dict["anchors"][0:1]
        batch_size = 1
        anchors_flattened = anchors.view(batch_size, -1, self.head.box_n_dim)
        cls_preds = preds_dict["cls_preds"].view(batch_size, -1, self.head.num_classes)  # [8, 70400, 1]
        reg_preds = preds_dict["box_preds"].view(batch_size, -1, self.head.box_coder.code_size)  # [batch_size, 70400, 7]
        iou_preds = preds_dict["iou_preds"].view(batch_size, -1, 1)
        coop_boxes = batch_dict["coop_boxes_in_egoCS"][0:1].view(batch_size, -1, self.head.box_coder.code_size)
        if self.head.use_direction_classifier:
            dir_preds = preds_dict["dir_cls_preds"].view(batch_size, -1, 2)
        else:
            dir_preds = None

        box_preds = self.head.box_coder.decode_torch(reg_preds[:, :, :self.head.box_coder.code_size],
                                                anchors_flattened)# .squeeze()
        detections = self.fusion_detect.det_head.get_task_detections(test_cfg,
                                              cls_preds, box_preds,
                                              dir_preds, iou_preds,
                                              batch_coop_boxes=coop_boxes,
                                              batch_anchors=anchors)
        return detections

    def post_processing_ego(self, batch_dict, test_cfg, det_all=False):
        preds_dict = batch_dict['preds_egos']
        anchors = batch_dict["anchors"]
        coop_boxes = batch_dict["coop_boxes"]
        batch_size = 1
        if det_all:
            batch_size = batch_dict['batch_size']
        else:
            anchors = anchors[0:1]
            if self.training:
                coop_boxes = batch_dict["coop_boxes"][:batch_size].view(batch_size, -1,
                                                                             self.head.box_coder.code_size)
        anchors_flattened = anchors.view(batch_size, -1, self.head.box_n_dim)
        cls_preds = preds_dict["cls_preds"][:batch_size].view(batch_size, -1, self.head.num_classes)  # [8, 70400, 1]
        reg_preds = preds_dict["box_preds"][:batch_size].view(batch_size, -1, self.head.box_coder.code_size)  # [batch_size, 70400, 7]
        iou_preds = preds_dict["iou_preds"][:batch_size].view(batch_size, -1, 1)

        if self.head.use_direction_classifier:
            dir_preds = preds_dict["dir_cls_preds"][:batch_size].view(batch_size, -1, 2)
        else:
            dir_preds = None

        box_preds = self.head.box_coder.decode_torch(reg_preds[:, :, :self.head.box_coder.code_size],
                                                anchors_flattened)# .squeeze()
        detections = self.head.get_task_detections(test_cfg,
                                              cls_preds, box_preds,
                                              dir_preds, iou_preds,
                                              batch_coop_boxes=coop_boxes,
                                              batch_anchors=anchors)

        return detections




