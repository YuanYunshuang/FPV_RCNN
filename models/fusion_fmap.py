from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
import torch
from models import *
import numpy as np
from models.utils import xavier_init
from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
import matplotlib.pyplot as plt


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


class FusionFmap(nn.Module):
    """
    This model is based on CIA-SSD. Each point cloud will be forwarded once to obtain
    logits features, which are then fused to refine the object detection.
    """
    def __init__(self, mcfg, cfg, dcfg):
        super(FusionFmap, self).__init__()
        self.mcfg = mcfg
        self.has_noise = dcfg.add_gps_noise
        self.test_cfg = cfg.TEST
        self.pc_range = dcfg.pc_range
        self.cpm_feature_size = np.array(dcfg.feature_map_size)[1:] * 2** len(mcfg.FUDET['upsample_channels'])
        self.cpm_feature_reso = dcfg.grid_size[:2] / self.cpm_feature_size * np.array(dcfg.voxel_size)[:2]

        self.vfe = MeanVFE(dcfg.n_point_features)
        self.spconv_block = VoxelBackBone8x(mcfg.SPCONV,
                                            input_channels=dcfg.n_point_features,
                                            grid_size=dcfg.grid_size)
        self.map_to_bev = HeightCompression(mcfg.MAP2BEV)
        self.ssfa = SSFA(mcfg.SSFA)
        self.cpm_enc = CPMEnc(**mcfg.CPMEnc)
        self.head = MultiGroupHead(mcfg.HEAD, dcfg.pc_range)
        self.vsa = VoxelSetAbstraction(mcfg.VSA, dcfg.voxel_size, dcfg.pc_range, num_bev_features=128,
                                       num_rawpoint_features=3)
        self.semseg = SemSeg(**mcfg.SEMSEG)
        self.matcher = MatcherV3(mcfg.MATCHER, dcfg.pc_range, has_noise=dcfg.add_gps_noise,
                                 search_range=dcfg.gps_noise_std[[0, 1, -1]] * dcfg.gps_noise_max_ratio[[0, 1, -1]])
        # self.matcher = MatcherV1(dcfg.pc_range, dcfg.add_gps_noise)
        self.fusion_detect = FUDET(mcfg, dcfg)
        # self.vsa = VoxelSetAbstraction(mcfg.VSA, dcfg.voxel_size, dcfg.pc_range, num_bev_features=128,
        #                                num_rawpoint_features=3)

        # self.set_trainable_parameters(mcfg.params_train)

    def set_trainable_parameters(self, block_names):
        for param in self.named_parameters():
            m = getattr(self, param[0].split('.')[0])
            if m.__class__.__name__ not in block_names:
                param[1].requires_grad = False

    def forward(self, batch_dict):
        n_coop = len(batch_dict['ids']) - 1
        data_dict = self.vfe(batch_dict)
        data_dict = self.spconv_block(data_dict)
        data_dict = self.map_to_bev(data_dict)
        logits_feature = self.ssfa(data_dict['spatial_features'])
        preds = self.head(logits_feature)
        batch_dict["preds_egos"] = preds
        batch_dict["detections"] = self.post_processing_ego(batch_dict, self.test_cfg, det_all=True)

        if n_coop > 0:
            if self.cpm_enc.encode_feature == 'spconv':
                encode_feature = data_dict['spatial_features']
            elif self.cpm_enc.encode_feature == 'ssfa':
                encode_feature = logits_feature
            else:
                raise NotImplementedError
            # elif self.cpm_enc.encode_feature == 'keypoints':
            #     batch_dict['det_boxes'] = []
            #     batch_dict['det_scores'] = []
            #     for dets in self.post_processing_ego(data_dict,
            #                                     self.test_cfg, det_all=True):
            #
            #         batch_dict['det_boxes'].append(dets['box_lidar'])
            #         batch_dict['det_scores'].append(dets['scores'])
            #
            #     batch_dict = self.vsa(batch_dict)

            batch_dict["cpm_features"] = self.cpm_enc(encode_feature)
            batch_dict, num_total_dets = self.get_preds(batch_dict)
            if num_total_dets>0 and self.has_noise:
                batch_dict = self.vsa(batch_dict)
                batch_dict = self.semseg(batch_dict)
            batch_dict = self.matcher(batch_dict)

            batch_dict = self.fusion_detect(batch_dict)

        return batch_dict

    def shift_and_filter_preds(self, batch_dict):
        """
        Transform coop pred. boxes into ego CS and remove boxes that are outside the detection range of ego vehicle
        """
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
        # batch_preds = batch_dict['batch_preds']
        # preds_ego = {
        #     'box_preds': torch.stack([pred['box_preds'][0] for pred in batch_preds], dim=0),
        #     'cls_preds': torch.stack([pred['cls_preds'][0] for pred in batch_preds], dim=0),
        #     'dir_cls_preds': torch.stack([pred['dir_cls_preds'][0] for pred in batch_preds], dim=0),
        #     'iou_preds': torch.stack([pred['iou_preds'][0] for pred in batch_preds], dim=0),
        #     'var_box_preds': torch.stack([pred['var_box_preds'][0] for pred in batch_preds], dim=0),
        # }
        batch_dict['preds_dict'] = batch_dict['preds_egos']
        loss_ego = self.head.loss(batch_dict)
        # supervise fusion detetion only for ego vehicle
        if 'preds_final' in batch_dict.keys():
            target_fused = batch_dict["target_fused"]
            batch_dict['preds_dict'] = batch_dict['preds_final']
            batch_dict['reg_targets'] = target_fused['reg_targets']
            batch_dict['reg_weights'] = target_fused['reg_weights']
            batch_dict['anchors'] = batch_dict['anchors'][0:1]
            batch_dict['labels'] = target_fused['labels']
            loss = self.fusion_detect.det_head.loss(batch_dict)

            loss.update({
                'loss_ego': loss_ego['loss'],
                'loss_fuse': loss['loss'],
                'loss': loss_ego['loss'] + loss['loss']
            })
            if self.has_noise:
                loss_kpts, accs = self.semseg.loss(batch_dict)
                loss.update(accs)
                loss.update({
                    'loss_kpts': loss_kpts,
                    'loss': loss_ego['loss'] + loss['loss'] + loss_kpts
                })
        else:
            loss = loss_ego
            loss.update({
                'loss_ego': loss_ego['loss'],
                'loss_fuse': 0,
                'loss': loss_ego['loss']
            })
        if hasattr(self.mcfg, 'RDFT') and self.mcfg.RDFT['use']:
            loss_rdf = self.deep_coral_loss(batch_dict['fused_features'], batch_dict['rdf_features'],
                                              batch_dict['target_fused']['reg_weights'] > 0)
            loss.update({
                'loss_rdf': loss_rdf,
                'loss': loss['loss'] + loss_rdf
            })

        # ######plot#####
        # import matplotlib.pyplot as plt
        # from vlib.point import draw_box_plt
        # predictions_dicts = self.head.post_processing(batch_dict, self.test_cfg)
        # pred_boxes = [pred_dict['box_lidar'] for pred_dict in predictions_dicts]
        # ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
        # ax.set_aspect('equal', 'box')
        # ax.set(xlim=(self.pc_range[0], self.pc_range[3]),
        #        ylim=(self.pc_range[1], self.pc_range[4]))
        # points = batch_dict['points'][0][0]
        # ax.plot(points[:, 0], points[:, 1], 'y.', markersize=0.3)
        # ax = draw_box_plt(batch_dict['gt_boxes'][0], ax, color='green', linewidth_scale=2)
        # ax = draw_box_plt(pred_boxes[0], ax, color='red')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.show()
        # plt.close()
        # ##############

        return loss

    def post_processing(self, batch_dict, test_cfg):
        if 'preds_final' in batch_dict.keys():
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
        translations = batch_dict['translations']

        coop_boxes_corrected = []
        if self.has_noise:
            coop_boxes = torch.cat(batch_dict["coop_boxes"][1:], dim=0)
            for cb, T, tf_local, tf in zip(coop_boxes, batch_dict['err_T_est'], batch_dict['err_tf_est_local'], translations[1:]):
                cur_box = cb.view(1, 7)
                if T is not None:
                    tmp = torch.cat([cur_box[:, :2], torch.ones((1, 1), device=cur_box.device)], dim=1)
                    cur_box[:, :2] = (T @ tmp.T)[:2, :].T
                    cur_box[:, -1] = cur_box[:, -1] + torch.atan2(T[1, 0], T[0, 1]) #tf_local[2]
                else:
                    cur_box[:, :2] = cur_box[:, :2] + tf[:2].reshape(1,2) - translations[0:1, :2]
                coop_boxes_corrected.append(cur_box)
            if len(coop_boxes_corrected) == 0:
                coop_boxes = None
            else:
                coop_boxes = torch.cat(coop_boxes_corrected, dim=0)
        else:
            coop_boxes = torch.cat(batch_dict["coop_boxes"], dim=0)
            coop_boxes[:, :2] = coop_boxes[:, :2] + translations[:, :2] - translations[:1, :2]

        coop_boxes = coop_boxes.view(batch_size, -1, self.head.box_coder.code_size)
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
                                              batch_anchors=anchors)[0]
        detections.update({
            'boxes_fused': batch_dict['boxes_fused'],
            'scores_fused': batch_dict['scores_fused'],
        })
        return detections

    def post_processing_ego(self, batch_dict, test_cfg, det_all=False):
        preds_dict = batch_dict['preds_egos']
        anchors = batch_dict["anchors"]
        coop_boxes = None
        batch_size = 1
        if det_all:
            batch_size = batch_dict['batch_size']
        else:
            anchors = anchors[0:1]
            translations = batch_dict['translations']
            coop_boxes = torch.cat(batch_dict["coop_boxes"][1:], dim=0)
            coop_boxes[:, :2] = coop_boxes[:, :2] + translations[1:, :2] - translations[:1, :2]
            coop_boxes = coop_boxes.view(batch_size, -1, self.head.box_coder.code_size)
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


class CPMEnc(nn.Module):
    """Collective Perception Message encoding module"""
    def __init__(self, in_channel, out_channel, n_layers=2, upsample=0, **kwargs):
        super(CPMEnc, self).__init__()
        if 'encode_feature' in kwargs:
            self.encode_feature = kwargs['encode_feature']
        cur_channels_in = in_channel
        cur_channels_out = out_channel

        assert n_layers>upsample
        block = []

        for i in range(n_layers - 1):
            cur_channels_out = cur_channels_in // 2 if cur_channels_in > 2*out_channel else out_channel
            conv_fn = _build_deconv_block if i<upsample else _build_conv_block
            block.extend(conv_fn(cur_channels_in, cur_channels_out))
            cur_channels_in = cur_channels_out
        self.encoder = Sequential(*block)

        self.conv_out = Sequential(*_build_conv_block(cur_channels_out, out_channel))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        x = self.encoder(x)
        out = self.conv_out(x)

        return out


class FUDET(nn.Module):
    def __init__(self, mcfg, dcfg):
        super(FUDET, self).__init__()
        self.fusion_res = mcfg.FUDET['fusion_resolution']
        self.fusion_score = mcfg.FUDET['fusion_score']
        self.pc_range = dcfg.pc_range
        self.has_noise = dcfg.add_gps_noise
        feature_dim = mcfg.CPMEnc['out_channel']
        det_dim = mcfg.CPMEnc['in_channel']

        cur_in = feature_dim

        convs = []
        for c in mcfg.FUDET['conv_head_channels']:
            convs.extend(_build_conv_block(cur_in, c))
            cur_in = c
        self.convs_head = Sequential(*convs)
        self.det_head = MultiGroupHead(mcfg.HEAD, dcfg.pc_range)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, batch_dict):
        feature_fused = self.fusion(batch_dict)
        out = self.convs_head(feature_fused)
        batch_dict['fused_features'] = out
        out = self.det_head(out)
        batch_dict['preds_final'] = out
        return batch_dict

    def fusion(self, data_dict):
        cpm_features = data_dict['cpm_features']
        # feat_mask = self.get_feat_mask(data_dict['detections'], cpm_features.permute(1, 0, 2, 3).shape[1:])
        # data_dict['feature_mask'] = feat_mask
        # masked_cpm_features = cpm_features
        # masked_cpm_features = masked_cpm_features * feat_mask.unsqueeze(1) # ego vehicle has full information
        # features_coop = masked_cpm_features[1:]
        # ego_features = masked_cpm_features[0]
        features_coop = cpm_features[1:]
        ego_features = cpm_features[0]
        # masked_cpm_features = masked_cpm_features * feat_mask.unsqueeze(1)
        translations = data_dict['translations']
        err_Ts_src = data_dict.get('err_T_est', None)
        # err_Ts_src = [torch.tensor(T, device='cuda:0') for T in data_dict['errs_T'][1:]]
        if err_Ts_src is not None:
            coop_node_mask = np.array([T is not None for T in err_Ts_src])
            if coop_node_mask.sum() > 0:
                err_Ts = torch.stack([T for T in err_Ts_src if T is not None], dim=0)
                # err_Ts[:, :2, 2] = err_Ts[:, :2, 2] / self.fusion_res
                # shifts_coop_to_ego = (translations[1:, :2][coop_node_mask] - translations[0:1, :2] + err_Ts[:, :2]) / self.fusion_res #  - err_Ts[:, :2]
                shifts_coop_to_ego = (err_Ts[:, :2, 2]) / self.fusion_res
                angles = torch.atan2(err_Ts[:, 1, 0], err_Ts[:, 0, 0]).reshape(-1, 1)
                # angles = err_Ts[:, 2].reshape(-1, 1)
                coop_features_transformed = self._affine_transform2(features_coop[coop_node_mask], -angles, -shifts_coop_to_ego)
                coop_features_fused = coop_features_transformed.sum(dim=0)
                features_add = cpm_features[0] + coop_features_fused
                # features_cat = torch.cat([features_ego, coop_features_fused], dim=0)

                # ii = coop_features_transformed[:2].sum(dim=1)  # + features_ego.sum(dim=1)
                # img = np.zeros((ii.shape[1], ii.shape[2], 3))
                # img[:, :, 0] = cpm_features[0].sum(dim=0).squeeze().detach().cpu().numpy()
                # img[:, :, 1:len(ii) + 1] = ii.detach().permute(1, 2, 0).cpu().numpy()
                # plt.imshow(img / img.max() * 2)
                # plt.show()
                return features_add.unsqueeze(0)
            else:
                return cpm_features[0:1]
        else:
            shifts_coop_to_ego = (translations[1:, :2] - translations[0:1, :2]) / self.fusion_res
            angles = torch.zeros_like(shifts_coop_to_ego[:, 0:1])
            coop_features_transformed = self._affine_transform2(features_coop, -angles, -shifts_coop_to_ego)
            coop_features_fused = coop_features_transformed.sum(dim=0)
            features_add = ego_features + coop_features_fused
            # # Fuse clouds
            # clouds = data_dict['points']
            # tfs = data_dict['translations']
            # clouds_fused = [clouds[clouds[:, 0] == i, 1:] for i in range(len(tfs))]
            # for i in range(1, len(tfs)):
            #     clouds_fused[i][:, :2] = clouds_fused[i][:, :2] + tfs[i, :2] - tfs[0, :2]
            # data_dict['points_fused'] = clouds_fused

        # ii = coop_features_transformed[:2].sum(dim=1)  # + features_ego.sum(dim=1)
        # img = np.zeros((ii.shape[1], ii.shape[2], 3))
        # img[:, :, 0] = ego_features.sum(dim=0).squeeze().detach().cpu().numpy()
        # img[:, :, 1:len(ii) + 1] = ii.detach().permute(1, 2, 0).cpu().numpy()
        # plt.imshow(img / img.max() * 2)
        # plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
        # # plt.imshow(shifted_cpm_features_coop[0].sum(dim=0).detach().cpu().numpy())
        # plt.close()
        return features_add.unsqueeze(0)

    def get_feat_mask(self, detections, out_shape):
        max_len = max([len(dets['box_lidar']) for dets in detections])
        batch_dets = detections[0]['box_lidar'].new_zeros((len(detections), max_len, 7))
        for i, dets in enumerate(detections):
            boxes = dets['box_lidar']
            batch_dets[i, :len(boxes)] = boxes
        batch_dets[:, :, 2] = 0
        x = torch.arange(self.pc_range[0] + 0.4, self.pc_range[3], 0.8).to(batch_dets.device)
        yy, xx = torch.meshgrid(x, x)
        points = batch_dets.new_zeros((len(xx.reshape(-1)), 3))
        points[:, 0] = xx.reshape(-1)
        points[:, 1] = yy.reshape(-1)
        batch_points = torch.stack([points] * len(detections), dim=0)
        box_idxs_of_pts = points_in_boxes_gpu(batch_points, batch_dets).view(out_shape) >= 0

        return box_idxs_of_pts

    def _shift2d(self, matrices, shifts):
        matrices_out = []
        for mat, shift in zip(matrices, shifts):
            dx, dy = int(shift[1]), int(shift[0])
            shifted_mat = torch.roll(mat, dx, 1)
            if dx < 0:
                shifted_mat[:, dx:, :] = 0
            elif dx > 0:
                shifted_mat[:, 0:dx, :] = 0
            shifted_mat = torch.roll(shifted_mat, dy, 2)
            if dy < 0:
                shifted_mat[:, :, dy:] = 0
            elif dy > 0:
                shifted_mat[:, :, 0:dy] = 0
            matrices_out.append(shifted_mat)
        return torch.stack(matrices_out, dim=0)

    def _affine_transform2(self, matrices, angles, shifts):
        ss = matrices.shape
        sina = torch.sin(angles)
        cosa = torch.cos(angles)
        x = shifts[:, 0:1]
        y = shifts[:, 1:]
        ones = torch.ones_like(sina)
        zeros = torch.zeros_like(sina)
        theta = torch.stack([torch.cat([cosa, -sina, zeros], dim=1),
                           torch.cat([sina, cosa, zeros], dim=1)], dim=1)
        grid = F.affine_grid(theta, ss, align_corners=False).float()
        res = F.grid_sample(matrices, grid, align_corners=False)
        theta = torch.stack([torch.cat([ones, zeros, x / ss[2] * 2], dim=1),
                           torch.cat([zeros, ones, y / ss[3] * 2], dim=1)], dim=1)
        grid = F.affine_grid(theta, ss, align_corners=False).float()
        res = F.grid_sample(res, grid, align_corners=False)
        return res

    def _affine_transform(self, matrices, T):
        ss = matrices.shape
        # T[:, :2, :2] = T[:, :2, :2].permute(0, 2, 1)
        T[:, 0, 2] = - T[:, 0, 2] / ss[2] * 2
        T[:, 1, 2] = - T[:, 1, 2] / ss[3] * 2
        grid = F.affine_grid(T, ss, align_corners=False).float()
        return F.grid_sample(matrices, grid, align_corners=False)




# sx_max = shifts_coop_to_ego.new_full((shifts_coop_to_ego.shape[0],), weights.shape[2])
# sy_max = shifts_coop_to_ego.new_full((shifts_coop_to_ego.shape[0],), weights.shape[3])
# x_max = torch.min(sx_max, weights.shape[2] + shifts_coop_to_ego[:, 0])
# y_max = torch.min(sy_max, weights.shape[3] + shifts_coop_to_ego[:, 1])
# sx_min = shifts_coop_to_ego.new_full((shifts_coop_to_ego.shape[0],), 0)
# sy_min = shifts_coop_to_ego.new_full((shifts_coop_to_ego.shape[0],), 0)
# x_min = torch.max(sx_min, shifts_coop_to_ego[:, 0])
# y_min = torch.max(sy_min, shifts_coop_to_ego[:, 1])
# xs = xs[inds > 0] + shifts_coop_to_ego[inds[inds > 0], 0]
# ys = ys[inds > 0] + shifts_coop_to_ego[inds[inds > 0], 1]

