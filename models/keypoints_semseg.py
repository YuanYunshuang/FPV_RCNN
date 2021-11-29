from torch import nn
from torch.nn import Sequential
import torch
from losses import build_loss
from models.utils import xavier_init
from ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils


class SemSeg(nn.Module):
    """Collective Perception Message encoding module"""
    def __init__(self, in_channels, num_cls, n_layers, valid_classes, **kwargs):
        super(SemSeg, self).__init__()
        self.valid_classes = valid_classes
        self.weights = kwargs['weight_code']
        fc_layers = [nn.Conv1d(in_channels, in_channels, 1), nn.ReLU()] * n_layers
        self.fc_layers = Sequential(*fc_layers)
        self.cls_layer = nn.Conv1d(in_channels, num_cls, 1)
        self.loss_fn = build_loss(kwargs['loss_cls'])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                xavier_init(m, distribution="uniform")

    def forward(self, batch_dict):
        point_features = batch_dict['point_features']
        point_coords = batch_dict['point_coords']
        x = torch.cat(point_features, dim=0).view(1, -1, point_features[0].shape[-1]).permute(0, 2, 1)
        x = self.fc_layers(x)
        out = self.cls_layer(x).permute(0, 2, 1)
        batch_dict['point_cls_logits'] = out
        preds = out.clone().detach().softmax(dim=-1).argmax(dim=-1).view(-1)
        batch_dict['kpts_preds'] = preds
        cur_idx = 0
        batch_dict['cpm_pts_features'] = []
        batch_dict['cpm_pts_coords'] = []
        batch_dict['cpm_pts_cls'] = []
        for i, points in enumerate(point_coords):
            cur_preds = preds[cur_idx:cur_idx + len(points)]
            mask = cur_preds>0
            cur_preds_masked = cur_preds[mask]
            cur_features = point_features[i][mask]
            cur_coords = point_coords[i][mask]

            wall_fence_mask = torch.logical_or(cur_preds_masked==1, cur_preds_masked==2)
            num_wall_fence_points = 50
            if wall_fence_mask.sum() > num_wall_fence_points:
                sampled_points = cur_coords[wall_fence_mask].unsqueeze(dim=0)  # (1, N, 3)
                # sample points with FPS
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), num_wall_fence_points
                ).long()

                wall_fence_points_down_sampled = sampled_points[0][cur_pt_idxs[0]]
                wall_fence_points_features = cur_features[wall_fence_mask][cur_pt_idxs[0]]
                tmp = torch.logical_not(wall_fence_mask)
                points_selected = torch.cat([cur_coords[tmp], wall_fence_points_down_sampled], dim=0)
                features_selected = torch.cat([cur_features[tmp], wall_fence_points_features], dim=0)
                cls_selected = torch.cat([cur_preds_masked[tmp], cur_preds_masked[wall_fence_mask][cur_pt_idxs[0]]])
            else:
                points_selected = cur_preds
                features_selected = cur_features
                cls_selected = cur_preds_masked

            batch_dict['cpm_pts_features'].append(features_selected)
            batch_dict['cpm_pts_coords'].append(points_selected)
            batch_dict['cpm_pts_cls'].append(cls_selected)
            cur_idx = cur_idx + len(points)
        return batch_dict

    def loss(self, batch_dict):
        tgt, weights = self.get_target(batch_dict)
        pred = batch_dict['point_cls_logits']
        loss = self.loss_fn(pred, tgt, weights=weights).sum(dim=-1).mean() * self.loss_fn._loss_weight
        preds = batch_dict['kpts_preds']
        tgt_lab = tgt.argmax(dim=-1)
        acc = (preds==tgt_lab).sum().cpu().numpy() / len(preds.squeeze())
        accs = {
        # 'acc_not_labeled': acc_not_labeled,
        # 'acc_fences':      acc_fences,
        # 'acc_poles':       acc_poles,
        # 'acc_vehicles':    acc_vehicles,
        # 'acc_walls':       acc_walls,
        'acc':             acc
        }
        return loss, accs

    def get_target(self, batch_dict):
        point_lbls = torch.cat(batch_dict['point_coords'], dim=0)[:, 3].view(-1, 1)
        tgt = torch.zeros((len(point_lbls), len(self.valid_classes) + 1), device=point_lbls.device)
        weights = torch.ones((len(point_lbls), 1), device=point_lbls.device) * self.weights[0]
        tgt[:, 0] = 1
        for i in range(1, len(self.valid_classes) + 1):
            mask = torch.zeros_like(point_lbls).bool()
            for l in self.valid_classes[i]:
                mask = torch.logical_or(point_lbls==l, mask)
            weights[mask] = self.weights[i]
            tgt[mask.view(-1), i] = 1
            tgt[mask.view(-1), 0] = 0

        tgt = tgt.view(1, -1, tgt.shape[-1])
        batch_dict['keypoints_tgt'] = tgt

        return tgt, weights
