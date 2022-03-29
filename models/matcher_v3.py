
import torch
from torch import nn
import numpy as np
import cv2
import itertools
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, giou3d
from sklearn.neighbors import NearestNeighbors
from vlib.point import draw_points_boxes_plt, draw_box_plt
import matplotlib.pyplot as plt
from utils.max_consensus import max_consunsus_hierarchical

pi = 3.141592653


def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period


class MatcherV3(nn.Module):
    def __init__(self, cfg, pc_range, has_noise=True, search_range=None):
        super(MatcherV3, self).__init__()
        self.pc_range = pc_range
        self.has_noise = has_noise
        self.max_cons_cfg = cfg['max_cons']
        self.max_cons_cfg['search_range'] = search_range

    @torch.no_grad()
    def forward(self, data_dict):
        tfs = data_dict['translations']
        keys = ['ids', 'translations', 'gt_boxes', 'coop_boxes', 'det_boxes', 'det_scores', 'point_coords', 'kpts_preds',
                'cpm_pts_coords', 'cpm_pts_cls']

        if self.has_noise:
            data_dict = self.err_correction(data_dict)
        else:
            boxes_ego_cs = []
            for i, boxes in enumerate(data_dict['det_boxes']):
                boxes[:, :2] = boxes[:, :2] + tfs[i, :2] - tfs[0, :2]
                boxes_ego_cs.append(boxes)
                if 'point_coords' in data_dict:
                    data_dict['point_coords'][i][:, :2] = data_dict['point_coords'][i][:, :2] + tfs[i, :2] - tfs[0, :2]

            data_dict['det_boxes_ego_coords'] = boxes_ego_cs
            if 'point_coords' in data_dict:
                data_dict['cpm_pts_features'] = data_dict['point_features']
                data_dict['cpm_pts_coords'] = data_dict['point_coords']

        clusters, scores = self.clustering(data_dict)
        # Ts_gt = data_dict['errs_T']
        data_dict['boxes_fused'], data_dict['scores_fused'] = self.cluster_fusion(clusters, scores)

        clouds = data_dict['points']
        clouds_fused = [clouds[clouds[:, 0]==i, 1:] for i in range(len(tfs))]
        Ts = data_dict['err_T_est'] if self.has_noise else []
        for i in range(1, len(tfs)):
            # clouds_fused[i][:, :2] = clouds_fused[i][:, :2] - tfs[0][:2] + tfs[i][:2]
            if self.has_noise and Ts[i-1] is not None:
                # correct the coop point clouds
                tmp = torch.cat([clouds_fused[i][:, :2], torch.ones((len(clouds_fused[i][:, :2]), 1),
                                 device=clouds_fused[i][:, :2].device)], dim=1)
                clouds_fused[i][:, :2] = (Ts[i-1] @ tmp.T)[:2, :].T
            else:
                clouds_fused[i][:, :2] = clouds_fused[i][:, :2] + tfs[i, :2] - tfs[0, :2]
            # t = T[2, :2]
            # theta = torch.atan2(T[1, 0], T[0, 0])
            # clouds_fused[i+1][:, :2] = clouds_fused[i+1][:, :2] + t - tfs[0][:2] + tfs[i+1][:2]
            # clouds_fused[i+1][:, :3] = self.rotate_points_along_z(clouds_fused[i+1], theta)
        data_dict['points_fused'] = clouds_fused
        # ##############################################################
        # import matplotlib.pyplot as plt
        # from vlib.point import draw_points_boxes_plt, draw_box_plt
        # det_boxes = data_dict['det_boxes_ego_coords']
        # fig = plt.figure(figsize=(10, 10))
        # gs = fig.add_gridspec(1, 1)
        # ax = fig.add_subplot(gs[0, 0])
        # ax.set_aspect('equal', 'box')
        # colors = ['g', 'b', 'r', 'c', 'y']
        # ax = draw_box_plt(det_boxes[0].cpu().numpy(), ax, 'r')
        # for i, (cld, boxes) in enumerate(zip(clouds_fused, det_boxes)):
            # ax = draw_points_boxes_plt(self.pc_range, cld.cpu().numpy(), points_c=colors[i] + '.',
            #                            boxes_pred=det_boxes[i], bbox_pred_c=colors[i],
            #                            ax=ax, return_ax=True)

            # ax = draw_box_plt(det_boxes[i].cpu().numpy(), ax, colors[i])
        # ax = draw_box_plt(data_dict['boxes_fused'].cpu().numpy(), ax, 'r')
        # ax = draw_box_plt(data_dict['gt_boxes_fused'].cpu().numpy(), ax, 'g')
        # for i, c in enumerate(clusters):
        #     # ax = plt.subplot(111)
        #     # ax.set_aspect('equal', 'box')
        #     ax = draw_box_plt(c.cpu().numpy(), ax, 'b')
        #     ax = draw_box_plt(data_dict['boxes_fused'][i].view(1, 7).cpu().numpy(), ax, 'r')
        #     plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
        # plt.close()
        # ##############################################################
        return data_dict

    def clustering(self, data_dict):
        """
        Assign predicted boxes to clusters according to their ious with each other
        """
        pred_scores = data_dict['det_scores']
        pred_boxes_cat = torch.cat(data_dict['det_boxes_ego_coords'], dim=0)
        # if torch.isnan(pred_boxes_cat).any():
        #     print('debug')
        assert not torch.isnan(pred_boxes_cat).any(), 'pred boxes contain nan.'
        pred_boxes_cat[:, -1] = limit_period(pred_boxes_cat[:, -1])
        pred_scores_cat = torch.cat(pred_scores, dim=0)
        ious = boxes_iou3d_gpu(pred_boxes_cat, pred_boxes_cat)
        cluster_indices = torch.zeros(len(ious)).int() # gt assignments of preds
        cur_cluster_id = 1
        while torch.any(cluster_indices == 0):
            cur_idx = torch.where(cluster_indices == 0)[0][0] # find the idx of the first pred which is not assigned yet
            cluster_indices[torch.where(ious[cur_idx] > 0.1)[0]] = cur_cluster_id
            cur_cluster_id += 1
        clusters = []
        scores = []
        for i in range(1, cluster_indices.max().item() + 1):
            clusters.append(pred_boxes_cat[cluster_indices==i])
            scores.append(pred_scores_cat[cluster_indices==i])
        if len(scores)==0:
            print('debug')

        return clusters, scores

    def cluster_fusion(self, clusters, scores):
        """
        Merge boxes in each cluster with scores as weights for merging
        """
        boxes_fused = []
        scores_fused = []
        for c, s in zip(clusters, scores):
            # reverse direction for non-dominant direction of boxes
            dirs = c[:, -1]
            max_score_idx = torch.argmax(s)
            dirs_diff = torch.abs(dirs - dirs[max_score_idx].item())
            lt_pi = (dirs_diff > pi).int()
            dirs_diff = dirs_diff * (1 - lt_pi) + (2 * pi - dirs_diff) * lt_pi
            score_lt_half_pi = s[dirs_diff > pi / 2].sum() # larger than
            score_set_half_pi = s[dirs_diff <= pi / 2].sum() # small equal than
            # select larger scored direction as final direction
            if score_lt_half_pi <= score_set_half_pi:
                dirs[dirs_diff > pi / 2] += pi
            else:
                dirs[dirs_diff <= pi / 2] += pi
            dirs = limit_period(dirs)
            s_normalized = s / s.sum()
            sint = torch.sin(dirs) * s_normalized
            cost = torch.cos(dirs) * s_normalized
            theta = torch.atan2(sint.sum(), cost.sum()).view(1,)
            center_dim = c[:, :-1] * s_normalized[:, None]
            boxes_fused.append(torch.cat([center_dim.sum(dim=0), theta]))
            s_sorted = torch.sort(s, descending=True).values
            s_fused = 0
            for i, ss in enumerate(s_sorted):
                s_fused += ss**(i+1)
            s_fused = torch.tensor([min(s_fused, 1.0)], device=s.device)
            scores_fused.append(s_fused)
        if len(boxes_fused) > 0:
            boxes_fused = torch.stack(boxes_fused, dim=0)
            scores_fused = torch.stack(scores_fused, dim=0)
        else:
            boxes_fused = None
            scores_fused = None
            print('debug')
        return boxes_fused, scores_fused

    def err_correction(self, data_dict):
        pred_boxes = data_dict['det_boxes']
        # kpts_coords = data_dict.get('point_coords', None)
        tfs = data_dict['translations']
        pts_feat = data_dict['cpm_pts_features']
        pts_coords = data_dict['cpm_pts_coords']
        pts_cls = data_dict['cpm_pts_cls']

        T_list = []
        tf_local_list = []
        corrected_boxes_list = [pred_boxes[0]]
        corrected_points_list = [pts_coords[0]]
        for boxes, points, tf, lbls in zip(pred_boxes[1:], pts_coords[1:], tfs[1:], pts_cls[1:]):
            T, tf_local = self.matching(pred_boxes[0], boxes, pts_coords[0],
                              points, tfs[0:1], tf.reshape(1, -1), pts_cls[0], lbls)
            if T is not None:
                # correct coords of points2
                tmp = torch.cat([points[:, :2], torch.ones((len(points), 1), device=points.device)], dim=1)
                points[:, :2] = (T @ tmp.T)[:2, :].T
                # correct coords of boxes2
                tmp = torch.cat([boxes[:, :2], torch.ones((len(boxes), 1), device=boxes.device)], dim=1)
                # cur_boxes[:, :2] = (tmp[:, :3] @ T)[: , :2]
                boxes[:, :2] = (T @ tmp.T)[:2, :].T
                boxes[:, -1] += torch.atan2(T[1, 0], T[0, 0])#tf_local[2]
            else:
                points[:, :2] = points[:, :2] + tf[:2] + tfs[0, :2]
                boxes[:, :2] = boxes[:, :2] + tf[:2] + tfs[0, :2]
            # fig = plt.figure(figsize=(8, 8))
            # ax = fig.add_subplot(111)
            # ax.set_aspect('equal')
            # ax.plot(points.cpu().numpy()[:, 0], points.cpu().numpy()[:, 1], '.', markersize=3)
            # ax.plot(pts_coords[0].cpu().numpy()[:, 0], pts_coords[0].cpu().numpy()[:, 1], '.', markersize=3)
            # plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
            # plt.close()
            T_list.append(T)
            tf_local_list.append(tf_local)
            corrected_boxes_list.append(boxes)
            corrected_points_list.append(points)
        # if cur coop has no overlap with coop vehicle, then try to match it with other coops
        # no_match = np.where([T is None for T in T_list])[0]
        # if len(no_match) > 0:
        #     coop_set = set(np.arange(1, len(T_list) + 1)) - set(no_match + 1)
        #     for i in no_match + 1:
        #         for j in coop_set:
        #             T, tf_local = self.matching(pred_boxes[j], pred_boxes[i], pts_coords[j],
        #                               pts_coords[i], tfs[j].reshape(1, -1), tfs[i].reshape(1, -1),
        #                                pts_cls[j], pts_cls[i])
        #             if T is not None:
        #                 points = pts_coords[i]
        #                 T = torch.tensor(T, device=points.device, dtype=points.dtype)
        #                 T = torch.matmul(T_list[j - 1], T)
        #                 T_list[i - 1] = T
        #                 tf_local = tf_local_list[j - 1] + tf_local
        #                 tf_local_list[i - 1] = tf_local
        #                 # correct coords of points2
        #                 # plt.plot(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), 'r.')
        #                 # plt.plot(pts_coords[j][:, 0].cpu().numpy(), pts_coords[j][:, 1].cpu().numpy(), 'g.')
        #                 # plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
        #                 # plt.close()
        #                 tmp = torch.cat([points[:, :2], torch.ones((len(points), 1), device=points.device)], dim=1)
        #                 points[:, :2] = (T @ tmp.T)[:2, :].T
        #                 corrected_boxes_list[i] = points
        #                 # correct coords of boxes2
        #                 boxes = pred_boxes[i]
        #                 tmp = torch.cat([boxes[:, :2], torch.ones((len(boxes), 1), device=boxes.device)], dim=1)
        #                 # cur_boxes[:, :2] = (tmp[:, :3] @ T)[: , :2]
        #                 boxes[:, :2] = (T @ tmp.T)[:2, :].T
        #                 boxes[:, -1] += tf_local[2]
        #                 corrected_boxes_list[i] = boxes

                        # plt.plot(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), 'r.')
                        # plt.plot(pts_coords[j][:, 0].cpu().numpy(), pts_coords[j][:, 1].cpu().numpy(), 'g.')
                        # plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
                        # plt.close()


        data_dict['det_boxes_ego_coords'] = corrected_boxes_list
        data_dict['cpm_pts_coords'] = corrected_points_list
        data_dict['err_T_est'] = T_list
        data_dict['err_tf_est_local'] = tf_local_list
        return data_dict

    def matching(self, boxes1, boxes2, points1, points2, loc1, loc2, lbls1, lbls2):
        """
        register boxes2 to boxes 1
        """
        ego_bbox_mask = torch.norm(boxes1[:, :2] + loc1[:, :2] - loc2[:, :2], dim=1) < 57.6
        coop_bbox_mask = torch.norm(boxes2[:, :2] + loc2[:, :2] - loc1[:, :2], dim=1) < 57.6
        ego_boxes_masked = boxes1[ego_bbox_mask]
        coop_boxes_masked = boxes2[coop_bbox_mask]
        # cost_bbox_loc = torch.cdist(ego_boxes_masked[:, :3], coop_boxes_masked[:, :3], p=1.)
        # cost_bbox_size = torch.cdist(ego_boxes_masked[:, 3:-1], coop_boxes_masked[:, 3:-1], p=1.)
        # cost_bbox_dir = torch.cdist(torch.cos(ego_boxes_masked[:, -1:]).__abs__(),
        #                             torch.cos(coop_boxes_masked[:, -1:]).__abs__(), p=1.)
        #
        # cost_giou = - giou3d(ego_boxes_masked, coop_boxes_masked)

        # C_boxes = 0.1 * cost_bbox_loc + 0.5 * cost_bbox_size + 20 * cost_bbox_dir + cost_giou

        cls_mask1 = torch.logical_and(lbls1>0, lbls1<4)
        cls_mask2 = torch.logical_and(lbls2>0, lbls2<4)
        points1_s = points1[cls_mask1]
        points2_s = points2[cls_mask2]
        mask1 = torch.norm(points1_s[:, :2] + loc1[:, :2] - loc2[:, :2], dim=1) < 57.6
        mask2 = torch.norm(points2_s[:, :2] + loc2[:, :2] - loc1[:, :2], dim=1) < 57.6
        dst = torch.cat([ego_boxes_masked[:, :2], points1_s[mask1, :2]], dim=0).cpu().numpy()
        src = torch.cat([coop_boxes_masked[:, :2], points2_s[mask2, :2]], dim=0).cpu().numpy()
        labels1 = np.concatenate([np.ones(len(ego_boxes_masked), dtype=np.int64) * 4, lbls1[cls_mask1][mask1].cpu().numpy()], axis=0)
        labels2 = np.concatenate([np.ones(len(coop_boxes_masked), dtype=np.int64) * 4, lbls2[cls_mask2][mask2].cpu().numpy()], axis=0)
        T, tf_local, _ = max_consunsus_hierarchical(dst, src, loc1.cpu().numpy(), loc2.cpu().numpy(),
                                                    point_labels=(labels1, labels2),
                                                    label_weights=[0, 1, 1, 2, 2], **self.max_cons_cfg)
        if T is None:
            return None, None
        T = torch.tensor(T, device=boxes1.device, dtype=boxes1.dtype)
        tf_local = torch.tensor(tf_local, device=boxes1.device, dtype=boxes1.dtype)
        # T = self.tfs_to_Tmat([torch.tensor(tf, device=boxes1.device, dtype=boxes1.dtype)])
        # T = self.icp(dst, src, num_cnt=[len(ego_boxes_masked), len(coop_boxes_masked)])

        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111)
        # ax.set_aspect('equal')
        # ax.plot(points1.cpu().numpy()[:, 0], points1.cpu().numpy()[:, 1], '.', markersize=1)
        # ax.plot(points2.cpu().numpy()[:, 0], points2.cpu().numpy()[:, 1], '.', markersize=1)
        # plt.savefig('/media/hdd/ophelia/tmp/tmp2.png')
        # plt.close()
        return T, tf_local

    def estimate_tf_2d(self, pointsl, pointsr):
        is_numpy = False
        if not isinstance(pointsl, torch.Tensor):
            pointsl = torch.tensor(pointsl)
            pointsr = torch.tensor(pointsr)
            is_numpy = True
        # 1 reduce by the center of mass
        l_mean = pointsl.mean(dim=0)
        r_mean = pointsr.mean(dim=0)
        l_reduced = pointsl - l_mean
        r_reduced = pointsr - r_mean
        # 2 compute the rotation
        Sxx = (l_reduced[:, 0] * r_reduced[:, 0]).sum()
        Syy = (l_reduced[:, 1] * r_reduced[:, 1]).sum()
        Sxy = (l_reduced[:, 0] * r_reduced[:, 1]).sum()
        Syx = (l_reduced[:, 1] * r_reduced[:, 0]).sum()
        theta = torch.atan2(Sxy - Syx, Sxx + Syy)  # / np.pi * 180
        t = r_mean.reshape(2, 1) - torch.tensor([[torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]], device=theta.device) @ l_mean.reshape(2, 1)
        if is_numpy:
            theta = theta.cpu().numpy()
            t = t.cpu().numpy()
        return theta, t.T

    def rotate_points_along_z(self, points, angle):
        """
        Args:
            points: (N, 2 + C)
            angle: float, angle along z-axis, angle increases x ==> y
        Returns:

        """
        out_dim = points.shape[1]
        if out_dim==2:
            points = torch.cat([points, torch.zeros((points.shape[0], 1), device=points.device)], dim=-1)
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        rot_matrix = torch.tensor([
            [cosa, -sina, 0],
            [sina, cosa, 0],
            [0, 0, 1]
        ]).float().to(angle.device)
        points_rot = torch.matmul(points[:, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, 3:]), dim=-1)
        points_out = points_rot[:, :2] if out_dim==2 else points_rot
        return points_out

    def tfs_to_Tmat(self, Ts):
        """
        Convert a list of transformations to one transformation matrix Tmat
        Ts: list of [dx, dy, theta]
        """
        Tmat = torch.eye(3).to(Ts[0].device)
        for T in Ts:
            cosa = torch.cos(T[2])
            sina = torch.sin(T[2])
            rot_matrix = torch.tensor([
                [cosa, -sina, T[0]],
                [sina, cosa, T[1]],
                [0, 0, 1]
            ]).float().to(T.device)
            Tmat = rot_matrix @ Tmat
        return Tmat

    def icp(self, a, b, num_cnt, init_pose=(0, 0, 0), no_iterations=15):
        '''
        The Iterative Closest Point estimator.
        Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
        their relative pose and the number of iterations
        Returns the affine transform that transforms
        the cloudpoint a to the cloudpoint b.
        Note:
            (1) This method works for cloudpoints with minor
            transformations. Thus, the result depents greatly on
            the initial pose estimation.
            (2) A large number of iterations does not necessarily
            ensure convergence. Contrarily, most of the time it
            produces worse results.
        '''

        dst = np.array([a], copy=True).astype(np.float32)
        src = np.array([b], copy=True).astype(np.float32)

        # Initialise with the initial pose estimation
        Tr = np.array([[np.cos(init_pose[2]), -np.sin(init_pose[2]), init_pose[0]],
                       [np.sin(init_pose[2]), np.cos(init_pose[2]), init_pose[1]],
                       [0, 0, 1]])

        src = cv2.transform(src, Tr[0:2])

        for i in range(no_iterations):
            # Find the nearest neighbours between the current source and the
            # destination cloudpoint
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
            distances, indices = nbrs.kneighbors(src[0, :, :2])
            mask = distances.squeeze() < max(5 * (4*no_iterations / (4 * no_iterations + 1**2)), 1)
            indices = indices[mask]

            if mask[:num_cnt[1]].sum() < 2 and mask[num_cnt[1]:].sum() < 10:
                # No enough overlassping points
                return None

            # Compute the transformation between the current source
            # and destination cloudpoint
            theta, t = self.estimate_tf_2d(src[0, mask, :2], dst[0][indices.squeeze()])
            T = self.tfs_to_Tmat([torch.tensor([t[0,0], t[0, 1], theta])]).numpy()

            # Transform the previous source and update the
            # current source cloudpoint
            src = cv2.transform(src, T)
            # Save the transformation from the actual source cloudpoint
            # to the destination
            Tr = np.dot(Tr, T)
        return Tr

