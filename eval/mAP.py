import os
from ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
import torch
import matplotlib.pyplot as plt
import numpy as np



class MetricAP():
    def __init__(self, cfg, run_path, name='none', device='cuda', bev=False):
        super(MetricAP, self).__init__()
        self.cfg = cfg
        self.n_coop = cfg['n_coop'] if 'n_coop' in list(cfg.keys()) else 0
        self.samples = []
        self.pred_boxes = {}
        self.gt_boxes = {}
        self.confidences = {}
        self.v_ids = {}
        self.device = device
        self.bev = bev
        self.iou_fn = boxes_iou_bev if self.bev else boxes_iou3d_gpu
        self.run_path = run_path
        file_test = run_path / 'thr{}_ncoop{}_{}.pth'.format(self.cfg['score_threshold'], self.n_coop, name)
        self.has_test_detections = False
        if file_test.exists():
            data = torch.load(file_test)
            self.samples = data['samples']
            self.pred_boxes = data['pred_boxes']
            self.gt_boxes = data['gt_boxes']
            self.confidences = data['confidences']
            self.v_ids = data['ids']
            self.has_test_detections = True

    def add_sample(self, name, pred_boxes, gt_boxes, confidences, ids=None):
        if len(pred_boxes)>0:
            assert pred_boxes.device.type==self.device
        if len(gt_boxes) > 0:
            assert gt_boxes.device.type==self.device
        self.samples.append(name)
        self.pred_boxes[name] = pred_boxes
        self.gt_boxes[name] = gt_boxes
        self.confidences[name] = confidences
        if ids is not None:
            self.v_ids[name] = ids

    @torch.no_grad()
    def add_samples(self, names, preds, gts, confs, ids=None):
        for i in range(len(names)):
            self.add_sample(names[i], preds[i].float(), gts[i].float(), confs[i], ids[i])

    def save_detections(self, filename):
        dict_detections = {
            'samples': self.samples,
            'pred_boxes': self.pred_boxes,
            'gt_boxes': self.gt_boxes,
            'confidences': self.confidences,
            'ids': self.v_ids
        }
        torch.save(dict_detections, str(self.run_path / filename
                                        .format(self.cfg['score_threshold'], self.n_coop)))
        self.has_test_detections = True

    def cal_precision_recall(self, IoU_thr=0.5):
        list_sample = []
        list_confidence = []
        list_tp = []
        N_gt = 0

        for sample in self.samples:
            if len(self.pred_boxes[sample])>0 and len(self.gt_boxes[sample])>0:
                ious = self.iou_fn(self.pred_boxes[sample], self.gt_boxes[sample])
                n, m = ious.shape
                list_sample.extend([sample] * n)
                list_confidence.extend(self.confidences[sample])
                N_gt += len(self.gt_boxes[sample])
                max_iou_pred_to_gts = ious.max(dim=1)
                max_iou_gt_to_preds = ious.max(dim=0)
                tp = max_iou_pred_to_gts[0] > IoU_thr
                is_best_match = max_iou_gt_to_preds[1][max_iou_pred_to_gts[1]] \
                                ==torch.tensor([i for i in range(len(tp))], device=tp.device)
                tp[torch.logical_not(is_best_match)] = False
                list_tp.extend(tp)
            elif len(self.pred_boxes[sample])==0:
                N_gt += len(self.gt_boxes[sample])
            elif len(self.gt_boxes[sample])==0:
                tp = torch.zeros(len(self.pred_boxes[sample]), device=self.pred_boxes[sample].device)
                list_tp.extend(tp.bool())
        order_inds = torch.tensor(list_confidence).argsort(descending=True)
        tp_all = torch.tensor(list_tp)[order_inds]
        list_accTP = tp_all.cumsum(dim=0)
        # list_accFP = torch.logical_not(tp_all).cumsum(dim=0)
        list_precision = list_accTP.float() / torch.arange(1, len(list_sample) + 1)
        list_recall = list_accTP.float() / N_gt
        # plt.plot(list_recall.numpy(), list_precision.numpy(), 'k.')
        # plt.savefig(str(self.run_path / 'auc_thr{}_ncoop{}.png'
        #                 .format(self.cfg['score_threshold'], self.n_coop)))
        # plt.close()

        return list_precision, list_recall

    #
    def cal_ap_all_point(self, IoU_thr=0.5):
        '''
        source: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/7c0bd0489e3fd4ae71fc0bc8f2a67dbab5dbdc9c/lib/Evaluator.py#L292
        '''

        prec, rec = self.cal_precision_recall(IoU_thr=IoU_thr)
        mrec = []
        mrec.append(0)
        [mrec.append(e.item()) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e.item()) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    def cal_ap_11_point(self, IoU_thr=0.5):
        '''
        source: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/7c0bd0489e3fd4ae71fc0bc8f2a67dbab5dbdc9c/lib/Evaluator.py#L315
        '''
        # 11-point interpolated average precision
        prec, rec = self.cal_precision_recall(IoU_thr=IoU_thr)
        mrec = []
        # mrec.append(0)
        [mrec.append(e.item()) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e.item()) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]



