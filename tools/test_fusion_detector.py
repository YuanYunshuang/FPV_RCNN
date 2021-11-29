import copy

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from vlib.image import draw_box_plt
from utils.train_utils import *
from models.utils import load_model_dict
import random


def test_net(cfgs):
    # set random seeds
    seed_num = 8
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)

    dcfg, mcfg, cfg = cfgs
    n_coop = cfg.TEST['n_coop'] if 'n_coop' in list(cfg.TEST.keys()) else 0
    com_range = cfg.TEST['com_range'] if 'com_range' in list(cfg.TEST.keys()) else 0
    print("Building test dataloader...")
    test_dataloader = build_dataloader(dcfg, cfg, train=False)
    print("\bfinished.")
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print("Building model...")
    model = build_model(mcfg, cfg, dcfg).to(device)
    print("\bfinished.")

    log_path = Path(cfg.PATHS['run'])
    if os.path.exists(str(log_path / 'epoch{:03}.pth'.format(cfg.TRAIN['total_epochs']))):
        test_out_path = log_path / 'test_ep{:2d}'.format(cfg.TRAIN['total_epochs'])
        ckpt_path = str(log_path / 'epoch{:03}.pth'.format(cfg.TRAIN['total_epochs']))
    else:
        test_out_path = log_path / 'test_latest'.format(com_range)
        ckpt_path = str(log_path / 'latest.pth')
    test_out_path.mkdir(exist_ok=True)

    # get metric
    from eval.mAP import MetricAP
    metric_fused = MetricAP(cfg.TEST, test_out_path, name='fused', device='cuda', bev=cfg.TEST['bev'])
    metric_obj = MetricAP(cfg.TEST, test_out_path, name='obj', device='cuda', bev=cfg.TEST['bev'])
    thrs = cfg.TEST['ap_ious']

    if metric_fused.has_test_detections:
        aps = [metric_fused.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
        with open(test_out_path / 'thr{}_ncoop{}_fused.txt'.format(cfg.TEST['score_threshold'], n_coop), 'w') as fh:
            for thr, ap in zip(thrs, aps):
                fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))

        aps = [metric_obj.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
        with open(test_out_path / 'thr{}_ncoop{}_obj.txt'.format(cfg.TEST['score_threshold'], n_coop), 'w') as fh:
            for thr,  ap in zip(thrs, aps):
                fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))
        return

    # load checkpoint
    pretrained_dict = torch.load(ckpt_path)
    model = load_model_dict(model, pretrained_dict)

    # dir for save test images
    if cfg.TEST['save_img']:
        images_path = (test_out_path / 'images_{}_{}'.format(cfg.TEST['score_threshold'], n_coop))
        images_path.mkdir(exist_ok=True)
    # dir for save cpms
    data_path = (test_out_path / 'data_{}_{}'.format(cfg.TEST['score_threshold'], n_coop))
    data_path.mkdir(exist_ok=True)
    load_data_to_device = load_data_to_gpu if device.type == 'cuda' else load_data_as_tensor
    model.eval()
    # direcs = []
    i = 0

    print("Start testing")
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader):
            i += 1

            boxes = batch_data['gt_boxes'][0]
            load_data_to_device(batch_data)

            # Forward pass
            batch_data = model(batch_data)

            predictions_dicts = model.post_processing(batch_data, cfg.TEST)
            pred_boxes_fused = predictions_dicts['box_lidar'] # 3d fusion
            scores_fused = predictions_dicts['scores']
            pred_boxes_obj = predictions_dicts['boxes_fused'] # obj. fusion
            scores_obj = predictions_dicts['scores_fused']
            # direcs.extend([boxes[:, -1] for boxes in pred_boxes_fused])

            if cfg.TEST['save_img'] and i % 1 == 0:
                ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
                ax.set_aspect('equal', 'box')
                ax.set(xlim=(dcfg.pc_range[0], dcfg.pc_range[3]),
                       ylim=(dcfg.pc_range[1], dcfg.pc_range[4]))
                points = torch.cat(batch_data['points_fused'], dim=0).cpu().numpy()
                ax.plot(points[:, 0], points[:, 1], 'y.', markersize=0.3)
                ax = draw_box_plt(boxes, ax, color='green', linewidth_scale=2)
                if pred_boxes_obj is not None:
                    ax = draw_box_plt(pred_boxes_obj, ax, color='red')
                ax = draw_box_plt(pred_boxes_fused, ax, color='blue')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(str(images_path / '{}.png'.format(batch_data['frame'])))
                plt.close()

            metric_fused.add_samples([batch_data['frame']], [pred_boxes_fused], [batch_data['gt_boxes_fused']], [scores_fused],
                               ids=[batch_data['ids']])
            metric_obj.add_samples([batch_data['frame']], [pred_boxes_obj], [batch_data['gt_boxes_fused']], [scores_obj],
                               ids=[batch_data['ids']])

    metric_fused.save_detections('thr{}_ncoop{}_fused.pth'.format(cfg.TEST['score_threshold'], n_coop))
    metric_obj.save_detections('thr{}_ncoop{}_obj.pth'.format(cfg.TEST['score_threshold'], n_coop))
    aps_fused = [metric_fused.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
    aps_obj = [metric_obj.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
    with open(test_out_path / 'thr{}_ncoop{}_fused.txt'.format(cfg.TEST['score_threshold'], n_coop), 'w') as fh:
        for thr, ap in zip(thrs, aps_fused):
            fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))
    with open(test_out_path / 'thr{}_ncoop{}_obj.txt'.format(cfg.TEST['score_threshold'], n_coop), 'w') as fh:
        for thr, ap in zip(thrs, aps_obj):
            fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))


if __name__=="__main__":
    cfgs = cfg_from_py("fusion_pvrcnn_comap")
    dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
    # cfg.PATHS['run'] = '/media/hdd/ophelia/koko/experiments-output/fusion-pvrcnn/rcnn_iou_reg_iou_resampling'
    for n in [2, 4]:
        # dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
        cfg.TEST['n_coop'] = n
        # print('debug')
        test_net(copy.deepcopy((dcfg, mcfg, cfg)))