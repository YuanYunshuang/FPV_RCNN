import logging, os
import numpy as np
from vlib.visulization import draw_points_boxes_bev_3d as visualization
from cfg import LABEL_COLORS

def update(dcfg_obj):
    dcfg_obj.n_classes = len(dcfg_obj.classes)
    dcfg_obj.grid_size = np.round((dcfg_obj.pc_range[3:6] - dcfg_obj.pc_range[:3]) /
                         np.array(dcfg_obj.voxel_size)).astype(np.int64)
    dcfg_obj.feature_map_size = [1, *(dcfg_obj.grid_size / 8).astype(np.int64).tolist()[:-1]]
    dcfg_obj.TARGET_ASSIGNER['feature_map_size'] = dcfg_obj.feature_map_size


class Dataset(object):
    LABEL_COLORS = LABEL_COLORS
    # AUGMENTOR = {
    #     'random_world_flip': ['x'],
    #     'random_world_rotation': [-45, 45],  # rotation range in degree
    #     'random_world_scaling': [0.95, 1.05]  # scale range
    # }

    BOX_CODER = {
        'type': 'GroundBox3dCoderTorch',
        'linear_dim': False,
        'n_dim': 7,
        'angle_vec_encode': False, # encode_angle_vector
        'angle_residual_encode': True # encode_angle_with_residual
    }

    def __init__(self):
        super(Dataset, self).__init__()
        self.name = 'comap'
        self.root = 'path/to/data'
        self.pc_range = np.array([-57.6, -57.6, -0.1, 57.6, 57.6, 3.9])
        self.com_range = 40
        self.range_clip_mode = 'circle'
        self.test_split = ['1148', '753', '599', '53',
                      '905', '245', '421', '509']
        self.train_val_split = ['829', '965', '224', '685', '924', '334', '1175', '139',
                           '1070', '1050', '1162']
        self.train_split_ratio = 0.8
        self.ego_cloud_name = 'cloud_ego' # 'noisy_cloud_ego'
        self.coop_cloud_name = 'cloud_coop_with_label' # 'noisy_cloud_coop'
        self.node_selection_mode = 'random_selection_40' # 'kmeans_selection_40'
        self.fuse_raw_data = False
        self.classes = {1: ['Vehicles'], 2: ['Roads', 'RoadLines']}  # 0 is reserved for not-defined class

        self.voxel_size = [0.1, 0.1, 0.1]
        self.max_points_per_voxel = 5
        self.max_num_voxels = 100000
        self.cal_voxel_mean_std = False
        self.n_point_features = 3  # x,y,z
        self.label_downsample = 4

        # This part induct info from the info provided above
        self.n_classes = len(self.classes)
        self.grid_size = np.round((self.pc_range[3:6] - self.pc_range[:3]) /
                             np.array(self.voxel_size)).astype(np.int64)
        self.feature_map_size = [1, *(self.grid_size / 8).astype(np.int64).tolist()[:-1]]

        self.add_gps_noise = True
        self.gps_noise_std = np.array([0.4, 0.4, 0.0, 0.0, 0.0, 4]) * 0.5 # [x, y, z, roll, pitch, yaw]
        self.gps_noise_max_ratio = np.array([2.5, 2.5, 0.0, 0.0, 0.0, 1.5])

        self.TARGET_ASSIGNER ={
            'anchor_generator': {
                'type': 'AnchorGeneratorRange',
                 'sizes': [4.41, 1.98, 1.64],
                 'rotations': [0, 1.57], # remember to change the num_dirs in HEAD cfg
                 'match_threshold': 0.6,
                 'unmatch_threshold': 0.45,
                 'class_name': 'Car'
            },
            'sample_positive_fraction': None,
            'sample_size': 512,
            'pos_area_threshold': -1,
            'box_coder': self.BOX_CODER,
            'out_size_factor': 8,
            'enable_similar_type': True,
            'feature_map_size': self.feature_map_size
        }

        self.process_fn = {
            'train': ['mask_points_in_range','rm_empty_gt_boxes', 'points_to_voxel', 'assign_target'],
            'test': ['mask_points_in_range', 'points_to_voxel', 'assign_target']
        }


    def update(self):
        self.n_classes = len(self.classes)
        self.grid_size = np.round((self.pc_range[3:6] - self.pc_range[:3]) /
                                      np.array(self.voxel_size)).astype(np.int64)
        self.feature_map_size = [1, *(self.grid_size / 8).astype(np.int64).tolist()[:-1]]
        self.TARGET_ASSIGNER['feature_map_size'] = self.feature_map_size


class Model:
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'fusion_pvrcnn'
        # self.params_train = ['CPMEnc', 'FUDET']
        self.VFE = None
        self.SPCONV = {
            'num_out_features': 64
        }
        self. MAP2BEV = {
            'num_features': 128
        }
        self.SSFA = {
            'layer_nums': [5],
            'ds_layer_strides': [1],
            'ds_num_filters': [128],
            'us_layer_strides': [1],
            'us_num_filters': [128],
            'num_input_features': 128,
            'norm_cfg': None
        }

        self.HEAD = {
                'type': 'MultiGroupHead',
                'mode': '3d',
                'in_channels': sum([128,]),
                'norm_cfg': None,
                'num_class': 1,
                'num_dirs': 2,
                'class_names': ['Car'],
                'weights': [1, ],
                'with_cls' : True,
                'with_reg' : True,
                'reg_class_agnostic' : False,
                'pred_var': False,
                'box_coder':  Dataset.BOX_CODER,
                'encode_background_as_zeros': True,
                'loss_norm': {'type': 'NormByNumPositives',
                              'pos_cls_weight': 50.0,
                              'neg_cls_weight': 1.0},
                'loss_cls': {'type': 'SigmoidFocalLoss',
                             'alpha': 0.25,
                             'gamma': 2.0,
                             'loss_weight': 1.0},
                'use_sigmoid_score': True,
                'loss_bbox': {'type': 'WeightedSmoothL1Loss',
                              'sigma': 3.0,
                              'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              'codewise': True,
                              'loss_weight': 2.0},
                'loss_iou': {'type': 'WeightedSmoothL1Loss',
                              'sigma': 3.0,
                              'code_weights': None,
                              'codewise': True,
                              'loss_weight': 1.0},
                'encode_rad_error_by_sin': True,
                'use_dir_classifier': True,
                'loss_aux': {'type': 'WeightedSoftmaxClassificationLoss',
                             'name': 'direction_classifier',
                             'loss_weight': 0.2},
                'direction_offset': 0.5,
                'nms': {
                    'name': 'normal', # 'iou_weighted',
                    'score_threshold': 0.3,
                    'cnt_threshold': 1,
                    'nms_pre_max_size': 1000,
                    'nms_post_max_size': 100,
                    'nms_iou_threshold': 0.01,
                },
                'logger': None
        }

        self.VSA = {
            'add_ego_mask_feature': False,
            'enlarge_selection_boxes': True,
            'point_source': 'raw_points',
            'num_keypoints': 2048,
            'num_out_features': 32,
            'features_source': ['bev', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4', 'raw_points'],
            'sa_layer': {
                'raw_points': {
                    'mlps': [[16, 16], [16, 16]],
                    'pool_radius': [0.4, 0.8],
                    'n_sample': [16, 16]
                },
                'x_conv1': {
                    'downsample_factor': 1,
                    'mlps': [[16, 16], [16, 16]],
                    'pool_radius': [0.4, 0.8],
                    'n_sample': [16, 16]
                },
                'x_conv2': {
                    'downsample_factor': 2,
                    'mlps': [[32, 32], [32, 32]],
                    'pool_radius': [0.8, 1.2],
                    'n_sample': [16, 32]
                },
                'x_conv3': {
                    'downsample_factor': 4,
                    'mlps': [[64, 64], [64, 64]],
                    'pool_radius': [1.2, 2.4],
                    'n_sample': [16, 32]
                },
                'x_conv4': {
                    'downsample_factor': 8,
                    'mlps': [[64, 64], [64, 64]],
                    'pool_radius': [2.4, 4.8],
                    'n_sample': [16, 32]
                },
            }
        }

        self.SEMSEG = {
            'valid_classes': {
                                1: [1, 11], # buildings, walls
                                2: [2], # fences
                                3: [5], # poles
                                4: [10], # vehicles
                             },
            'weight_code': [1, 1.0, 1.0, 1.0, 1.0],
            'num_cls': 5,
            'in_channels': 32,
            'n_layers': 2,
            'loss_cls': {'type': 'SigmoidFocalLoss',
                         'loss_weight': 20.0},
        }

        self.MATCHER = {
            'max_cons': {
                'resolution': [0.5, 0.5, 1], # resolution for dividing searching range
                'radius': 1, # nearest neighbor consensus searching radius
                'min_cons': 10, # min. number of consensus score
                'min_match_acc_points': 2, # min. number of matched pole and vehicle center points
            }
        }

        self.ROI_HEAD = {
            'num_cls': 1,
            'input_channels': 32,
            'shared_fc': [256, 256],
            'cls_fc': [256, 256],
            'reg_fc': [256, 256],
            'dp_ratio': 0.3,
            'roi_grid_pool': {
                'grid_size': 6,
                'mlps': [[64, 64], [64, 64]],
                'pool_radius': [0.8, 1.6],
                'nsample': [16, 16],
                'pool_method': 'max_pool',
            },
            'det_range': 57.6,
            'loss_cls': {'type': 'WeightedSigmoidBinaryCELoss',
                         'loss_weight': 1.0},
            'use_sigmoid_score': True,

            'loss_iou': {'type': 'WeightedSmoothL1Loss',
                         'sigma': 3.0,
                         'code_weights': [1.0],
                         'codewise': True,
                         'loss_weight': 1.0},
            'loss_bbox': {'type': 'WeightedSmoothL1Loss',
                          'sigma': 3.0,
                          'reduction': 'mean',
                          'code_weights': [20.0] * 7,
                          'codewise': True,
                          'loss_weight': 1.0},
        }


class Optimization:
    def __init__(self):
        self.TRAIN = {
        'project_name': None, #None, # 'cia-ssd',
        'visualization_func': visualization,
        'batch_size': 1,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'betas': [0.95, 0.999],
        'scheduler_step': 150,
        'scheduler_gamma': 0.5,
        'resume': True,
        'epoch': 0,
        'total_epochs': 10,
        'log_every': 20,
        'save_ckpt_every': 10
    }
        self.TEST = {
        'save_img': True,
        'bev': False,
        'n_coop': 4,
        'com_range': 40,
        'score_threshold': 0.3,
        'cnt_threshold': 2,
        'nms_pre_max_size': 1000,
        'nms_post_max_size': 100,
        'nms_iou_threshold': 0.01,
        'ap_ious': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
        self.PATHS = {
         'run': '/path/for/logging/output'
    }



