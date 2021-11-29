from torch import nn
from models import *


class CIASSD(nn.Module):
    def __init__(self, mcfg, cfg, dcfg):
        super(CIASSD, self).__init__()
        self.vfe = MeanVFE(dcfg.n_point_features)
        self.spconv_block = VoxelBackBone8x(mcfg.SPCONV,
                                            input_channels=dcfg.n_point_features,
                                            grid_size=dcfg.grid_size)
        self.map_to_bev = HeightCompression(mcfg.MAP2BEV)
        self.ssfa = SSFA(mcfg.SSFA)
        self.head = MultiGroupHead(mcfg.HEAD, dcfg.pc_range)

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        out = self.ssfa(batch_dict['spatial_features'])
        out = self.head(out)
        batch_dict['preds_dict'] = out[0]

        return batch_dict

    def loss(self, batch_dict):
        return self.head.loss(batch_dict)

    def post_processing(self, batch_dict, test_cfg):
        return self.head.post_processing(batch_dict, test_cfg)





