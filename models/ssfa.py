import torch
from torch import nn
from torch.nn import Sequential

from models.utils import xavier_init, build_norm_layer



class SSFA(nn.Module):
    def __init__(self, cfg):
        super(SSFA, self).__init__()

        self._layer_strides = cfg['ds_layer_strides']  # [1,]
        self._num_filters = cfg['ds_num_filters']      # [128,]
        self._layer_nums = cfg['layer_nums']           # [5,]
        self._upsample_strides = cfg['us_layer_strides']      # [1,]
        self._num_upsample_filters = cfg['us_num_filters']    # [128,]
        self._num_input_features = cfg['num_input_features']  # 128

        if cfg['norm_cfg'] is None:  # True
            cfg['norm_cfg'] = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = cfg['norm_cfg']

        self.bottom_up_block_0 = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, stride=1, bias=False),
            build_norm_layer(self._norm_cfg, 128)[1],
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.bottom_up_block_1 = Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 256, )[1],
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 256, )[1],
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 256, )[1],
            nn.ReLU(),

        )

        self.trans_0 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.trans_1 = Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            build_norm_layer(self._norm_cfg, 256, )[1],
            nn.ReLU(),
        )

        self.deconv_block_0 = Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.deconv_block_1 = Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.conv_0 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.w_0 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            build_norm_layer(self._norm_cfg, 1, )[1],
        )

        self.conv_1 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.w_1 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            build_norm_layer(self._norm_cfg, 1, )[1],
        )


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        x_0 = self.bottom_up_block_0(x)
        x_1 = self.bottom_up_block_1(x_0)
        x_trans_0 = self.trans_0(x_0)
        x_trans_1 = self.trans_1(x_1)
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0
        x_middle_1 = self.deconv_block_1(x_trans_1)
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]

        return x_output.contiguous()