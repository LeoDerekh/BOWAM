import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.test.CBAM import ChannelAttention, SpatialAttention
from models.test.torch_wavelets import DWT_2D, IDWT_2D
from timm.layers import create_conv2d

# class WTConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
#         super(WTConv2d, self).__init__()

#         assert in_channels == out_channels

#         self.in_channels = in_channels
#         self.wt_levels = wt_levels
#         self.stride = stride
#         self.dilation = 1

#         self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
#         self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
#         self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

#         self.wt_function = partial(wavelet.wavelet_transform, filters = self.wt_filter)
#         self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters = self.iwt_filter)

#         self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
#         self.base_scale = _ScaleModule([1,in_channels,1,1])

#         self.wavelet_convs = nn.ModuleList(
#             [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
#         )
#         self.wavelet_scale = nn.ModuleList(
#             [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
#         )

#     def forward(self, x):

#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []

#         curr_x_ll = x

#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
#             if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
#                 curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
#                 curr_x_ll = F.pad(curr_x_ll, curr_pads)

#             curr_x = self.wt_function(curr_x_ll)
#             curr_x_ll = curr_x[:,:,0,:,:]

#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
#             curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
#             curr_x_tag = curr_x_tag.reshape(shape_x)

#             x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
#             x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

#         next_x_ll = 0

#         for i in range(self.wt_levels-1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()

#             curr_x_ll = curr_x_ll + next_x_ll

#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
#             next_x_ll = self.iwt_function(curr_x)

#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

#         x_tag = next_x_ll
#         assert len(x_ll_in_levels) == 0

#         x = self.base_scale(self.base_conv(x))
#         x = x + x_tag

#         return x

# class _ScaleModule(nn.Module):
#     def __init__(self, dims, init_scale=1.0, init_bias=0):
#         super(_ScaleModule, self).__init__()
#         self.dims = dims
#         self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
#         self.bias = None

#     def forward(self, x):
#         return torch.mul(self.weight, x)


# class WTConv2d_v1(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wave='haar'):
#         super(WTConv2d_v1, self).__init__()

#         assert in_channels == out_channels
#         self.in_channels = in_channels
#         self.stride = stride
        
#         self.dwt = DWT_2D(wave=wave)
#         self.idwt = IDWT_2D(wave=wave)

#         self.outconv_bn_relu_L = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.outconv_bn_relu_H = nn.Sequential(
#             nn.Conv2d(in_channels * 3, out_channels * 3, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_channels * 3),
#             nn.ReLU(inplace=True),
#         )

#         # self.cbam = CBAM(in_channels * 3)
#         self.channel_attention = ChannelAttention(in_channels * 3, ratio=16)

#         self.base_conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#         )
#         # self.base_scale = _ScaleModule([1, in_channels, 1, 1])

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # print("x shape: ", x.shape)
#         flag = False
#         if x.shape[2] == 7:
#             flag = True
#             x = F.interpolate(x, size=(6, 6), mode='bilinear', align_corners=False)
#         x_dwt = self.dwt(x)
#         # print("x_dwt shape: ", x_dwt.shape)
#         x_ll, x_lh, x_hl, x_hh = torch.chunk(x_dwt, 4, dim=1)

#         y_H = torch.cat([x_lh, x_hl, x_hh], dim=1)
#         y_L = x_ll
#         # print("y_L shape: ", y_L.shape)

        

#         y_L = self.outconv_bn_relu_L(y_L)
#         # y_H = self.cbam(y_H)
#         # y_H = self.channel_attention(y_H)
#         # y_H = self.conv_bn_relu(y_H)
#         y_H = self.outconv_bn_relu_H(y_H)

        

#         # print("y_H shape: ", y_H.shape)


#         x_dwt = torch.cat([y_L, y_H], dim=1)

#         x_idwt = self.idwt(x_dwt)
#         if flag:
#             x_idwt = F.interpolate(x_idwt, size=(7, 7), mode='bilinear', align_corners=False)
#         # print("x_idwt shape: ", x_idwt.shape)
#         x_idwt = self.base_conv_bn_relu(x_idwt)

#         # x_idwt = self.base_scale(x_idwt)

#         return x_idwt
    


class WTConv2d_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wave='haar'):
        super(WTConv2d_v2, self).__init__()
        assert in_channels == out_channels
        self.dwt = DWT_2D(wave=wave)
        self.idwt = IDWT_2D(wave=wave)
        # self.filter = nn.Sequential(
        #     nn.Conv2d(in_channels*4 , in_channels, kernel_size=1, stride=1, groups=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(),

        #     # SqueezeExcitation(input_c=in_channels, expand_c=in_channels, squeeze_factor=16),

        #     nn.Conv2d(in_channels , in_channels*4, kernel_size=1, stride=1, groups=1),
        #     nn.BatchNorm2d(in_channels*4),
        #     nn.ReLU(inplace=True),
        # )

        # self.filter = nn.Sequential(
        #     nn.Conv2d(in_channels*4 , in_channels*4, kernel_size=1, stride=1, groups=1),
        #     nn.BatchNorm2d(in_channels*4),
        #     nn.ReLU(inplace=True),
        # )

        # self.outconv_bn_relu_L = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        # )
        # self.outconv_bn_relu_H = nn.Sequential(
        #     nn.Conv2d(in_channels * 3, out_channels * 3, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(out_channels * 3),
        #     nn.ReLU(inplace=True),
        # )
        self.ChannelAttention = ChannelAttention(in_channels*3, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

        self.base_conv_bn_relu = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias),
            # nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, bias=bias),
            create_conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=1, depthwise=True, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )


        
        # self.cbam = CBAM(in_channels*3)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        flag = False
        if x.shape[2] == 7:
            flag = True
            x = F.interpolate(x, size=(6, 6), mode='bilinear', align_corners=False)
        x_dwt = self.dwt(x)

        x_ll, x_lh, x_hl, x_hh = torch.chunk(x_dwt, 4, dim=1)

        y_H = torch.cat([x_lh, x_hl, x_hh], dim=1)
        y_L = x_ll

        y_H = self.ChannelAttention(y_H)
        y_L = self.SpatialAttention(y_L)

        # y_L = self.outconv_bn_relu_L(y_L)
        # y_H = self.outconv_bn_relu_H(y_H)

        x_dwt = torch.cat([y_L, y_H], dim=1)

        x_idwt = self.idwt(x_dwt)
        if flag:
            x_idwt = F.interpolate(x_idwt, size=(7, 7), mode='bilinear', align_corners=False)

        x_idwt = self.base_conv_bn_relu(x_idwt)

        return x_idwt
