import torch
from torch import nn
import torch.nn.functional as F
from networks.vnet import *

class FeaturePerturbation3D(nn.Module):
    def __init__(self, lam=0.9, kap=0.2, eps=1e-6, use_gpu=True):
        super(FeaturePerturbation3D, self).__init__()
        # self.num_features = num_features
        self.eps = eps
        self.lam = lam
        self.kap = kap
        self.use_gpu = use_gpu

    def forward(self, x):
        # normalization
        mu = x.mean(dim=[2, 3, 4], keepdim=True)  # [B, C, 1, 1, 1]
        var = x.var(dim=[2, 3, 4], keepdim=True)  # [B, C, 1, 1, 1]
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        batch_mu = mu.mean(dim=[0], keepdim=True)  # [1,C,1,1]
        batch_psi = (mu.var(dim=[0], keepdim=True) + self.eps).sqrt()  # [1,C,1,1]
        batch_sig = sig.mean(dim=[0], keepdim=True)  # [1,C,1,1]
        batch_phi = (sig.var(dim=[0], keepdim=True) + self.eps).sqrt()  # [1,C,1,1]
        epsilon = torch.empty(1).uniform_(-self.kap, self.kap)
        epsilon = epsilon.cuda()
        gamma = self.lam * sig + (1 - self.lam) * batch_sig + epsilon * batch_phi
        gamma = gamma.cuda()
        beta = self.lam * mu + (1 - self.lam) * batch_mu + epsilon * batch_psi
        beta = beta.cuda()
        x_aug = gamma * x_normed + beta
        return x_aug


# Mutual learning MLRPL
class ConvBlock(nn.Module):

    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):

    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):

    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):

    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Encoder(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res

class Decoder_v1(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False,
                 up_type=0):
        super(Decoder_v1, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16,
                                                 n_filters * 8,
                                                 normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8,
                                                n_filters * 4,
                                                normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4,
                                                  n_filters * 2,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2,
                                                  n_filters,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)

        out_seg = self.out_conv(x9)

        return out_seg, x9
    


# # # class Decoder_v2(nn.Module):
# # #     # this decoder is used for the weak strong feature perturbations module. 
# # #     def __init__(self,
# # #                  n_channels=3,
# # #                  n_classes=2,
# # #                  n_filters=16,
# # #                  normalization='none',
# # #                  has_dropout=False,
# # #                  has_residual=False,
# # #                  up_type=0,
# # #                  perturb_fn=None):
        
# # #         super(Decoder_v2, self).__init__()
# # #         self.has_dropout = has_dropout
# # #         self.perturb_fn = perturb_fn


# # #         convBlock = ConvBlock if not has_residual else ResidualConvBlock

# # #         self.block_five_up = Upsampling_function(n_filters * 16,
# # #                                                  n_filters * 8,
# # #                                                  normalization=normalization,
# # #                                                  mode_upsampling=up_type)

# # #         self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
# # #         self.block_six_up = Upsampling_function(n_filters * 8,
# # #                                                 n_filters * 4,
# # #                                                 normalization=normalization,
# # #                                                 mode_upsampling=up_type)

# # #         self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
# # #         self.block_seven_up = Upsampling_function(n_filters * 4,
# # #                                                   n_filters * 2,
# # #                                                   normalization=normalization,
# # #                                                   mode_upsampling=up_type)

# # #         self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
# # #         self.block_eight_up = Upsampling_function(n_filters * 2,
# # #                                                   n_filters,
# # #                                                   normalization=normalization,
# # #                                                   mode_upsampling=up_type)

# # #         self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
# # #         self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

# # #         self.dropout = nn.Dropout3d(p=0.5, inplace=False)

# # #     def forward(self, features):
# # #         x1, x2, x3, x4, x5 = features

# # #         x5_up = self.block_five_up(x5)
# # #         if self.perturb_fn: x5_up = self.perturb_fn(x5_up)
# # #         x5_up = x5_up + x4

# # #         x6 = self.block_six(x5_up)
# # #         if self.perturb_fn: x6 = self.perturb_fn(x6)
# # #         x6_up = self.block_six_up(x6)
# # #         if self.perturb_fn: x6_up = self.perturb_fn(x6_up)
# # #         x6_up = x6_up + x3

# # #         x7 = self.block_seven(x6_up)
# # #         if self.perturb_fn: x7 = self.perturb_fn(x7)
# # #         x7_up = self.block_seven_up(x7)
# # #         if self.perturb_fn: x7_up = self.perturb_fn(x7_up)
# # #         x7_up = x7_up + x2

# # #         x8 = self.block_eight(x7_up)
# # #         if self.perturb_fn: x8 = self.perturb_fn(x8)
# # #         x8_up = self.block_eight_up(x8)
# # #         if self.perturb_fn: x8_up = self.perturb_fn(x8_up)
# # #         x8_up = x8_up + x1

# # #         x9 = self.block_nine(x8_up)
# # #         if self.has_dropout:
# # #             x9 = self.dropout(x9)

# # #         out_seg = self.out_conv(x9)
# # #         return out_seg, x9


class Vnet_MLRPL(nn.Module):
    # name Changed from Mine3d_v1 to Vnet_MLRPL for clarity. 

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False,
                 ):
        
        super(Vnet_MLRPL, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        out_seg2, f2 = self.decoder2(features)
        if self.training:
            return [out_seg1, out_seg2], [f1, f2]
        else:
            return out_seg1, out_seg2
        

class Vnet_MLRPL_ours(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        
        super(Vnet_MLRPL_ours, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)  
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=2) 

    # def forward(self, input_weak, input_strong=None):
    def forward(self, input):
        outputs = {}
        features1 = self.encoder(input)
        out_seg1, f1 = self.decoder1(features1)
        out_seg2, f2 = self.decoder2(features1)

        outputs['out_seg1'] = out_seg1
        outputs['out_seg2'] = out_seg2
        outputs['features1'] = f1
        outputs['features2'] = f2

        # if input_strong is not None:
        #     features_strong = self.encoder(input_strong)
        #     out_seg3, f3 = self.decoder3(features_strong)
        #     outputs['out_seg3'] = out_seg3
        #     outputs['features_strong'] = f3

        return outputs
    

class Decoder_v3(nn.Module):
    # this decoder is created for the model for 2 decoders (ml) and with 2 heads (segmentation and border segmentation)

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False,
                 up_type=0):
        super(Decoder_v3, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16,
                                                 n_filters * 8,
                                                 normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8,
                                                n_filters * 4,
                                                normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4,
                                                  n_filters * 2,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2,
                                                  n_filters,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)

        return x9


class Vnet_MLRPL_border(nn.Module):
    # model has 2 ML decoders as original but it has 2 heads. Onee for semantic segmentation and one for only border segmentation.

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        
        super(Vnet_MLRPL_border, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v3(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)  
        self.decoder2 = Decoder_v3(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=2) 

        self.seg_head1 = nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)
        self.boundary_head1 = nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)

        self.seg_head2 = nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)
        self.boundary_head2 = nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)
    
    def forward(self, x):
        outputs = {}
        features = self.encoder(x)
        f1 = self.decoder1(features)
        f2 = self.decoder2(features)

        out_seg1 = self.seg_head1(f1)
        out_boundary1 = self.boundary_head1(f1)

        out_seg2 = self.seg_head2(f2)
        out_boundary2 = self.boundary_head2(f2)

        outputs['out_seg1'] = out_seg1
        outputs['out_seg2'] = out_seg2
        outputs['features1'] = f1
        outputs['features2'] = f2
        outputs['boundary1'] = out_boundary1
        outputs['boundary2'] = out_boundary2

        return outputs
    


class Vnet_base(nn.Module):
    # The base Vnet model that is same as the ML model's setting (in terms of number of layers and parameters etc.)

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        
        super(Vnet_base, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)  
    
    def forward(self, x):
        outputs = {}
        features = self.encoder(x)
        out_seg1, f1 = self.decoder1(features)
        outputs['out_seg1'] = out_seg1
        outputs['features1'] = f1
        return outputs

class Vnet_MLRPL_3decoder(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        
        super(Vnet_MLRPL_3decoder, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)  
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=2)  
        self.decoder3 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=1)
        

    def forward(self, input):
        outputs = {}
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        out_seg2, f2 = self.decoder2(features)
        out_seg3, f3 = self.decoder3(features)

        outputs['out_seg1'] = out_seg1
        outputs['out_seg2'] = out_seg2
        outputs['out_seg3'] = out_seg3
        outputs['features1'] = f1
        outputs['features2'] = f2
        outputs['features3'] = f3

        return outputs



class Decoder_v2(nn.Module):
    # this decoder is used for the weak strong feature perturbations module. 
    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False,
                 up_type=0,
                 perturb_fn=None):
        
        super(Decoder_v2, self).__init__()
        self.has_dropout = has_dropout
        self.perturb_fn = perturb_fn


        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16,
                                                 n_filters * 8,
                                                 normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8,
                                                n_filters * 4,
                                                normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4,
                                                  n_filters * 2,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2,
                                                  n_filters,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.cont_conv_a = nn.Conv3d(n_filters, n_filters, kernel_size=1, padding=0) # the projection had will have 16 filters. 
        self.cont_conv_b = nn.Conv3d(n_filters, n_filters, kernel_size=1, padding=0)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)
        if self.perturb_fn: x5_up = self.perturb_fn(x5_up)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        if self.perturb_fn: x6 = self.perturb_fn(x6)
        x6_up = self.block_six_up(x6)
        if self.perturb_fn: x6_up = self.perturb_fn(x6_up)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        if self.perturb_fn: x7 = self.perturb_fn(x7)
        x7_up = self.block_seven_up(x7)
        if self.perturb_fn: x7_up = self.perturb_fn(x7_up)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        if self.perturb_fn: x8 = self.perturb_fn(x8)
        x8_up = self.block_eight_up(x8)
        if self.perturb_fn: x8_up = self.perturb_fn(x8_up)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)

        out_seg = self.out_conv(x9)
        cont = self.cont_conv_a(x9)
        cont_output = self.cont_conv_b(cont)

        return out_seg, x9, cont_output


class Vnet_MLRPL_weakstrong(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False, test=False):
        
        super(Vnet_MLRPL_weakstrong, self).__init__()
        kaps = [0.067, 0.134, 0.2]

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        
        
        # Three decoders for different perturbations they are all the same
        self.decoder_np = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=None) # no perturbation
        
        if test:
            self.decoder1 = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=None)
            self.decoder2 = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=None) 
            self.decoder3 = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=None) 
            
        else:
            self.decoder1 = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=FeaturePerturbation3D(kap=kaps[0])) # wealy perturbated
            self.decoder2 = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=FeaturePerturbation3D(kap=kaps[1])) # medium perturbated
            self.decoder3 = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=FeaturePerturbation3D(kap=kaps[2])) # strong perturbated


    def forward(self, input):
        outputs = {}
        features = self.encoder(input)
        out_seg, f, cont_out = self.decoder_np(features)
        out_seg1, f1, cont_out1 = self.decoder1(features)
        out_seg2, f2, cont_out2 = self.decoder2(features)
        out_seg3, f3, cont_out3 = self.decoder3(features)

        outputs['out_segnp'] = out_seg
        outputs['out_seg1'] = out_seg1
        outputs['out_seg2'] = out_seg2
        outputs['out_seg3'] = out_seg3
        outputs['cont_out'] = cont_out
        outputs['cont_out1'] = cont_out1
        outputs['cont_out2'] = cont_out2
        outputs['cont_out3'] = cont_out3
        outputs['features'] = f
        outputs['features1'] = f1
        outputs['features2'] = f2
        outputs['features3'] = f3

        return outputs



class Vnet_ours(nn.Module):
    # 2 decoders from ml model
    # onde decoder from FP model.

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False, test=False):
        
        super(Vnet_ours, self).__init__()
        kaps = [0.067, 0.134, 0.2]

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
 
        self.decoder3 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=1)
        
        if test:
            self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)  
            self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=2) 
            self.decoder3 = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=None) 

        else:
            self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)  
            self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=2) 
            self.decoder3 = Decoder_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0, perturb_fn=FeaturePerturbation3D(kap=kaps[1])) # medium perturbated


    def forward(self, input):
        outputs = {}
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        out_seg2, f2 = self.decoder2(features)
        out_seg3, f3, cont_out3 = self.decoder3(features)

        outputs['out_seg1'] = out_seg1
        outputs['out_seg2'] = out_seg2
        outputs['out_seg3'] = out_seg3
        outputs['cont_out3'] = cont_out3
        outputs['features1'] = f1
        outputs['features2'] = f2
        outputs['features3'] = f3

        return outputs



### 

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features, with_feature=False):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        if with_feature:
            return out_seg, x9
        else:
            return out_seg


class VNet_caml(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_caml, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1


class InterSampleAttention(torch.nn.Module):
    """
        Implementation for inter-sample self-attention
        input size for the encoder_layers: [batch, h x w x d, dim]
    """
    def __init__(self, input_dim=256, hidden_dim=1024):
        super(InterSampleAttention, self).__init__()
        self.input_dim = input_dim
        self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)

    def forward(self, feature):
        if self.training:
            b, c, h, w, d = feature.shape
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(b, h * w * d, c)
            feature = self.encoder_layers(feature)
            feature = feature.view(b, h, w, d, c)
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature


class IntraSampleAttention(torch.nn.Module):
    """
    Implementation for intra-sample self-attention
    input size for the encoder_layers: [h x w x d, batch, dim]
    """
    def __init__(self, input_dim=256, hidden_dim=1024):
        super(IntraSampleAttention, self).__init__()
        self.input_dim = input_dim
        self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)

    def forward(self, feature):
        if self.training:
            b, c, h, w, d = feature.shape
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(b, h * w * d, c)
            feature = feature.permute(1, 0, 2).contiguous()
            feature = self.encoder_layers(feature)
            feature = feature.permute(1, 0, 2).contiguous()
            feature = feature.view(b, h, w, d, c)
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature


class EncoderAuxiliary(nn.Module):
    """
    encoder for auxiliary model with CMA
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4):
        super(EncoderAuxiliary, self).__init__()
        self.insert_idx = insert_idx
        self.cma_type = cma_type

        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # print(self.get_dim(self.insert_idx))
        if self.cma_type == 'v2+':
            self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 4:
            return self.block_four.conv[6].weight.shape[0]

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        # cma layers
        if self.insert_idx == 4:
            if self.cma_type == "v2+":
                x4 = self.intra_attention(x4)
            x4 = self.inter_attention(x4)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class CAML_vnet(nn.Module):
    """ 
    caml model with one aux encoder and one decoder
    """

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4, feat_dim=32):
        
        super(CAML_vnet, self).__init__()
        self.cma_type = cma_type
        self.insert_idx = insert_idx
        assert self.insert_idx == 4

        self.encoder2 = EncoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                                  cma_type=self.cma_type, insert_idx=self.insert_idx)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

        self.projection_head2 = nn.Sequential(
            nn.Linear(n_filters, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head2 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, input):
        features2 = self.encoder2(input)
        out_seg2, embedding2 = self.decoder2(features2, with_feature=True)
        return out_seg2

class CAML3d_v1(nn.Module):
    """
    Use CMA on Encoder layer 4
    With different upsample
    """

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4, feat_dim=32):
        super(CAML3d_v1, self).__init__()
        self.cma_type = cma_type
        self.insert_idx = insert_idx
        assert self.insert_idx == 4
        self.encoder1 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.encoder2 = EncoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                                  cma_type=self.cma_type, insert_idx=self.insert_idx)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.projection_head1 = nn.Sequential(
            nn.Linear(n_filters, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head1 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.projection_head2 = nn.Sequential(
            nn.Linear(n_filters, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head2 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, input):
        features1 = self.encoder1(input)
        features2 = self.encoder2(input)
        out_seg1, embedding1 = self.decoder1(features1, with_feature=True)
        out_seg2, embedding2 = self.decoder2(features2, with_feature=True)
        return out_seg1, out_seg2, embedding1, embedding2