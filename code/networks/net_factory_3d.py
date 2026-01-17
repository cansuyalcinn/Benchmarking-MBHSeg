from networks.unet_3D import unet_3D
from networks.vnet import VNet
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from networks.nnunet import initialize_network
from networks.vnet2 import VNet2
from networks.vnet_other_models import Vnet_MLRPL, Vnet_MLRPL_ours, Vnet_MLRPL_weakstrong, Vnet_MLRPL_3decoder, Vnet_ours, Vnet_MLRPL_border, Vnet_base, CAML_vnet, VNet_caml
from networks.DCNet_Unet3D import MCNet3d_KD
from networks.Vnet_SSNET import VNet_SSNet
from networks.Vnet_BCP import VNet_BCP
from networks.vnetmismatch import VNetMisMatch 
from networks.Caml_vnet import CAML3d_v1

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, mode="train"):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
        
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, 
                   n_classes=class_num,
                   normalization='batchnorm', 
                   has_dropout=True).cuda()
        
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, 
                   n_classes=class_num,
                   normalization='batchnorm', 
                   has_dropout=False).cuda()
        
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
        
    elif net_type == "vnet_MLRPL" and mode == "train":
        net = Vnet_MLRPL(n_channels=in_chns,
                 n_classes=class_num,
                 n_filters=16,
                 normalization='batchnorm',
                 has_dropout=True,
                 has_residual=False).cuda()
        
    elif net_type == "vnet_MLRPL" and mode == "test":
        net = Vnet_MLRPL(n_channels=in_chns,
                 n_classes=class_num,
                 n_filters=16,
                 normalization='batchnorm',
                 has_dropout=False,
                 has_residual=False).cuda()
        
    elif net_type == "vnet_MLRPL_ours" and mode == "train":
            net = Vnet_MLRPL_ours(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False).cuda()
            
    elif net_type == "Vnet_MLRPL_3decoder" and mode == "train":
            net = Vnet_MLRPL_3decoder(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False).cuda()
            
    elif net_type == "Vnet_MLRPL_3decoder" and mode == "test":
            net = Vnet_MLRPL_3decoder(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=False,
                has_residual=False).cuda()

    elif net_type == "Vnet_MLRPL_weakstrong" and mode == "train":
            net = Vnet_MLRPL_weakstrong(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False, test=False).cuda()
            
    elif net_type == "Vnet_MLRPL_weakstrong" and mode == "test":
            net = Vnet_MLRPL_weakstrong(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=False,
                has_residual=False, test=True).cuda()
            

    elif net_type == "Vnet_ours" and mode == "train":
            net = Vnet_ours(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False, test=False).cuda()
            
    elif net_type == "Vnet_ours" and mode == "test":
            net = Vnet_ours(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=False,
                has_residual=False, test=True).cuda()
            
    elif net_type == "Vnet_MLRPL_border" and mode == "train":
            net = Vnet_MLRPL_border(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False).cuda()
            
    elif net_type == "Vnet_MLRPL_border" and mode == "test":
            net = Vnet_MLRPL_border(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=False,
                has_residual=False).cuda()
            
    elif net_type == "Vnet_base" and mode == "train":
            # Our baseline model that is same as the mlrpl encoder + decoder 1. 
            net = Vnet_base(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False).cuda()
            
    elif net_type == "Vnet_base" and mode == "test":
            net = Vnet_base(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=False,
                has_residual=False).cuda()
            
    elif net_type == "VNet_caml" and mode == "train":
            # Our baseline model that is same as the mlrpl encoder + decoder 1. 
            net = VNet_caml(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False).cuda()
            
    elif net_type == "CAML_vnet" and mode == "train":
            # Our baseline model that is same as the mlrpl encoder + decoder 1. 
            net = CAML_vnet(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False).cuda()
            
    elif net_type == "VNet_caml" and mode == "train":
            # Our baseline model that is same as the mlrpl encoder + decoder 1. 
            net = VNet_caml(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False).cuda()
        
    elif net_type == "CAML3d_v1" and mode == "train":
            # Our baseline model that is same as the mlrpl encoder + decoder 1. 
            net = CAML3d_v1(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=True,
                has_residual=False).cuda()

    elif net_type == "CAML3d_v1" and mode == "test":
            # Our baseline model that is same as the mlrpl encoder + decoder 1. 
            net = CAML3d_v1(n_channels=in_chns,
                n_classes=class_num,
                n_filters=16,
                normalization='batchnorm',
                has_dropout=False,
                has_residual=False).cuda()

    elif net_type =='mcnet_kd':
        net = MCNet3d_KD(in_chns=in_chns, class_num=class_num).cuda()

    elif net_type == "vnet_SSNet" and mode == "train":
        net = VNet_SSNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()

    elif net_type == "vnet_SSNet" and mode == "test":
        net = VNet_SSNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()


    elif net_type == "vnet_bcp" and mode == "train":
        net = VNet_BCP(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()

    elif net_type == "vnet_bcp" and mode == "test":
        net = VNet_BCP(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "vnet_mismatch" and mode == "train":
        net = VNetMisMatch(n_channels=in_chns, 
                           n_classes=class_num, 
                           normalization='batchnorm', 
                           n_filters=16,
                           has_dropout=False).cuda()

    else:
        net = None
    return net