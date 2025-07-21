import os
from functools import partial

from timm.models.deit import deit_tiny_patch16_224
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from torch import nn

from backbone.model.EfficientMod import efficientMod_xxs
from backbone.model.PC_vit import PC_vit
from backbone.model.convnext import convnext_t
from backbone.model.resnet import *

from torchvision.transforms import Resize

from models.Spatial_vit import VisionTransformer_POS

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'


class res_DRT(nn.Module):
    def __init__(self, num_classes=5):
        super(res_DRT, self).__init__()
        ##Position Calibration Module(subbranch)
        self.vit_pos = VisionTransformer_POS(img_size=14,
        patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize = Resize([14, 14])
        ##main branch consisting of CA blocks
        self.main_branch = ResNet18(num_classes=num_classes)
        # self.s_branch = RMT_N4(num_class=num_classes)
    def forward(self, on, apex, off):
        # onset:x1 apex:x5
        B = on.shape[0]
        B_b = apex.shape[0]
        # Position Calibration Module (subbranch)
        # 前分支
        POS = self.vit_pos(self.resize(on)).transpose(1, 2).view(B, 512, 14, 14)
        act = apex - on
        out = self.main_branch(act, POS)
        # 后分支
        act_b = apex - off
        POS_b = self.vit_pos(self.resize(apex)).transpose(1, 2).view(B_b, 512, 14, 14)
        out_b = self.main_branch(act_b, POS_b)

        return out, out_b


class cov_DRT(nn.Module):
    def __init__(self, num_classes=5):
        super(cov_DRT, self).__init__()
        ##Position Calibration Module(subbranch)
        self.vit_pos = VisionTransformer_POS(img_size=7,
        patch_size=1, embed_dim=768, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize = Resize([7, 7])
        ##main branch consisting of CA blocks
        self.main_branch = convnext_t(num_classes=num_classes)
        # self.s_branch = RMT_N4(num_class=num_classes)
    def forward(self, on, apex, off):
        # onset:x1 apex:x5
        B = on.shape[0]
        B_b = apex.shape[0]
        # Position Calibration Module (subbranch)
        # 前分支
        POS = self.vit_pos(self.resize(on)).transpose(1, 2).view(B, 768, 7, 7)
        # print("POS", POS.shape)
        act = apex - on
        out = self.main_branch(act, POS)
        # 后分支
        act_b = apex - off
        POS_b = self.vit_pos(self.resize(apex)).transpose(1, 2).view(B_b, 768, 7, 7)
        out_b = self.main_branch(act_b, POS_b)

        return out, out_b


class de_DRT(nn.Module):
    def __init__(self, num_classes=5):
        super(de_DRT, self).__init__()
        ##Position Calibration Module(subbranch)
        self.vit_pos = PC_vit(img_size=14,patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4,
                                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             drop_path_rate=0.)
        self.resize = Resize([14, 14])
        ##main branch consisting of CA blocks
        self.main_branch = efficientMod_xxs(num_classes=num_classes)
        # self.s_branch = RMT_N4(num_class=num_classes)

    def forward(self, on, apex, off):
        # onset:x1 apex:x5
        B = on.shape[0]
        B_b = apex.shape[0]
        # Position Calibration Module (subbranch)
        # 前分支
        POS = self.vit_pos(self.resize(on)).transpose(1, 2).view(B, 512, 14, 14)
        act = apex - on
        out = self.main_branch(act, POS)
        # 后分支
        act_b = apex - off
        POS_b = self.vit_pos(self.resize(apex)).transpose(1, 2).view(B_b, 512, 14, 14)
        out_b = self.main_branch(act_b, POS_b)

        return out, out_b