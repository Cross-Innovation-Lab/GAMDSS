import torch
from torch import nn
from torchvision.transforms import Resize
from functools import partial
from models.Spatial_vit import VisionTransformer
from models.RMT import *

# 前模型
class DRT(nn.Module):
    def __init__(self, num_classes=5):
        super(DRT, self).__init__()
        ##Position Calibration Module(subbranch)
        self.vit_pos = VisionTransformer(img_size=14,
        patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize = Resize([14, 14])
        ##main branch consisting of CA blocks
        self.main_branch = RMT_T3(num_class=num_classes)
        # self.s_branch = RMT_N4(num_class=num_classes)
    def forward(self, on, apex, off):
        # onset:x1 apex:x5
        B = on.shape[0]
        # Position Calibration Module (subbranch)
        POS = self.vit_pos(self.resize(on)).transpose(1, 2).view(B, 512, 14, 14)
        act = apex - on
        # act_b = apex - off
        # s = self.s_branch(act_b, POS)
        out = self.main_branch(act, POS)
        # out_b = self.main_branch(act_b, POS)

        # return out, out_b
        return out

# 前加后模型
class BDRT(nn.Module):
    def __init__(self, num_classes=5):
        super(BDRT, self).__init__()
        ##Position Calibration Module(subbranch)
        self.vit_pos = VisionTransformer(img_size=14,
        patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize = Resize([14, 14])
        ##main branch consisting of CA blocks
        self.main_branch = RMT_T3(num_class=num_classes)
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

# 后模型
class ADRT(nn.Module):
    def __init__(self, num_classes=5):
        super(ADRT, self).__init__()
        ##Position Calibration Module(subbranch)
        self.vit_pos = VisionTransformer(img_size=14,
        patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize = Resize([14, 14])
        ##main branch consisting of CA blocks
        self.main_branch = RMT_T3(num_class=num_classes)
        # self.s_branch = RMT_N4(num_class=num_classes)
    def forward(self, on, apex, off):
        # onset:x1 apex:x5
        B = on.shape[0]
        B_b = apex.shape[0]
        # Position Calibration Module (subbranch)
        # 前分支
        # POS = self.vit_pos(self.resize(on)).transpose(1, 2).view(B, 512, 14, 14)
        # act = apex - on
        # out = self.main_branch(act, POS)
        # 后分支
        act_b = apex - off
        POS_b = self.vit_pos(self.resize(apex)).transpose(1, 2).view(B_b, 512, 14, 14)
        out_b = self.main_branch(act_b, POS_b)

        # return out, out_b
        return out_b
