# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：LDF baseline model
# Author: Yinda XU

import torch
from torch import Tensor
from torch import nn
from .resnet_mod import resnet_mod56

class LDF_baseline(nn.Module):
    def __init__(self, arr_ClassAttr):
        super(LDF_baseline, self).__init__()
        num_classes, dim_ClassAttr = arr_ClassAttr.shape
        self.resnet_mod = resnet_mod56(num_classes=dim_ClassAttr)
        self.ts_ClassAttr_t = Tensor(arr_ClassAttr).transpose(0, 1).cuda()
    def forward(self, X):
        X = self.resnet_mod(X)
        ret = torch.matmul(X, self.ts_ClassAttr_t)
        return ret
