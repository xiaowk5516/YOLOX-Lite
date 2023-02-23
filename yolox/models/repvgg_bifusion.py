#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .repvgg import RepVGG
from .network_blocks import SimConv, CSPLayer, GSConv, Transpose


class RepVGGBiFusion(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_channels=[256, 512, 1024],
        gs=False,
        act="silu",
    ):
        super().__init__()
        self.in_channels = [int(x * width) for x in in_channels]
        Conv = GSConv if gs else SimConv

        self.backbone = RepVGG(depth, width)    # 9
        
        # top-down f1
        self.f1_p5 = Conv(self.in_channels[2], self.in_channels[1], 1, 1) # 10
        
        # top-down f2
        self.f2_up = Transpose(self.in_channels[1], self.in_channels[1]) # 11
        self.f2_p4 = Conv(self.in_channels[1], self.in_channels[0], 1, 1) # 12
        self.f2_p3 = Conv(self.in_channels[0], self.in_channels[0], 3, 2) # 143
        # Concate(self.in_channels[0], self.in_channels[0], self.in_channels[1])
        self.f2_conv1 = Conv(self.in_channels[1] + self.in_channels[0]*2, self.in_channels[1], 1, 1) # 15
        self.f2_C3 = CSPLayer(self.in_channels[1], self.in_channels[1], round(12 * depth), False, act=act)  # 16
        self.f2_conv2 = Conv(self.in_channels[1], self.in_channels[0], 1, 1) # 17
        
        # top-down f3
        self.f3_up = Transpose(self.in_channels[0], self.in_channels[0]) 
        # Concate(self.in_channels[0], self.in_channels[1])
        self.f3_conv1 = Conv(self.in_channels[1], self.in_channels[0], 1, 1)
        self.f3_C3 = CSPLayer(self.in_channels[0], self.in_channels[0], round(12 * depth), False, act=act)
        
        # bottom-up p2
        self.p2_down = Conv(self.in_channels[0], self.in_channels[0], 3, 2)
        # Concate(self.in_channels[0], self.in_channels[0], self.in_channels[1])
        self.p2_conv = Conv(self.in_channels[2], self.in_channels[1], 1, 1)
        self.p2_C3 = CSPLayer(self.in_channels[1], self.in_channels[1], round(12 * depth), False, act=act)

        # bottom-up p1
        self.p1_down = Conv(self.in_channels[1], self.in_channels[1], 3, 2)
        # Concate(self.in_channels[1], self.in_channels[1])
        self.p1_C3 = CSPLayer(self.in_channels[2], self.in_channels[2], round(12 * depth), False, act=act)
        
        
    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        [p1, p2, p3, p4, p5] = out_features

        # top-down f1
        f1_x = self.f1_p5(p5)
        
        # top-down f2
        f2_up_x = self.f2_up(f1_x)
        f2_p4_x = self.f2_p4(p4)
        f2_p3_x = self.f2_p3(p3)
        f2_x = torch.cat([f2_up_x, f2_p4_x, f2_p3_x], 1)
        f2_x = self.f2_conv1(f2_x)
        f2_x = self.f2_C3(f2_x)
        f2_x = self.f2_conv2(f2_x)
        
        # top-down f3
        f3_up = self.f3_up(f2_x)
        f3_x = torch.cat([f3_up, p3], 1)
        f3_x = self.f3_conv1(f3_x)
        f3_x = self.f3_C3(f3_x)
        
        # bottom-up p2
        p2_down = self.p2_down(f3_x)
        p2_x = torch.cat([p2_down, f2_x, p4], 1)
        p2_x = self.p2_conv(p2_x)
        p2_x = self.p2_C3(p2_x)
        
        # bottom-up p3
        p1_down_x = self.p1_down(p2_x)
        p1_x = torch.cat([p1_down_x, f1_x], 1)
        p1_x = self.p1_C3(p1_x)
        
        outputs = (f3_x, p2_x, p1_x)
        
        return outputs
