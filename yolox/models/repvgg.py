#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import RepVGGBlock, RepBlock, SimSPPF


class RepVGG(nn.Module):
    def __init__(self, dep_mul, wid_mul):
        super().__init__()
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 6), 1)  # 6
        
        # p1
        self.p1 = RepVGGBlock(3, base_channels, 3, 2)
        
        # p2
        self.p2 = nn.Sequential(
            RepVGGBlock(base_channels, base_channels*2, 3, 2),
            RepBlock(base_channels*2, base_channels*2, n=base_depth)
        )

        # p3
        self.p3 = nn.Sequential(
            RepVGGBlock(base_channels*2, base_channels*4, 3, 2),
            RepBlock(base_channels*4, base_channels*4, n=base_depth*2)
        )
        
        # p4
        self.p4 = nn.Sequential(
            RepVGGBlock(base_channels*4, base_channels*8, 3, 2),
            RepBlock(base_channels*8, base_channels*8, n=base_depth*3)
        )
        
        # p5
        self.p5 = nn.Sequential(
            RepVGGBlock(base_channels*8, base_channels*16, 3, 2),
            RepBlock(base_channels*16, base_channels*16, n=base_depth),
            SimSPPF(in_channels=base_channels*16, out_channels=base_channels*16)
        )
        
            
    def forward(self, x):
        x_p1 = self.p1(x)
        
        x_p2 = self.p2(x_p1)
        
        x_p3 = self.p3(x_p2)
        
        x_p4 = self.p4(x_p3)
        
        x_p5 = self.p5(x_p4)
        
        return [x_p1, x_p2, x_p3, x_p4, x_p5]
    
