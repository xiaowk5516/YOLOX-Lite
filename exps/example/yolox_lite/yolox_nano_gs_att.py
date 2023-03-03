#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (640, 640)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (640, 640)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.eval_interval = 1
        self.max_epoch = 330
        self.no_aug_epochs = 30
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, RepVGGPAFPN, RepGSAttFusion
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = RepGSAttFusion(
                self.depth, self.width, in_channels=in_channels,
                gs=True,
                act="relu"
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act="relu"
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
