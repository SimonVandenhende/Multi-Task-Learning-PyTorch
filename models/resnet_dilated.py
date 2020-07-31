#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch.nn as nn


class ResnetDilated(nn.Module):
    """ ResNet backbone with dilated convolutions """
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu
        
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def forward_stage(self, x, stage):
        assert(stage in ['conv','layer1','layer2','layer3','layer4', 'layer1_without_conv'])
        
        if stage == 'conv':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            return x

        elif stage == 'layer1':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            return x
        
        elif stage == 'layer1_without_conv':
            x = self.layer1(x)
            return x

        else: # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer(x)

    def forward_stage_except_last_block(self, x, stage):
        assert(stage in ['layer1','layer2','layer3','layer4'])

        if stage == 'layer1':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1[:-1](x)
            return x

        else: # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer[:-1](x)
    
    def forward_stage_last_block(self, x, stage):
        assert(stage in ['layer1','layer2','layer3','layer4'])

        if stage == 'layer1':
            x = self.layer1[-1](x)
            return x

        else: # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer[-1](x)

