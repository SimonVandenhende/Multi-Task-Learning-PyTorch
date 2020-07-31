#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of PAD-Net.
    https://arxiv.org/abs/1805.04409
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import Bottleneck
from models.layers import SEBlock, SABlock


class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """
    def __init__(self, p, tasks, input_channels, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__() 
        self.tasks = tasks 
        layers = {}
        conv_out = {}
        
        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                    stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = Bottleneck(input_channels, intermediate_channels//4, downsample=downsample)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels//4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = conv_out_

        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)


    def forward(self, x):
        out = {}
        
        for task in self.tasks:
            out['features_%s' %(task)] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' %(task)])
        
        return out 


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}
        
        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)


    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}
        out = {t: x['features_%s' %(t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out


class PADNet(nn.Module):
    def __init__(self, p, backbone, backbone_channels):
        super(PADNet, self).__init__()
        # General
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.channels = backbone_channels

        # Backbone
        self.backbone = backbone

        # Task-specific heads for initial prediction 
        self.initial_task_prediction_heads = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels)

        # Multi-modal distillation
        self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, 256)

        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(256, 256//4, downsample=None)
            bottleneck2 = Bottleneck(256, 256//4, downsample=None)
            conv_out_ = nn.Conv2d(256, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)
    

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        
        # Backbone
        x = self.backbone(x)

        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(x)
        for task in self.auxilary_tasks:
            out['initial_%s' %(task)] = x[task]
 
        # Refine features through multi-modal distillation
        x = self.multi_modal_distillation(x)

        # Make final prediction with task-specific heads
        for task in self.tasks:
            out[task] = F.interpolate(self.heads[task](x[task]), img_size, mode='bilinear')

        return out
