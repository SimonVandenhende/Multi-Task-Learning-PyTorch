#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

""" 
    Implementation of cross-stitch networks
    https://arxiv.org/abs/1604.03539
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseMultiply(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWiseMultiply, self).__init__()
        self.param = nn.Parameter(torch.FloatTensor(num_channels), requires_grad=True)

    def init_value(self, value):
        with torch.no_grad():
            self.param.data.fill_(value)

    def forward(self, x):
        return torch.mul(self.param.view(1,-1,1,1), x)


class CrossStitchUnit(nn.Module):
    def __init__(self, tasks, num_channels, alpha, beta):
        super(CrossStitchUnit, self).__init__()
        self.cross_stitch_unit = nn.ModuleDict({t: nn.ModuleDict({t: ChannelWiseMultiply(num_channels) for t in tasks}) for t in tasks})

        for t_i in tasks:
            for t_j in tasks:
                if t_i == t_j:
                    self.cross_stitch_unit[t_i][t_j].init_value(alpha)
                else:
                    self.cross_stitch_unit[t_i][t_j].init_value(beta)

    def forward(self, task_features):
        out = {}
        for t_i in task_features.keys():
            prod = torch.stack([self.cross_stitch_unit[t_i][t_j](task_features[t_j]) for t_j in task_features.keys()])
            out[t_i] = torch.sum(prod, dim=0)
        return out
           

class CrossStitchNetwork(nn.Module):
    """ 
        Implementation of cross-stitch networks.
        We insert a cross-stitch unit, to combine features from the task-specific backbones
        after every stage.
       
        Argument: 
            backbone: 
                nn.ModuleDict object which contains pre-trained task-specific backbones.
                {task: backbone for task in p.TASKS.NAMES}
        
            heads: 
                nn.ModuleDict object which contains the task-specific heads.
                {task: head for task in p.TASKS.NAMES}
        
            stages: 
                list of stages where we instert a cross-stitch unit between the task-specific backbones.
                Note: the backbone modules require a method 'forward_stage' to get feature representations
                at the respective stages.
        
            channels: 
                dict which contains the number of channels in every stage
        
            alpha, beta: 
                floats for initializing cross-stitch units (see paper)
        
    """
    def __init__(self, p, backbone: nn.ModuleDict, heads: nn.ModuleDict, 
                    stages: list, channels: dict, alpha: float, beta: float):
        super(CrossStitchNetwork, self).__init__()

        # Tasks, backbone and heads
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.heads = heads
        self.stages = stages

        # Cross-stitch units
        self.cross_stitch = nn.ModuleDict({stage: CrossStitchUnit(self.tasks, channels[stage], alpha, beta) for stage in stages})


    def forward(self, x):
        img_size = x.size()[-2:]
        x = {task: x for task in self.tasks} # Feed as input to every single-task network

        # Backbone
        for stage in self.stages:
    
            # Forward through next stage of task-specific network
            for task in self.tasks:
                x[task] = self.backbone[task].forward_stage(x[task], stage)
            
            # Cross-stitch the task-specific features
            x = self.cross_stitch[stage](x)

        # Task-specific heads
        out = {task: self.heads[task](x[task]) for task in self.tasks}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear') for task in self.tasks} 

        return out
