#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of NDDR-CNN
    https://arxiv.org/abs/1801.08297
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NDDRLayer(nn.Module):
    def __init__(self, tasks, channels, alpha, beta):
        super(NDDRLayer, self).__init__()
        self.tasks = tasks
        self.layer = nn.ModuleDict({task: nn.Sequential(
                                        nn.Conv2d(len(tasks) * channels, channels, 1, 1, 0, bias=False), nn.BatchNorm2d(channels, momentum=0.05), nn.ReLU()) for task in self.tasks}) # Momentum set as NDDR-CNN repo
        
        # Initialize
        for i, task in enumerate(self.tasks):
            layer = self.layer[task]
            t_alpha = torch.diag(torch.FloatTensor([alpha for _ in range(channels)])) # C x C
            t_beta = torch.diag(torch.FloatTensor([beta for _ in range(channels)])).repeat(1, len(self.tasks)) # C x (C x T)
            t_alpha = t_alpha.view(channels, channels, 1, 1)
            t_beta = t_beta.view(channels, channels * len(self.tasks), 1, 1)
    
            layer[0].weight.data.copy_(t_beta)
            layer[0].weight.data[:,int(i*channels):int((i+1)*channels)].copy_(t_alpha)
            layer[1].weight.data.fill_(1.0)
            layer[1].bias.data.fill_(0.0)


    def forward(self, x):
        x = torch.cat([x[task] for task in self.tasks], 1) # Use self.tasks to retain order!
        output = {task: self.layer[task](x) for task in self.tasks}
        return output


class NDDRCNN(nn.Module):
    """ 
        Implementation of NDDR-CNN.
        We insert a nddr-layer to fuse the features from the task-specific backbones after every
        stage.
       
        Argument: 
            backbone: 
                nn.ModuleDict object which contains pre-trained task-specific backbones.
                {task: backbone for task in p.TASKS.NAMES}
        
            heads: 
                nn.ModuleDict object which contains the task-specific heads.
                {task: head for task in p.TASKS.NAMES}

            all_stages:        

            nddr_stages: 
                list of stages where we instert a nddr-layer between the task-specific backbones.
                Note: the backbone modules require a method 'forward_stage' to get feature representations
                at the respective stages.
        
            channels: 
                dict which contains the number of channels in every stage
        
            alpha, beta: 
                floats for initializing the nddr-layers (see paper)
        
    """
    def __init__(self, p, backbone: nn.ModuleDict, heads: nn.ModuleDict, 
                    all_stages: list, nddr_stages: list, channels: dict, alpha: float, beta: float):
        super(NDDRCNN, self).__init__()
        
        # Tasks, backbone and heads
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.heads = heads
        self.all_stages = all_stages
        self.nddr_stages = nddr_stages

        # NDDR-CNN units
        self.nddr = nn.ModuleDict({stage: NDDRLayer(self.tasks, channels[stage], alpha, beta) for stage in nddr_stages})


    def forward(self, x):
        img_size = x.size()[-2:]
        x = {task: x for task in self.tasks} # Feed as input to every single-task network

        # Backbone
        for stage in self.all_stages:
    
            # Forward through next stage of task-specific network
            for task in self.tasks:
                x[task] = self.backbone[task].forward_stage(x[task], stage)
            
            if stage in self.nddr_stages:
                # Fuse task-specific features through NDDR-layer.
                x = self.nddr[stage](x)

        # Task-specific heads
        out = {task: self.heads[task](x[task]) for task in self.tasks}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear') for task in self.tasks} 

        return out
