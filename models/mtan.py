#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

""" 
    Implementation of MTAN  
    https://arxiv.org/abs/1803.10704 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet, conv1x1, Bottleneck
from models.resnet_dilated import ResnetDilated


class AttentionLayer(nn.Sequential):
    """ 
        Attention layer: Takes a feature representation as input and generates an attention mask 
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(AttentionLayer, self).__init__(
                    nn.Conv2d(in_channels=in_channels, 
                        out_channels=mid_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=mid_channels, 
                        out_channels=out_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(out_channels),
                    nn.Sigmoid())


class RefinementBlock(nn.Sequential):
    """
        Refinement block uses a single Bottleneck layer to refine the features after applying task-specific attention.
    """
    def __init__(self, in_channels, out_channels):
        downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride=1),
                                    nn.BatchNorm2d(out_channels))
        super(RefinementBlock, self).__init__(Bottleneck(in_channels, out_channels//4, downsample=downsample))


class MTAN(nn.Module):
    """ 
        Implementation of MTAN  
        https://arxiv.org/abs/1803.10704 

        Note: The implementation is based on a ResNet backbone.
        
        Argument: 
            backbone: 
                nn.ModuleDict object which contains pre-trained task-specific backbones.
                {task: backbone for task in p.TASKS.NAMES}
        
            heads: 
                nn.ModuleDict object which contains the task-specific heads.
                {task: head for task in p.TASKS.NAMES}
        
            stages: 
                a list of the different stages in the network 
                ['layer1', 'layer2', 'layer3', 'layer4']
 
            channels: 
                dict which contains the number of channels in every stage
            
            downsample:
                dict which tells where to apply 2 x 2 downsampling in the model        

    """
    def __init__(self, p, backbone, heads: nn.ModuleDict, 
                stages: list, channels: dict, downsample: dict): 
        super(MTAN, self).__init__()
        
        # Initiate model                  
        self.tasks = p.TASKS.NAMES
        assert(isinstance(backbone, ResNet) or isinstance(backbone, ResnetDilated))
        self.backbone = backbone
        self.heads = heads
        assert(set(stages) == {'layer1','layer2','layer3','layer4'})
        self.stages = stages
        self.channels = channels

        # Task-specific attention modules
        self.attention_1 = nn.ModuleDict({task: AttentionLayer(channels['layer1'], channels['layer1']//4,
                                                        channels['layer1']) for task in self.tasks})
        self.attention_2 = nn.ModuleDict({task: AttentionLayer(2*channels['layer2'], channels['layer2']//4,
                                                        channels['layer2']) for task in self.tasks})
        self.attention_3 = nn.ModuleDict({task: AttentionLayer(2*channels['layer3'], channels['layer3']//4,
                                                        channels['layer3']) for task in self.tasks})
        self.attention_4 = nn.ModuleDict({task: AttentionLayer(2*channels['layer4'], channels['layer4']//4,
                                                        channels['layer4']) for task in self.tasks})

        # Shared refinement
        self.refine_1 = RefinementBlock(channels['layer1'], channels['layer2'])
        self.refine_2 = RefinementBlock(channels['layer2'], channels['layer3'])
        self.refine_3 = RefinementBlock(channels['layer3'], channels['layer4'])
        
        # Downsample
        self.downsample = {stage: nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity() for stage, downsample in downsample.items()}


    def forward(self, x):
        img_size = x.size()[-2:]
        
        # Shared backbone
        # In case of ResNet we apply attention over the last bottleneck in each block.
        # Other backbones can be included by implementing the forward_stage_except_last_block
        # and forward_stage_last_block
        u_1_b = self.backbone.forward_stage_except_last_block(x, 'layer1')
        u_1_t = self.backbone.forward_stage_last_block(u_1_b, 'layer1')  

        u_2_b = self.backbone.forward_stage_except_last_block(u_1_t, 'layer2')
        u_2_t = self.backbone.forward_stage_last_block(u_2_b, 'layer2')  
        
        u_3_b = self.backbone.forward_stage_except_last_block(u_2_t, 'layer3')
        u_3_t = self.backbone.forward_stage_last_block(u_3_b, 'layer3')  
        
        u_4_b = self.backbone.forward_stage_except_last_block(u_3_t, 'layer4')
        u_4_t = self.backbone.forward_stage_last_block(u_4_b, 'layer4') 

        ## Apply attention over the first Resnet Block -> Over last bottleneck
        a_1_mask = {task: self.attention_1[task](u_1_b) for task in self.tasks}
        a_1 = {task: a_1_mask[task] * u_1_t for task in self.tasks}
        a_1 = {task: self.downsample['layer1'](self.refine_1(a_1[task])) for task in self.tasks}
        
        ## Apply attention over the second Resnet Block -> Over last bottleneck
        a_2_mask = {task: self.attention_2[task](torch.cat((u_2_b, a_1[task]), 1)) for task in self.tasks}
        a_2 = {task: a_2_mask[task] * u_2_t for task in self.tasks}
        a_2 = {task: self.downsample['layer2'](self.refine_2(a_2[task])) for task in self.tasks}
        
        ## Apply attention over the third Resnet Block -> Over last bottleneck
        a_3_mask = {task: self.attention_3[task](torch.cat((u_3_b, a_2[task]), 1)) for task in self.tasks}
        a_3 = {task: a_3_mask[task] * u_3_t for task in self.tasks}
        a_3 = {task: self.downsample['layer3'](self.refine_3(a_3[task])) for task in self.tasks}
        
        ## Apply attention over the last Resnet Block -> No more refinement since we have task-specific
        ## heads anyway. Testing with extra self.refin_4 did not result in any improvements btw.
        a_4_mask = {task: self.attention_4[task](torch.cat((u_4_b, a_3[task]), 1)) for task in self.tasks}
        a_4 = {task: a_4_mask[task] * u_4_t for task in self.tasks}

        # Task-specific heads
        out = {task: self.heads[task](a_4[task]) for task in self.tasks}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear') for task in self.tasks} 
    
        return out
