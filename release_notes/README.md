# Multi-Task Learning

This repo aims to implement several multi-task learning models and training strategies in PyTorch. The code complements the following works:
> [**Revisting Multi-Task Learning in the Deep Learning Era**](https://arxiv.org/abs/2004.13379)
>
> [Simon Vandenhende](https://twitter.com/svandenh1), [Stamatios Georgoulis](https://twitter.com/stam_g), Marc Proesmans, Dengxin Dai and Luc Van Gool.

> [**MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning**](https://arxiv.org/abs/2001.06902)
>
> [Simon Vandenhende](https://twitter.com/svandenh1), [Stamatios Georgoulis](https://twitter.com/stam_g) and Luc Van Gool.

In the current version (v1), several models have been implemented on NYUD and PASCAL. Notice that there can be small changes in the results compared to the numbers in the papers. A list of differences in the implementation can be found at the bottom of this document. Importantly, the conclusions drawn from comparing the implemented architectures are identical. 

## Version v1

(1) The following datasets and tasks are supported.

| Dataset | Sem. Seg. | Depth | Normals | Edge | Saliency | Human Parts |
|---------|-----------|-------|---------|----------------|----------|-------------|
| PASCAL  |     Y     |   N   |    Y    |       Y        |    Y     |      Y      |
| NYUD    |     Y     |   Y   |    Y    |       Y        |    N     |      N      |

(2) The following models are supported

| Backbone | HRNet-18 | ResNet-50 |
|----------|----------|-----------|
| Single-Task |  Y    |  Y |
| Multi-Task | Y | Y |
| Cross-Stitch | | Y |
| NDDR-CNN | | Y |
| MTAN | | Y |
| PAD-Net | Y | |
| MTI-Net | Y | |

## Version v2

For version v2, we plan the following:

- Include additional models.
- Include the training strategies tested in our survey paper. 
- Possibly include support for Cityscapes.


## Differences with other works

Although the conclusions are the same as in our papers, the numbers can differ a bit.
We list the most important differences below:

- A different augmentation strategy was implemented for NYUDv2 when writing the papers.
Here, we went with the augmentation strategy used in ASTMT.

- The evaluation of the depth prediction is done in an image-wise fashion, while in the papers, we averaged
in a pixel-wise fashion.

- The MTAN model was updated after a discussion with the authors. The results will be updated in the revised version
of our paper as well. 


## FAQ

### When will you release the rest of the code base?  
We are currently working on a major revision of our survey paper. We plan to upload a revised version of the paper in September 2020. Version V2 of the repository is planned to be released around the same time. 
