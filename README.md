# Multi-Task Learning

This repo aims to implement several multi-task learning models and training strategies in PyTorch. The code base complements the following works: 
> [**Revisting Multi-Task Learning in the Deep Learning Era**](https://arxiv.org/abs/2004.13379)
>
> [Simon Vandenhende](https://twitter.com/svandenh1), [Stamatios Georgoulis](https://twitter.com/stam_g), Marc Proesmans, Dengxin Dai and Luc Van Gool.

> [**MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning**](https://arxiv.org/abs/2001.06902)
>
> [Simon Vandenhende](https://twitter.com/svandenh1), [Stamatios Georgoulis](https://twitter.com/stam_g) and Luc Van Gool.

<strong> Note: </strong> Have a look at the release notes to see what architectures and datasets are currently supported. We are working on a revision of our survey. Missing models or training strategies will be uploaded together with the revised version of the paper (est. time: September 2020). 

## Installation
The code runs with recent Pytorch version, e.g. 1.4.
Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/), the most important packages can be installed as:
```shell
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install imageio scikit-image		   	   # Image operations
conda install -c conda-forge opencv		           # OpenCV
conda install pyyaml easydict                 		   # Configurations
conda install termcolor                       		   # Colorful print statements
```
We refer to the `requirements.txt` file for an overview of the package versions in our own environment.

## Usage

### Setup 
The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/pascal/`.
- Specify the output directory in `configs/your_env.yml`. All results will be stored under this directory.
- The [seism](https://github.com/jponttuset/seism) repository is needed to perform the edge evaluation. See the README in `./evaluation/seism/`.
- If you want to use the HRNet backbones, please download the pre-trained weights [here](https://github.com/HRNet/HRNet-Image-Classification). 
The provided config files use an HRNet-18 backbone. Download the `hrnet_w18_small_model_v2.pth` and save it to the directory `./models/pretrained_models/`.

The datasets will be downloaded automatically to the specified paths when running the code for the first time.

### Train model
The configuration files to train the model can be found in the `configs/` directory. For example, run the following commands to train a model.

```shell
python main.py --config_env configs/env.yml --config_exp configs/$DATASET/$MODEL.yml
```

### Evaluate model
The best model is evaluated at the end of training. The multi-task evaluation criterion is based on Equation 10 from our survey paper and requires to pre-train a set of single-tasking networks first. It is possible to only validate the model during the last 10 epochs to speed-up training by adding the following line to your config file:

```python
eval_final_10_epochs_only: True
``` 

## Support
The following datasets and tasks are supported.

| Dataset | Sem. Seg. | Depth | Normals | Edge | Saliency | Human Parts |
|---------|-----------|-------|---------|----------------|----------|-------------|
| PASCAL  |     Y     |   N   |    Y    |       Y        |    Y     |      Y      |
| NYUD    |     Y     |   Y   |    Y    |       Y        |    N     |      N      |


## Results
The following results were obtained by running the included config files.

### HRNet-18 backbone (NYUD)

Models using surface normals estimation and edge detection as auxilary tasks are indicated between brackets.

| Model        | Seg. (mIoU) | Depth (rmse) | MTL Perf. (%)|
|--------------|-------------|--------------|----------|
| Single-Task  | 34.5        | 0.610   	    | + 0.00 |
| Multi-Task   | 33.9        | 0.610        | - 0.75 |
| PAD-Net      | 35.4	     | 0.614        | + 1.13 |	       |
| PAD-Net (E+N)| 35.5	     | 0.593	    | + 2.90 |
| MTI-Net      | 36.2        | 0.563        | + 6.36 |   
| MTI-Net (E+N)| 37.6        | 0.539        | + 10.32 |

### ResNet-50 backbone (NYUD)

| Model        | Seg. (mIoU) | Depth (rmse) | MTL Perf. (%)|
|--------------|-------------|--------------|----------|
| Single-Task  | 40.1 	     | 0.571        | + 0.00 |
| Multi-Task   | 39.8        | 0.573        | - 0.55 |
| Cross-stitch | 39.9        | 0.565        | + 0.26 |  
| NDDR-CNN     | running     |              |        |
| MTAN         | 40.0 	     | 0.572 	    | - 0.17 |

### HRNet-18 backbone (PASCAL)

| Model	          | Seg. (mIoU) | Parts (mIoU) | Sal (mIoU) | Edge (odsF) | Norm (mean) | MTL Perf. (%) |
|-----------------|-------------|--------------|------------|-------------|-------------|-----------|
| Single-Task     | 59.4	| 60.3 	       | 67.0       | 69.2        | 14.6        | + 0.00 |
| Multi-Task (small) | 56.3	| 60.4 	       | 65.8	    | -           | -           | - 2.26 |
| Multi-Task (all) | 55.4	| 59.4	       | 65.4       | 71.6	  | 15.0	| - 1.98 | 
| PAD-Net (small) | 52.7        | 60.5         | 66.0       | -           | -           | - 4.08 |
| PAD-Net (all) | running | | | | | | 
| MTI-Net (small) | 63.1 	| 62.1         | 67.2	    | -		  | -           | + 3.20 | 
| MTI-Net (all) | 62.8 		| 62.2	       | 67.4	    | 73.0        | 14.8        | + 2.72 | 


## References
This code repository is heavily based on the [ASTMT](https://github.com/facebookresearch/astmt) repository. In particular, the evaluation and dataloaders were taken from there.

 
## Citation
If you find this repo useful for your research, please consider citing the following works:

```
@article{vandenhende2020revisiting,
  title={Revisiting Multi-Task Learning in the Deep Learning Era},
  author={Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Dai, Dengxin and Van Gool, Luc},
  journal={arXiv preprint arXiv:2004.13379},
  year={2020}
}

@article{vandenhende2020mti,
  title={MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning},
  author={Vandenhende, Simon and Georgoulis, Stamatios and Van Gool, Luc},
  journal={arXiv preprint arXiv:2001.06902},
  year={2020}
}

@InProceedings{MRK19,
  Author    = {Kevis-Kokitsi Maninis and Ilija Radosavovic and Iasonas Kokkinos},
  Title     = {Attentive Single-Tasking of Multiple Tasks},
  Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  Year      = {2019}
}

@article{pont2015supervised,
  title={Supervised evaluation of image segmentation and object proposal techniques},
  author={Pont-Tuset, Jordi and Marques, Ferran},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={38},
  number={7},
  pages={1465--1478},
  year={2015},
  publisher={IEEE}
}
```

## FAQ
Have a look at the release notes for additional remarks.

## License
This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).

## Acknoledgements
The authors acknowledge support by Toyota via the TRACE project and MACCHINA (KULeuven, C14/18/065).
