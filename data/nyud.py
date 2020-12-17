# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import sys
import tarfile
import cv2

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

from utils.mypath import MyPath
from utils.utils import mkdir_if_missing
from data.google_drive import download_file_from_google_drive

class NYUD_MT(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes semantic segmentation and depth prediction.

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """

    GOOGLE_DRIVE_ID = '14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw'
    FILE = 'NYUD_MT.tgz'

    def __init__(self,
                 root=MyPath.db_root_dir('NYUD_MT'),
                 download=True,
                 split='val',
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_edge=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 ):

        self.root = root

        if download:
            self._download()

        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'images')
        
        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        _edge_gt_dir = os.path.join(root, 'edge')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals
        self.do_normals = do_normals
        self.normals = []
        _normal_gt_dir = os.path.join(root, 'normals')

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                # Images
                _image = os.path.join(_image_dir, line + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(self.root, _edge_gt_dir, line + '.npy')
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = os.path.join(self.root, _semseg_gt_dir, line + '.png')
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                # Surface Normals
                _normal = os.path.join(self.root, _normal_gt_dir, line + '.npy')
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

                # Depth Prediction
                _depth = os.path.join(self.root, _depth_gt_dir, line + '.npy')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            if _edge.shape != _img.shape[:2]:
                _edge = cv2.resize(_edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['edge'] = _edge

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                print('RESHAPE SEMSEG')
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            if _normals.shape[:2] != _img.shape[:2]:
                _normals = cv2.resize(_normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            if _depth.shape[:2] != _img.shape[:2]:
                print('RESHAPE DEPTH')
                _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_edge(self, index):
        _edge = np.load(self.edges[index]).astype(np.float32)
        return _edge

    def _load_semseg(self, index):
        # Note: We ignore the background class as other related works.
        _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)
        _semseg[_semseg == 0] = 256
        _semseg = _semseg - 1
        return _semseg

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        return _depth

    def _load_normals(self, index):
        _normals = np.load(self.normals[index])
        return _normals

    def _download(self):
        _fpath = os.path.join(MyPath.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(MyPath.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'


def test_mt():
    import torch
    import data.custom_transforms as tr
    import  matplotlib.pyplot as plt 
    from torchvision import transforms
    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.ScaleNRotate(rots=(-2, 2), scales=(.75, 1.25),
                                                    flagvals={'image': cv2.INTER_CUBIC,
                                                              'edge': cv2.INTER_NEAREST,
                                                              'semseg': cv2.INTER_NEAREST,
                                                              'normals': cv2.INTER_LINEAR,
                                                              'depth': cv2.INTER_LINEAR}),
                                    tr.FixedResize(resolutions={'image': (512, 512),
                                                                'edge': (512, 512),
                                                                'semseg': (512, 512),
                                                                'normals': (512, 512),
                                                                'depth': (512, 512)},
                                                   flagvals={'image': cv2.INTER_CUBIC,
                                                             'edge': cv2.INTER_NEAREST,
                                                             'semseg': cv2.INTER_NEAREST,
                                                             'normals': cv2.INTER_LINEAR,
                                                             'depth': cv2.INTER_LINEAR}),
                                    tr.AddIgnoreRegions(),
                                    tr.ToTensor()])
    dataset = NYUD_MT(split='train', transform=transform, retname=True,
                      do_edge=True,
                      do_semseg=True,
                      do_normals=True,
                      do_depth=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=5)

    for i, sample in enumerate(dataloader):
        print(i)
        for j in range(sample['image'].shape[0]):
            f, ax_arr = plt.subplots(5)
            for k in range(len(ax_arr)):
                ax_arr[k].cla()
            ax_arr[0].imshow(np.transpose(sample['image'][j], (1,2,0)))
            ax_arr[1].imshow(sample['edge'][j,0])
            ax_arr[2].imshow(sample['semseg'][j,0]/40)
            ax_arr[3].imshow(np.transpose(sample['normals'][j], (1,2,0)))
            max_depth = torch.max(sample['depth'][j,0][sample['depth'][j,0] != 255]).item()
            ax_arr[4].imshow(sample['depth'][j,0]/max_depth) # Not ideal. Better is to show inverse depth.

            plt.show()
        break


if __name__ == '__main__':
    test_mt()
