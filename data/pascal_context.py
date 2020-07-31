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
import json
import cv2

import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
from skimage.morphology import thin
from six.moves import urllib

from utils.mypath import MyPath, PROJECT_ROOT_DIR

class PASCALContext(data.Dataset):
    """
    PASCAL-Context dataset, for multiple tasks
    Included tasks:
        1. Edge detection,
        2. Semantic Segmentation,
        3. Human Part Segmentation,
        4. Surface Normal prediction (distilled),
        5. Saliency (distilled)
    """

    URL = 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz'
    FILE = 'PASCAL_MT.tgz'

    HUMAN_PART = {1: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 1,
                      'lhand': 1, 'llarm': 1, 'llleg': 1, 'luarm': 1, 'luleg': 1, 'mouth': 1,
                      'neck': 1, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 1,
                      'rhand': 1, 'rlarm': 1, 'rlleg': 1, 'ruarm': 1, 'ruleg': 1, 'torso': 1},
                  4: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 4,
                      'lhand': 3, 'llarm': 3, 'llleg': 4, 'luarm': 3, 'luleg': 4, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 4,
                      'rhand': 3, 'rlarm': 3, 'rlleg': 4, 'ruarm': 3, 'ruleg': 4, 'torso': 2},
                  6: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 6,
                      'lhand': 4, 'llarm': 4, 'llleg': 6, 'luarm': 3, 'luleg': 5, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 6,
                      'rhand': 4, 'rlarm': 4, 'rlleg': 6, 'ruarm': 3, 'ruleg': 5, 'torso': 2},
                  14: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 14,
                       'lhand': 8, 'llarm': 7, 'llleg': 13, 'luarm': 6, 'luleg': 12, 'mouth': 1,
                       'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 11,
                       'rhand': 5, 'rlarm': 4, 'rlleg': 10, 'ruarm': 3, 'ruleg': 9, 'torso': 2}
                  }

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    CONTEXT_CATEGORY_LABELS = [0,
                               2, 23, 25, 31, 34,
                               45, 59, 65, 72, 98,
                               397, 113, 207, 258, 284,
                               308, 347, 368, 416, 427]

    def __init__(self,
                 root=MyPath.db_root_dir('PASCAL_MT'),
                 download=True,
                 split='val',
                 transform=None,
                 area_thres=0,
                 retname=True,
                 overfit=False,
                 do_edge=True,
                 do_human_parts=False,
                 do_semseg=False,
                 do_normals=False,
                 do_sal=False,
                 num_human_parts=6,
                 ):

        self.root = root
        if download:
            self._download()

        image_dir = os.path.join(self.root, 'JPEGImages')
        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.area_thres = area_thres
        self.retname = retname

        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        edge_gt_dir = os.path.join(self.root, 'pascal-context', 'trainval')

        # Semantic Segmentation
        self.do_semseg = do_semseg
        self.semsegs = []

        # Human Part Segmentation
        self.do_human_parts = do_human_parts
        part_gt_dir = os.path.join(self.root, 'human_parts')
        self.parts = []
        self.human_parts_category = 15
        print(PROJECT_ROOT_DIR)
        self.cat_part = json.load(open(os.path.join(PROJECT_ROOT_DIR, 'data/db_info/pascal_part.json'), 'r'))
        self.cat_part["15"] = self.HUMAN_PART[num_human_parts]
        self.parts_file = os.path.join(os.path.join(self.root, 'ImageSets', 'Parts'),
                                       ''.join(self.split) + '.txt')

        # Surface Normal Estimation
        self.do_normals = do_normals
        _normal_gt_dir = os.path.join(self.root, 'normals_distill')
        self.normals = []
        if self.do_normals:
            with open(os.path.join(PROJECT_ROOT_DIR, 'data/db_info/nyu_classes.json')) as f:
                cls_nyu = json.load(f)
            with open(os.path.join(PROJECT_ROOT_DIR, 'data/db_info/context_classes.json')) as f:
                cls_context = json.load(f)

            self.normals_valid_classes = []
            for cl_nyu in cls_nyu:
                if cl_nyu in cls_context and cl_nyu != 'unknown':
                    self.normals_valid_classes.append(cls_context[cl_nyu])

            # Custom additions due to incompatibilities
            self.normals_valid_classes.append(cls_context['tvmonitor'])

        # Saliency
        self.do_sal = do_sal
        _sal_gt_dir = os.path.join(self.root, 'sal_distill')
        self.sals = []

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(self.root, 'ImageSets', 'Context')

        self.im_ids = []
        self.images = []

        print("Initializing dataloader for PASCAL {} set".format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                # Images
                _image = os.path.join(image_dir, line + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(edge_gt_dir, line + ".mat")
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = self._get_semseg_fname(line)
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                # Human Parts
                _human_part = os.path.join(self.root, part_gt_dir, line + ".mat")
                assert os.path.isfile(_human_part)
                self.parts.append(_human_part)

                _normal = os.path.join(self.root, _normal_gt_dir, line + ".png")
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

                _sal = os.path.join(self.root, _sal_gt_dir, line + ".png")
                assert os.path.isfile(_sal)
                self.sals.append(_sal)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_human_parts:
            assert (len(self.images) == len(self.parts))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))
        if self.do_sal:
            assert (len(self.images) == len(self.sals))

        if not self._check_preprocess_parts():
            print('Pre-processing PASCAL dataset for human parts, this will take long, but will be done only once.')
            self._preprocess_parts()

        if self.do_human_parts:
            # Find images which have human parts
            self.has_human_parts = []
            for ii in range(len(self.im_ids)):
                if self.human_parts_category in self.part_obj_dict[self.im_ids[ii]]:
                    self.has_human_parts.append(1)
                else:
                    self.has_human_parts.append(0)

            # If the other tasks are disabled, select only the images that contain human parts, to allow batching
            if not self.do_edge and not self.do_semseg and not self.do_sal and not self.do_normals:
                print('Ignoring images that do not contain human parts')
                for i in range(len(self.parts) - 1, -1, -1):
                    if self.has_human_parts[i] == 0:
                        del self.im_ids[i]
                        del self.images[i]
                        del self.parts[i]
                        del self.has_human_parts[i]
            print('Number of images with human parts: {:d}'.format(np.sum(self.has_human_parts)))

        #  Overfit to n_of images
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]
            if self.do_edge:
                self.edges = self.edges[:n_of]
            if self.do_semseg:
                self.semsegs = self.semsegs[:n_of]
            if self.do_human_parts:
                self.parts = self.parts[:n_of]
            if self.do_normals:
                self.normals = self.normals[:n_of]
            if self.do_sal:
                self.sals = self.sals[:n_of]

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

        if self.do_human_parts:
            _human_parts, _ = self._load_human_parts(index)
            if _human_parts.shape != _img.shape[:2]:
                _human_parts = cv2.resize(_human_parts, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['human_parts'] = _human_parts

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals_distilled(index)
            if _normals.shape[:2] != _img.shape[:2]:
                _normals = cv2.resize(_normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_sal:
            _sal = self._load_sal_distilled(index)
            if _sal.shape[:2] != _img.shape[:2]:
                _sal = cv2.resize(_sal, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['sal'] = _sal

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
        # Read Target object
        _tmp = sio.loadmat(self.edges[index])
        _edge = cv2.Laplacian(_tmp['LabelMap'], cv2.CV_64F)
        _edge = thin(np.abs(_edge) > 0).astype(np.float32)
        return _edge

    def _load_human_parts(self, index):
        if self.has_human_parts[index]:

            # Read Target object
            _part_mat = sio.loadmat(self.parts[index])['anno'][0][0][1][0]

            _inst_mask = _target = None

            for _obj_ii in range(len(_part_mat)):

                has_human = _part_mat[_obj_ii][1][0][0] == self.human_parts_category
                has_parts = len(_part_mat[_obj_ii][3]) != 0

                if has_human and has_parts:
                    if _inst_mask is None:
                        _inst_mask = _part_mat[_obj_ii][2].astype(np.float32)
                        _target = np.zeros(_inst_mask.shape)
                    else:
                        _inst_mask = np.maximum(_inst_mask, _part_mat[_obj_ii][2].astype(np.float32))

                    n_parts = len(_part_mat[_obj_ii][3][0])
                    for part_i in range(n_parts):
                        cat_part = str(_part_mat[_obj_ii][3][0][part_i][0][0])
                        mask_id = self.cat_part[str(self.human_parts_category)][cat_part]
                        mask = _part_mat[_obj_ii][3][0][part_i][1].astype(bool)
                        _target[mask] = mask_id

            if _target is not None:
                _target, _inst_mask = _target.astype(np.float32), _inst_mask.astype(np.float32)
            else:
                _target, _inst_mask = np.zeros((512, 512), dtype=np.float32), np.zeros((512, 512), dtype=np.float32)

            return _target, _inst_mask

        else:
            return np.zeros((512, 512), dtype=np.float32), np.zeros((512, 512), dtype=np.float32)

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)

        return _semseg

    def _load_normals_distilled(self, index):
        _tmp = np.array(Image.open(self.normals[index])).astype(np.float32)
        _tmp = 2.0 * _tmp / 255.0 - 1.0

        labels = sio.loadmat(os.path.join(self.root, 'pascal-context', 'trainval', self.im_ids[index] + '.mat'))
        labels = labels['LabelMap']

        _normals = np.zeros(_tmp.shape, dtype=np.float)

        for x in np.unique(labels):
            if x in self.normals_valid_classes:
                _normals[labels == x, :] = _tmp[labels == x, :]

        return _normals

    def _load_sal_distilled(self, index):
        _sal = np.array(Image.open(self.sals[index])).astype(np.float32) / 255.
        _sal = (_sal > 0.5).astype(np.float32)

        return _sal

    def _get_semseg_fname(self, fname):
        fname_voc = os.path.join(self.root, 'semseg', 'VOC12', fname + '.png')
        fname_context = os.path.join(self.root, 'semseg', 'pascal-context', fname + '.png')
        if os.path.isfile(fname_voc):
            seg = fname_voc
        elif os.path.isfile(fname_context):
            seg = fname_context
        else:
            seg = None
            print('Segmentation for im: {} was not found'.format(fname))

        return seg

    def _check_preprocess_parts(self):
        _obj_list_file = self.parts_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.part_obj_dict = json.load(open(_obj_list_file, 'r'))

            return list(np.sort([str(x) for x in self.part_obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess_parts(self):
        self.part_obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            if ii % 100 == 0:
                print("Processing image: {}".format(ii))
            part_mat = sio.loadmat(
                os.path.join(self.root, 'human_parts', '{}.mat'.format(self.im_ids[ii])))
            n_obj = len(part_mat['anno'][0][0][1][0])

            # Get the categories from these objects
            _cat_ids = []
            for jj in range(n_obj):
                obj_area = np.sum(part_mat['anno'][0][0][1][0][jj][2])
                obj_cat = int(part_mat['anno'][0][0][1][0][jj][1])
                if obj_area > self.area_thres:
                    _cat_ids.append(int(part_mat['anno'][0][0][1][0][jj][1]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.part_obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.parts_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.part_obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(
                    ',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.part_obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing for parts finished')

    def _download(self):
        _fpath = os.path.join(MyPath.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading ' + self.URL + ' to ' + _fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)

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
        return 'PASCAL_MT(split=' + str(self.split) + ')'


def test_all():
    import matplotlib.pyplot as plt
    import torch
    import data.custom_transforms as tr
    from torchvision import transforms
    from utils.custom_collate import collate_mil

    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.ScaleNRotate(rots=(-90, 90), scales=(1., 1.),
                                                    flagvals={'image': cv2.INTER_CUBIC,
                                                              'edge': cv2.INTER_NEAREST,
                                                              'semseg': cv2.INTER_NEAREST,
                                                              'human_parts': cv2.INTER_NEAREST,
                                                              'normals': cv2.INTER_CUBIC,
                                                              'sal': cv2.INTER_NEAREST}),
                                    tr.FixedResize(resolutions={'image': (512, 512),
                                                                'edge': (512, 512),
                                                                'semseg': (512, 512),
                                                                'human_parts': (512, 512),
                                                                'normals': (512, 512),
                                                                'sal': (512, 512)},
                                                   flagvals={'image': cv2.INTER_CUBIC,
                                                             'edge': cv2.INTER_NEAREST,
                                                             'semseg': cv2.INTER_NEAREST,
                                                             'human_parts': cv2.INTER_NEAREST,
                                                             'normals': cv2.INTER_CUBIC,
                                                             'sal': cv2.INTER_NEAREST}),
                                    tr.AddIgnoreRegions(),
                                    tr.ToTensor()])
    dataset = PASCALContext(split='train', transform=transform, retname=True,
                            do_edge=True,
                            do_semseg=True,
                            do_human_parts=True,
                            do_normals=True,
                            do_sal=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        print(i)
        for j in range(sample['image'].shape[0]):
            f, ax_arr = plt.subplots(2, 3)

            for k in range(len(ax_arr)):
                for l in range(len(ax_arr[k])):
                    ax_arr[k][l].cla()

            ax_arr[0][0].imshow(np.transpose(sample['image'][j], (1, 2, 0)))
            ax_arr[0][1].imshow(np.transpose(sample['edge'][j], (1, 2, 0))[:, :, 0])
            ax_arr[0][2].imshow(np.transpose(sample['semseg'][j], (1, 2, 0))[:, :, 0] / 20.)
            ax_arr[1][0].imshow(np.transpose(sample['human_parts'][j], (1, 2, 0))[:, :, 0] / 6.)
            ax_arr[1][1].imshow(np.transpose(sample['normals'][j], (1, 2, 0)))
            ax_arr[1][2].imshow(np.transpose(sample['sal'][j], (1, 2, 0))[:, :, 0])

            plt.show()
        break


if __name__ == '__main__':
    test_all()
