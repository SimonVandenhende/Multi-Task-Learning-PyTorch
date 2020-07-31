# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import os.path
import glob
import json
import numpy as np
import torch
from PIL import Image

VOC_CATEGORY_NAMES = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


NYU_CATEGORY_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair',
                      'sofa', 'table', 'door', 'window', 'bookshelf',
                      'picture', 'counter', 'blinds', 'desk', 'shelves',
                      'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
                      'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                      'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
                      'person', 'night stand', 'toilet', 'sink', 'lamp',
                      'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']


def eval_semseg(loader, folder, n_classes=20, has_bg=True):

    n_classes = n_classes + int(has_bg)

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32)

        gt = sample['semseg']
        valid = (gt != 255)

        if mask.shape != gt.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # TP, FP, and FN evaluation
        for i_part in range(0, n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (mask == i_part)
            tp[i_part] += np.sum(tmp_gt & tmp_pred & valid)
            fp[i_part] += np.sum(~tmp_gt & tmp_pred & valid)
            fn[i_part] += np.sum(tmp_gt & ~tmp_pred & valid)

    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)

    return eval_result


class SemsegMeter(object):
    def __init__(self, database):
        if database == 'PASCALContext':
            n_classes = 20
            cat_names = VOC_CATEGORY_NAMES
            has_bg = True
             
        elif database == 'NYUD':
            n_classes = 40
            cat_names = NYU_CATEGORY_NAMES
            has_bg = False
        
        else:
            raise NotImplementedError
        
        self.n_classes = n_classes + int(has_bg)
        self.cat_names = cat_names
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.squeeze()
        gt = gt.squeeze()
        valid = (gt != 255)
    
        for i_part in range(0, self.n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes
            
    def get_score(self, verbose=True):
        jac = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)


        if verbose:
            print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
            class_IoU = eval_result['jaccards_all_categs']
            for i in range(len(class_IoU)):
                spaces = ''
                for j in range(0, 20 - len(self.cat_names[i])):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result


def eval_semseg_predictions(database, save_dir, overfit=False):
    """ Evaluate the segmentation maps that are stored in the save dir """

    # Dataloaders
    if database == 'PASCALContext':
        from data.pascal_context import PASCALContext
        n_classes = 20
        cat_names = VOC_CATEGORY_NAMES
        has_bg = True
        gt_set = 'val'
        db = PASCALContext(split=gt_set, do_edge=False, do_human_parts=False, do_semseg=True,
                                          do_normals=False, overfit=overfit)
   
    elif database == 'NYUD':
        from data.nyud import NYUD_MT
        n_classes = 40
        cat_names = NYU_CATEGORY_NAMES
        has_bg = False
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_semseg=True, overfit=overfit)
    
    else:
        raise NotImplementedError
    
    base_name = database + '_' + 'test' + '_semseg'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (semseg)')
    eval_results = eval_semseg(db, os.path.join(save_dir, 'semseg'), n_classes=n_classes, has_bg=has_bg)
    with open(fname, 'w') as f:
        json.dump(eval_results, f)
        
    # Print results
    class_IoU = eval_results['jaccards_all_categs']
    mIoU = eval_results['mIoU']

    print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * mIoU))
    for i in range(len(class_IoU)):
        spaces = ''
        for j in range(0, 15 - len(cat_names[i])):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(cat_names[i], spaces, 100 * class_IoU[i]))

    return eval_results
