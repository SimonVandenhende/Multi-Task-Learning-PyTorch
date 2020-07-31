# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import glob
import json
import os.path
import numpy as np
import torch
from PIL import Image

PART_CATEGORY_NAMES = ['background', 'head', 'torso', 'uarm', 'larm', 'uleg', 'lleg']


def eval_human_parts(loader, folder, n_parts=6):

    tp = [0] * (n_parts + 1)
    fp = [0] * (n_parts + 1)
    fn = [0] * (n_parts + 1)

    counter = 0
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        if 'human_parts' not in sample:
            continue

        # Check for valid pixels
        gt = sample['human_parts']
        uniq = np.unique(gt)
        if len(uniq) == 1 and (uniq[0] == 255 or uniq[0] == 0):
            continue

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32)

        # Case of a binary (probability) result
        if n_parts == 1:
            mask = (mask > 0.5 * 255).astype(np.float32)

        counter += 1
        valid = (gt != 255)

        if mask.shape != gt.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # TP, FP, and FN evaluation
        for i_part in range(0, n_parts + 1):
            tmp_gt = (gt == i_part)
            tmp_pred = (mask == i_part)
            tp[i_part] += np.sum(tmp_gt & tmp_pred & (valid))
            fp[i_part] += np.sum(~tmp_gt & tmp_pred & (valid))
            fn[i_part] += np.sum(tmp_gt & ~tmp_pred & (valid))

    print('Successful evaluation for {} images'.format(counter))
    jac = [0] * (n_parts + 1)
    for i_part in range(0, n_parts + 1):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)

    return eval_result


class HumanPartsMeter(object):
    def __init__(self, database):
        assert(database == 'PASCALContext')
        self.database = database
        self.cat_names = PART_CATEGORY_NAMES
        self.n_parts = 6
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)

    @torch.no_grad() 
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = (gt != 255)
        
        for i_part in range(self.n_parts + 1):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & (valid)).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & (valid)).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & (valid)).item()

    def reset(self):
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)
 
    def get_score(self, verbose=True):
        jac = [0] * (self.n_parts + 1)
        for i_part in range(0, self.n_parts + 1):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)
        
        print('\nHuman Parts mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
        class_IoU = jac
        for i in range(len(class_IoU)):
            spaces = ''
            for j in range(0, 15 - len(self.cat_names[i])):
                spaces += ' '
            print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result


def eval_human_parts_predictions(database, save_dir, overfit=False):
    """ Evaluate the human parts predictions that are stored in the save dir """

    # Dataloaders
    if database == 'PASCALContext':
        from data.pascal_context import PASCALContext
        gt_set = 'val'
        db = PASCALContext(split=gt_set, do_edge=False, do_human_parts=True, do_semseg=False,
                                          do_normals=False, do_sal=False, overfit=overfit)
    
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_human_parts'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (human parts)')
    eval_results = eval_human_parts(db, os.path.join(save_dir, 'human_parts'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print Results
    class_IoU = eval_results['jaccards_all_categs']
    mIoU = eval_results['mIoU']

    print('\nHuman Parts mIoU: {0:.4f}\n'.format(100 * mIoU))
    for i in range(len(class_IoU)):
        spaces = ''
        for j in range(0, 15 - len(PART_CATEGORY_NAMES[i])):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(PART_CATEGORY_NAMES[i], spaces, 100 * class_IoU[i]))

    return eval_results
