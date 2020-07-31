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
import numpy as np
import glob
import json
import torch
from PIL import Image

import evaluation.jaccard as evaluation


def eval_sal(loader, folder, mask_thres=None):
    if mask_thres is None:
        mask_thres = [0.5]

    eval_result = dict()
    eval_result['all_jaccards'] = np.zeros((len(loader), len(mask_thres)))
    eval_result['prec'] = np.zeros((len(loader), len(mask_thres)))
    eval_result['rec'] = np.zeros((len(loader), len(mask_thres)))

    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample["meta"]["image"] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32) / 255.

        gt = sample["sal"]

        if mask.shape != gt.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
        
        for j, thres in enumerate(mask_thres):
            #gt = (gt > thres).astype(np.float32) # Removed this from ASTMT code. GT is already binarized. 
            mask_eval = (mask > thres).astype(np.float32)
            eval_result['all_jaccards'][i, j] = evaluation.jaccard(gt, mask_eval)
            eval_result['prec'][i, j], eval_result['rec'][i, j] = evaluation.precision_recall(gt, mask_eval)

    # Average for each thresholds
    eval_result['mIoUs'] = np.mean(eval_result['all_jaccards'], 0)

    eval_result['mPrec'] = np.mean(eval_result['prec'], 0)
    eval_result['mRec'] = np.mean(eval_result['rec'], 0)
    eval_result['F'] = 2 * eval_result['mPrec'] * eval_result['mRec'] / \
                       (eval_result['mPrec'] + eval_result['mRec'] + 1e-12)

    # Maximum of averages (maxF, maxmIoU)
    eval_result['mIoU'] = np.max(eval_result['mIoUs'])
    eval_result['maxF'] = np.max(eval_result['F'])

    eval_result = {x: eval_result[x].tolist() for x in eval_result}

    return eval_result


class SaliencyMeter(object):
    def __init__(self):
        self.mask_thres = np.linspace(0.2, 0.9, 15) # As below
        self.all_jacards = []
        self.prec = []
        self.rec = []    
    
    @torch.no_grad()
    def update(self, pred, gt):
        # Predictions and ground-truth
        b = pred.size(0)
        pred = pred.float().squeeze() / 255.
        gt = gt.squeeze().cpu().numpy()

        # Allocate memory for batch results 
        jaccards = np.zeros((b, len(self.mask_thres)))
        prec = np.zeros((b, len(self.mask_thres)))
        rec = np.zeros((b, len(self.mask_thres)))

        for j, thres in enumerate(self.mask_thres):
            #gt_eval = (gt > thres).cpu().numpy() # Removed this from ASTMT code. GT is already binarized.
            mask_eval = (pred > thres).cpu().numpy()
            for i in range(b):
                jaccards[i, j] = evaluation.jaccard(gt[i], mask_eval[i])
                prec[i, j], rec[i, j] = evaluation.precision_recall(gt[i], mask_eval[i])

        self.all_jacards.append(jaccards)
        self.prec.append(prec)
        self.rec.append(rec)

    def reset(self):
        self.all_jacards = []
        self.prec = []
        self.rec = []

    def get_score(self, verbose=True):
        eval_result = dict()
        
        # Concatenate batched results
        eval_result['all_jaccards'] = np.concatenate(self.all_jacards)
        eval_result['prec'] = np.concatenate(self.prec)
        eval_result['rec'] = np.concatenate(self.rec)

        # Average for each threshold
        eval_result['mIoUs'] = np.mean(eval_result['all_jaccards'], 0)

        eval_result['mPrec'] = np.mean(eval_result['prec'], 0)
        eval_result['mRec'] = np.mean(eval_result['rec'], 0)
        eval_result['F'] = 2 * eval_result['mPrec'] * eval_result['mRec'] / \
                           (eval_result['mPrec'] + eval_result['mRec'] + 1e-12)

        # Maximum of averages (maxF, maxmIoU)
        eval_result['mIoU'] = np.max(eval_result['mIoUs'])
        eval_result['maxF'] = np.max(eval_result['F'])

        eval_result = {x: eval_result[x].tolist() for x in eval_result}

        if verbose:
            # Print the results 
            print('Results for Saliency Estimation')
            print('mIoU: {0:.3f}'.format(100*eval_result['mIoU']))
            print('maxF: {0:.3f}'.format(100*eval_result['maxF']))

        return eval_result
    

def eval_sal_predictions(database, save_dir, overfit=False):
    """ Evaluate the saliency estimation maps that are stored in the save dir """

    # Dataloaders
    if database == 'PASCALContext':
        from data.pascal_context import PASCALContext
        split = 'val'
        db = PASCALContext(split=split, do_edge=False, do_human_parts=False, do_semseg=False,
                                          do_normals=False, do_sal=True, overfit=overfit)
    
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_sal'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evalute the saved images (saliency)')
    eval_results = eval_sal(db, os.path.join(save_dir, 'sal'), mask_thres=np.linspace(0.2, 0.9, 15))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print the results 
    print('Results for Saliency Estimation')
    print('mIoU: {0:.3f}'.format(100*eval_results['mIoU']))
    print('maxF: {0:.3f}'.format(100*eval_results['maxF']))

    return eval_results
