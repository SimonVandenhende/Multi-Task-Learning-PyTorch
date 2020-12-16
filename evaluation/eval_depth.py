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
import torch
import json
import scipy.io as sio


def eval_depth(loader, folder):

    total_rmses = 0.0
    total_log_rmses = 0.0
    n_valid = 0.0

    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating depth: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.mat')
        pred = sio.loadmat(filename)['depth'].astype(np.float32)
        label = sample['depth']
        
        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_LINEAR)

        valid_mask = (label != 0)
        n_valid += np.sum(valid_mask)

        label[label == 0] = 1e-9 # Avoid overflow/underflow
        pred[pred <= 0] = 1e-9

        log_rmse_tmp = (np.log(label[valid_mask]) - np.log(pred[valid_mask])) ** 2
        total_log_rmses += np.sum(log_rmse_tmp)

        rmse_tmp = (label[valid_mask] - pred[valid_mask]) ** 2
        total_rmses += np.sum(rmse_tmp)

    eval_result = dict()
    eval_result['rmse'] = np.sqrt(total_rmses / n_valid)
    eval_result['log_rmse'] = np.sqrt(total_log_rmses / n_valid)

    return eval_result


class DepthMeter(object):
    def __init__(self):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        
        # Determine valid mask
        mask = (gt != 255).bool()
        self.n_valid += mask.float().sum().item() # Valid pixels per image
        
        # Only positive depth values are possible
        pred = torch.clamp(pred, min=1e-9)

        # Per pixel rmse and log-rmse.
        log_rmse_tmp = torch.pow(torch.log(gt) - torch.log(pred), 2)
        log_rmse_tmp = torch.masked_select(log_rmse_tmp, mask)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        rmse_tmp = torch.pow(gt - pred, 2)
        rmse_tmp = torch.masked_select(rmse_tmp, mask)
        self.total_rmses += rmse_tmp.sum().item()

    def reset(self):
        self.rmses = []
        self.log_rmses = []
        
    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)

        if verbose:
            print('Results for depth prediction')
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result
        

def eval_depth_predictions(database, save_dir, overfit=False):

    # Dataloaders
    if database == 'NYUD':
        from data.nyud import NYUD_MT 
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_depth=True, overfit=overfit)
    
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_depth'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (depth)')
    eval_results = eval_depth(db, os.path.join(save_dir, 'depth'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results
    print('Results for Depth Estimation')
    for x in eval_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))

    return eval_results
