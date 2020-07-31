#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import glob
import json
import torch
import numpy as np
from utils.utils import mkdir_if_missing
from losses.loss_functions import BalancedCrossEntropyLoss
from utils.mypath import MyPath, PROJECT_ROOT_DIR

class EdgeMeter(object):
    def __init__(self, pos_weight):
        self.loss = 0
        self.n = 0
        self.loss_function = BalancedCrossEntropyLoss(size_average=True, pos_weight=pos_weight)
        
    @torch.no_grad()
    def update(self, pred, gt):
        gt = gt.squeeze()
        pred = pred.float().squeeze() / 255.
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0
        self.n = 0

    def get_score(self, verbose=True):
        eval_dict = {'loss': self.loss / self.n}

        if verbose:
            print('\n Edge Detection Evaluation')
            print('Edge Detection Loss %.3f' %(eval_dict['loss']))

        return eval_dict


def eval_edge_predictions(p, database, save_dir):
    """ The edge are evaluated through seism """

    print('Evaluate the edge prediction using seism ... This can take a while ...')

    # DataLoaders
    if database == 'PASCALContext':
        from data.pascal_context import PASCALContext
        split = 'val'
        db = PASCALContext(split=split, do_edge=True, do_human_parts=False, do_semseg=False,
                            do_normals=False, do_sal=True, overfit=False)

    else:
        raise NotImplementedError

    # First check if all files are there
    files = glob.glob(os.path.join(save_dir, 'edge/*png'))
    
    assert(len(files) == len(db))

    # rsync the results to the seism root
    print('Rsync the results to the seism root ...')
    exp_name = database + '_' + p['setup'] + '_' + p['model']
    seism_root = MyPath.seism_root()
    result_dir = os.path.join(seism_root, 'datasets/%s/%s/'%(database, exp_name))
    mkdir_if_missing(result_dir)
    os.system('rsync -a %s %s' %(os.path.join(save_dir, 'edge/*'), result_dir))
    print('Done ...')

    v = list(np.arange(0.01, 1.00, 0.01))
    parameters_location = os.path.join(seism_root, 'parameters/%s.txt' %(exp_name))
    with open(parameters_location, 'w') as f:
        for l in v:
            f.write('%.2f\n' %(l))

    # generate a seism script that we will run.
    print('Generate seism script to perform the evaluation ...')
    seism_base = os.path.join(PROJECT_ROOT_DIR, 'evaluation/seism/pr_curves_base.m')
    with open(seism_base) as f:
        seism_file = f.readlines()
    seism_file = [line.strip() for line in seism_file]
    output_file = [seism_file[0]]

        ## Add experiments parameters (TODO)
    output_file += ['addpath(\'%s\')' %(os.path.join(seism_root, 'src/scripts/'))]
    output_file += ['addpath(\'%s\')' %(os.path.join(seism_root, 'src/misc/'))]
    output_file += ['addpath(\'%s\')' %(os.path.join(seism_root, 'src/tests/'))]
    output_file += ['addpath(\'%s\')' %(os.path.join(seism_root, 'src/gt_wrappers/'))]
    output_file += ['addpath(\'%s\')' %(os.path.join(seism_root, 'src/io/'))]
    output_file += ['addpath(\'%s\')' %(os.path.join(seism_root, 'src/measures/'))]
    output_file += ['addpath(\'%s\')' %(os.path.join(seism_root, 'src/piotr_edges/'))]
    output_file += ['addpath(\'%s\')' %(os.path.join(seism_root, 'src/segbench/'))]
    output_file.extend(seism_file[1:18])
     
        ## Add method (TODO)
    output_file += ['methods(end+1).name = \'%s\'; methods(end).io_func = @read_one_png; methods(end).legend =     methods(end).name;  methods(end).type = \'contour\';' %(exp_name)]
    output_file.extend(seism_file[19:61])

        ## Add path to save output
    output_file += ['filename = \'%s\'' %(os.path.join(save_dir, database + '_' + 'test' + '_edge.txt'))]
    output_file += seism_file[62:]

    # save the file to the seism dir
    output_file_path = os.path.join(seism_root, exp_name + '.m')
    with open(output_file_path, 'w') as f:
        for line in output_file:
            f.write(line + '\n')
    
    # go to the seism dir and perform evaluation
    print('Go to seism root dir and run the evaluation ... This takes time ...')
    cwd = os.getcwd()
    os.chdir(seism_root)
    os.system("matlab -nodisplay -nosplash -nodesktop -r \"addpath(\'%s\');%s;exit\"" %(seism_root, exp_name))
    os.chdir(cwd)
    
    # write to json
    print('Finished evaluation in seism ... Write results to JSON ...')
    with open(os.path.join(save_dir, database + '_' + 'test' + '_edge.txt'), 'r') as f:
        seism_result = [line.strip() for line in f.readlines()]

    eval_dict = {}
    for line in seism_result:
        metric, score = line.split(':')
        eval_dict[metric] = float(score)

    with open(os.path.join(save_dir, database + '_' + 'test' + '_edge.json'), 'w') as f:
        json.dump(eval_dict, f)
    
    # print
    print('Edge Detection Evaluation')
    for k, v in eval_dict.items():
        spaces = ''
        for j in range(0, 10 - len(k)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(k, spaces, 100*v))

    # cleanup - Important. Else Matlab will reuse the files.
    print('Cleanup files in seism ...')
    result_rm = os.path.join(seism_root, 'results/%s/%s/' %(database, exp_name))
    data_rm = os.path.join(seism_root, 'datasets/%s/%s/' %(database, exp_name))
    os.system("rm -rf %s" %(result_rm))
    os.system("rm -rf %s" %(data_rm))
    print('Finished cleanup ...')

    return eval_dict
