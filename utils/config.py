# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import cv2
import yaml
from easydict import EasyDict as edict
from utils.utils import mkdir_if_missing


def parse_task_dictionary(db_name, task_dictionary):
    """ 
        Return a dictionary with task information. 
        Additionally we return a dict with key, values to be added to the main dictionary
    """

    task_cfg = edict()
    other_args = dict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}
    task_cfg.FLAGVALS = {'image': cv2.INTER_CUBIC}
    task_cfg.INFER_FLAGVALS = {}

    if 'include_semseg' in task_dictionary.keys() and task_dictionary['include_semseg']:
        # Semantic segmentation
        tmp = 'semseg'
        task_cfg.NAMES.append('semseg')
        if db_name == 'PASCALContext':
            task_cfg.NUM_OUTPUT[tmp] = 21
        elif db_name == 'NYUD':
            task_cfg.NUM_OUTPUT[tmp] = 40
        else:
            raise NotImplementedError
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST 
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if 'include_human_parts' in task_dictionary.keys() and task_dictionary['include_human_parts']:
        # Human Parts Segmentation
        assert(db_name == 'PASCALContext')
        tmp = 'human_parts'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 7
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if 'include_sal' in task_dictionary.keys() and task_dictionary['include_sal']:
        # Saliency Estimation
        assert(db_name == 'PASCALContext')
        tmp = 'sal'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    if 'include_normals' in task_dictionary.keys() and task_dictionary['include_normals']:
        # Surface Normals 
        tmp = 'normals'
        assert(db_name in ['PASCALContext', 'NYUD'])
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 3
        task_cfg.FLAGVALS[tmp] = cv2.INTER_CUBIC
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['normloss'] = 1  # Hard-coded L1 loss for normals

    if 'include_edge' in task_dictionary.keys() and task_dictionary['include_edge']:
        # Edge Detection
        assert(db_name in ['PASCALContext', 'NYUD'])
        tmp = 'edge'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['edge_w'] = task_dictionary['edge_w']
        other_args['eval_edge'] = False

    if 'include_depth' in task_dictionary.keys() and task_dictionary['include_depth']:
        # Depth
        assert(db_name == 'NYUD')
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['depthloss'] = 'l1'
 
    return task_cfg, other_args


def create_config(env_file, exp_file):
    
    # Read the files 
    with open(env_file, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(exp_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    
    # Copy all the arguments
    cfg = edict()
    for k, v in config.items():
        cfg[k] = v

    
    # Parse the task dictionary separately
    cfg.TASKS, extra_args = parse_task_dictionary(cfg['train_db_name'], cfg['task_dictionary'])

    for k, v in extra_args.items():
        cfg[k] = v
    
    cfg.ALL_TASKS = edict() # All tasks = Main tasks
    cfg.ALL_TASKS.NAMES = []
    cfg.ALL_TASKS.NUM_OUTPUT = {}
    cfg.ALL_TASKS.FLAGVALS = {'image': cv2.INTER_CUBIC}
    cfg.ALL_TASKS.INFER_FLAGVALS = {}

    for k in cfg.TASKS.NAMES:
        cfg.ALL_TASKS.NAMES.append(k)
        cfg.ALL_TASKS.NUM_OUTPUT[k] = cfg.TASKS.NUM_OUTPUT[k]
        cfg.ALL_TASKS.FLAGVALS[k] = cfg.TASKS.FLAGVALS[k]
        cfg.ALL_TASKS.INFER_FLAGVALS[k] = cfg.TASKS.INFER_FLAGVALS[k]


    # Parse auxiliary dictionary separately
    if 'auxilary_task_dictionary' in cfg.keys():
        cfg.AUXILARY_TASKS, extra_args = parse_task_dictionary(cfg['train_db_name'], 
                                                                cfg['auxilary_task_dictionary'])
        for k, v in extra_args.items():
            cfg[k] = v

        for k in cfg.AUXILARY_TASKS.NAMES: # Add auxilary tasks to all tasks
            if not k in cfg.ALL_TASKS.NAMES:
                cfg.ALL_TASKS.NAMES.append(k)
                cfg.ALL_TASKS.NUM_OUTPUT[k] = cfg.AUXILARY_TASKS.NUM_OUTPUT[k]
                cfg.ALL_TASKS.FLAGVALS[k] = cfg.AUXILARY_TASKS.FLAGVALS[k]
                cfg.ALL_TASKS.INFER_FLAGVALS[k] = cfg.AUXILARY_TASKS.INFER_FLAGVALS[k]

    
    # Other arguments 
    if cfg['train_db_name'] == 'PASCALContext':
        cfg.TRAIN = edict()
        cfg.TRAIN.SCALE = (512, 512)
        cfg.TEST = edict()
        cfg.TEST.SCALE = (512, 512)

    elif cfg['train_db_name'] == 'NYUD':
        cfg.TRAIN = edict()
        cfg.TRAIN.SCALE = (480, 640)
        cfg.TEST = edict()
        cfg.TEST.SCALE = (480, 640)

    else:
        raise NotImplementedError


    # Location of single-task performance dictionaries (For multi-task learning evaluation)
    if cfg['setup'] == 'multi_task':
        cfg.TASKS.SINGLE_TASK_TEST_DICT = edict()
        cfg.TASKS.SINGLE_TASK_VAL_DICT = edict()
        for task in cfg.TASKS.NAMES:
            task_dir = os.path.join(root_dir, cfg['train_db_name'], cfg['backbone'], 'single_task', task)
            val_dict = os.path.join(task_dir, 'results', '%s_val_%s.json' %(cfg['val_db_name'], task))
            test_dict = os.path.join(task_dir, 'results', '%s_test_%s.json' %(cfg['val_db_name'], task))
            cfg.TASKS.SINGLE_TASK_TEST_DICT[task] = test_dict
            cfg.TASKS.SINGLE_TASK_VAL_DICT[task] = val_dict

    # Overfitting (Useful for debugging -> Overfit on small partition of the data)
    if not 'overfit' in cfg.keys():
        cfg['overfit'] = False

    # Determine output directory
    if cfg['setup'] == 'single_task':
        output_dir = os.path.join(root_dir, cfg['train_db_name'], cfg['backbone'], cfg['setup'])
        output_dir = os.path.join(output_dir, cfg.TASKS.NAMES[0])

    elif cfg['setup'] == 'multi_task':
        if cfg['model'] == 'baseline':
            output_dir = os.path.join(root_dir, cfg['train_db_name'], cfg['backbone'], 'multi_task_baseline')
        else:
            output_dir = os.path.join(root_dir, cfg['train_db_name'], cfg['backbone'], cfg['model'])
    else:
        raise NotImplementedError
        

    cfg['root_dir'] = root_dir
    cfg['output_dir'] = output_dir
    cfg['save_dir'] = os.path.join(output_dir, 'results')
    cfg['checkpoint'] = os.path.join(output_dir, 'checkpoint.pth.tar')
    cfg['best_model'] = os.path.join(output_dir, 'best_model.pth.tar')
    mkdir_if_missing(cfg['output_dir'])
    mkdir_if_missing(cfg['save_dir'])
    return cfg
