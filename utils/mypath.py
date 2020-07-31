# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]


class MyPath(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/path/to/databases/'
        db_names = {'PASCAL_MT', 'NYUD_MT'}

        if database in db_names:
            return os.path.join(db_root, database)
        
        elif not database:
            return db_root
        
        else:
            raise NotImplementedError

    @staticmethod
    def seism_root():
        return '/path/to/seism/'
