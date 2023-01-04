import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
import math
import shutil


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.
    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    os.chdir(source)
    all_source_files_list = []
    for file in glob.glob("*.tfrecord"):
        all_source_files_list.append(file)
    
    # randomize file order by loading to set 
    all_source_files_set = set(all_source_files_list)
    
    train_ratio = 60
    val_ratio = 20
    test_ratio = 20
    
    number_of_train_files = int(math.ceil(len(all_source_files_set) * (train_ratio/100)))
    number_of_val_files = int(math.ceil(len(all_source_files_set) * (val_ratio/100)))
    number_of_test_files = int(len(all_source_files_set) * (test_ratio/100))
        
    logger.info(f'number_of_train_files : {number_of_train_files}')
    logger.info(f'number_of_val_files : {number_of_val_files}')
    logger.info(f'number_of_test_files : {number_of_test_files}')
    
    
    train_dir_path=f"train"
    val_dir_path=f"val"
    test_dir_path=f"test"
    if os.path.exists(train_dir_path):
        shutil.rmtree(train_dir_path)
        os.makedirs(train_dir_path)
    else:
        os.makedirs(train_dir_path)
        
    if os.path.exists(val_dir_path):
        shutil.rmtree(val_dir_path)
        os.makedirs(val_dir_path)
    else:
        os.makedirs(val_dir_path)
        
    if os.path.exists(test_dir_path):
        shutil.rmtree(test_dir_path)
        os.makedirs(test_dir_path)
    else:
        os.makedirs(test_dir_path)
        
    logger.info(f'sub folders created')

          
    cnt = 0 
    for fname in all_source_files_set:
        if cnt < number_of_train_files:
            shutil.move(f'{fname}',f'{train_dir_path}/{fname}')
        elif cnt >=number_of_train_files and cnt < number_of_val_files +number_of_train_files :
            shutil.move(f'{fname}',f'{val_dir_path}/{fname}')
        else:
            shutil.move(f'{fname}',f'{test_dir_path}/{fname}')
        cnt+=1
        
    if number_of_train_files ==0:
        logger.info(f'there are no files to move')
    else:
        logger.info(f'files moved to train,test,val folders')


if __name__ == "__main__":
    # python create_splits.py --source data_split_test  --destination data_split_test
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
