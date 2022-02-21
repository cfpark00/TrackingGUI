import os
import sys
sys.path.append(os.getcwd())

from src.Dataset import Dataset

import argparse

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('n_add', type=int, help='Number of beurons to add')
parser.add_argument('--overwrite', default=True, help='overwrites file (default:True)')

args=parser.parse_args()

assert args.n_add>0
assert args.overwrite

dataset=Dataset(args.file_path)
dataset.open()
dataset.add_points(args.n_add)
dataset.close()


