import os

current_dir = os.path.dirname(os.path.abspath(__file__))

MAIN_PROJECT_DIR = os.path.abspath(os.path.join(current_dir, os.pardir))
DATASET_DIR = os.path.join(MAIN_PROJECT_DIR, 'dataset')

