import os

current_dir = os.path.dirname(os.path.abspath(__file__))

MAIN_PROJECT_DIR = os.path.abspath(os.path.join(current_dir, os.pardir))
DATASET_DIR = os.path.join(MAIN_PROJECT_DIR, 'dataset')

CLASS_INFO_PATH = os.path.join(MAIN_PROJECT_DIR, 'class_info.json')

CLASS_MAPPING = {
    'Ei': 'elliptical', 'Er': 'elliptical', 'Ec': 'elliptical',
    'Sa': 'spiral', 'Sb': 'spiral', 'Sc': 'spiral', 'Sd': 'spiral',
    'SBa': 'spiral barred', 'SBb': 'spiral barred', 'SBc': 'spiral barred', 'SBd': 'spiral barred',
    'Sen': 'spiral edge-on', 'Ser': 'spiral edge-on', 'Seb': 'spiral edge-on',
    'A': 'artifact'
}

GALAXY_MEAN = [8.9483, 6.2549, 5.1239]
GALAXY_STD = [14.9211, 12.5903, 8.8366]