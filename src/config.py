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
