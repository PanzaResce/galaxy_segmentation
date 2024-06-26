import os

current_dir = os.path.dirname(os.path.abspath(__file__))

MAIN_PROJECT_DIR = os.path.abspath(os.path.join(current_dir, os.pardir))
DATASET_DIR = os.path.join(MAIN_PROJECT_DIR, 'dataset')

CLASS_MAPPING = {
    'Ei': 'E', 'Er': 'E', 'Ec': 'E',
    'Sa': 'S', 'Sb': 'S', 'Sc': 'S', 'Sd': 'S',
    'SBa': 'SB', 'SBb': 'SB', 'SBc': 'SB', 'SBd': 'SB',
    'Sen': 'Se', 'Ser': 'Se', 'Seb': 'Se'
}

CLASS_TO_ID = {
    'E': 1,
    'S': 2,
    'SB': 3,
    'Se': 4,
    'A': 5
}