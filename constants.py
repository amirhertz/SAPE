import os
import sys


IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = f'{PROJECT_ROOT}/assets/'
RAW_IMAGES = f'{DATA_ROOT}images/'
RAW_MESHES = f'{DATA_ROOT}meshes/'
RAW_SILHOUETTES = f'{DATA_ROOT}silhouettes/'
CHECKPOINTS_ROOT = f'{DATA_ROOT}checkpoints'
