import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Tuple, List, Union, Callable, Type, Iterator, Dict, Set, Optional, Any, Sized
from types import DynamicClassAttribute
from enum import Enum
import torch.optim.optimizer
import torch.utils.data


N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Union[TS, N]
TA = Union[T, ARRAY]

V_Mesh = Tuple[V, V]
T_Mesh = Tuple[T, Optional[T]]
T_Mesh_T = Union[T_Mesh, T]
COLORS = Union[T, V, Tuple[int, int, int]]

D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device
Optimizer = torch.optim.Adam
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader
Subset = torch.utils.data.Subset


class EncodingType(enum.Enum):

    @DynamicClassAttribute
    def value(self) -> str:
        return super(EncodingType, self).value

    PE = 'pe'
    FF = 'ff'
    NoEnc = 'no_encoding'
    RBF = 'rbf'
    PRBF = 'periodic_rbf'


class ControllerType(enum.Enum):

    @DynamicClassAttribute
    def value(self) -> str:
        return super(ControllerType, self).value

    NoControl = 'no_control'
    GlobalProgression = 'global_progression'
    SpatialProgressionStashed = 'spatial_progression_stashed'


class DeformType(Enum):
    HARMONIC = 'harmonic'
    RIGID = 'rigid'
    BEAUTY = 'beauty'


