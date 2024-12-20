import torch
from .cables import *

cable_map = {}
cable_map.update(
    {
        'leakyrelu' : LeakyReLUCable
    }
)