import torch
from .cables import *
from .models import *

cable_map = {}
cable_map.update(
    {
        'conv1d' : Conv1dCable
    }
)
cable_map.update(
    {
        'leakyrelu' : LeakyReLUCable
    }
)