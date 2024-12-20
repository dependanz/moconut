import torch
import moconut
from typing import Optional

class ModuleCable(torch.nn.Module):
    def __init__(
        self,
        op_type : str,
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        device  : Optional[str] = None
    ):
        super(ModuleCable, self).__init__()
        self.op_type = op_type
        self.inlets  = inlets
        self.outlets = outlets
        self.config  = config
        self.device  = device
        
        self.op = moconut.pytorch.module_map[self.op_type](
            *moconut.pytorch.pack_config_map[self.op_type](
                config = config
            )
        )

class LeakyReLUCable(ModuleCable):
    def __init__(
        self,
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        device  : Optional[str] = None
    ):
        super().__init__('leakyrelu', inlets, outlets, config, device)
        
    def send(self, inlet_group : list):
        return [self.op(inlet_group[i]) for i in range(len(inlet_group))]
    
class Conv1dCable(ModuleCable):
    def __init__(
        self,
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        device  : Optional[str] = None
    ):
        super().__init__('conv1d', inlets, outlets, config, device)
        
    def send(self, inlet_group : list):
        return [self.op(inlet_group[i]) for i in range(len(inlet_group))]