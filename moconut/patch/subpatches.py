import torch
import moconut
from typing import Optional

class ModuleSubPatch(torch.nn.Module):
    def __init__(
        self,
        op_type : str,
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        device  : Optional[str] = None
    ):
        super(ModuleSubPatch, self).__init__()
        self.op_type = op_type
        self.inlets  = inlets
        self.outlets = outlets
        self.config  = config
        self.device  = device
        
        self.op = moconut.module_map[self.op_type](
            *moconut.pack_config_map[self.op_type](
                config = config
            )
        )

class LeakyReLUSubPatch(ModuleSubPatch):
    def __init__(
        self,
        parents  : Optional[list[torch.nn.Module]],
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        device  : Optional[str] = None
    ):
        super().__init__('leakyrelu', inlets, outlets, config, device)
        
    def send(self, inlet_group : list):
        return [self.op(inlet_group[i]) for i in range(len(inlet_group))]
    
class Conv1dSubPatch(ModuleSubPatch):
    def __init__(
        self,
        parents  : Optional[list[torch.nn.Module]],
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        device  : Optional[str] = None
    ):
        super().__init__('conv1d', inlets, outlets, config, device)
        
    def send(self, inlet_group : list):
        return [self.op(inlet_group[i]) for i in range(len(inlet_group))]

################################################################
# TODO: Generalize the SubPatch.
################################################################
class RepeatingSubPatch(torch.nn.Module):
    def __init__(
        self,
        parents  : Optional[list[torch.nn.Module]],
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        device  : Optional[str]  = None
    ):
        super(RepeatingSubPatch, self).__init__()
        self.op_type = 'repeating_subpatch'
        self.inlets  = inlets
        self.outlets = outlets
        self.config  = config
        self.device  = device
        
        # TODO: Constraints for patches to be truly subpatches (i.e. needs parent info to setup)
        assert parents is not None
        
        ################################################################
        # Parse Required Config Args
        ################################################################
        required = {
            'repeat_over'   : moconut.AttributeName,
            'compute_graph' : list[dict]
        }
        for arg in required:
            if arg in config:
                if not isinstance(config[arg], required[arg]):
                    raise ValueError(f"{self.__class__.__name__}::RequiredArgTypeMismatchError - '{arg}'")
                setattr(self, arg, config[arg])
            else:
                raise ValueError(f"{self.__class__.__name__}::RequirementError - '{arg}'")
            
        ################################################################
        # Setup Compute Graph
        ################################################################
        attr_found = False
        for parent in parents:
            if hasattr(parent, self.repeat_over):
                setattr(self, self.repeat_over, getattr(parent, self.repeat_over))
                attr_found = True
                break
        if not attr_found:
            raise AttributeError(f"{self.__class__.__name__}::AttributeNotFound - {self.repeat_over} not found in parent patches.")
        
        self.operations = torch.nn.ModuleList()
        for i in range(len(getattr(self, self.repeat_over))):
            for subpatch in self.compute_graph:
                self.operations.append(
                    moconut.subpatch_map[subpatch['op_type']](
                        parents = [*parents, self],
                        inlets  = subpatch['inlets'],
                        outlets = subpatch['outlets'],
                        config  = subpatch['config'],
                    )
                )
        
    def send(self, in_messages : dict) -> dict:
        messages = in_messages
        for i in range(len(self.operations)):
            messages = self.operations[i](messages)

        out_messages = {}
        for outlet_name in self.outlets:
            out_messages[outlet_name] = messages[outlet_name]
        
        return out_messages