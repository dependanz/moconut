import copy
import torch
import moconut
from typing import Optional, List

class Subpatch(torch.nn.Module):
    def __init__(
        self,
        op_type : str,
        inlets  : List[str],
        outlets : List[str],
        config  : dict,
        parents : Optional[List[torch.nn.Module]] = None,
        device  : Optional[str] = None
    ):
        super(Subpatch, self).__init__()
        self.op_type = op_type
        self.inlets  = inlets
        self.outlets = outlets
        self.config  = copy.deepcopy(config)
        self.device  = device
            
    def forward(self, in_messages : dict) -> dict:
        messages = in_messages
        
        out_messages = {}
        for outlet_name in self.outlets:
            out_messages[outlet_name] = messages[outlet_name]
        
        return messages

class LeakyReLUSubpatch(Subpatch):
    def __init__(
        self,
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        parents : Optional[list[torch.nn.Module]] = None,
        device  : Optional[str] = None
    ): 
        super(LeakyReLUSubpatch, self).__init__(
            'leakyrelu', 
            parents, 
            inlets, 
            outlets, 
            config, 
            device
        )
        
        # Subpatch Constraints
        if len(self.inlets) != len(self.outlets):
            raise AttributeError(f"{self.__class__.__name__}::SubpatchConstraintError - Subpatch '{self.op_type}' requires the same number of inlets as outlets")
  			
        # Replace moconut.AttributeName with appropriate parent information
        for arg in config:
            if isinstance(config[arg], moconut.AttributeName):
                attr_found = False
                for parent in parents[::-1]:
                    if hasattr(parent, config[arg].name):
                        config[arg] = getattr(parent, config[arg].name)
                        attr_found = True
                        break
                                
                if not attr_found:
                    raise AttributeError(f"{self.__class__.__name__}::AttributeNotFound - {config[arg].name} not found in parent patches.")
        
        # Set operation used for the Subpatch
        self.op = moconut.module_map[self.op_type](
            *moconut.pack_config_map[self.op_type](
                config = self.config
            )
        )
        
    def forward(self, in_messages : dict) -> dict:
        if len(in_messages) != len(self.inlets):
            raise RuntimeError(f"{self.__class__.__name__}::SubpatchInputMessageError - Subpatch '{self.op_type}' was given {len(in_messages)} inputs, but it has {len(self.inlets)} inlets.")

        out_messages = {}
        for i in range(len(self.outlets)):
            inlet_name = self.inlets[i]
            outlet_name = self.outlets[i]
            out_messages[outlet_name] = self.op(in_messages[inlet_name])
                
        return out_messages
				
    
class Conv1dSubpatch(Subpatch):
    def __init__(
        self,
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        parents : Optional[list[torch.nn.Module]] = None,
        device  : Optional[str] = None
    ):
        super(Conv1dSubpatch, self).__init__(
            'conv1d', 
            parents, 
            inlets, 
            outlets, 
            config, 
            device
        )
        
        # Subpatch Constraints
        if len(self.inlets) != len(self.outlets):
            raise AttributeError(f"{self.__class__.__name__}::SubpatchConstraintError - Subpatch '{self.op_type}' requires the same number of inlets as outlets")
  			
        # Replace moconut.AttributeName with appropriate parent information
        for arg in self.config:
            if isinstance(self.config[arg], moconut.AttributeName):
                attr_found = False
                if self.config[arg].indexed:
                    for parent in parents[::-1]:
                        if hasattr(parent, self.config[arg].name):
                            self.config[arg] = getattr(parent, self.config[arg].name)[self.config[arg].index]
                            attr_found = True
                            break
                else:
                    for parent in parents[::-1]:
                        if hasattr(parent, config[arg].name):
                            self.config[arg] = getattr(parent, self.config[arg].name)
                            attr_found = True
                            break
                                
                if not attr_found:
                    raise AttributeError(f"{self.__class__.__name__}::AttributeNotFound - {self.config[arg].name} not found in parent patches.")
        
        # Set operation used for the Subpatch
        self.op = moconut.module_map[self.op_type](
            *moconut.pack_config_map[self.op_type](
                config = self.config
            )
        )
    
    def forward(self, in_messages : dict) -> dict:
        if len(in_messages) != len(self.inlets):
            raise RuntimeError(f"{self.__class__.__name__}::SubpatchInputMessageError - Subpatch '{self.op_type}' was given {len(in_messages)} inputs, but it has {len(self.inlets)} inlets.")

        out_messages = {}
        for i in range(len(self.outlets)):
            inlet_name = self.inlets[i]
            outlet_name = self.outlets[i]
            out_messages[outlet_name] = self.op(in_messages[inlet_name])
                
        return out_messages

################################################################
# TODO: Generalize the Subpatch.
################################################################
class RepeatingSubpatch(torch.nn.Module):
    def __init__(
        self,
        inlets  : list[str],
        outlets : list[str],
        config  : dict,
        parents : Optional[list[torch.nn.Module]] = None,
        device  : Optional[str]  = None
    ):
        super(RepeatingSubpatch, self).__init__()
        self.op_type = 'repeating_subpatch'
        self.inlets  = inlets
        self.outlets = outlets
        self.config  = copy.deepcopy(config)
        self.device  = device
        
        # TODO: Constraints for patches to be truly subpatches (i.e. needs parent info to setup)
        assert parents is not None
        
        ################################################################
        # Parse Required Config Args
        ################################################################
        required = {
            'repeat_over'   : moconut.AttributeName,
            'compute_graph' : list
        }
        for arg in required:
            if arg in self.config:
                if not isinstance(self.config[arg], required[arg]):
                    raise ValueError(f"{self.__class__.__name__}::RequiredArgTypeMismatchError - '{arg}'")
                setattr(self, arg, self.config[arg])
            else:
                raise ValueError(f"{self.__class__.__name__}::RequirementError - '{arg}'")
            
        ################################################################
        # Setup Compute Graph
        ################################################################
        attr_found = False
        for parent in parents:
            if hasattr(parent, self.repeat_over.name):
                setattr(self, 'repeat_over', getattr(parent, self.repeat_over.name))
                attr_found = True
                break
        if not attr_found:
            raise AttributeError(f"{self.__class__.__name__}::AttributeNotFound - {self.repeat_over} not found in parent patches.")
        
        self.operations = torch.nn.ModuleList()
        for i in range(len(self.repeat_over)):
            for subpatch in self.compute_graph:
                
                # Evaluate Ellipsis AttributeNames
                subpatch_config = copy.deepcopy(subpatch['config'])
                for arg in subpatch_config:
                    if isinstance(subpatch_config[arg], moconut.AttributeName):
                        if subpatch_config[arg].indexed:
                            subpatch_config[arg].index = i
                
                self.operations.append(
                    moconut.subpatch_map[subpatch['op_type']](
                        parents = [*parents, self],
                        inlets  = subpatch['inlets'],
                        outlets = subpatch['outlets'],
                        config  = subpatch_config,
                    )
                )
        
    def forward(self, in_messages : dict) -> dict:
        messages = in_messages
        for i in range(len(self.operations)):
            messages.update(
                self.operations[i](messages)
            )
        
        return messages
