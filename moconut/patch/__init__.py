import torch
from .subpatches import *
from .models import *

################################################################
# keyword to moconut.patch.ModuleSubPatch
################################################################
subpatch_map = {}
subpatch_map.update(
    {
        'conv1d' : Conv1dSubpatch
    }
)
subpatch_map.update(
    {
        'leakyrelu' : LeakyReLUSubpatch
    }
)
subpatch_map.update(
    {
        'repeating_subpatch' : RepeatingSubpatch
    }
)

################################################################
# Patch
################################################################
def Patch(
    op_type       : str,
    required      : dict[str, type],
    independent   : dict,
    dependent     : dict,
    constraints   : Optional[list] = None,
    compute_graph : list[dict] = []
):
    """
        The docstring
    """
    class MoconutPytorchPatch(torch.nn.Module):
        """
            MoconutPytorchPatch
        """
        def __init__(
            self,
            inlets  : List[str],
            outlets : List[str],
            config  : dict,
            parents : Optional[List[torch.nn.Module]] = None,
            device  : Optional[str] = None
        ):
            super(MoconutPytorchPatch, self).__init__()
            self.op_type = op_type
            self.inlets  = inlets
            self.outlets = outlets
            self.config  = copy.deepcopy(config)
            self.device  = device
            if parents is None:
                parents = []

            ################################################################
            # Parse Required Config Args
            ################################################################
            for arg in required:
                if arg in self.config:
                    setattr(self, arg, self.config[arg])
                else:
                    raise ValueError(f"{self.__class__.__name__}::RequirementError - '{arg}'")

            ################################################################
            # Parse Independent Config Args
            ################################################################
            for arg in independent:
                setattr(
                    self,
                    arg,
                    self.config[arg] if arg in self.config else independent[arg]
                )
            
            ################################################################
            # Parse Dependent Config Args
            ################################################################
            for arg in dependent:
                if arg in self.config:
                    setattr(self, arg, self.config[arg])
                elif not isinstance(dependent[arg], moconut.DependentDefault):
                    raise ValueError(f"{self.__class__.__name__}::DependenceError - '{arg}' needs to be associated to a moconut.DependentDefault.")
                else:
                    parents = []
                    for parent in dependent[arg].parents:
                        if not hasattr(self, parent):
                            raise ValueError(f"{self.__class__.__name__}::DependenceError - '{arg}' is dependent on '{parent}' but the latter doesn't exist.")
                        parents.append(getattr(self, parent))

                    setattr(
                        self,
                        arg,
                        dependent[arg].dependence(*parents)
                    )
            
            ################################################################
            # Enforce Config Arg Constraints
            ################################################################
            if constraints:
                for constraint in constraints:
                    constraint.evaluate(self)

            ################################################################
            # Setup Compute Graph
            ################################################################
            if not compute_graph:
                raise ValueError(f"moconut.Patch::RequirementError - moconut.Patch requires the argument 'compute_graph'. It is either None or an empty list.")
            if not all([isinstance(config,dict) for config in compute_graph]):
                raise ValueError(f"moconut.Patch::RequiredArgTypeMismatch - 'compute_graph' should be a list of dicts.")
            self.compute_graph = compute_graph
            
            self.operations = torch.nn.ModuleList()
            for subpatch in self.compute_graph:
                subpatch_config = subpatch['config']
                self.operations.append(
                    moconut.subpatch_map[subpatch['op_type']](
                        inlets  = subpatch['inlets'],
                        outlets = subpatch['outlets'],
                        config  = subpatch_config,
                        parents = [*parents, self],
                    )
                )
                
        def forward(self, in_messages : dict) -> dict:
            messages = in_messages
            for i in range(len(self.operations)):
                messages.update(
                    self.operations[i](messages)
                )
            
            return messages

    return MoconutPytorchPatch
