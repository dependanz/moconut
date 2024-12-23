import torch
from .subpatches import *
from .models import *

################################################################
# keyword to moconut.patch.ModuleSubPatch
################################################################
subpatch_map = {}
subpatch_map.update(
    {
        'conv1d' : Conv1dSubPatch
    }
)
subpatch_map.update(
    {
        'leakyrelu' : LeakyReLUSubPatch
    }
)
subpatch_map.update(
    {
        'repeating_subpatch' : LeakyReLUSubPatch
    }
)

################################################################
# Patch
################################################################
def Patch(
    required    : dict[str, type],
    independent : dict,
    dependent   : dict,
    constraints : Optional[list] = None
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
            config,
            device : Optional[str] = None
        ):
            super(MoconutPytorchPatch, self).__init__()
            self.config = config
            self.device = device

            ################################################################
            # Parse Required Config Args
            ################################################################
            for arg in required:
                if arg in config:
                    setattr(self, arg, config[arg])
                else:
                    raise ValueError(f"{self.__class__.__name__}::RequirementError - '{arg}'")

            ################################################################
            # Parse Independent Config Args
            ################################################################
            for arg in independent:
                setattr(
                    self,
                    arg,
                    config[arg] if arg in config else independent[arg]
                )
            
            ################################################################
            # Parse Dependent Config Args
            ################################################################
            for arg in dependent:
                if arg in config:
                    setattr(self, arg, config[arg])
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


    return MoconutPytorchPatch
