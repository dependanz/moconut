import torch
import moconut
from typing import Optional

class ComputeGraphModule(torch.nn.Module):
    """
        ComputeGraphModule

        :config:
        compute_graph -> list[dict], a compute graph where each element in the list is a config for 
        a patch object
    """
    def __init__(
        self,
        config,
        device : Optional[str] = None
    ):
        super(ComputeGraphModule, self).__init__()
        self.config = config
        self.device = device

        ################################################################
        # Setup Compute Graph
        ################################################################
        self.operations = torch.nn.ModuleList()
        for subpatch in config['compute_graph']:
            self.operations.append(
                moconut.pytorch.patch.subpatch_map[subpatch['op_type']](
                    inlets  = subpatch['inlets'],
                    outlets = subpatch['outlets'],
                    config  = subpatch['config'],
                )
            )

    def forward(self, messages):
        for operation in self.operations:
            outlet_group = operation.send(
                inlet_group = [messages[inlet] for inlet in operation.inlets]
            )
            for i, outlet in enumerate(operation.outlets):
                messages[outlet] = outlet_group[i]
        
        return messages