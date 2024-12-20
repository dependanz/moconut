import torch
import moconut
from typing import Optional

class ComputeGraphModule(torch.nn.Module):
    """
        ComputeGraphModule

        :config:
        compute_graph -> list[dict], a compute graph where each element in the list is a config for 
        a "patch cable"
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
        for cable in config['compute_graph']:
            self.operations.append(
                moconut.pytorch.patch.cable_map[cable['op_type']](
                    inlets  = cable['inlets'],
                    outlets = cable['outlets'],
                    config  = cable['config'],
                )
            )

    def forward(self, nodes):
        for operation in self.operations:
            outlet_group = operation.send(
                inlet_group = [nodes[inlet] for inlet in operation.inlets]
            )
            for i, outlet in enumerate(operation.outlets):
                nodes[outlet] = outlet_group[i]
        
        return nodes
    
