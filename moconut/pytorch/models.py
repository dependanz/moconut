import torch
import moconut
from typing import Optional

class PyTorchSequentialModule(torch.nn.Module):
    def __init__(
        self,
        config : dict,
        device : Optional[str] = None
    ):
        self.module_list = torch.nn.ModuleList()
        for module_config in config['modules']:
            if module_config['module'] in moconut.pytorch.module_map:
                self.module_list.append(
                    moconut.pytorch.module_map[module_config['module']](
                        *moconut.pytorch.pack_config_map[module_config['module']](
                            config = module_config
                        )
                    )
                )
            elif module_config['module'] in moconut.user_module_map:
                self.module_list.append(
                    moconut.user_module_map[module_config['module']](
                        config = module_config['config'], 
                        device = device
                    )
                )
            else:
                raise NotImplementedError(f'Module "{module_config["module"]}" not implemented')
        self.module_list.to(device)
        
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x