import torch
import moconut

class MoconutModule(torch.nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super(MoconutModule, self).__init__()
        self.device = device
        
        self.inlets  = config['inlets']
        self.outlets = config['outlets']
        self.ops = torch.nn.ModuleList()
        for i in range(len(config['operations'])):
            

def test_simple_resnet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mlp_config = {
        'inlets'     : ['in'],
        'operations' : [
            {
                'inlets'    : ['in'],
                'outlets'   : ['x0'],
                'op_name'   : 'linear',
                'op_config' : {
                    'in_features'  : 784,
                    'out_features' : 256
                }
            },
            {
                'inlets'  : ['x0'],
                'outlets' : ['x0'],
                'op_name' : 'selu'
            },
            {
                'inlets'  : ['x0'],
                'outlets' : ['x1'],
                'op_name'   : 'linear',
                'op_config' : {
                    'in_features'  : 256,
                    'out_features' : 256
                }
            },
            {
                'inlets'  : ['x1'],
                'outlets' : ['x1'],
                'op_name' : 'selu'
            },
            {
                'inlets'  : ['x0', 'x1'],
                'outlets' : ['x1'],
                'op_name' : 'add'
            },
            {
                'inlets'  : ['x1'],
                'outlets' : ['x2'],
                'op_name'   : 'linear',
                'op_config' : {
                    'in_features'  : 256,
                    'out_features' : 256
                }
            },
            {
                'inlets'  : ['x2'],
                'outlets' : ['x2'],
                'op_name' : 'selu'
            },
            {
                'inlets'  : ['x1', 'x2'],
                'outlets' : ['x2'],
                'op_name' : 'add'
            },
            {
                'inlets'  : ['x2'],
                'outlets' : ['out'],
                'op_name'   : 'linear',
                'op_config' : {
                    'in_features'  : 256,
                    'out_features' : 10
                }
            },
            {
                'inlets'  : ['out'],
                'outlets' : ['out'],
                'op_name' : 'softmax',
                'op_config' : {
                    'dim' : -1
                }
            },
        ],
        'outlets'    : ['out']
    }
    
    model = MoconutModule(
        config = mlp_config,
        device = device
    )
    