import torch
import moconut
import unittest
from typing import Optional

class TestPatching(unittest.TestCase):

    def test_patching(self):
        #######################################
        # Create a sample Patch
        #######################################
        class ConvResBlock1DStack(moconut.Patch(
            op_type = 'convresblock1dstack',
            required    = {
                'in_dim'       : int,
                'kernel_sizes' : list,
            },
            independent = {
                'activation' : {'module' : 'leakyrelu'}
            },
            dependent   = {
                'dilations'  : moconut.DependentDefault(
                    parents = ['kernel_sizes'],
                    dependence = moconut.dependence.list.repeat_match_len(
                        data = 1
                    )
                )
            },
            compute_graph = [
                {
                    'op_type' : 'repeating_subpatch',
                    'inlets'  : ['x'],
                    'outlets' : ['x'],
                    'config'  : {
                        'repeat_over' : moconut.AttributeName('kernel_sizes'),
                        'compute_graph' : [
                            {
                                    'op_type' : 'leakyrelu',
                                    'inlets'  : ['x'],
                                    'outlets' : ['x'],
                                    'config'  : {}
                            },
                            {
                                'op_type' : 'conv1d',
                                'inlets'  : ['x'],
                                'outlets' : ['x'],
                                'config'  : {
                                    'in_channels'  : moconut.AttributeName('in_dim'),
                                    'out_channels' : moconut.AttributeName('in_dim'),
                                    'kernel_size'  : moconut.AttributeName('kernel_sizes')[...],
                                    'stride'       : 1,
                                    'padding'      : 'valid',
                                    'dilation'     : moconut.AttributeName('dilations')[...],
                                    'groups'       : 1,
                                    'bias'         : True,
                                    'padding_mode' : 'zeros'
                                }
                            },
                        ]
                    }
                },
            ]
        )):
            def __init__(
                self,
                config,
                device : Optional[str] = None
            ):
                super(ConvResBlock1DStack, self).__init__(config, device)

        #######################################
        # Pass a sample config for the patch
        #######################################
        model = ConvResBlock1DStack(
            inlets  = ['x'],
            outlets = ['x'],
            config = {
                'in_dim'  : 512,
                'kernel_sizes' : [3 for _ in range(10)],
                # 'dilations'    : [1, 1],
                # 'activation'   : {'module' : 'leakyrelu'}
            }
        )
        
        #######################################
        # Test the patch
        #######################################
        # print(model)
        test_input = {'x' : torch.randn((5,512,356))}
        test_output = model(test_input)
        
        assert test_output['x'].shape[-1] == 336
    
    def test_custom_subpatch(self):
        # Add an MLP patch
        moconut.Add('mlp', moconut.Patch(
            required = {
                'dims'   : list
            },
            independent = {},
            dependent   = {},
            constraints = [],
            compute_graph = [
                {
                    'op_type' : 'repeating_subpatch',
                    'inlets'  : ['r'],
                    'outlets' : ['r'],
                    'config'  : {
                        'repeat_over' : moconut.AttributeName('dims'),
                        'compute_graph' : [
                            {
                                'op_type' : 'linear',
                                'inlets'  : ['r'],
                                'outlets' : ['r'],
                                'config'  : {
                                    'in_features'  : moconut.AttributeName('dims')[:-1],
                                    'out_features' : moconut.AttributeName('dims')[1:],
                                    'bias'         : True,
                                }
                            },
                            {
                                'op_type' : 'relu',
                                'inlets'  : ['r'],
                                'outlets' : ['r'],
                                'config'  : {}
                            },
                        ]
                    }
                }
            ]
        ))
        
        # Create a simple ResNet
        class ResNet3(moconut.Patch(
            required = {
                'in_dim' : int
            },
            independent = {
                
            },
            dependent = {
                
            }, 
            constraints = [],
            compute_graph = [
                {
                    'op_type' : 'linear',
                    'inlets'  : ['x'],
                    'outlets' : ['r'],
                    'config'  : {
                        
                    }
                },
                {
                    'op_type' : 'relu',
                    'inlets'  : ['r'],
                    'outlets' : ['r'],
                    'config'  : {
                        
                    }
                },
                {
                    'op_type' : 'linear',
                    'inlets'  : ['r'],
                    'outlets' : ['r'],
                    'config'  : {
                        
                    }
                },
                {
                    'op_type' : 'linear',
                    'inlets'  : ['r'],
                    'outlets' : ['r'],
                    'config'  : {
                        
                    }
                }
            ]
        )):
            def __init__(self, config, device):
                super(ResNet3, self).__init__(config, device)
        
        # Initialize the ResNet
        model = ResNet3(
            inlets  = ['x'],
            outlets = ['x'],
            config  = {
                'in_dim' : 784
            }
        )
        
if __name__ == '__main__':
    unittest.main()