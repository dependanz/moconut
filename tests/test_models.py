import torch
import moconut
import unittest
from typing import Optional

class TestPatching(unittest.TestCase):

    def test_linear_patching(self):
        #######################################
        # Create a sample Patch model
        #######################################
        class ConvResBlock1DStack(moconut.Patch(
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
            }
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
            config = {
                'in_dim'  : 512,
                'kernel_sizes' : [3, 3],
                # 'dilations'    : [1, 1],
                'activation'   : {'module' : 'leakyrelu'},

                'compute_graph' : [
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
                                        'padding'      : 'same',
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
            }
        )
        
        #######################################
        # Test the patch
        #######################################
        ...
        
if __name__ == '__main__':
    unittest.main()