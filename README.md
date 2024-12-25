# Model Construction Utils (moconut)

<!-- `m.o.c.o.n.u.t. oversees construction of new universal templates` ... nah. -->

## Abstraction level 1

```Python
import moconut

# Add a patch to the current patch library
moconut.AddPatch('mlp3',moconut.Patch(
	required = {
		'in_dim' : int,
		'dims'   : list
	},
	independent = {},
	dependent   = {},
	constraints = [
		moconut.constraint.io.injective(), # num_inlets == num_outlets (and each outlet has to be given a value.)
		moconut.constraint.list.has_length(3)('dims')
	],
	compute_graph = {
		'objects' : [
			{
				'name'    : 'linear0',
				'op_type' : 'linear',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {
					'in_features'  : moconut.AttributeName('in_dim'),
					'out_features' : moconut.AttributeName('dims')[0],
					'bias'         : True
				}
			},
			{
				'name'    : 'linear1',
				'op_type' : 'linear',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {
					'in_features'  : moconut.AttributeName('dims')[0],
					'out_features' : moconut.AttributeName('dims')[1],
					'bias'         : True
				}
			},
			{
				'name'    : 'linear2',
				'op_type' : 'linear',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {
					'in_features'  : moconut.AttributeName('dims')[1],
					'out_features' : moconut.AttributeName('dims')[2],
					'bias'         : True
				}
			},
			{
				'name'    : 'activation0',
				'op_type' : 'relu',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {}
			},
			{
				'name'    : 'activation1',
				'op_type' : 'relu',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {}
			},
			{
				'name'    : 'activation2',
				'op_type' : 'tanh',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {}
			}
		],
		'topology' : [
			(['in[0]'], ['linear0[0]']),
			(['linear0[0]'], ['activation0[0]']),
			(['activation0[0]'], ['linear1[0]']),
			(['linear1[0]'], ['activation1[0]']),
			(['activation1[0]'], ['linear2[0]']),
			(['linear2[0]'], ['activation2[0]']),
			(['activation2[0]'], ['out[0]']),
		]
	}
))

moconut.AddPatch('simpleconv1d_classifier',moconut.Patch(
	required = {
		'dims'         : list,
		'kernel_sizes' : list,
	},
	independent = {
		'in_dim' : 1
	},
	dependent   = {
		'strides' : moconut.DependentDefault(
			parents = ['kernel_sizes'],
			dependence = moconut.dependence.list.repeat_match_parent_len(
				data = 1
			)
		)
	},
	constraints = [
		moconut.constraint.list.has_length(3)('dims', 'kernel_sizes')
	],
	compute_graph = {
		'objects' : [
			{
				'name'    : 'conv0',
				'op_type' : 'conv1d',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {
					'in_channels'  : moconut.AttributeName('in_dim'),
					'out_channels' : moconut.AttributeName('dims')[0],
					'kernel_size'  : moconut.AttributeName('kernel_sizes')[0],
					'stride'       : 1,
					'padding'      : 0,
					'dilation'     : 1,
					'groups'       : 1,
					'bias'         : True,
					'padding_mode' : 'zeros'
				}
			},
			{
				'name'    : 'conv1',
				'op_type' : 'conv1d',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {
					'in_channels'  : moconut.AttributeName('dims')[0],
					'out_channels' : moconut.AttributeName('dims')[1],
					'kernel_size'  : moconut.AttributeName('kernel_sizes')[1],
					'stride'       : 1,
					'padding'      : 0,
					'dilation'     : 1,
					'groups'       : 1,
					'bias'         : True,
					'padding_mode' : 'zeros'
				}
			},
			{
				'name'    : 'conv2',
				'op_type' : 'conv1d',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {
					'in_channels'  : moconut.AttributeName('dims')[1],
					'out_channels' : moconut.AttributeName('dims')[2],
					'kernel_size'  : moconut.AttributeName('kernel_sizes')[2],
					'stride'       : 1,
					'padding'      : 0,
					'dilation'     : 1,
					'groups'       : 1,
					'bias'         : True,
					'padding_mode' : 'zeros'
				}
			},
			{
				'name'    : 'mlp3_0',
				'op_type' : 'mlp3',
				'inlets'  : 1,
				'outlets' : 1,
				'config'  : {
					'in_channels'  : moconut.AttributeName('dims')[1],
					'out_channels' : moconut.AttributeName('dims')[2],
					'kernel_size'  : moconut.AttributeName('kernel_sizes')[2],
					'stride'       : 1,
					'padding'      : 0,
					'dilation'     : 1,
					'groups'       : 1,
					'bias'         : True,
					'padding_mode' : 'zeros'
				}
			}
		],
		'topology' : [
			(['in[0]'], ['conv0[0]']),
			
			(['conv0[0]'], ['activation0[0]']),
			(['activation0[0]'], ['conv1[0]']),
			
			(['conv1[0]'], ['activation1[0]']),
			(['activation1[0]'], ['conv2[0]']),
			
			(['conv1[0]'], ['activation2[0]']),

			(['activation2[0]'], ['flatten0[0]']),
			(['flatten0[0]'], ['mlp3_0[0]']),

			(['mlp3_0[0]'], ['out[0]'])
		]
	}
))
```

## TODO:
- [ ] Abstraction level 1
- [ ] DSL Parser
- [ ] Generalize to other AD libs