# Model Construction Utils (moconut)

```Python
    import moconut

    model = moconut([
        "in -> (x0:t<:,28,28>)",
        "(x0:t<:,28,28>) -> flatten -> (x0:t<:,784>)",
        "(x0:t<:,784>) -> linear<784,256> -> layernorm<256> -> relu -> (x0:t<:,256>)",
        "(x0:t<:,256>) -> linear<256,256> -> layernorm<256> -> relu -> (x0:t<:,256>)",
        "(x0:t<:,256>) -> linear<256,256> -> layernorm<256> -> relu -> (x0:t<:,256>)",
        "(x0:t<:,256>) -> linear<256,256> -> layernorm<256> -> relu -> (x0:t<:,256>)",
        "(x0:t<:,256>) -> out"
    ])
```

## Proposed format:
1. ```object_name<*params>[*inlet_names] -> [*outlet_names]```
2. Param special values:
	- `:` denotes "doesn't matter" or "ignore".
	- `a` denotes auto, or infer argument from previous object.

```Python
	import moconut
	
	# 'object_name'<*params>[*inlets] -> [*outlets]
	model = moconut({
			'conv_relu' : [
					"in<_t<:,a,a,a>>[x0]",
					"conv<3,3>[x0] -> [x0]",
					"relu[x0] -> [x0]",
					"out<_t<:,a,a,a>>[x0]"
			],
			'op' : [
					"in<_t<:,572,572,1>>[x0]",
					"conv_relu(2)[x0] -> [x0]",
					"maxpool<2,2>[x0] -> [x1]",
					"conv_relu(2)[x1] -> [x1]",
					"maxpool<2,2>[x1] -> [x2]",
					"conv_relu(2)[x2] -> [x2]",
					"maxpool<2,2>[x2] -> [x3]",
					"conv_relu(2)[x3] -> [x3]",
					"maxpool<2,2>[x3] -> [x4]",
					"conv_relu(2)[x4] -> [x4]",
					
					"upconv<2,2>[x4] -> [x5]",
					"concat[x3,x5] -> [x5]",
					"conv_relu(2)[x5] -> [x5]",
					"upconv<2,2>[x5] -> [x6]",
					"concat[x2,x6] -> [x6]",
					"conv_relu(2)[x6] -> [x6]",
					"upconv<2,2>[x6] -> [x7]",
					"concat[x1,x7] -> [x7]",
					"conv_relu(2)[x7] -> [x7]",
					"upconv<2,2>[x7] -> [x8]",
					"concat[x0,x8] -> [x8]",
					"conv_relu(2)[x8] -> [x8]",
					
					"conv<1,1>[x8] -> [x8]"
					"out<_t<:,a,a,a>>[x8]"
			]
	})
```
