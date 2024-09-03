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
					"in<:,a,a,a>[x0]",
					"conv<3,3>[x0] -> [x0]",
					"relu[x0] -> [x0]",
					"out<_t>[x0]"
			],
			'op' : [
					"in<_t<:,572,572,1>>[x0]",
					"conv_relu[x0] -> [x1]",
					"conv_relu[x1] -> [x2]",
					"conv_relu[x2] -> [x3]"
			]
	})
```
