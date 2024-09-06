# Model Construction Utils (moconut)

## Proposed format:
1. Objects:
	- ```{object_name(num_times),...}(num_times)<*params>[*inlet_names] -> [*outlet_names]```
2. Types:
	- ```_type<*params>(repeats)```
3. Param special values:
	- `:` denotes "doesn't matter" or "ignore".
	- `a` denotes auto, or infer argument from previous object.
4. Inlet/Outlet special values:
	- `$number`, denotes a placeholder of variable names based on what's passed into the inlets and which variables the outlets will connect to.

```Python
	import moconut
	
	# 'object_name'<*params>[*inlets] -> [*outlets]
	unet = moconut({
		'conv_relu' : [
			"in<_t<:,a,a,a>>[$1]",
			"conv<3,3>[$1] -> [$1]",
			"relu[$1] -> [$1]",
			"out<_t<:,a,a,a>>[$1]"
		],
		'conv_relu_maxpool' : [
			"in<_t<:,a,a,a>>[$1]",
			"conv_relu(2)[$1] -> [$1]",
			"maxpool<2,2>[$1] -> [$2]",
			"out<_t<:,a,a,a>(2)>[$1, $2]"
		],
		'upconv_concat_conv_relu' : [
			"in<_t<:,a,a,a>(2)>[$1, $2]",
			"upconv<2,2>[$2] -> [$3]",
			"concat[$1,$3] -> [$3]",
			"conv_relu(2)[$3] -> [$3]",
			"out<_t<:,a,a,a>(2)>[$3]"
		],
		'op' : [
			"in<_t<:,572,572,1>>[x0]",
			"conv_relu_maxpool[x0] -> [x0, x1]",
			"conv_relu_maxpool[x1] -> [x1, x2]",
			"conv_relu_maxpool[x2] -> [x2, x3]",
			"conv_relu_maxpool[x3] -> [x3, x4]",
			"conv_relu(2)[x4] -> [x4]",
			
			"upconv_concat_conv_relu[x3,x4] -> [x5]",
			"upconv_concat_conv_relu[x2,x5] -> [x6]",
			"upconv_concat_conv_relu[x1,x6] -> [x7]",
			"upconv_concat_conv_relu[x0,x7] -> [x8]"
			
			"conv<1,1>[x8] -> [x8]"
			"out<_t<:,a,a,a>>[x8]"
		]
	})
```
