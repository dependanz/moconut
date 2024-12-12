from moconut.utils import pack_args_in_order_with_defaults

################################################################
# Activations
################################################################
def pack_relu_config(config : dict):
    arg_order = ['inplace']
    arg_defaults = {'inplace' : False}

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)