from moconut.pack.utils import pack_args_in_order_with_defaults

################################################################
# Control/Utility
################################################################
def pack_flatten_config(config : dict):
    arg_order = ['start_dim', 'end_dim']
    arg_defaults = {
        'start_dim' : 1,
        'end_dim' : -1
    }

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)