from moconut.utils import pack_args_in_order_with_defaults

################################################################
# Normalizations
################################################################
def pack_batchnorm2d_config(config : dict):
    arg_order = ['num_features', 'eps', 'momentum', 'affine', 'track_running_stats']
    arg_defaults = {
        'eps'                 : 0.00001, 
        'momentum'            : 0.1, 
        'affine'              : True, 
        'track_running_stats' : True
    }

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)