################################################################
# Argument Packing Util
################################################################
def pack_args_in_order_with_defaults(config, arg_order, arg_defaults):
    args = []
    for arg in arg_order:
        if arg in config:
            args.append(config[arg])
        elif arg in arg_defaults:
            args.append(arg_defaults[arg])
        else:
            raise ValueError(f'Missing required argument: "{arg}"')
    return args