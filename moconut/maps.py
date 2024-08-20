import torch

"""
  Module Map for DAG-based construction of parametric models 
    (essentially a dictionary mapping strings to torch modules)
"""
module_map = {
    ################################################################################################################
    # CONVOLUTION                                                                                                  #
    ################################################################################################################
    # torch.nn.Conv1d                                                                                              #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - in_channels  : int                  - Number of channels in input                                      #
    #     - out_channels : int                  - Number of channels in output                                     #
    #     - kernel_size  : int | tuple          - Size of convolving kernel                                        #
    #     - stride       : [int | tuple]?       - Stride of the convolution                     (Default: 1)       #
    #     - padding      : [int | tuple | str]? - Padding added to both sides of input          (Default: 0)       #
    #     - padding_mode : [str]?               - 'zeros', 'reflect', 'replicate' or 'circular' (Default: 'zeros') #
    #     - dilation     : [int | tuple]?       - Spacing between kernel elements               (Default: 1)       #
    #     - groups       : [int]?               - Number of blocked connections from input                         #
    #                                             channels to output channels                   (Default: 1)       #
    #     - bias         : [bool]?              - If True, adds learnable bias to output        (Default: True)    #
    # ---------------                                                                                              #
    # Variables                                                                                                    #
    # ---------------                                                                                              #
    #     - weight       : torch.Tensor         - Learnable weights of module                                      #
    #                                                 dim(out_channels, in_channels / groups, kernel_size)         #
    #     - bias         : torch.Tensor         - Learnable bias of module                                         #
    #                                                 dim(out_channels)                                            #
    ################################################################################################################
    'conv1d'   : torch.nn.Conv1d,
    ################################################################################################################
    # torch.nn.Conv2d                                                                                              #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - in_channels  : int                  - Number of channels in input                                      #
    #     - out_channels : int                  - Number of channels in output                                     #
    #     - kernel_size  : int | tuple          - Size of convolving kernel                                        #
    #     - stride       : [int | tuple]?       - Stride of the convolution                     (Default: 1)       #
    #     - padding      : [int | tuple | str]? - Padding added to all sides of input           (Default: 0)       #
    #     - padding_mode : [str]?               - 'zeros', 'reflect', 'replicate' or 'circular' (Default: 'zeros') #
    #     - dilation     : [int | tuple]?       - Spacing between kernel elements               (Default: 1)       #
    #     - groups       : [int]?               - Number of blocked connections from input                         #
    #                                             channels to output channels                   (Default: 1)       #
    #     - bias         : [bool]?              - If True, adds learnable bias to output        (Default: True)    #
    # ---------------                                                                                              #
    # Variables                                                                                                    #
    # ---------------                                                                                              #
    #     - weight       : torch.Tensor         - Learnable weights of module                                      #
    #                                     dim(out_channels, in_channels / groups, kernel_size[0], kernel_size[1])  #
    #     - bias         : torch.Tensor         - Learnable bias of module                                         #
    #                                                 dim(out_channels)                                            #
    ################################################################################################################
    'conv2d'   : torch.nn.Conv2d,
    ################################################################################################################
    # torch.nn.Conv3d                                                                                              #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - in_channels  : int                  - Number of channels in input                                      #
    #     - out_channels : int                  - Number of channels in output                                     #
    #     - kernel_size  : int | tuple          - Size of convolving kernel                                        #
    #     - stride       : [int | tuple]?       - Stride of the convolution                     (Default: 1)       #
    #     - padding      : [int | tuple | str]? - Padding added to all sides of input           (Default: 0)       #
    #     - padding_mode : [str]?               - 'zeros', 'reflect', 'replicate' or 'circular' (Default: 'zeros') #
    #     - dilation     : [int | tuple]?       - Spacing between kernel elements               (Default: 1)       #
    #     - groups       : [int]?               - Number of blocked connections from input                         #
    #                                             channels to output channels                   (Default: 1)       #
    #     - bias         : [bool]?              - If True, adds learnable bias to output        (Default: True)    #
    # ---------------                                                                                              #
    # Variables                                                                                                    #
    # ---------------                                                                                              #
    #     - weight       : torch.Tensor         - Learnable weights of module                                      #
    #                     dim(out_channels, in_channels / groups, kernel_size[0], kernel_size[1], kernel_size[2])  #
    #     - bias         : torch.Tensor         - Learnable bias of module                                         #
    #                                                 dim(out_channels)                                            #
    ################################################################################################################
    'conv3d'   : torch.nn.Conv3d,
    ################################################################################################################
    # torch.nn.ConvTranspose1d                                                                                     #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - in_channels  : int                  - Number of channels in input                                      #
    #     - out_channels : int                  - Number of channels in output                                     #
    #     - kernel_size  : int | tuple          - Size of convolving kernel                                        #
    #     - stride       : [int | tuple]?       - Stride of the convolution                     (Default: 1)       #
    #     - padding      : [int | tuple]? - dilation * (kernel_size - 1) - padding                                 #
    #                                             zero-padding will be added to both sides of input (Default: 0)   #
    #     - output_padding : [int | tuple]?     - Additional size added to one side of the output shape            #
    #                                             (Default: 0)                                                     #
    #     - groups       : [int]?               - Number of blocked connections from input                         #
    #                                             channels to output channels                   (Default: 1)       #
    #     - bias         : [bool]?              - If True, adds learnable bias to output        (Default: True)    #
    #     - dilation     : [int | tuple]?       - Spacing between kernel elements               (Default: 1)       #
    # ---------------                                                                                              #
    # Variables                                                                                                    #
    # ---------------                                                                                              #
    #     - weight       : torch.Tensor         - Learnable weights of module                                      #
    #                                             dim(in_channels, out_channels / groups, kernel_size)             #
    #     - bias         : torch.Tensor         - Learnable bias of module                                         #
    #                                                 dim(out_channels)                                            #
    ################################################################################################################
    'convtranspose1d'   : torch.nn.ConvTranspose1d,
    ################################################################################################################
    # torch.nn.ConvTranspose2d                                                                                     #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - in_channels  : int                  - Number of channels in input                                      #
    #     - out_channels : int                  - Number of channels in output                                     #
    #     - kernel_size  : int | tuple          - Size of convolving kernel                                        #
    #     - stride       : [int | tuple]?       - Stride of the convolution                     (Default: 1)       #
    #     - padding      : [int | tuple]? - dilation * (kernel_size - 1) - padding                                 #
    #                                             zero-padding will be added to both sides of input (Default: 0)   #
    #     - output_padding : [int | tuple]?     - Additional size added to one side of the output shape            #
    #                                             (Default: 0)                                                     #
    #     - groups       : [int]?               - Number of blocked connections from input                         #
    #                                             channels to output channels                   (Default: 1)       #
    #     - bias         : [bool]?              - If True, adds learnable bias to output        (Default: True)    #
    #     - dilation     : [int | tuple]?       - Spacing between kernel elements               (Default: 1)       #
    # ---------------                                                                                              #
    # Variables                                                                                                    #
    # ---------------                                                                                              #
    #     - weight       : torch.Tensor         - Learnable weights of module                                      #
    #                                             dim(in_channels, out_channels / groups,                          #
    #                                                                  kernel_size[0], kernel_size[1])             #
    #     - bias         : torch.Tensor         - Learnable bias of module                                         #
    #                                                 dim(out_channels)                                            #
    ################################################################################################################
    'convtranspose2d'   : torch.nn.ConvTranspose2d,
    ################################################################################################################
    # torch.nn.ConvTranspose3d                                                                                     #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - in_channels  : int                  - Number of channels in input                                      #
    #     - out_channels : int                  - Number of channels in output                                     #
    #     - kernel_size  : int | tuple          - Size of convolving kernel                                        #
    #     - stride       : [int | tuple]?       - Stride of the convolution                     (Default: 1)       #
    #     - padding      : [int | tuple]? - dilation * (kernel_size - 1) - padding                                 #
    #                                             zero-padding will be added to both sides of input (Default: 0)   #
    #     - output_padding : [int | tuple]?     - Additional size added to one side of the output shape            #
    #                                             (Default: 0)                                                     #
    #     - groups       : [int]?               - Number of blocked connections from input                         #
    #                                             channels to output channels                   (Default: 1)       #
    #     - bias         : [bool]?              - If True, adds learnable bias to output        (Default: True)    #
    #     - dilation     : [int | tuple]?       - Spacing between kernel elements               (Default: 1)       #
    # ---------------                                                                                              #
    # Variables                                                                                                    #
    # ---------------                                                                                              #
    #     - weight       : torch.Tensor         - Learnable weights of module                                      #
    #                                             dim(in_channels, out_channels / groups,                          #
    #                                                     kernel_size[0], kernel_size[1], kernel_size[2])          #
    #     - bias         : torch.Tensor         - Learnable bias of module                                         #
    #                                                 dim(out_channels)                                            #
    ################################################################################################################
    'convtranspose3d'   : torch.nn.ConvTranspose3d,
    ################################################################################################################
    # torch.nn.LazyConv1d, torch.nn.LazyConv2d, torch.nn.LazyConv3d                                                #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - out_channels : int                  - Number of channels produced by the convolution                   #
    #     - kernel_size  : int | tuple          - Size of the convolving kernel                                    #
    #     - stride       : [int | tuple]?       - Stride of the convolution                     (Default: 1)       #
    #     - padding      : [int | tuple]?       - Zero-padding added to both sides of the input.(Default: 0)       #
    #     - padding_mode : str?                 - 'zeros', 'reflect', 'replicate', 'circular'   (Default: 'zeros') #
    #     - dilation     : [int | tuple]?       - Spacing between kernel elements               (Default: 1)       #
    #     - groups       : [int]?               - Number of blocked connections from input                         #
    #                                             channels to output channels                   (Default: 1)       #
    #     - bias         : [bool]?              - If True, adds learnable bias to output        (Default: True)    #
    ################################################################################################################
    'lazyconv1d'   : torch.nn.LazyConv1d,
    'lazyconv2d'   : torch.nn.LazyConv2d,
    'lazyconv3d'   : torch.nn.LazyConv3d,
    ################################################################################################################
    # torch.nn.LazyConvTranspose1d, torch.nn.LazyConvTranspose2d, torch.nn.LazyConvTranspose3d                     #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - out_channels : int                  - Number of channels produced by the convolution                   #
    #     - kernel_size  : int | tuple          - Size of the convolving kernel                                    #
    #     - stride       : [int | tuple]?       - Stride of the convolution                     (Default: 1)       #
    #     - padding      : [int | tuple]?       - dilation * (kernel_size - 1) - padding                           #
    #                                             zero-padding will be added to both sides of input (Default: 0)   #
    #     - output_padding : [int | tuple]?     - Additional size added to one side of the output shape            #
    #                                             (Default: 0)                                                     #
    #     - groups       : [int]?               - Number of blocked connections from input                         #
    #                                             channels to output channels                   (Default: 1)       #
    #     - bias         : [bool]?              - If True, adds learnable bias to output        (Default: True)    #
    #     - dilation     : [int | tuple]?       - Spacing between kernel elements               (Default: 1)       #
    ################################################################################################################
    'lazyconvtranspose1d'   : torch.nn.LazyConvTranspose1d,
    'lazyconvtranspose2d'   : torch.nn.LazyConvTranspose2d,
    'lazyconvtranspose3d'   : torch.nn.LazyConvTranspose3d,
    ################################################################################################################
    # torch.nn.Unfold - Extracts sliding local blocks from a batched input tensor.                                 #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - kernel_size : int | tuple    - the size of the sliding blocks                                          #
    #     - dilation    : [int | tuple]? - a parameter that controls stride of elements within neighborhood        #
    #                                      (Default: 1)                                                            #
    #     - padding     : [int | tuple]? - implicit zero padding addedon both sides of input (Default: 0)          #
    #     - stride      : [int | tuple]? - the stride of the sliding blocks in the input spatial dimensions        #
    #                                      (Default: 1)                                                            #
    ################################################################################################################
    'unfold' : torch.nn.Unfold,
    ################################################################################################################
    # torch.nn.Fold - Combines an array of sliding local blocks into a large containing tensor.                    #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - output_size : int | tuple    - shape of spatial dimensions of the output (i.e. output.sizes()[2:])     #
    #     - kernel_size : int | tuple    - the size of the sliding blocks                                          #
    #     - dilation    : [int | tuple]? - a parameter that controls stride of elements within neighborhood        #
    #                                      (Default: 1)                                                            #
    #     - padding     : [int | tuple]? - implicit zero padding addedon both sides of input (Default: 0)          #
    #     - stride      : [int | tuple]? - the stride of the sliding blocks in the input spatial dimensions        #
    #                                      (Default: 1)                                                            #
    ################################################################################################################
    'fold' : torch.nn.Unfold,
    
    ################################################################################################################
    # POOLING                                                                                                      #
    ################################################################################################################
    # torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d                                                   #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - kernel_size    : int | tuple[int] - The size of the sliding window, must be > 0.                       #
    #     - stride         : int | tuple[int] - The stride of the sliding window, must be > 0.                     #
    #                                           (Default: kernel_size)                                             #
    #     - padding        : int | tuple[int] - Implicit negative infinity padding to be                           #
    #                                           added on both sides, must be >= 0 and <= kernel_size / 2.          #
    #     - dilation       : int | tuple[int] - The stride between elements within a sliding window, must be > 0.  #
    #     - return_indices : bool             - If True, will return the argmax along with the max values.         #
    #                                           Useful for torch.nn.MaxUnpool1d later                              #
    #     - ceil_mode      : bool             - If True, will use ceil instead of floor to compute the output      #
    #                                           shape. This ensures that every element in the input tensor is      #
    #                                           covered by a sliding window.                                       #
    ################################################################################################################
    'maxpool1d' : torch.nn.MaxPool1d,
    'maxpool2d' : torch.nn.MaxPool2d,
    'maxpool3d' : torch.nn.MaxPool3d,
    ################################################################################################################
    # torch.nn.MaxUnpool1d, torch.nn.MaxUnpool2d, torch.nn.MaxUnpool3d                                             #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - kernel_size : int | tuple - Size of max pooling window                                                 #
    #     - stride      : int | tuple - Stride of the max pooling window.  (Default: kernel_size)                  #
    #     - padding     : int | tuple - Padding that was added to the input                                        #
    ################################################################################################################
    'maxunpool1d' : torch.nn.MaxUnpool1d,
    'maxunpool2d' : torch.nn.MaxUnpool2d,
    'maxunpool3d' : torch.nn.MaxUnpool3d,
    ################################################################################################################
    # torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d                                                   #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - kernel_size       : int | tuple[int] - the size of the window                                          #
    #     - stride            : int | tuple[int] - the stride of the window. (Default: kernel_size)                #
    #     - padding           : int | tuple[int] - implicit zero padding to be added on both sides                 #
    #     - ceil_mode         : bool             - when True, will use ceil instead of floor to compute            #
    #                                              the output shape                                                #
    #     - count_include_pad : bool             - when True, will include the zero-padding in the                 #
    #                                              averaging calculation                                           #
    ################################################################################################################
    'avgpool1d' : torch.nn.AvgPool1d,
    'avgpool2d' : torch.nn.AvgPool2d,
    'avgpool3d' : torch.nn.AvgPool3d,
    ################################################################################################################
    # torch.nn.FractionalMaxPool2d, torch.nn.FractionalMaxPool2d                                                   #
    #       - Applies a [2D | 3D] fractional max pooling over an input signal composed of several input planes.    #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - kernel_size    : int | tuple[(int){2,3}]      - the size of the window to take a max over.             #
    #     - output_size    : int | tuple[(int){2,3}]      - the target output size of the image of                 #
    #                                                       the form oH x oW                                       #
    #     - output_ratio   : float | tuple[(float){2,3}]  - If one wants to have an output size as a ratio of the  #
    #                                                       input size, this option can be given.                  #
    #                                                       Must be in range (0,1)                                 #
    #     - return_indices : bool                         - if True, will return the indices along                 #
    #                                                       with the outputs.                                      #
    ################################################################################################################
    'fractionalmaxpool2d' : torch.nn.FractionalMaxPool2d,
    'fractionalmaxpool3d' : torch.nn.FractionalMaxPool3d,
    ################################################################################################################
    # torch.nn.LPPool1d, torch.nn.LPPool2d, torch.nn.LPPool3d                                                      #
    #       - Applies a power-average pooling over an input signal composed of several input planes.               #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - kernel_size : int | tuple[(int){1,3}] - the size of the window.                                        #
    #     - stride      : int | tuple[(int){1,3}] - the stride of the window. Default value is kernel_size         #
    #     - ceil_mode   : bool                    - when True, will use ceil instead of floor to compute           #
    #                                               the output shape                                               #
    ################################################################################################################
    'lppool1d' : torch.nn.LPPool1d,
    'lppool2d' : torch.nn.LPPool2d,
    'lppool3d' : torch.nn.LPPool3d,
    ################################################################################################################
    # torch.nn.AdaptiveMaxPool1d, torch.nn.AdaptiveMaxPool2d, torch.nn.AdaptiveMaxPool3d                           #
    #       - Applies an adaptive max pooling over an input signal composed of several input planes.               #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - output_size    : int | tuple[(int){1,3}] - target output size L_{out}                                  #
    #     - return_indices : bool                    - if True, will return the indices along with the outputs.    #
    #                                                  Useful to pass to nn.MaxUnpool1d. (Default: False)          #
    ################################################################################################################
    'adaptivemaxpool1d' : torch.nn.AdaptiveMaxPool1d,
    'adamaxpool1d'      : torch.nn.AdaptiveMaxPool1d,
    'adaptivemaxpool2d' : torch.nn.AdaptiveMaxPool2d,
    'adamaxpool2d'      : torch.nn.AdaptiveMaxPool2d,
    'adaptivemaxpool3d' : torch.nn.AdaptiveMaxPool3d,
    'adamaxpool3d'      : torch.nn.AdaptiveMaxPool3d,
    ################################################################################################################
    # torch.nn.AdaptiveMaxPool1d, torch.nn.AdaptiveMaxPool2d, torch.nn.AdaptiveMaxPool3d                           #
    #       - Applies an adaptive avg pooling over an input signal composed of several input planes.               #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - output_size    : int | tuple[(int){1,3}] - target output size L_{out}                                  #
    ################################################################################################################
    'adaptiveavgpool1d' : torch.nn.AdaptiveAvgPool1d,
    'adaavgpool1d'      : torch.nn.AdaptiveAvgPool1d,
    'adaptiveavgpool2d' : torch.nn.AdaptiveAvgPool2d,
    'adaavgpool2d'      : torch.nn.AdaptiveAvgPool2d,
    'adaptiveavgpool3d' : torch.nn.AdaptiveAvgPool3d,
    'adaavgpool3d'      : torch.nn.AdaptiveAvgPool3d,

    ################################################################################################################
    # PADDING                                                                                                      #
    ################################################################################################################
    # torch.nn.ReflectionPad1d, torch.nn.ReflectionPad2d, torch.nn.ReflectionPad3d                                 #
    #       - Pads the input tensor using the reflection of the input boundary.                                    #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - padding : int | tuple - the size of the padding. If is int, uses the same padding in all boundaries.   #
    ################################################################################################################
    'reflectionpad1d' : torch.nn.ReflectionPad1d,
    'reflectionpad2d' : torch.nn.ReflectionPad2d,
    'reflectionpad3d' : torch.nn.ReflectionPad3d,
    ################################################################################################################
    # torch.nn.ReplicationPad1d, torch.nn.ReplicationPad2d, torch.nn.ReplicationPad3d                              #
    #       - Pads the input tensor using replication of the input boundary.                                       #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - padding : int | tuple - the size of the padding. If is int, uses the same padding in all boundaries.   #
    ################################################################################################################
    'replicationpad1d' : torch.nn.ReplicationPad1d,
    'replicationpad2d' : torch.nn.ReplicationPad2d,
    'replicationpad3d' : torch.nn.ReplicationPad3d,
    ################################################################################################################
    # torch.nn.ZeroPad1d, torch.nn.ZeroPad2d, torch.nn.ZeroPad3d                                                   #
    #       - Pads the input tensor boundaries with zero.                                                          #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - padding : int | tuple - the size of the padding. If is int, uses the same padding in all boundaries.   #
    ################################################################################################################
    'zeropad1d' : torch.nn.ZeroPad1d,
    'zeropad2d' : torch.nn.ZeroPad2d,
    'zeropad3d' : torch.nn.ZeroPad3d,
    ################################################################################################################
    # torch.nn.ConstantPad1d, torch.nn.ConstantPad2d, torch.nn.ConstantPad3d                                       #
    #       - Pads the input tensor boundaries with a constant value.                                              #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - padding : int | tuple - the size of the padding. If is int, uses the same padding in all boundaries.   #
    #     - value   : float       - the constant used for padding                                                  #
    ################################################################################################################
    'constpad1d'    : torch.nn.ConstantPad1d,
    'constantpad1d' : torch.nn.ConstantPad1d,
    'constpad2d'    : torch.nn.ConstantPad2d,
    'constantpad2d' : torch.nn.ConstantPad2d,
    'constpad3d'    : torch.nn.ConstantPad3d,
    'constantpad3d' : torch.nn.ConstantPad3d,
    ################################################################################################################
    # torch.nn.CircularPad1d, torch.nn.CircularPad2d, torch.nn.CircularPad3d                                       #
    #       - Pads the input tensor using circular padding of the input boundary.                                  #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - padding : int | tuple - the size of the padding. If is int, uses the same padding in all boundaries.   #
    ################################################################################################################
    'circlepad1d'   : torch.nn.CircularPad1d,
    'circularpad1d' : torch.nn.CircularPad1d,
    'circlepad2d'   : torch.nn.CircularPad2d,
    'circularpad2d' : torch.nn.CircularPad2d,
    'circlepad3d'   : torch.nn.CircularPad3d,
    'circularpad3d' : torch.nn.CircularPad3d,
    
    ################################################################################################################
    # NON-LINEAR ACTIVATIONS (WEIGHTED SUM, NONLINEARITY)                                                          #
    ################################################################################################################
    # torch.nn.ELU                                                                                                 #
    #       - Exponential Linear Unit (ELU) (https://arxiv.org/abs/1511.07289)                                     #
    # ---------------                                                                                              #
    # Parameters                                                                                                   #
    # ---------------                                                                                              #
    #     - alpha : float - (default: 1.0)                                                                         #
    ################################################################################################################






    ######################################################
    # torch.nn.Identity:                                 #
    #    - args   : Any - any argument (unused)          #
    #    - kwargs : Any - any keyword argument (unused)  #
    ######################################################
    None       : torch.nn.Identity,     # Identity - Args{args: any argument; kwargs: any keyword argument}
    'identity' : torch.nn.Identity,

    'linear'   : torch.nn.Linear,       # Linear   - Args{args: in_features, out_features}, Kwargs{bias=True, device=None, dtype=None}
    'bilinear' : torch.nn.Bilinear,     
}