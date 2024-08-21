import typing
from inspect import _empty
from types import NoneType

import torch
from torch import nn

def pack_optionally_iterable(a):
    if isinstance(a, (list, tuple)):
        return [*a]
    else:
        return [a]

################################################################################################################
# N-ARY OPS                                                                                                    #
################################################################################################################

class MoconutNeg(nn.Module):
    def __init__(self):
        super(MoconutNeg, self).__init__()
    
    def forward(self, x):
        return [-x]

class MoconutAdd(nn.Module):
    def __init__(
        self, 
        num_inlets : int = 2
    ):
        super(MoconutAdd, self).__init__()
        self.num_inlets = num_inlets
    
    def forward(
        self, 
        inlets: list
    ):
        assert len(inlets) == self.num_inlets
        x = inlets[0]
        for i in range(1, self.num_inlets):
            x += inlets[i]
        return [x]

class MoconutProd(nn.Module):
    def __init__(
        self, 
        num_inlets : int = 2
    ):
        super(MoconutProd, self).__init__()
        self.num_inlets = num_inlets
    
    def forward(
        self, 
        inlets: list
    ):
        assert len(inlets) == self.num_inlets
        x = inlets[0]
        for i in range(1, self.num_inlets):
            x *= inlets[i]
        return [x]
    
################################################################################################################
# MODULES FROM TORCH.NN                                                                                        #
################################################################################################################

class MoconutAdaptiveAvgPool1d(nn.Module):
    def __init__(
        self,
        output_size : typing.Union[int, NoneType, typing.Tuple[typing.Optional[int], ...]],
    ):
        super(MoconutAdaptiveAvgPool1d, self).__init__()
        self.adaptiveavgpool1d = nn.AdaptiveAvgPool1d(
            output_size = output_size,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.adaptiveavgpool1d(*inlets))

class MoconutAdaptiveAvgPool2d(nn.Module):
    def __init__(
        self,
        output_size : typing.Union[int, NoneType, typing.Tuple[typing.Optional[int], ...]],
    ):
        super(MoconutAdaptiveAvgPool2d, self).__init__()
        self.adaptiveavgpool2d = nn.AdaptiveAvgPool2d(
            output_size = output_size,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.adaptiveavgpool2d(*inlets))

class MoconutAdaptiveAvgPool3d(nn.Module):
    def __init__(
        self,
        output_size : typing.Union[int, NoneType, typing.Tuple[typing.Optional[int], ...]],
    ):
        super(MoconutAdaptiveAvgPool3d, self).__init__()
        self.adaptiveavgpool3d = nn.AdaptiveAvgPool3d(
            output_size = output_size,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.adaptiveavgpool3d(*inlets))

class MoconutAdaptiveLogSoftmaxWithLoss(nn.Module):
    def __init__(
        self,
        in_features,
        n_classes,
        cutoffs : typing.Sequence[int],
        div_value = 4.0,
        head_bias = False,
        device = None,
        dtype = None,
    ):
        super(MoconutAdaptiveLogSoftmaxWithLoss, self).__init__()
        self.adaptivelogsoftmaxwithloss = nn.AdaptiveLogSoftmaxWithLoss(
            in_features = in_features,
            n_classes = n_classes,
            cutoffs = cutoffs,
            div_value = div_value,
            head_bias = head_bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.adaptivelogsoftmaxwithloss(*inlets))

class MoconutAdaptiveMaxPool1d(nn.Module):
    def __init__(
        self,
        output_size : typing.Union[int, NoneType, typing.Tuple[typing.Optional[int], ...]],
        return_indices = False,
    ):
        super(MoconutAdaptiveMaxPool1d, self).__init__()
        self.adaptivemaxpool1d = nn.AdaptiveMaxPool1d(
            output_size = output_size,
            return_indices = return_indices,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.adaptivemaxpool1d(*inlets))

class MoconutAdaptiveMaxPool2d(nn.Module):
    def __init__(
        self,
        output_size : typing.Union[int, NoneType, typing.Tuple[typing.Optional[int], ...]],
        return_indices = False,
    ):
        super(MoconutAdaptiveMaxPool2d, self).__init__()
        self.adaptivemaxpool2d = nn.AdaptiveMaxPool2d(
            output_size = output_size,
            return_indices = return_indices,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.adaptivemaxpool2d(*inlets))

class MoconutAdaptiveMaxPool3d(nn.Module):
    def __init__(
        self,
        output_size : typing.Union[int, NoneType, typing.Tuple[typing.Optional[int], ...]],
        return_indices = False,
    ):
        super(MoconutAdaptiveMaxPool3d, self).__init__()
        self.adaptivemaxpool3d = nn.AdaptiveMaxPool3d(
            output_size = output_size,
            return_indices = return_indices,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.adaptivemaxpool3d(*inlets))

class MoconutAlphaDropout(nn.Module):
    def __init__(
        self,
        p = 0.5,
        inplace = False,
    ):
        super(MoconutAlphaDropout, self).__init__()
        self.alphadropout = nn.AlphaDropout(
            p = p,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.alphadropout(*inlets))

class MoconutAvgPool1d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int]],
        stride : typing.Union[int, typing.Tuple[int]] = None,
        padding : typing.Union[int, typing.Tuple[int]] = 0,
        ceil_mode = False,
        count_include_pad = True,
    ):
        super(MoconutAvgPool1d, self).__init__()
        self.avgpool1d = nn.AvgPool1d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            ceil_mode = ceil_mode,
            count_include_pad = count_include_pad,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.avgpool1d(*inlets))

class MoconutAvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, int]],
        stride : typing.Union[int, typing.Tuple[int, int], NoneType] = None,
        padding : typing.Union[int, typing.Tuple[int, int]] = 0,
        ceil_mode = False,
        count_include_pad = True,
        divisor_override : typing.Optional[int] = None,
    ):
        super(MoconutAvgPool2d, self).__init__()
        self.avgpool2d = nn.AvgPool2d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            ceil_mode = ceil_mode,
            count_include_pad = count_include_pad,
            divisor_override = divisor_override,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.avgpool2d(*inlets))

class MoconutAvgPool3d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, int, int]],
        stride : typing.Union[int, typing.Tuple[int, int, int], NoneType] = None,
        padding : typing.Union[int, typing.Tuple[int, int, int]] = 0,
        ceil_mode = False,
        count_include_pad = True,
        divisor_override : typing.Optional[int] = None,
    ):
        super(MoconutAvgPool3d, self).__init__()
        self.avgpool3d = nn.AvgPool3d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            ceil_mode = ceil_mode,
            count_include_pad = count_include_pad,
            divisor_override = divisor_override,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.avgpool3d(*inlets))

class MoconutBCELoss(nn.Module):
    def __init__(
        self,
        weight : typing.Optional[torch.Tensor] = None,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutBCELoss, self).__init__()
        self.bceloss = nn.BCELoss(
            weight = weight,
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.bceloss(*inlets))

class MoconutBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        weight : typing.Optional[torch.Tensor] = None,
        size_average = None,
        reduce = None,
        reduction = 'mean',
        pos_weight : typing.Optional[torch.Tensor] = None,
    ):
        super(MoconutBCEWithLogitsLoss, self).__init__()
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss(
            weight = weight,
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
            pos_weight = pos_weight,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.bcewithlogitsloss(*inlets))

class MoconutBatchNorm1d(nn.Module):
    def __init__(
        self,
        num_features,
        eps = 1e-05,
        momentum : typing.Optional[float] = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutBatchNorm1d, self).__init__()
        self.batchnorm1d = nn.BatchNorm1d(
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.batchnorm1d(*inlets))

class MoconutBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        eps = 1e-05,
        momentum : typing.Optional[float] = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutBatchNorm2d, self).__init__()
        self.batchnorm2d = nn.BatchNorm2d(
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.batchnorm2d(*inlets))

class MoconutBatchNorm3d(nn.Module):
    def __init__(
        self,
        num_features,
        eps = 1e-05,
        momentum : typing.Optional[float] = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutBatchNorm3d, self).__init__()
        self.batchnorm3d = nn.BatchNorm3d(
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.batchnorm3d(*inlets))

class MoconutBilinear(nn.Module):
    def __init__(
        self,
        in1_features,
        in2_features,
        out_features,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutBilinear, self).__init__()
        self.bilinear = nn.Bilinear(
            in1_features = in1_features,
            in2_features = in2_features,
            out_features = out_features,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.bilinear(*inlets))

class MoconutCELU(nn.Module):
    def __init__(
        self,
        alpha = 1.0,
        inplace = False,
    ):
        super(MoconutCELU, self).__init__()
        self.celu = nn.CELU(
            alpha = alpha,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.celu(*inlets))

class MoconutCTCLoss(nn.Module):
    def __init__(
        self,
        blank = 0,
        reduction = 'mean',
        zero_infinity = False,
    ):
        super(MoconutCTCLoss, self).__init__()
        self.ctcloss = nn.CTCLoss(
            blank = blank,
            reduction = reduction,
            zero_infinity = zero_infinity,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.ctcloss(*inlets))

class MoconutChannelShuffle(nn.Module):
    def __init__(
        self,
        groups,
    ):
        super(MoconutChannelShuffle, self).__init__()
        self.channelshuffle = nn.ChannelShuffle(
            groups = groups,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.channelshuffle(*inlets))

class MoconutCircularPad1d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int]],
    ):
        super(MoconutCircularPad1d, self).__init__()
        self.circularpad1d = nn.CircularPad1d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.circularpad1d(*inlets))

class MoconutCircularPad2d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int]],
    ):
        super(MoconutCircularPad2d, self).__init__()
        self.circularpad2d = nn.CircularPad2d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.circularpad2d(*inlets))

class MoconutCircularPad3d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int, int, int]],
    ):
        super(MoconutCircularPad3d, self).__init__()
        self.circularpad3d = nn.CircularPad3d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.circularpad3d(*inlets))

class MoconutConstantPad1d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int]],
        value,
    ):
        super(MoconutConstantPad1d, self).__init__()
        self.constantpad1d = nn.ConstantPad1d(
            padding = padding,
            value = value,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.constantpad1d(*inlets))

class MoconutConstantPad2d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int]],
        value,
    ):
        super(MoconutConstantPad2d, self).__init__()
        self.constantpad2d = nn.ConstantPad2d(
            padding = padding,
            value = value,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.constantpad2d(*inlets))

class MoconutConstantPad3d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int, int, int]],
        value,
    ):
        super(MoconutConstantPad3d, self).__init__()
        self.constantpad3d = nn.ConstantPad3d(
            padding = padding,
            value = value,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.constantpad3d(*inlets))

class MoconutContainer(nn.Module):
    def __init__(
        self,
        kwargs : typing.Any,
    ):
        super(MoconutContainer, self).__init__()
        self.container = nn.Container(
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.container(*inlets))

class MoconutConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int]],
        stride : typing.Union[int, typing.Tuple[int]] = 1,
        padding : typing.Union[str, int, typing.Tuple[int]] = 0,
        dilation : typing.Union[int, typing.Tuple[int]] = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutConv1d, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.conv1d(*inlets))

class MoconutConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int, int]],
        stride : typing.Union[int, typing.Tuple[int, int]] = 1,
        padding : typing.Union[str, int, typing.Tuple[int, int]] = 0,
        dilation : typing.Union[int, typing.Tuple[int, int]] = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutConv2d, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.conv2d(*inlets))

class MoconutConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int, int, int]],
        stride : typing.Union[int, typing.Tuple[int, int, int]] = 1,
        padding : typing.Union[str, int, typing.Tuple[int, int, int]] = 0,
        dilation : typing.Union[int, typing.Tuple[int, int, int]] = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutConv3d, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.conv3d(*inlets))

class MoconutConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int]],
        stride : typing.Union[int, typing.Tuple[int]] = 1,
        padding : typing.Union[int, typing.Tuple[int]] = 0,
        output_padding : typing.Union[int, typing.Tuple[int]] = 0,
        groups = 1,
        bias = True,
        dilation : typing.Union[int, typing.Tuple[int]] = 1,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutConvTranspose1d, self).__init__()
        self.convtranspose1d = nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.convtranspose1d(*inlets))

class MoconutConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int, int]],
        stride : typing.Union[int, typing.Tuple[int, int]] = 1,
        padding : typing.Union[int, typing.Tuple[int, int]] = 0,
        output_padding : typing.Union[int, typing.Tuple[int, int]] = 0,
        groups = 1,
        bias = True,
        dilation : typing.Union[int, typing.Tuple[int, int]] = 1,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutConvTranspose2d, self).__init__()
        self.convtranspose2d = nn.ConvTranspose2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.convtranspose2d(*inlets))

class MoconutConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int, int, int]],
        stride : typing.Union[int, typing.Tuple[int, int, int]] = 1,
        padding : typing.Union[int, typing.Tuple[int, int, int]] = 0,
        output_padding : typing.Union[int, typing.Tuple[int, int, int]] = 0,
        groups = 1,
        bias = True,
        dilation : typing.Union[int, typing.Tuple[int, int, int]] = 1,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutConvTranspose3d, self).__init__()
        self.convtranspose3d = nn.ConvTranspose3d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.convtranspose3d(*inlets))

class MoconutCosineEmbeddingLoss(nn.Module):
    def __init__(
        self,
        margin = 0.0,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutCosineEmbeddingLoss, self).__init__()
        self.cosineembeddingloss = nn.CosineEmbeddingLoss(
            margin = margin,
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.cosineembeddingloss(*inlets))

class MoconutCosineSimilarity(nn.Module):
    def __init__(
        self,
        dim = 1,
        eps = 1e-08,
    ):
        super(MoconutCosineSimilarity, self).__init__()
        self.cosinesimilarity = nn.CosineSimilarity(
            dim = dim,
            eps = eps,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.cosinesimilarity(*inlets))

class MoconutCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight : typing.Optional[torch.Tensor] = None,
        size_average = None,
        ignore_index = -100,
        reduce = None,
        reduction = 'mean',
        label_smoothing = 0.0,
    ):
        super(MoconutCrossEntropyLoss, self).__init__()
        self.crossentropyloss = nn.CrossEntropyLoss(
            weight = weight,
            size_average = size_average,
            ignore_index = ignore_index,
            reduce = reduce,
            reduction = reduction,
            label_smoothing = label_smoothing,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.crossentropyloss(*inlets))

class MoconutCrossMapLRN2d(nn.Module):
    def __init__(
        self,
        size,
        alpha = 0.0001,
        beta = 0.75,
        k = 1,
    ):
        super(MoconutCrossMapLRN2d, self).__init__()
        self.crossmaplrn2d = nn.CrossMapLRN2d(
            size = size,
            alpha = alpha,
            beta = beta,
            k = k,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.crossmaplrn2d(*inlets))

class MoconutDropout(nn.Module):
    def __init__(
        self,
        p = 0.5,
        inplace = False,
    ):
        super(MoconutDropout, self).__init__()
        self.dropout = nn.Dropout(
            p = p,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.dropout(*inlets))

class MoconutDropout1d(nn.Module):
    def __init__(
        self,
        p = 0.5,
        inplace = False,
    ):
        super(MoconutDropout1d, self).__init__()
        self.dropout1d = nn.Dropout1d(
            p = p,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.dropout1d(*inlets))

class MoconutDropout2d(nn.Module):
    def __init__(
        self,
        p = 0.5,
        inplace = False,
    ):
        super(MoconutDropout2d, self).__init__()
        self.dropout2d = nn.Dropout2d(
            p = p,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.dropout2d(*inlets))

class MoconutDropout3d(nn.Module):
    def __init__(
        self,
        p = 0.5,
        inplace = False,
    ):
        super(MoconutDropout3d, self).__init__()
        self.dropout3d = nn.Dropout3d(
            p = p,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.dropout3d(*inlets))

class MoconutELU(nn.Module):
    def __init__(
        self,
        alpha = 1.0,
        inplace = False,
    ):
        super(MoconutELU, self).__init__()
        self.elu = nn.ELU(
            alpha = alpha,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.elu(*inlets))

class MoconutEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx : typing.Optional[int] = None,
        max_norm : typing.Optional[float] = None,
        norm_type = 2.0,
        scale_grad_by_freq = False,
        sparse = False,
        _weight : typing.Optional[torch.Tensor] = None,
        _freeze = False,
        device = None,
        dtype = None,
    ):
        super(MoconutEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim,
            padding_idx = padding_idx,
            max_norm = max_norm,
            norm_type = norm_type,
            scale_grad_by_freq = scale_grad_by_freq,
            sparse = sparse,
            _weight = _weight,
            _freeze = _freeze,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.embedding(*inlets))

class MoconutEmbeddingBag(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        max_norm : typing.Optional[float] = None,
        norm_type = 2.0,
        scale_grad_by_freq = False,
        mode = 'mean',
        sparse = False,
        _weight : typing.Optional[torch.Tensor] = None,
        include_last_offset = False,
        padding_idx : typing.Optional[int] = None,
        device = None,
        dtype = None,
    ):
        super(MoconutEmbeddingBag, self).__init__()
        self.embeddingbag = nn.EmbeddingBag(
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim,
            max_norm = max_norm,
            norm_type = norm_type,
            scale_grad_by_freq = scale_grad_by_freq,
            mode = mode,
            sparse = sparse,
            _weight = _weight,
            include_last_offset = include_last_offset,
            padding_idx = padding_idx,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.embeddingbag(*inlets))

class MoconutFeatureAlphaDropout(nn.Module):
    def __init__(
        self,
        p = 0.5,
        inplace = False,
    ):
        super(MoconutFeatureAlphaDropout, self).__init__()
        self.featurealphadropout = nn.FeatureAlphaDropout(
            p = p,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.featurealphadropout(*inlets))

class MoconutFlatten(nn.Module):
    def __init__(
        self,
        start_dim = 1,
        end_dim = -1,
    ):
        super(MoconutFlatten, self).__init__()
        self.flatten = nn.Flatten(
            start_dim = start_dim,
            end_dim = end_dim,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.flatten(*inlets))

class MoconutFold(nn.Module):
    def __init__(
        self,
        output_size : typing.Union[int, typing.Tuple[int, ...]],
        kernel_size : typing.Union[int, typing.Tuple[int, ...]],
        dilation : typing.Union[int, typing.Tuple[int, ...]] = 1,
        padding : typing.Union[int, typing.Tuple[int, ...]] = 0,
        stride : typing.Union[int, typing.Tuple[int, ...]] = 1,
    ):
        super(MoconutFold, self).__init__()
        self.fold = nn.Fold(
            output_size = output_size,
            kernel_size = kernel_size,
            dilation = dilation,
            padding = padding,
            stride = stride,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.fold(*inlets))

class MoconutFractionalMaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, int]],
        output_size : typing.Union[int, typing.Tuple[int, int], NoneType] = None,
        output_ratio : typing.Union[float, typing.Tuple[float, float], NoneType] = None,
        return_indices = False,
        _random_samples = None,
    ):
        super(MoconutFractionalMaxPool2d, self).__init__()
        self.fractionalmaxpool2d = nn.FractionalMaxPool2d(
            kernel_size = kernel_size,
            output_size = output_size,
            output_ratio = output_ratio,
            return_indices = return_indices,
            _random_samples = _random_samples,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.fractionalmaxpool2d(*inlets))

class MoconutFractionalMaxPool3d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, int, int]],
        output_size : typing.Union[int, typing.Tuple[int, int, int], NoneType] = None,
        output_ratio : typing.Union[float, typing.Tuple[float, float, float], NoneType] = None,
        return_indices = False,
        _random_samples = None,
    ):
        super(MoconutFractionalMaxPool3d, self).__init__()
        self.fractionalmaxpool3d = nn.FractionalMaxPool3d(
            kernel_size = kernel_size,
            output_size = output_size,
            output_ratio = output_ratio,
            return_indices = return_indices,
            _random_samples = _random_samples,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.fractionalmaxpool3d(*inlets))

class MoconutGELU(nn.Module):
    def __init__(
        self,
        approximate = 'none',
    ):
        super(MoconutGELU, self).__init__()
        self.gelu = nn.GELU(
            approximate = approximate,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.gelu(*inlets))

class MoconutGLU(nn.Module):
    def __init__(
        self,
        dim = -1,
    ):
        super(MoconutGLU, self).__init__()
        self.glu = nn.GLU(
            dim = dim,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.glu(*inlets))

class MoconutGRU(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutGRU, self).__init__()
        self.gru = nn.GRU(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.gru(*inlets))

class MoconutGRUCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutGRUCell, self).__init__()
        self.grucell = nn.GRUCell(
            input_size = input_size,
            hidden_size = hidden_size,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.grucell(*inlets))

class MoconutGaussianNLLLoss(nn.Module):
    def __init__(
        self,
        full = False,
        eps = 1e-06,
        reduction = 'mean',
    ):
        super(MoconutGaussianNLLLoss, self).__init__()
        self.gaussiannllloss = nn.GaussianNLLLoss(
            full = full,
            eps = eps,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.gaussiannllloss(*inlets))

class MoconutGroupNorm(nn.Module):
    def __init__(
        self,
        num_groups,
        num_channels,
        eps = 1e-05,
        affine = True,
        device = None,
        dtype = None,
    ):
        super(MoconutGroupNorm, self).__init__()
        self.groupnorm = nn.GroupNorm(
            num_groups = num_groups,
            num_channels = num_channels,
            eps = eps,
            affine = affine,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.groupnorm(*inlets))

class MoconutHardshrink(nn.Module):
    def __init__(
        self,
        lambd = 0.5,
    ):
        super(MoconutHardshrink, self).__init__()
        self.hardshrink = nn.Hardshrink(
            lambd = lambd,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.hardshrink(*inlets))

class MoconutHardsigmoid(nn.Module):
    def __init__(
        self,
        inplace = False,
    ):
        super(MoconutHardsigmoid, self).__init__()
        self.hardsigmoid = nn.Hardsigmoid(
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.hardsigmoid(*inlets))

class MoconutHardswish(nn.Module):
    def __init__(
        self,
        inplace = False,
    ):
        super(MoconutHardswish, self).__init__()
        self.hardswish = nn.Hardswish(
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.hardswish(*inlets))

class MoconutHardtanh(nn.Module):
    def __init__(
        self,
        min_val = -1.0,
        max_val = 1.0,
        inplace = False,
        min_value : typing.Optional[float] = None,
        max_value : typing.Optional[float] = None,
    ):
        super(MoconutHardtanh, self).__init__()
        self.hardtanh = nn.Hardtanh(
            min_val = min_val,
            max_val = max_val,
            inplace = inplace,
            min_value = min_value,
            max_value = max_value,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.hardtanh(*inlets))

class MoconutHingeEmbeddingLoss(nn.Module):
    def __init__(
        self,
        margin = 1.0,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutHingeEmbeddingLoss, self).__init__()
        self.hingeembeddingloss = nn.HingeEmbeddingLoss(
            margin = margin,
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.hingeembeddingloss(*inlets))

class MoconutHuberLoss(nn.Module):
    def __init__(
        self,
        reduction = 'mean',
        delta = 1.0,
    ):
        super(MoconutHuberLoss, self).__init__()
        self.huberloss = nn.HuberLoss(
            reduction = reduction,
            delta = delta,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.huberloss(*inlets))

class MoconutIdentity(nn.Module):
    def __init__(
        self,
        args : typing.Any,
        kwargs : typing.Any,
    ):
        super(MoconutIdentity, self).__init__()
        self.identity = nn.Identity(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.identity(*inlets))

class MoconutInstanceNorm1d(nn.Module):
    def __init__(
        self,
        num_features,
        eps = 1e-05,
        momentum = 0.1,
        affine = False,
        track_running_stats = False,
        device = None,
        dtype = None,
    ):
        super(MoconutInstanceNorm1d, self).__init__()
        self.instancenorm1d = nn.InstanceNorm1d(
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.instancenorm1d(*inlets))

class MoconutInstanceNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        eps = 1e-05,
        momentum = 0.1,
        affine = False,
        track_running_stats = False,
        device = None,
        dtype = None,
    ):
        super(MoconutInstanceNorm2d, self).__init__()
        self.instancenorm2d = nn.InstanceNorm2d(
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.instancenorm2d(*inlets))

class MoconutInstanceNorm3d(nn.Module):
    def __init__(
        self,
        num_features,
        eps = 1e-05,
        momentum = 0.1,
        affine = False,
        track_running_stats = False,
        device = None,
        dtype = None,
    ):
        super(MoconutInstanceNorm3d, self).__init__()
        self.instancenorm3d = nn.InstanceNorm3d(
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.instancenorm3d(*inlets))

class MoconutKLDivLoss(nn.Module):
    def __init__(
        self,
        size_average = None,
        reduce = None,
        reduction = 'mean',
        log_target = False,
    ):
        super(MoconutKLDivLoss, self).__init__()
        self.kldivloss = nn.KLDivLoss(
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
            log_target = log_target,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.kldivloss(*inlets))

class MoconutL1Loss(nn.Module):
    def __init__(
        self,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutL1Loss, self).__init__()
        self.l1loss = nn.L1Loss(
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.l1loss(*inlets))

class MoconutLPPool1d(nn.Module):
    def __init__(
        self,
        norm_type,
        kernel_size : typing.Union[int, typing.Tuple[int, ...]],
        stride : typing.Union[int, typing.Tuple[int, ...], NoneType] = None,
        ceil_mode = False,
    ):
        super(MoconutLPPool1d, self).__init__()
        self.lppool1d = nn.LPPool1d(
            norm_type = norm_type,
            kernel_size = kernel_size,
            stride = stride,
            ceil_mode = ceil_mode,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lppool1d(*inlets))

class MoconutLPPool2d(nn.Module):
    def __init__(
        self,
        norm_type,
        kernel_size : typing.Union[int, typing.Tuple[int, ...]],
        stride : typing.Union[int, typing.Tuple[int, ...], NoneType] = None,
        ceil_mode = False,
    ):
        super(MoconutLPPool2d, self).__init__()
        self.lppool2d = nn.LPPool2d(
            norm_type = norm_type,
            kernel_size = kernel_size,
            stride = stride,
            ceil_mode = ceil_mode,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lppool2d(*inlets))

class MoconutLPPool3d(nn.Module):
    def __init__(
        self,
        norm_type,
        kernel_size : typing.Union[int, typing.Tuple[int, ...]],
        stride : typing.Union[int, typing.Tuple[int, ...], NoneType] = None,
        ceil_mode = False,
    ):
        super(MoconutLPPool3d, self).__init__()
        self.lppool3d = nn.LPPool3d(
            norm_type = norm_type,
            kernel_size = kernel_size,
            stride = stride,
            ceil_mode = ceil_mode,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lppool3d(*inlets))

class MoconutLSTM(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutLSTM, self).__init__()
        self.lstm = nn.LSTM(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lstm(*inlets))

class MoconutLSTMCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLSTMCell, self).__init__()
        self.lstmcell = nn.LSTMCell(
            input_size = input_size,
            hidden_size = hidden_size,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lstmcell(*inlets))

class MoconutLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape : typing.Union[int, typing.List[int], torch.Size],
        eps = 1e-05,
        elementwise_affine = True,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(
            normalized_shape = normalized_shape,
            eps = eps,
            elementwise_affine = elementwise_affine,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.layernorm(*inlets))

class MoconutLazyBatchNorm1d(nn.Module):
    def __init__(
        self,
        eps = 1e-05,
        momentum = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLazyBatchNorm1d, self).__init__()
        self.lazybatchnorm1d = nn.LazyBatchNorm1d(
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazybatchnorm1d(*inlets))

class MoconutLazyBatchNorm2d(nn.Module):
    def __init__(
        self,
        eps = 1e-05,
        momentum = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLazyBatchNorm2d, self).__init__()
        self.lazybatchnorm2d = nn.LazyBatchNorm2d(
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazybatchnorm2d(*inlets))

class MoconutLazyBatchNorm3d(nn.Module):
    def __init__(
        self,
        eps = 1e-05,
        momentum = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLazyBatchNorm3d, self).__init__()
        self.lazybatchnorm3d = nn.LazyBatchNorm3d(
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazybatchnorm3d(*inlets))

class MoconutLazyConv1d(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int]],
        stride : typing.Union[int, typing.Tuple[int]] = 1,
        padding : typing.Union[int, typing.Tuple[int]] = 0,
        dilation : typing.Union[int, typing.Tuple[int]] = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutLazyConv1d, self).__init__()
        self.lazyconv1d = nn.LazyConv1d(
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyconv1d(*inlets))

class MoconutLazyConv2d(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int, int]],
        stride : typing.Union[int, typing.Tuple[int, int]] = 1,
        padding : typing.Union[int, typing.Tuple[int, int]] = 0,
        dilation : typing.Union[int, typing.Tuple[int, int]] = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutLazyConv2d, self).__init__()
        self.lazyconv2d = nn.LazyConv2d(
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyconv2d(*inlets))

class MoconutLazyConv3d(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int, int, int]],
        stride : typing.Union[int, typing.Tuple[int, int, int]] = 1,
        padding : typing.Union[int, typing.Tuple[int, int, int]] = 0,
        dilation : typing.Union[int, typing.Tuple[int, int, int]] = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutLazyConv3d, self).__init__()
        self.lazyconv3d = nn.LazyConv3d(
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyconv3d(*inlets))

class MoconutLazyConvTranspose1d(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int]],
        stride : typing.Union[int, typing.Tuple[int]] = 1,
        padding : typing.Union[int, typing.Tuple[int]] = 0,
        output_padding : typing.Union[int, typing.Tuple[int]] = 0,
        groups = 1,
        bias = True,
        dilation : typing.Union[int, typing.Tuple[int]] = 1,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutLazyConvTranspose1d, self).__init__()
        self.lazyconvtranspose1d = nn.LazyConvTranspose1d(
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyconvtranspose1d(*inlets))

class MoconutLazyConvTranspose2d(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int, int]],
        stride : typing.Union[int, typing.Tuple[int, int]] = 1,
        padding : typing.Union[int, typing.Tuple[int, int]] = 0,
        output_padding : typing.Union[int, typing.Tuple[int, int]] = 0,
        groups = 1,
        bias = True,
        dilation = 1,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutLazyConvTranspose2d, self).__init__()
        self.lazyconvtranspose2d = nn.LazyConvTranspose2d(
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyconvtranspose2d(*inlets))

class MoconutLazyConvTranspose3d(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size : typing.Union[int, typing.Tuple[int, int, int]],
        stride : typing.Union[int, typing.Tuple[int, int, int]] = 1,
        padding : typing.Union[int, typing.Tuple[int, int, int]] = 0,
        output_padding : typing.Union[int, typing.Tuple[int, int, int]] = 0,
        groups = 1,
        bias = True,
        dilation : typing.Union[int, typing.Tuple[int, int, int]] = 1,
        padding_mode = 'zeros',
        device = None,
        dtype = None,
    ):
        super(MoconutLazyConvTranspose3d, self).__init__()
        self.lazyconvtranspose3d = nn.LazyConvTranspose3d(
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyconvtranspose3d(*inlets))

class MoconutLazyInstanceNorm1d(nn.Module):
    def __init__(
        self,
        eps = 1e-05,
        momentum = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLazyInstanceNorm1d, self).__init__()
        self.lazyinstancenorm1d = nn.LazyInstanceNorm1d(
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyinstancenorm1d(*inlets))

class MoconutLazyInstanceNorm2d(nn.Module):
    def __init__(
        self,
        eps = 1e-05,
        momentum = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLazyInstanceNorm2d, self).__init__()
        self.lazyinstancenorm2d = nn.LazyInstanceNorm2d(
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyinstancenorm2d(*inlets))

class MoconutLazyInstanceNorm3d(nn.Module):
    def __init__(
        self,
        eps = 1e-05,
        momentum = 0.1,
        affine = True,
        track_running_stats = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLazyInstanceNorm3d, self).__init__()
        self.lazyinstancenorm3d = nn.LazyInstanceNorm3d(
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazyinstancenorm3d(*inlets))

class MoconutLazyLinear(nn.Module):
    def __init__(
        self,
        out_features,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLazyLinear, self).__init__()
        self.lazylinear = nn.LazyLinear(
            out_features = out_features,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.lazylinear(*inlets))

class MoconutLeakyReLU(nn.Module):
    def __init__(
        self,
        negative_slope = 0.01,
        inplace = False,
    ):
        super(MoconutLeakyReLU, self).__init__()
        self.leakyrelu = nn.LeakyReLU(
            negative_slope = negative_slope,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.leakyrelu(*inlets))

class MoconutLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutLinear, self).__init__()
        self.linear = nn.Linear(
            in_features = in_features,
            out_features = out_features,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.linear(*inlets))

class MoconutLocalResponseNorm(nn.Module):
    def __init__(
        self,
        size,
        alpha = 0.0001,
        beta = 0.75,
        k = 1.0,
    ):
        super(MoconutLocalResponseNorm, self).__init__()
        self.localresponsenorm = nn.LocalResponseNorm(
            size = size,
            alpha = alpha,
            beta = beta,
            k = k,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.localresponsenorm(*inlets))

class MoconutLogSigmoid(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutLogSigmoid, self).__init__()
        self.logsigmoid = nn.LogSigmoid(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.logsigmoid(*inlets))

class MoconutLogSoftmax(nn.Module):
    def __init__(
        self,
        dim : typing.Optional[int] = None,
    ):
        super(MoconutLogSoftmax, self).__init__()
        self.logsoftmax = nn.LogSoftmax(
            dim = dim,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.logsoftmax(*inlets))

class MoconutMSELoss(nn.Module):
    def __init__(
        self,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutMSELoss, self).__init__()
        self.mseloss = nn.MSELoss(
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.mseloss(*inlets))

class MoconutMarginRankingLoss(nn.Module):
    def __init__(
        self,
        margin = 0.0,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutMarginRankingLoss, self).__init__()
        self.marginrankingloss = nn.MarginRankingLoss(
            margin = margin,
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.marginrankingloss(*inlets))

class MoconutMaxPool1d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, ...]],
        stride : typing.Union[int, typing.Tuple[int, ...], NoneType] = None,
        padding : typing.Union[int, typing.Tuple[int, ...]] = 0,
        dilation : typing.Union[int, typing.Tuple[int, ...]] = 1,
        return_indices = False,
        ceil_mode = False,
    ):
        super(MoconutMaxPool1d, self).__init__()
        self.maxpool1d = nn.MaxPool1d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            return_indices = return_indices,
            ceil_mode = ceil_mode,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.maxpool1d(*inlets))

class MoconutMaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, ...]],
        stride : typing.Union[int, typing.Tuple[int, ...], NoneType] = None,
        padding : typing.Union[int, typing.Tuple[int, ...]] = 0,
        dilation : typing.Union[int, typing.Tuple[int, ...]] = 1,
        return_indices = False,
        ceil_mode = False,
    ):
        super(MoconutMaxPool2d, self).__init__()
        self.maxpool2d = nn.MaxPool2d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            return_indices = return_indices,
            ceil_mode = ceil_mode,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.maxpool2d(*inlets))

class MoconutMaxPool3d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, ...]],
        stride : typing.Union[int, typing.Tuple[int, ...], NoneType] = None,
        padding : typing.Union[int, typing.Tuple[int, ...]] = 0,
        dilation : typing.Union[int, typing.Tuple[int, ...]] = 1,
        return_indices = False,
        ceil_mode = False,
    ):
        super(MoconutMaxPool3d, self).__init__()
        self.maxpool3d = nn.MaxPool3d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            return_indices = return_indices,
            ceil_mode = ceil_mode,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.maxpool3d(*inlets))

class MoconutMaxUnpool1d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int]],
        stride : typing.Union[int, typing.Tuple[int], NoneType] = None,
        padding : typing.Union[int, typing.Tuple[int]] = 0,
    ):
        super(MoconutMaxUnpool1d, self).__init__()
        self.maxunpool1d = nn.MaxUnpool1d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.maxunpool1d(*inlets))

class MoconutMaxUnpool2d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, int]],
        stride : typing.Union[int, typing.Tuple[int, int], NoneType] = None,
        padding : typing.Union[int, typing.Tuple[int, int]] = 0,
    ):
        super(MoconutMaxUnpool2d, self).__init__()
        self.maxunpool2d = nn.MaxUnpool2d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.maxunpool2d(*inlets))

class MoconutMaxUnpool3d(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, int, int]],
        stride : typing.Union[int, typing.Tuple[int, int, int], NoneType] = None,
        padding : typing.Union[int, typing.Tuple[int, int, int]] = 0,
    ):
        super(MoconutMaxUnpool3d, self).__init__()
        self.maxunpool3d = nn.MaxUnpool3d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.maxunpool3d(*inlets))

class MoconutMish(nn.Module):
    def __init__(
        self,
        inplace = False,
    ):
        super(MoconutMish, self).__init__()
        self.mish = nn.Mish(
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.mish(*inlets))

class MoconutModule(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutModule, self).__init__()
        self.module = nn.Module(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.module(*inlets))

class MoconutModuleDict(nn.Module):
    def __init__(
        self,
        modules : typing.Optional[typing.Mapping[str, torch.nn.modules.module.Module]] = None,
    ):
        super(MoconutModuleDict, self).__init__()
        self.moduledict = nn.ModuleDict(
            modules = modules,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.moduledict(*inlets))

class MoconutModuleList(nn.Module):
    def __init__(
        self,
        modules : typing.Optional[typing.Iterable[torch.nn.modules.module.Module]] = None,
    ):
        super(MoconutModuleList, self).__init__()
        self.modulelist = nn.ModuleList(
            modules = modules,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.modulelist(*inlets))

class MoconutMultiLabelMarginLoss(nn.Module):
    def __init__(
        self,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutMultiLabelMarginLoss, self).__init__()
        self.multilabelmarginloss = nn.MultiLabelMarginLoss(
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.multilabelmarginloss(*inlets))

class MoconutMultiLabelSoftMarginLoss(nn.Module):
    def __init__(
        self,
        weight : typing.Optional[torch.Tensor] = None,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutMultiLabelSoftMarginLoss, self).__init__()
        self.multilabelsoftmarginloss = nn.MultiLabelSoftMarginLoss(
            weight = weight,
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.multilabelsoftmarginloss(*inlets))

class MoconutMultiMarginLoss(nn.Module):
    def __init__(
        self,
        p = 1,
        margin = 1.0,
        weight : typing.Optional[torch.Tensor] = None,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutMultiMarginLoss, self).__init__()
        self.multimarginloss = nn.MultiMarginLoss(
            p = p,
            margin = margin,
            weight = weight,
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.multimarginloss(*inlets))

class MoconutMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout = 0.0,
        bias = True,
        add_bias_kv = False,
        add_zero_attn = False,
        kdim = None,
        vdim = None,
        batch_first = False,
        device = None,
        dtype = None,
    ):
        super(MoconutMultiheadAttention, self).__init__()
        self.multiheadattention = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = dropout,
            bias = bias,
            add_bias_kv = add_bias_kv,
            add_zero_attn = add_zero_attn,
            kdim = kdim,
            vdim = vdim,
            batch_first = batch_first,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.multiheadattention(*inlets))

class MoconutNLLLoss(nn.Module):
    def __init__(
        self,
        weight : typing.Optional[torch.Tensor] = None,
        size_average = None,
        ignore_index = -100,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutNLLLoss, self).__init__()
        self.nllloss = nn.NLLLoss(
            weight = weight,
            size_average = size_average,
            ignore_index = ignore_index,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.nllloss(*inlets))

class MoconutNLLLoss2d(nn.Module):
    def __init__(
        self,
        weight : typing.Optional[torch.Tensor] = None,
        size_average = None,
        ignore_index = -100,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutNLLLoss2d, self).__init__()
        self.nllloss2d = nn.NLLLoss2d(
            weight = weight,
            size_average = size_average,
            ignore_index = ignore_index,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.nllloss2d(*inlets))

class MoconutPReLU(nn.Module):
    def __init__(
        self,
        num_parameters = 1,
        init = 0.25,
        device = None,
        dtype = None,
    ):
        super(MoconutPReLU, self).__init__()
        self.prelu = nn.PReLU(
            num_parameters = num_parameters,
            init = init,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.prelu(*inlets))

class MoconutPairwiseDistance(nn.Module):
    def __init__(
        self,
        p = 2.0,
        eps = 1e-06,
        keepdim = False,
    ):
        super(MoconutPairwiseDistance, self).__init__()
        self.pairwisedistance = nn.PairwiseDistance(
            p = p,
            eps = eps,
            keepdim = keepdim,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.pairwisedistance(*inlets))

class MoconutParameterDict(nn.Module):
    def __init__(
        self,
        parameters : typing.Any = None,
    ):
        super(MoconutParameterDict, self).__init__()
        self.parameterdict = nn.ParameterDict(
            parameters = parameters,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.parameterdict(*inlets))

class MoconutParameterList(nn.Module):
    def __init__(
        self,
        values : typing.Optional[typing.Iterable[typing.Any]] = None,
    ):
        super(MoconutParameterList, self).__init__()
        self.parameterlist = nn.ParameterList(
            values = values,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.parameterlist(*inlets))

class MoconutPixelShuffle(nn.Module):
    def __init__(
        self,
        upscale_factor,
    ):
        super(MoconutPixelShuffle, self).__init__()
        self.pixelshuffle = nn.PixelShuffle(
            upscale_factor = upscale_factor,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.pixelshuffle(*inlets))

class MoconutPixelUnshuffle(nn.Module):
    def __init__(
        self,
        downscale_factor,
    ):
        super(MoconutPixelUnshuffle, self).__init__()
        self.pixelunshuffle = nn.PixelUnshuffle(
            downscale_factor = downscale_factor,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.pixelunshuffle(*inlets))

class MoconutPoissonNLLLoss(nn.Module):
    def __init__(
        self,
        log_input = True,
        full = False,
        size_average = None,
        eps = 1e-08,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutPoissonNLLLoss, self).__init__()
        self.poissonnllloss = nn.PoissonNLLLoss(
            log_input = log_input,
            full = full,
            size_average = size_average,
            eps = eps,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.poissonnllloss(*inlets))

class MoconutRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape : typing.Union[int, typing.List[int], torch.Size],
        eps : typing.Optional[float] = None,
        elementwise_affine = True,
        device = None,
        dtype = None,
    ):
        super(MoconutRMSNorm, self).__init__()
        self.rmsnorm = nn.RMSNorm(
            normalized_shape = normalized_shape,
            eps = eps,
            elementwise_affine = elementwise_affine,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.rmsnorm(*inlets))

class MoconutRNN(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutRNN, self).__init__()
        self.rnn = nn.RNN(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.rnn(*inlets))

class MoconutRNNBase(nn.Module):
    def __init__(
        self,
        mode,
        input_size,
        hidden_size,
        num_layers = 1,
        bias = True,
        batch_first = False,
        dropout = 0.0,
        bidirectional = False,
        proj_size = 0,
        device = None,
        dtype = None,
    ):
        super(MoconutRNNBase, self).__init__()
        self.rnnbase = nn.RNNBase(
            mode = mode,
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = bias,
            batch_first = batch_first,
            dropout = dropout,
            bidirectional = bidirectional,
            proj_size = proj_size,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.rnnbase(*inlets))

class MoconutRNNCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias = True,
        nonlinearity = 'tanh',
        device = None,
        dtype = None,
    ):
        super(MoconutRNNCell, self).__init__()
        self.rnncell = nn.RNNCell(
            input_size = input_size,
            hidden_size = hidden_size,
            bias = bias,
            nonlinearity = nonlinearity,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.rnncell(*inlets))

class MoconutRNNCellBase(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias,
        num_chunks,
        device = None,
        dtype = None,
    ):
        super(MoconutRNNCellBase, self).__init__()
        self.rnncellbase = nn.RNNCellBase(
            input_size = input_size,
            hidden_size = hidden_size,
            bias = bias,
            num_chunks = num_chunks,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.rnncellbase(*inlets))

class MoconutRReLU(nn.Module):
    def __init__(
        self,
        lower = 0.125,
        upper = 0.3333333333333333,
        inplace = False,
    ):
        super(MoconutRReLU, self).__init__()
        self.rrelu = nn.RReLU(
            lower = lower,
            upper = upper,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.rrelu(*inlets))

class MoconutReLU(nn.Module):
    def __init__(
        self,
        inplace = False,
    ):
        super(MoconutReLU, self).__init__()
        self.relu = nn.ReLU(
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.relu(*inlets))

class MoconutReLU6(nn.Module):
    def __init__(
        self,
        inplace = False,
    ):
        super(MoconutReLU6, self).__init__()
        self.relu6 = nn.ReLU6(
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.relu6(*inlets))

class MoconutReflectionPad1d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int]],
    ):
        super(MoconutReflectionPad1d, self).__init__()
        self.reflectionpad1d = nn.ReflectionPad1d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.reflectionpad1d(*inlets))

class MoconutReflectionPad2d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int]],
    ):
        super(MoconutReflectionPad2d, self).__init__()
        self.reflectionpad2d = nn.ReflectionPad2d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.reflectionpad2d(*inlets))

class MoconutReflectionPad3d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int, int, int]],
    ):
        super(MoconutReflectionPad3d, self).__init__()
        self.reflectionpad3d = nn.ReflectionPad3d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.reflectionpad3d(*inlets))

class MoconutReplicationPad1d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int]],
    ):
        super(MoconutReplicationPad1d, self).__init__()
        self.replicationpad1d = nn.ReplicationPad1d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.replicationpad1d(*inlets))

class MoconutReplicationPad2d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int]],
    ):
        super(MoconutReplicationPad2d, self).__init__()
        self.replicationpad2d = nn.ReplicationPad2d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.replicationpad2d(*inlets))

class MoconutReplicationPad3d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int, int, int]],
    ):
        super(MoconutReplicationPad3d, self).__init__()
        self.replicationpad3d = nn.ReplicationPad3d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.replicationpad3d(*inlets))

class MoconutSELU(nn.Module):
    def __init__(
        self,
        inplace = False,
    ):
        super(MoconutSELU, self).__init__()
        self.selu = nn.SELU(
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.selu(*inlets))

class MoconutSequential(nn.Module):
    def __init__(
        self,
        args,
    ):
        super(MoconutSequential, self).__init__()
        self.sequential = nn.Sequential(
            args = args,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.sequential(*inlets))

class MoconutSiLU(nn.Module):
    def __init__(
        self,
        inplace = False,
    ):
        super(MoconutSiLU, self).__init__()
        self.silu = nn.SiLU(
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.silu(*inlets))

class MoconutSigmoid(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.sigmoid(*inlets))

class MoconutSmoothL1Loss(nn.Module):
    def __init__(
        self,
        size_average = None,
        reduce = None,
        reduction = 'mean',
        beta = 1.0,
    ):
        super(MoconutSmoothL1Loss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
            beta = beta,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.smoothl1loss(*inlets))

class MoconutSoftMarginLoss(nn.Module):
    def __init__(
        self,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutSoftMarginLoss, self).__init__()
        self.softmarginloss = nn.SoftMarginLoss(
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.softmarginloss(*inlets))

class MoconutSoftmax(nn.Module):
    def __init__(
        self,
        dim : typing.Optional[int] = None,
    ):
        super(MoconutSoftmax, self).__init__()
        self.softmax = nn.Softmax(
            dim = dim,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.softmax(*inlets))

class MoconutSoftmax2d(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutSoftmax2d, self).__init__()
        self.softmax2d = nn.Softmax2d(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.softmax2d(*inlets))

class MoconutSoftmin(nn.Module):
    def __init__(
        self,
        dim : typing.Optional[int] = None,
    ):
        super(MoconutSoftmin, self).__init__()
        self.softmin = nn.Softmin(
            dim = dim,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.softmin(*inlets))

class MoconutSoftplus(nn.Module):
    def __init__(
        self,
        beta = 1.0,
        threshold = 20.0,
    ):
        super(MoconutSoftplus, self).__init__()
        self.softplus = nn.Softplus(
            beta = beta,
            threshold = threshold,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.softplus(*inlets))

class MoconutSoftshrink(nn.Module):
    def __init__(
        self,
        lambd = 0.5,
    ):
        super(MoconutSoftshrink, self).__init__()
        self.softshrink = nn.Softshrink(
            lambd = lambd,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.softshrink(*inlets))

class MoconutSoftsign(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutSoftsign, self).__init__()
        self.softsign = nn.Softsign(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.softsign(*inlets))

class MoconutSyncBatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        eps = 1e-05,
        momentum : typing.Optional[float] = 0.1,
        affine = True,
        track_running_stats = True,
        process_group : typing.Optional[typing.Any] = None,
        device = None,
        dtype = None,
    ):
        super(MoconutSyncBatchNorm, self).__init__()
        self.syncbatchnorm = nn.SyncBatchNorm(
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            process_group = process_group,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.syncbatchnorm(*inlets))

class MoconutTanh(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutTanh, self).__init__()
        self.tanh = nn.Tanh(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.tanh(*inlets))

class MoconutTanhshrink(nn.Module):
    def __init__(
        self,
        args,
        kwargs,
    ):
        super(MoconutTanhshrink, self).__init__()
        self.tanhshrink = nn.Tanhshrink(
            args = args,
            kwargs = kwargs,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.tanhshrink(*inlets))

class MoconutThreshold(nn.Module):
    def __init__(
        self,
        threshold,
        value,
        inplace = False,
    ):
        super(MoconutThreshold, self).__init__()
        self.threshold = nn.Threshold(
            threshold = threshold,
            value = value,
            inplace = inplace,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.threshold(*inlets))

class MoconutTransformer(nn.Module):
    def __init__(
        self,
        d_model = 512,
        nhead = 8,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        dim_feedforward = 2048,
        dropout = 0.1,
        activation : typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]] = torch.relu,
        custom_encoder : typing.Optional[typing.Any] = None,
        custom_decoder : typing.Optional[typing.Any] = None,
        layer_norm_eps = 1e-05,
        batch_first = False,
        norm_first = False,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            custom_encoder = custom_encoder,
            custom_decoder = custom_decoder,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
            norm_first = norm_first,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.transformer(*inlets))

class MoconutTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer : nn.TransformerDecoderLayer,
        num_layers,
        norm : typing.Optional[torch.nn.modules.module.Module] = None,
    ):
        super(MoconutTransformerDecoder, self).__init__()
        self.transformerdecoder = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = num_layers,
            norm = norm,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.transformerdecoder(*inlets))

class MoconutTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward = 2048,
        dropout = 0.1,
        activation : typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]] = torch.relu,
        layer_norm_eps = 1e-05,
        batch_first = False,
        norm_first = False,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutTransformerDecoderLayer, self).__init__()
        self.transformerdecoderlayer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
            norm_first = norm_first,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.transformerdecoderlayer(*inlets))

class MoconutTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer : nn.TransformerEncoderLayer,
        num_layers,
        norm : typing.Optional[torch.nn.modules.module.Module] = None,
        enable_nested_tensor = True,
        mask_check = True,
    ):
        super(MoconutTransformerEncoder, self).__init__()
        self.transformerencoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = num_layers,
            norm = norm,
            enable_nested_tensor = enable_nested_tensor,
            mask_check = mask_check,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.transformerencoder(*inlets))

class MoconutTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward = 2048,
        dropout = 0.1,
        activation : typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]] = torch.relu,
        layer_norm_eps = 1e-05,
        batch_first = False,
        norm_first = False,
        bias = True,
        device = None,
        dtype = None,
    ):
        super(MoconutTransformerEncoderLayer, self).__init__()
        self.transformerencoderlayer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
            norm_first = norm_first,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.transformerencoderlayer(*inlets))

class MoconutTripletMarginLoss(nn.Module):
    def __init__(
        self,
        margin = 1.0,
        p = 2.0,
        eps = 1e-06,
        swap = False,
        size_average = None,
        reduce = None,
        reduction = 'mean',
    ):
        super(MoconutTripletMarginLoss, self).__init__()
        self.tripletmarginloss = nn.TripletMarginLoss(
            margin = margin,
            p = p,
            eps = eps,
            swap = swap,
            size_average = size_average,
            reduce = reduce,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.tripletmarginloss(*inlets))

class MoconutTripletMarginWithDistanceLoss(nn.Module):
    def __init__(
        self,
        distance_function : typing.Optional[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        margin = 1.0,
        swap = False,
        reduction = 'mean',
    ):
        super(MoconutTripletMarginWithDistanceLoss, self).__init__()
        self.tripletmarginwithdistanceloss = nn.TripletMarginWithDistanceLoss(
            distance_function = distance_function,
            margin = margin,
            swap = swap,
            reduction = reduction,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.tripletmarginwithdistanceloss(*inlets))

class MoconutUnflatten(nn.Module):
    def __init__(
        self,
        dim : typing.Union[int, str],
        unflattened_size : typing.Union[torch.Size, typing.List[int], typing.Tuple[int, ...], typing.Tuple[typing.Tuple[str, int]]],
    ):
        super(MoconutUnflatten, self).__init__()
        self.unflatten = nn.Unflatten(
            dim = dim,
            unflattened_size = unflattened_size,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.unflatten(*inlets))

class MoconutUnfold(nn.Module):
    def __init__(
        self,
        kernel_size : typing.Union[int, typing.Tuple[int, ...]],
        dilation : typing.Union[int, typing.Tuple[int, ...]] = 1,
        padding : typing.Union[int, typing.Tuple[int, ...]] = 0,
        stride : typing.Union[int, typing.Tuple[int, ...]] = 1,
    ):
        super(MoconutUnfold, self).__init__()
        self.unfold = nn.Unfold(
            kernel_size = kernel_size,
            dilation = dilation,
            padding = padding,
            stride = stride,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.unfold(*inlets))

class MoconutUpsample(nn.Module):
    def __init__(
        self,
        size : typing.Union[int, typing.Tuple[int, ...], NoneType] = None,
        scale_factor : typing.Union[float, typing.Tuple[float, ...], NoneType] = None,
        mode = 'nearest',
        align_corners : typing.Optional[bool] = None,
        recompute_scale_factor : typing.Optional[bool] = None,
    ):
        super(MoconutUpsample, self).__init__()
        self.upsample = nn.Upsample(
            size = size,
            scale_factor = scale_factor,
            mode = mode,
            align_corners = align_corners,
            recompute_scale_factor = recompute_scale_factor,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.upsample(*inlets))

class MoconutUpsamplingBilinear2d(nn.Module):
    def __init__(
        self,
        size : typing.Union[int, typing.Tuple[int, int], NoneType] = None,
        scale_factor : typing.Union[float, typing.Tuple[float, float], NoneType] = None,
    ):
        super(MoconutUpsamplingBilinear2d, self).__init__()
        self.upsamplingbilinear2d = nn.UpsamplingBilinear2d(
            size = size,
            scale_factor = scale_factor,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.upsamplingbilinear2d(*inlets))

class MoconutUpsamplingNearest2d(nn.Module):
    def __init__(
        self,
        size : typing.Union[int, typing.Tuple[int, int], NoneType] = None,
        scale_factor : typing.Union[float, typing.Tuple[float, float], NoneType] = None,
    ):
        super(MoconutUpsamplingNearest2d, self).__init__()
        self.upsamplingnearest2d = nn.UpsamplingNearest2d(
            size = size,
            scale_factor = scale_factor,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.upsamplingnearest2d(*inlets))

class MoconutZeroPad1d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int]],
    ):
        super(MoconutZeroPad1d, self).__init__()
        self.zeropad1d = nn.ZeroPad1d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.zeropad1d(*inlets))

class MoconutZeroPad2d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int]],
    ):
        super(MoconutZeroPad2d, self).__init__()
        self.zeropad2d = nn.ZeroPad2d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.zeropad2d(*inlets))

class MoconutZeroPad3d(nn.Module):
    def __init__(
        self,
        padding : typing.Union[int, typing.Tuple[int, int, int, int, int, int]],
    ):
        super(MoconutZeroPad3d, self).__init__()
        self.zeropad3d = nn.ZeroPad3d(
            padding = padding,
        )

    def forward(self, inlets : list):
	    return pack_optionally_iterable(self.zeropad3d(*inlets))