import torch
from .ops import *

"""
  [TODO] Finish "docstrings"
  Module Map for DAG-based construction of parametric models 
    (essentially a dictionary mapping strings to torch modules)
"""
pytorch_module_map = {
    ################################################################################################################
    # N-ARY OPS                                                                                                    #
    ################################################################################################################
    'neg'  : MoconutNeg,
    '~'    : MoconutNeg,
    'add'  : MoconutAdd,
    '+'    : MoconutAdd,
    'prod' : MoconutProd,
    '*'    : MoconutProd,
    
    ################################################################################################################
    # CONVOLUTION                                                                                                  #
    ################################################################################################################
    'conv1d'   : MoconutConv1d,
    'conv2d'   : MoconutConv2d,
    'conv3d'   : MoconutConv3d,
    'convtranspose1d'   : MoconutConvTranspose1d,
    'convtranspose2d'   : MoconutConvTranspose2d,
    'convtranspose3d'   : MoconutConvTranspose3d,
    'lazyconv1d'   : MoconutLazyConv1d,
    'lazyconv2d'   : MoconutLazyConv2d,
    'lazyconv3d'   : MoconutLazyConv3d,
    'lazyconvtranspose1d'   : MoconutLazyConvTranspose1d,
    'lazyconvtranspose2d'   : MoconutLazyConvTranspose2d,
    'lazyconvtranspose3d'   : MoconutLazyConvTranspose3d,
    'unfold' : MoconutUnfold,
    'fold' : MoconutUnfold,
    
    ################################################################################################################
    # POOLING                                                                                                      #
    ################################################################################################################
    'maxpool1d' : MoconutMaxPool1d,
    'maxpool2d' : MoconutMaxPool2d,
    'maxpool3d' : MoconutMaxPool3d,
    'maxunpool1d' : MoconutMaxUnpool1d,
    'maxunpool2d' : MoconutMaxUnpool2d,
    'maxunpool3d' : MoconutMaxUnpool3d,
    'avgpool1d' : MoconutAvgPool1d,
    'avgpool2d' : MoconutAvgPool2d,
    'avgpool3d' : MoconutAvgPool3d,
    'fractionalmaxpool2d' : MoconutFractionalMaxPool2d,
    'fractionalmaxpool3d' : MoconutFractionalMaxPool3d,
    'lppool1d' : MoconutLPPool1d,
    'lppool2d' : MoconutLPPool2d,
    'lppool3d' : MoconutLPPool3d,
    'adaptivemaxpool1d' : MoconutAdaptiveMaxPool1d,
    'adamaxpool1d'      : MoconutAdaptiveMaxPool1d,
    'adaptivemaxpool2d' : MoconutAdaptiveMaxPool2d,
    'adamaxpool2d'      : MoconutAdaptiveMaxPool2d,
    'adaptivemaxpool3d' : MoconutAdaptiveMaxPool3d,
    'adamaxpool3d'      : MoconutAdaptiveMaxPool3d,
    'adaptiveavgpool1d' : MoconutAdaptiveAvgPool1d,
    'adaavgpool1d'      : MoconutAdaptiveAvgPool1d,
    'adaptiveavgpool2d' : MoconutAdaptiveAvgPool2d,
    'adaavgpool2d'      : MoconutAdaptiveAvgPool2d,
    'adaptiveavgpool3d' : MoconutAdaptiveAvgPool3d,
    'adaavgpool3d'      : MoconutAdaptiveAvgPool3d,

    ################################################################################################################
    # PADDING                                                                                                      #
    ################################################################################################################
    'reflectionpad1d' : MoconutReflectionPad1d,
    'reflectionpad2d' : MoconutReflectionPad2d,
    'reflectionpad3d' : MoconutReflectionPad3d,
    'replicationpad1d' : MoconutReplicationPad1d,
    'replicationpad2d' : MoconutReplicationPad2d,
    'replicationpad3d' : MoconutReplicationPad3d,
    'zeropad1d' : MoconutZeroPad1d,
    'zeropad2d' : MoconutZeroPad2d,
    'zeropad3d' : MoconutZeroPad3d,
    'constpad1d'    : MoconutConstantPad1d,
    'constantpad1d' : MoconutConstantPad1d,
    'constpad2d'    : MoconutConstantPad2d,
    'constantpad2d' : MoconutConstantPad2d,
    'constpad3d'    : MoconutConstantPad3d,
    'constantpad3d' : MoconutConstantPad3d,
    'circlepad1d'   : MoconutCircularPad1d,
    'circularpad1d' : MoconutCircularPad1d,
    'circlepad2d'   : MoconutCircularPad2d,
    'circularpad2d' : MoconutCircularPad2d,
    'circlepad3d'   : MoconutCircularPad3d,
    'circularpad3d' : MoconutCircularPad3d,

    ################################################################################################################
    # NON-LINEAR ACTIVATIONS (WEIGHTED SUM, NONLINEARITY)                                                          #
    ################################################################################################################
    'elu' : MoconutELU,
    'hardshrink' : MoconutHardshrink,
    'hardsig'     : MoconutHardsigmoid,
    'hardsigmoid' : MoconutHardsigmoid,
    'hardtanh'     : MoconutHardtanh,
    'hardswish'     : MoconutHardswish,
    'lrelu'     : MoconutLeakyReLU,
    'leakyrelu' : MoconutLeakyReLU,
    'logsig'     : MoconutLogSigmoid,
    'logsigmoid' : MoconutLogSigmoid,
    'mha'                : MoconutMultiheadAttention,
    'multiheadattention' : MoconutMultiheadAttention,
    'prelu'     : MoconutPReLU,
    'relu'     : MoconutReLU,
    'relu6'     : MoconutReLU6,
    'rrelu'     : MoconutRReLU,
    'selu'     : MoconutSELU,
    'celu'     : MoconutCELU,
    'gelu'     : MoconutGELU,
    'sig'     : MoconutSigmoid,
    'sigmoid' : MoconutSigmoid,
    'silu'     : MoconutSiLU,
    'mish'     : MoconutMish,
    'softplus'     : MoconutSoftplus,
    'softshrink'     : MoconutSoftshrink,
    'softsign' : MoconutSoftsign,
    'tanh' : MoconutTanh,
    'tanhshrink' : MoconutTanhshrink,
    'thresh' : MoconutThreshold,
    'threshold' : MoconutThreshold,
    'glu' : MoconutGLU,
    
    ################################################################################################################
    # NON-LINEAR ACTIVATIONS (OTHER)                                                                               #
    ################################################################################################################
    'softmin'    : MoconutSoftmin,
    'softmax'    : MoconutSoftmax,
    'logsoftmax' : MoconutLogSoftmax,
    'softmax2d' : MoconutSoftmax2d,
    'adaptivelogsoftmaxwithloss' : MoconutAdaptiveLogSoftmaxWithLoss,

    ################################################################################################################
    # NORMALIZATION LAYERS                                                                                         #
    ################################################################################################################
    'batchnorm1d'    : MoconutBatchNorm1d,
    'batchnorm2d'    : MoconutBatchNorm2d,
    'batchnorm3d'    : MoconutBatchNorm3d,
    'lazybatchnorm1d'    : MoconutLazyBatchNorm1d,
    'lazybatchnorm2d'    : MoconutLazyBatchNorm2d,
    'lazybatchnorm3d'    : MoconutLazyBatchNorm3d,
    'groupnorm'    : MoconutGroupNorm,
    'syncbatchnorm' : MoconutSyncBatchNorm,
    'instancenorm1d' : MoconutInstanceNorm1d,
    'instancenorm2d' : MoconutInstanceNorm2d,
    'instancenorm3d' : MoconutInstanceNorm3d,
    'lazyinstancenorm1d' : MoconutLazyInstanceNorm1d,
    'lazyinstancenorm2d' : MoconutLazyInstanceNorm2d,
    'lazyinstancenorm3d' : MoconutLazyInstanceNorm3d,
    'layernorm' : MoconutLayerNorm,
    'localresponsenorm' : MoconutLocalResponseNorm,
    
    ################################################################################################################
    # RECURRENT LAYERS                                                                                             #
    ################################################################################################################
    'rnnbase'  : MoconutRNNBase,
    'rnn'      : MoconutRNN,
    'lstm'     : MoconutLSTM,
    'gru'      : MoconutGRU,
    'rnncell'  : MoconutRNNCell,
    'lstmcell' : MoconutLSTMCell,
    'grucell'  : MoconutGRUCell,
    
    ################################################################################################################
    # TRANSFORMER LAYERS                                                                                           #
    ################################################################################################################
    'transformer'             : MoconutTransformer,
    'transformerencoder'      : MoconutTransformerEncoder,
    'transformerdecoder'      : MoconutTransformerDecoder,
    'transformerencoderlayer' : MoconutTransformerEncoderLayer,
    'transformerdecoderlayer' : MoconutTransformerDecoderLayer,
    
    ################################################################################################################
    # LINEAR LAYERS                                                                                                #
    ################################################################################################################
    None         : MoconutIdentity,
    'identity'   : MoconutIdentity,
    'linear'     : MoconutLinear,
    'bilinear'   : MoconutBilinear,
    'lazylinear' : MoconutLazyLinear,
    
    ################################################################################################################
    # DROPOUT LAYERS                                                                                               #
    ################################################################################################################
    'dropout'             : MoconutDropout,
    'dropout1d'           : MoconutDropout1d,
    'dropout2d'           : MoconutDropout2d,
    'dropout3d'           : MoconutDropout3d,
    'alphadropout'        : MoconutAlphaDropout,
    'featurealphadropout' : MoconutFeatureAlphaDropout,
    
    ################################################################################################################
    # SPARSE LAYERS                                                                                                #
    ################################################################################################################
    'emb'          : MoconutEmbedding,
    'embed'        : MoconutEmbedding,
    'embedding'    : MoconutEmbedding,
    'embbag'       : MoconutEmbeddingBag,
    'embedbag'     : MoconutEmbeddingBag,
    'embeddingbag' : MoconutEmbeddingBag,
    
    ################################################################################################################
    # DISTANCE FUNCTIONS                                                                                           #
    ################################################################################################################
    'cosinesimilarity' : MoconutCosineSimilarity,
    'pairwisedistance' : MoconutPairwiseDistance,
    
    ################################################################################################################
    # LOSS FUNCTIONS                                                                                               #
    ################################################################################################################
    'l1loss'                        : MoconutL1Loss,
    'mseloss'                       : MoconutMSELoss,
    'crossentropyloss'              : MoconutCrossEntropyLoss,
    'ctcloss'                       : MoconutCTCLoss,
    'nllloss'                       : MoconutNLLLoss,
    'poissonnllloss'                : MoconutPoissonNLLLoss,
    'gaussiannllloss'               : MoconutGaussianNLLLoss,
    'kldivloss'                     : MoconutKLDivLoss,
    'bceloss'                       : MoconutBCELoss,
    'bcewithlogitsloss'             : MoconutBCEWithLogitsLoss,
    'marginrankingloss'             : MoconutMarginRankingLoss,
    'hingeembeddingloss'            : MoconutHingeEmbeddingLoss,
    'multilabelmarginloss'          : MoconutMultiLabelMarginLoss,
    'huberloss'                     : MoconutHuberLoss,
    'smoothl1loss'                  : MoconutSmoothL1Loss,
    'softmarginloss'                : MoconutSoftMarginLoss,
    'multilabelsoftmarginloss'      : MoconutMultiLabelSoftMarginLoss,
    'cosineembeddingloss'           : MoconutCosineEmbeddingLoss,
    'multimarginloss'               : MoconutMultiMarginLoss,
    'tripletmarginloss'             : MoconutTripletMarginLoss,
    'tripletmarginwithdistanceloss' : MoconutTripletMarginWithDistanceLoss,
    
    ################################################################################################################
    # VISION LAYERS                                                                                                #
    ################################################################################################################
    'pixelshuffle'          : MoconutPixelShuffle,
    'pixelunshuffle'        : MoconutPixelUnshuffle,
    'upsample'              : MoconutUpsample,
    'upsamplingnearest2d'   : MoconutUpsamplingNearest2d,
    'upsamplingbilinear2d'  : MoconutUpsamplingBilinear2d,
    
    ################################################################################################################
    # SHUFFLE LAYERS                                                                                               #
    ################################################################################################################
    'channelshuffle' : MoconutChannelShuffle,
    
    ################################################################################################################
    # DATAPARALLEL LAYERS (MULTI-GPU, DISTRIBUTED)                                                                 #
    ################################################################################################################
    # 'dataparallel' : MoconutDataParallel,
    # 'distributeddataparallel' : Moconutparallel.DistributedDataParallel,
    
    ################################################################################################################
    # UTILITIES                                                                                                    #
    ################################################################################################################
    'flatten'   : MoconutFlatten,
    'unflatten' : MoconutUnflatten
}