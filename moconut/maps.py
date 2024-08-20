import torch

"""
  [TODO] Finish "docstrings"
  Module Map for DAG-based construction of parametric models 
    (essentially a dictionary mapping strings to torch modules)
"""
module_map = {
    ################################################################################################################
    # CONVOLUTION                                                                                                  #
    ################################################################################################################
    'conv1d'   : torch.nn.Conv1d,
    'conv2d'   : torch.nn.Conv2d,
    'conv3d'   : torch.nn.Conv3d,
    'convtranspose1d'   : torch.nn.ConvTranspose1d,
    'convtranspose2d'   : torch.nn.ConvTranspose2d,
    'convtranspose3d'   : torch.nn.ConvTranspose3d,
    'lazyconv1d'   : torch.nn.LazyConv1d,
    'lazyconv2d'   : torch.nn.LazyConv2d,
    'lazyconv3d'   : torch.nn.LazyConv3d,
    'lazyconvtranspose1d'   : torch.nn.LazyConvTranspose1d,
    'lazyconvtranspose2d'   : torch.nn.LazyConvTranspose2d,
    'lazyconvtranspose3d'   : torch.nn.LazyConvTranspose3d,
    'unfold' : torch.nn.Unfold,
    'fold' : torch.nn.Unfold,
    
    ################################################################################################################
    # POOLING                                                                                                      #
    ################################################################################################################
    'maxpool1d' : torch.nn.MaxPool1d,
    'maxpool2d' : torch.nn.MaxPool2d,
    'maxpool3d' : torch.nn.MaxPool3d,
    'maxunpool1d' : torch.nn.MaxUnpool1d,
    'maxunpool2d' : torch.nn.MaxUnpool2d,
    'maxunpool3d' : torch.nn.MaxUnpool3d,
    'avgpool1d' : torch.nn.AvgPool1d,
    'avgpool2d' : torch.nn.AvgPool2d,
    'avgpool3d' : torch.nn.AvgPool3d,
    'fractionalmaxpool2d' : torch.nn.FractionalMaxPool2d,
    'fractionalmaxpool3d' : torch.nn.FractionalMaxPool3d,
    'lppool1d' : torch.nn.LPPool1d,
    'lppool2d' : torch.nn.LPPool2d,
    'lppool3d' : torch.nn.LPPool3d,
    'adaptivemaxpool1d' : torch.nn.AdaptiveMaxPool1d,
    'adamaxpool1d'      : torch.nn.AdaptiveMaxPool1d,
    'adaptivemaxpool2d' : torch.nn.AdaptiveMaxPool2d,
    'adamaxpool2d'      : torch.nn.AdaptiveMaxPool2d,
    'adaptivemaxpool3d' : torch.nn.AdaptiveMaxPool3d,
    'adamaxpool3d'      : torch.nn.AdaptiveMaxPool3d,
    'adaptiveavgpool1d' : torch.nn.AdaptiveAvgPool1d,
    'adaavgpool1d'      : torch.nn.AdaptiveAvgPool1d,
    'adaptiveavgpool2d' : torch.nn.AdaptiveAvgPool2d,
    'adaavgpool2d'      : torch.nn.AdaptiveAvgPool2d,
    'adaptiveavgpool3d' : torch.nn.AdaptiveAvgPool3d,
    'adaavgpool3d'      : torch.nn.AdaptiveAvgPool3d,

    ################################################################################################################
    # PADDING                                                                                                      #
    ################################################################################################################
    'reflectionpad1d' : torch.nn.ReflectionPad1d,
    'reflectionpad2d' : torch.nn.ReflectionPad2d,
    'reflectionpad3d' : torch.nn.ReflectionPad3d,
    'replicationpad1d' : torch.nn.ReplicationPad1d,
    'replicationpad2d' : torch.nn.ReplicationPad2d,
    'replicationpad3d' : torch.nn.ReplicationPad3d,
    'zeropad1d' : torch.nn.ZeroPad1d,
    'zeropad2d' : torch.nn.ZeroPad2d,
    'zeropad3d' : torch.nn.ZeroPad3d,
    'constpad1d'    : torch.nn.ConstantPad1d,
    'constantpad1d' : torch.nn.ConstantPad1d,
    'constpad2d'    : torch.nn.ConstantPad2d,
    'constantpad2d' : torch.nn.ConstantPad2d,
    'constpad3d'    : torch.nn.ConstantPad3d,
    'constantpad3d' : torch.nn.ConstantPad3d,
    'circlepad1d'   : torch.nn.CircularPad1d,
    'circularpad1d' : torch.nn.CircularPad1d,
    'circlepad2d'   : torch.nn.CircularPad2d,
    'circularpad2d' : torch.nn.CircularPad2d,
    'circlepad3d'   : torch.nn.CircularPad3d,
    'circularpad3d' : torch.nn.CircularPad3d,

    ################################################################################################################
    # NON-LINEAR ACTIVATIONS (WEIGHTED SUM, NONLINEARITY)                                                          #
    ################################################################################################################
    'elu' : torch.nn.ELU,
    'hardshrink' : torch.nn.Hardshrink,
    'hardsig'     : torch.nn.Hardsigmoid,
    'hardsigmoid' : torch.nn.Hardsigmoid,
    'hardtanh'     : torch.nn.Hardtanh,
    'hardswish'     : torch.nn.Hardswish,
    'lrelu'     : torch.nn.LeakyReLU,
    'leakyrelu' : torch.nn.LeakyReLU,
    'logsig'     : torch.nn.LogSigmoid,
    'logsigmoid' : torch.nn.LogSigmoid,
    'mha'                : torch.nn.MultiheadAttention,
    'multiheadattention' : torch.nn.MultiheadAttention,
    'prelu'     : torch.nn.PReLU,
    'relu'     : torch.nn.ReLU,
    'relu6'     : torch.nn.ReLU6,
    'rrelu'     : torch.nn.RReLU,
    'selu'     : torch.nn.SELU,
    'celu'     : torch.nn.CELU,
    'gelu'     : torch.nn.GELU,
    'sig'     : torch.nn.Sigmoid,
    'sigmoid' : torch.nn.Sigmoid,
    'silu'     : torch.nn.SiLU,
    'mish'     : torch.nn.Mish,
    'softplus'     : torch.nn.Softplus,
    'softshrink'     : torch.nn.Softshrink,
    'softsign' : torch.nn.Softsign,
    'tanh' : torch.nn.Tanh,
    'tanhshrink' : torch.nn.Tanhshrink,
    'thresh' : torch.nn.Threshold,
    'threshold' : torch.nn.Threshold,
    'glu' : torch.nn.GLU,
    
    ################################################################################################################
    # NON-LINEAR ACTIVATIONS (OTHER)                                                                               #
    ################################################################################################################
    'softmin'    : torch.nn.Softmin,
    'softmax'    : torch.nn.Softmax,
    'logsoftmax' : torch.nn.LogSoftmax,
    'softmax2d' : torch.nn.Softmax2d,
    'adaptivelogsoftmaxwithloss' : torch.nn.AdaptiveLogSoftmaxWithLoss,

    ################################################################################################################
    # NORMALIZATION LAYERS                                                                                         #
    ################################################################################################################
    'batchnorm1d'    : torch.nn.BatchNorm1d,
    'batchnorm2d'    : torch.nn.BatchNorm2d,
    'batchnorm3d'    : torch.nn.BatchNorm3d,
    'lazybatchnorm1d'    : torch.nn.LazyBatchNorm1d,
    'lazybatchnorm2d'    : torch.nn.LazyBatchNorm2d,
    'lazybatchnorm3d'    : torch.nn.LazyBatchNorm3d,
    'groupnorm'    : torch.nn.GroupNorm,
    'syncbatchnorm' : torch.nn.SyncBatchNorm,
    'instancenorm1d' : torch.nn.InstanceNorm1d,
    'instancenorm2d' : torch.nn.InstanceNorm2d,
    'instancenorm3d' : torch.nn.InstanceNorm3d,
    'lazyinstancenorm1d' : torch.nn.LazyInstanceNorm1d,
    'lazyinstancenorm2d' : torch.nn.LazyInstanceNorm2d,
    'lazyinstancenorm3d' : torch.nn.LazyInstanceNorm3d,
    'layernorm' : torch.nn.LayerNorm,
    'localresponsenorm' : torch.nn.LocalResponseNorm,
    'rmsnorm' : torch.nn.RMSNorm,
    
    ################################################################################################################
    # RECURRENT LAYERS                                                                                             #
    ################################################################################################################
    'rnnbase'  : torch.nn.RNNBase,
    'rnn'      : torch.nn.RNN,
    'lstm'     : torch.nn.LSTM,
    'gru'      : torch.nn.GRU,
    'rnncell'  : torch.nn.RNNCell,
    'lstmcell' : torch.nn.LSTMCell,
    'grucell'  : torch.nn.GRUCell,
    
    ################################################################################################################
    # TRANSFORMER LAYERS                                                                                           #
    ################################################################################################################
    'transformer'             : torch.nn.Transformer,
    'transformerencoder'      : torch.nn.TransformerEncoder,
    'transformerdecoder'      : torch.nn.TransformerDecoder,
    'transformerencoderlayer' : torch.nn.TransformerEncoderLayer,
    'transformerdecoderlayer' : torch.nn.TransformerDecoderLayer,
    
    ################################################################################################################
    # LINEAR LAYERS                                                                                                #
    ################################################################################################################
    None         : torch.nn.Identity,
    'identity'   : torch.nn.Identity,
    'linear'     : torch.nn.Linear,
    'bilinear'   : torch.nn.Bilinear,
    'lazylinear' : torch.nn.LazyLinear,
    
    ################################################################################################################
    # DROPOUT LAYERS                                                                                               #
    ################################################################################################################
    'dropout'             : torch.nn.Dropout,
    'dropout1d'           : torch.nn.Dropout1d,
    'dropout2d'           : torch.nn.Dropout2d,
    'dropout3d'           : torch.nn.Dropout3d,
    'alphadropout'        : torch.nn.AlphaDropout,
    'featurealphadropout' : torch.nn.FeatureAlphaDropout,
    
    ################################################################################################################
    # SPARSE LAYERS                                                                                                #
    ################################################################################################################
    'emb'          : torch.nn.Embedding,
    'embed'        : torch.nn.Embedding,
    'embedding'    : torch.nn.Embedding,
    'embbag'       : torch.nn.EmbeddingBag,
    'embedbag'     : torch.nn.EmbeddingBag,
    'embeddingbag' : torch.nn.EmbeddingBag,
    
    ################################################################################################################
    # DISTANCE FUNCTIONS                                                                                           #
    ################################################################################################################
    'cosinesimilarity' : torch.nn.CosineSimilarity,
    'pairwisedistance' : torch.nn.PairwiseDistance,
    
    ################################################################################################################
    # LOSS FUNCTIONS                                                                                               #
    ################################################################################################################
    'l1loss'                        : torch.nn.L1Loss,
    'mseloss'                       : torch.nn.MSELoss,
    'crossentropyloss'              : torch.nn.CrossEntropyLoss,
    'ctcloss'                       : torch.nn.CTCLoss,
    'nllloss'                       : torch.nn.NLLLoss,
    'poissonnllloss'                : torch.nn.PoissonNLLLoss,
    'gaussiannllloss'               : torch.nn.GaussianNLLLoss,
    'kldivloss'                     : torch.nn.KLDivLoss,
    'bceloss'                       : torch.nn.BCELoss,
    'bcewithlogitsloss'             : torch.nn.BCEWithLogitsLoss,
    'marginrankingloss'             : torch.nn.MarginRankingLoss,
    'hingeembeddingloss'            : torch.nn.HingeEmbeddingLoss,
    'multilabelmarginloss'          : torch.nn.MultiLabelMarginLoss,
    'huberloss'                     : torch.nn.HuberLoss,
    'smoothl1loss'                  : torch.nn.SmoothL1Loss,
    'softmarginloss'                : torch.nn.SoftMarginLoss,
    'multilabelsoftmarginloss'      : torch.nn.MultiLabelSoftMarginLoss,
    'cosineembeddingloss'           : torch.nn.CosineEmbeddingLoss,
    'multimarginloss'               : torch.nn.MultiMarginLoss,
    'tripletmarginloss'             : torch.nn.TripletMarginLoss,
    'tripletmarginwithdistanceloss' : torch.nn.TripletMarginWithDistanceLoss,
    
    ################################################################################################################
    # VISION LAYERS                                                                                                #
    ################################################################################################################
    'pixelshuffle'          : torch.nn.PixelShuffle,
    'pixelunshuffle'        : torch.nn.PixelUnshuffle,
    'upsample'              : torch.nn.Upsample,
    'upsamplingnearest2d'   : torch.nn.UpsamplingNearest2d,
    'upsamplingbilinear2d'  : torch.nn.UpsamplingBilinear2d,
    
    ################################################################################################################
    # SHUFFLE LAYERS                                                                                               #
    ################################################################################################################
    'channelshuffle' : torch.nn.ChannelShuffle,
    
    ################################################################################################################
    # DATAPARALLEL LAYERS (MULTI-GPU, DISTRIBUTED)                                                                 #
    ################################################################################################################
    'dataparallel' : torch.nn.DataParallel,
    'distributeddataparallel' : torch.nn.parallel.DistributedDataParallel,
    
    ################################################################################################################
    # UTILITIES                                                                                                    #
    ################################################################################################################
    'flatten'   : torch.nn.Flatten,
    'unflatten' : torch.nn.Unflatten
}