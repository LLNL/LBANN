import lbann
import lbann.modules

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: Pool("avg", stride),
    'max_pool_3x3': lambda C, stride, affine: Pool("max", stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),    
}

class BatchNorm(lbann.modules.Module):

    def __init__(self, eps=1e-5, momentum=0.1, affine=True, statistics_group_size=1):
        super().__init__()
        self.decay = 1 - momentum
        self.epsilon = eps
        self.statistics_group_size = statistics_group_size
        self.weights =[
            lbann.Weights(initializer=lbann.ConstantInitializer(value=1)),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0)),
        ]
        if not affine:
            for w in self.weights:
                w.optimizer = lbann.NoOptimizer()

    def forward(self, x):
        return lbann.BatchNormalization(x,
                                        weights = self.weights,
                                        decay = self.decay,
                                        epsilon = self.epsilon,
                                        statistics_group_size = self.statistics_group_size)

class ReLUConvBN(lbann.modules.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.conv = lbann.modules.Convolution2dModule(out_channels = C_out,
                                                      kernel_size = kernel_size,
                                                      stride = stride,
                                                      padding = padding,
                                                      bias = False)
        self.bn = BatchNorm(affine = affine)

    def forward(self, x):
        x = lbann.Relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class DilConv(lbann.modules.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=False):
        super().__init__()
        self.conv1 = lbann.modules.Convolution2dModule(out_channels = C_in,
                                                       kernel_size = kernel_size,
                                                       stride = stride,
                                                       padding = padding,
                                                       dilation = dilation,
                                                       groups = C_in,
                                                       bias = False)
        self.conv2 = lbann.modules.Convolution2dModule(out_channels = C_out,
                                                       kernel_size = kernel_size,
                                                       padding = 0,
                                                       bias = False) 
        self.bn = BatchNorm(affine = affine)

    def forward(self, x):
        x = lbann.Relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x

class SepConv(lbann.modules.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.relu = lbann.Relu
        self.conv1 = lbann.modules.Convolution2dModule(out_channels = C_in,
                                                       kernel_size = kernel_size,
                                                       stride = stride,
                                                       padding = padding,
                                                       groups = C_in,
                                                       bias = False)
        self.conv2 = lbann.modules.Convolution2dModule(out_channels = C_in,
                                                       kernel_size = 1,
                                                       padding = 0,
                                                       bias = False)
        self.bn1 = BatchNorm(affine = affine)
        self.conv3 = lbann.modules.Convolution2dModule(out_channels = C_in,
                                                       kernel_size = kernel_size,
                                                       stride = 1,
                                                       padding = padding,
                                                       groups = C_in,
                                                       bias = False)
        self.conv4 = lbann.modules.Convolution2dModule(out_channels = C_out,
                                                       kernel_size = 1,
                                                       padding = 0,
                                                       bias = False)
        self.bn2 = BatchNorm(affine = affine)

    def forward(self, x):
        x = lbann.Relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = lbann.Relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn2(x)
        return x

class Identity(lbann.modules.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Zero(lbann.modules.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return lbann.Constant(hint_layer=x)
        raise NotImplementedError
        return x[:,:,::self.stride,::self.stride] * 0.

class FactorizedReduce(lbann.modules.Module):

    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        assert C_out % 2 == 0
        self.conv1 = lbann.modules.Convolution2dModule(out_channels = C_out // 2,
                                                       kernel_size = 1,
                                                       stride = 2,
                                                       padding = 0,
                                                       bias = False)
        self.conv2 = lbann.modules.Convolution2dModule(out_channels = C_out // 2,
                                                       kernel_size = 1,
                                                       stride = 2,
                                                       padding = 0,
                                                       bias = False)

        #NSGA repo defines conv2 with exact specs as conv1 which I dont think is necessary
        self.bn = BatchNorm(affine = affine)

    def forward(self, x):
        x = lbann.Relu(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x) ### @todo Input should be x[:,1:,1:]
        x = lbann.Concatenation(x1, x2, axis=0)
        x = self.bn(x)
        return x

class Pool(lbann.modules.Module):

    def __init__(self, mode, stride):
        self.mode = mode
        self.stride = stride

    def forward(self, x):
        if self.mode == "avg":
            return lbann.Pooling(x, 
                                 num_dims = 2,
                                 pool_dims_i = 3,
                                 pool_strides_i = self.stride,
                                 pool_pads_i = 1,
                                 pool_mode = "average_no_pad")
        elif self.mode == "max":
            return lbann.Pooling(x,
                                 num_dims = 2,
                                 pool_dims_i = 3,
                                 pool_strides_i = self.stride,
                                 pool_pads_i = 1,
                                 pool_mode = "max")
                                  
def main():
    x = lbann.Constant(num_neurons='3 224 224')
    ReLUConvBN(2, 4, 3, 1, 0)(x)
    DilConv(2, 4, 3, 1, 0, 0)(x)
    SepConv(2, 4, 3, 1, 0)(x)
    Identity()(x)
    Zero(1)(x)
    FactorizedReduce(2, 4)(x)
    Pool("avg", 1)(x)

if __name__ == '__main__':
    main()   
