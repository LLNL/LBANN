from operations import *
from genotypes import *

#import lbann
#import lbann.modules
import lbann.models
import lbann.models.resnet

class Cell(lbann.modules.Module):

   def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
       super().__init__()

       if reduction_prev:
           self.preprocess0 = FactorizedReduce(C_prev_prev, C)
       else:
           self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
       self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

       if reduction:
           op_names, indices = zip(*genotype.reduce)
           concat = genotype.reduce_concat
       else:
           op_names, indices = zip(*genotype.normal)
           concat = genotype.normal_concat
       self._compile(C, op_names, indices, concat, reduction)

   def _compile(self, C, op_names, indices, concat, reduction):
       assert len(op_names) == len(indices)
       self._steps = len(op_names) // 2
       self._concat = concat
       self.multiplier = len(concat)

       self._ops = []
       for name, index in zip(op_names, indices):
           stride = 2 if reduction and index < 2 else 1
           op = OPS[name](C, stride, True)
           self._ops += [op]
       self._indices = indices

   def forward(self, s0, s1): # add drop_prob later
       s0 = self.preprocess0(s0)
       s1 = self.preprocess1(s1)

       states = [s0, s1]
       for i in range(self._steps):
           h1 = states[self._indices[2 * i]]
           h2 = states[self._indices[2 * i + 1]]
           op1 = self._ops[2 * i]
           op2 = self._ops[2 * i + 1]
           #print(h1)
           #print(op1)
           h1 = op1(h1)
           h2 = op2(h2)
           s = lbann.Sum(h1, h2) #h1 + h2
           states += [s]

       return lbann.Concatenation([states[i] for i in self._concat], axis=0)

class NetworkCIFAR(lbann.modules.Module):

   def __init__(self, C, num_classes, layers, auxiliary, genotype):
       super().__init__()
       self._num_classes = num_classes
       self._layers = layers
       self._auxiliary = auxiliary

       stem_multiplier = 3
       C_curr = stem_multiplier * C
       self.stem1 = lbann.modules.Convolution2dModule(out_channels = C_curr,
                                                      kernel_size = 3,
                                                      stride = 1,
                                                      bias = False)
       #self.stem2 = lbann.BatchNormalization

       C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
       self.cells = []
       reduction_prev = False
       for i in range(layers):
           if i in [layers // 3, 2 * layers // 3]:
               C_curr *= 2
               reduction = True
           else:
               reduction = False

           cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

           reduction_prev = reduction
           self.cells += [cell]
           C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
           if i == 2 * layers // 3:
               C_to_auxiliary = C_prev

   def forward(self, input):
       logits_aux = None
       s0 = s1 = self.stem1(input)
       #input2 = self.stem1(input)
       #s0 = s1 = self.stem2(input2)
       for i, cell in enumerate(self.cells):
           s0, s1 = s1, cell(s0, s1)
           if i == 2 * self._layers // 3:
               if self._auxiliary and self.training:
                   logits_aux = self.auxiliary_head(s1)

       out = lbann.ChannelwiseMean(s1)

       logits = lbann.FullyConnected(out, num_neurons = self._num_classes)
       return logits, logits_aux

if __name__ == '__main__':

    # Dummy input
    reader = lbann.reader_pb2.DataReader()
    reader_ = reader.reader.add()
    reader_.name = 'synthetic'
    reader_.role = 'train'
    reader_.num_samples = 1
    reader_.num_labels = 1
    reader_.synth_dimensions = '1'
    reader_.percent_of_data_to_use = 1.0
    input_ = lbann.Input(data_field='samples')
   
    # Construct model with random input 
    genome = NASNet
    mymodel = NetworkCIFAR(32, 10, 20, False, genome) # nsga uses 34 instead of 32
    x = lbann.Gaussian(neuron_dims='3 32 32')
    y = mymodel(x)

    # Run LBANN
    model = lbann.Model(epochs=1, layers=[input_, x])
    trainer = lbann.Trainer(mini_batch_size=1)
    lbann.run(trainer=trainer,
              model=model,
              data_reader=reader,
              optimizer=lbann.SGD(),
              job_name='lbann_nasnet_test')
