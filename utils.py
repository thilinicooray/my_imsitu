import math
import numpy as np
import torch.nn as nn

def initLinear(linear, val = None):
  if val is None:
    fan = linear.in_features +  linear.out_features 
    spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
  else:
    spread = val
  linear.weight.data.uniform_(-spread,spread)
  linear.bias.data.uniform_(-spread,spread)

def init_gru_cell(input):

  weight = eval('input.weight_ih')
  bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
  nn.init.uniform_(weight, -bias, bias)
  weight = eval('input.weight_hh')
  bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
  nn.init.uniform_(weight, -bias, bias)

  if input.bias:
    weight = eval('input.bias_ih' )
    weight.data.zero_()
    weight.data[input.hidden_size: 2 * input.hidden_size] = 1
    weight = eval('input.bias_hh')
    weight.data.zero_()
    weight.data[input.hidden_size: 2 * input.hidden_size] = 1
