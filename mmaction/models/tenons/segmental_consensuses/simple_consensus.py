import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES


class _SimpleConsensus(torch.autograd.Function):
    """Simplest segmental consensus module"""

    def __init__(self,
                 consensus_type='avg',
                 dim=1):
        super(_SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = 'avg'#consensus_type
        self.dim = 1#dim
        self.shape = None

    @staticmethod
    def forward(self, x):
        self.shape = x.size()
        #if self.consensus_type == 'avg':
        #output = x.mean(dim=self.dim, keepdim=True)
        output = x.mean(dim=1, keepdim=True)
        #else:
         #   output = None
        return output

    @staticmethod
    def backward(self, grad_output):
        #if self.consensus_type == 'avg':
        #grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        grad_in = grad_output.expand(self.shape) / float(self.shape[1])
        #else:
        #    grad_in = None
        return grad_in


@SEGMENTAL_CONSENSUSES.register_module
class SimpleConsensus(nn.Module):
    def __init__(self, consensus_type='avg', dim=1):
        super(SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        #return _SimpleConsensus(self.consensus_type, self.dim).apply(input)
        return _SimpleConsensus.apply(input)

