import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import numpy as np
from functools import reduce
from torch.autograd import Variable

def edit_tensor_in_numpy(input, trafo):
    # Kept in case tensor transformations should be done in numpy rather than pytorch (might be slightly faster, but is ugly and might break code..)
    is_cuda = input.is_cuda
    if is_cuda:
        input_np = input.cpu().data.numpy()
    else:
        input_np = input.data.numpy()
    del input
    input_np = trafo(input_np)
    input = Variable(torch.from_numpy(input_np))
    if is_cuda:
        input = input.cuda()
    return input

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn
        #
    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

class ReCodeAlphabet(nn.Module):
    def __init__(self):
        super(ReCodeAlphabet, self).__init__()
        #
    def forward(self, input):
        # Swap ACGT to AGCT
        # array has shape (N, 4, 1, 1000)
        # pytorch doesn't support full indexing at the moment, at some point this should work: [:,:,torch.LongTensor([0,2,1,3])]
        input_reordered = [input[:,i,...] for i in [0,2,1,3]]
        input = torch.stack(input_reordered, dim=1)
        # slightly faster but ugly:
        #input = edit_tensor_in_numpy(input, lambda x: x[:,[0,2,1,3], ...])
        return input

class ConcatenateRC(nn.Module):
    def __init__(self):
        super(ConcatenateRC, self).__init__()
        #
    def forward(self, input):
        # array has shape (N, 4, 1, 1000)
        # return the sequence + its RC concatenated
        # create inverted indices
        invert_dims = [1,3]
        input_bkup = input
        for idim in invert_dims:
            idxs = [i for i in range(input.size(idim)-1, -1, -1)]
            idxs_var = Variable(torch.LongTensor(idxs))
            if input.is_cuda:
                idxs_var =idxs_var.cuda()
            input = input.index_select(idim, idxs_var)
        #
        input = torch.cat([input_bkup, input], dim=0)
        #
        # Using numpy:
        #input = edit_tensor_in_numpy(input, lambda x: np.concatenate([x, x[:,::-1, : ,::-1]],axis=0))
        return input

class AverageRC(nn.Module):
    def __init__(self):
        super(AverageRC, self).__init__()
        #
    def forward(self, input):
        # average over fwd and RC
        input = input[:int(input.shape[0]/2)] /2  + input[int(input.shape[0]/2):] /2
        #
        # Using numpy:
        #input = edit_tensor_in_numpy(input, lambda x: x[:int(x.shape[0]/2),:]/2.0+x[int(x.shape[0]/2):,:]/2.0)
        return input

def get_model(load_weights = True):
    deepsea_cpu = nn.Sequential( # Sequential,
        nn.Conv2d(4,320,(1, 8),(1, 1)),
        nn.Threshold(0, 1e-06),
        nn.MaxPool2d((1, 4),(1, 4)),
        nn.Dropout(0.2),
        nn.Conv2d(320,480,(1, 8),(1, 1)),
        nn.Threshold(0, 1e-06),
        nn.MaxPool2d((1, 4),(1, 4)),
        nn.Dropout(0.2),
        nn.Conv2d(480,960,(1, 8),(1, 1)),
        nn.Threshold(0, 1e-06),
        nn.Dropout(0.5),
        Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(50880,925)), # Linear,
        nn.Threshold(0, 1e-06),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(925,919)), # Linear,
        nn.Sigmoid(),
    )
    if load_weights:
        deepsea_cpu.load_state_dict(torch.load('model_files/deepsea_cpu.pth'))
    return nn.Sequential(ReCodeAlphabet(), deepsea_cpu)


def get_seqpred_model(load_weights = True):
    deepsea_cpu = nn.Sequential( # Sequential,
        nn.Conv2d(4,320,(1, 8),(1, 1)),
        nn.Threshold(0, 1e-06),
        nn.MaxPool2d((1, 4),(1, 4)),
        nn.Dropout(0.2),
        nn.Conv2d(320,480,(1, 8),(1, 1)),
        nn.Threshold(0, 1e-06),
        nn.MaxPool2d((1, 4),(1, 4)),
        nn.Dropout(0.2),
        nn.Conv2d(480,960,(1, 8),(1, 1)),
        nn.Threshold(0, 1e-06),
        nn.Dropout(0.5),
        Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(50880,925)), # Linear,
        nn.Threshold(0, 1e-06),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(925,919)), # Linear,
        nn.Sigmoid(),
    )
    if load_weights:
        deepsea_cpu.load_state_dict(torch.load('model_files/deepsea_cpu.pth'))
    return nn.Sequential(ReCodeAlphabet(), ConcatenateRC(), deepsea_cpu, AverageRC())


def save_seqpred_model_weights(fname = 'model_files/deepsea_variant_effects.pth'):
    m = get_model()
    torch.save(m.state_dict(), fname)


def save_seqpred_model_weights(fname = 'model_files/deepsea_predict.pth'):
    spm = get_seqpred_model()
    torch.save(spm.state_dict(), fname)

model = get_model(load_weights = False)