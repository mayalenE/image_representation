import math
import os
import torch
from torch.nn.init import kaiming_normal_, kaiming_uniform_, xavier_normal_, xavier_uniform_, uniform_, eye_, _calculate_fan_in_and_fan_out

def get_weights_init(initialization_name):
    '''
    initialization_name: string such that the function called is weights_init_<initialization_name>
    '''
    initialization_name = initialization_name.lower()
    return eval("ME_weights_init_{}".format(initialization_name))


def ME_weights_init_pretrain(network, checkpoint_filepath):
    if os.path.exists(checkpoint_filepath):
        saved_model = torch.load(checkpoint_filepath, map_location='cpu')
        network.load_state_dict(saved_model['network_state_dict'])
    else:
        print("WARNING: the checkpoint filepath for a pretrain initialization has not been found, skipping the initialization")
    return network


def ME_weights_init_null(m):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        m.kernel.fill_(0)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if classname.find('MinkowskiLinear') != -1:
        m.linear.weight.data.fill_(0)
        if m.linear.bias is not None:
            m.linear.bias.data.fill_(0)
    if classname.find('MinkowskiBatchNorm') != -1:
        m.bn.weight.data.fill_(0)
        if m.bias is not None:
            m.bias.data.fill_(0)

def ME_weights_init_connections_identity(m):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        m.kernel.fill_(1) # for 1*1 convolution is equivalent to identity
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('MinkowskiLinear') != -1:
        eye_(m.linear.weight.data)
        if m.linear.bias is not None:
            m.linear.bias.data.fill_(0)
    elif classname.find('MinkowskiBatchNorm') != -1:
        m.bn.reset_parameters()

def ME_weights_init_uniform(m, a=0., b=0.01):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        uniform_(m.kernel, a, b)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('MinkowskiLinear') != -1:
        uniform_(m.linear.weight.data, a, b)
        if m.linear.bias is not None:
            m.linear.bias.data.fill_(0)
    elif classname.find('MinkowskiBatchNorm') != -1:
        m.bn.reset_parameters()

def ME_weights_init_pytorch(m):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        kaiming_uniform_(m.kernel, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(m.kernel)
            bound = 1 / math.sqrt(fan_in)
            uniform_(m.bias, -bound, bound)

    if classname.find('MinkowskiLinear') != -1:
        m.linear.reset_parameters()

    if (classname.find('MinkowskiBatchNorm') != -1):
        m.bn.reset_parameters()

        
def ME_weights_init_xavier_uniform(m):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        xavier_uniform_(m.kernel)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if classname.find('MinkowskiLinear') != -1:
        xavier_uniform_(m.linear.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if (classname.find('MinkowskiBatchNorm') != -1):
        m.bn.reset_parameters()
            
def ME_weights_init_xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        xavier_normal_(m.kernel)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if classname.find('MinkowskiLinear') != -1:
        xavier_normal_(m.linear.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if (classname.find('MinkowskiBatchNorm') != -1):
        m.bn.reset_parameters()
            
def ME_weights_init_kaiming_uniform(m):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        kaiming_uniform_(m.kernel)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if classname.find('MinkowskiLinear') != -1:
        kaiming_uniform_(m.linear.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if (classname.find('MinkowskiBatchNorm') != -1):
        m.bn.reset_parameters()
            
def ME_weights_init_kaiming_normal(m):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        kaiming_normal_(m.kernel)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if classname.find('MinkowskiLinear') != -1:
        kaiming_normal_(m.linear.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if (classname.find('MinkowskiBatchNorm') != -1):
        m.bn.reset_parameters()

def ME_weights_init_custom_uniform(m):
    classname = m.__class__.__name__
    if classname.find('MinkowskiConvolution') != -1:
        m.kernel.uniform_(-1 / (m.kernel.size(2)), 1 / (m.kernel.size(2)))
        if m.bias is not None:
            m.bias.data.uniform_(-0.1, 0.1)
    if classname.find('MinkowskiLinear') != -1:
        m.linear.weight.data.uniform_(-1 / math.sqrt(m.linear.weight.size(0)), 1 / math.sqrt(m.linear.weight.size(0)))
        if m.linear.bias is not None:
            m.linear.bias.data.uniform_(-0.1, 0.1)
    if (classname.find('MinkowskiBatchNorm') != -1):
        m.bn.reset_parameters()
