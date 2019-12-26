import math
from torch.nn.init import kaiming_normal_, kaiming_uniform_, xavier_normal_, xavier_uniform_
        

def get_initialization(initialization_name):
    '''
    initialization_name: string such that the function called is weights_init_<initialization_name>
    '''
    initialization_name = initialization_name.lower()
    return eval("weights_init_{}".format(initialization_name))



def weights_init_pytorch(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        m.reset_parameters()
        
def weights_init_xavier_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_kaiming_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_kaiming_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def weights_init_custom_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.uniform_(-1,1)
        m.weight.data.uniform_(-1/(m.weight.size(2)), 1/(m.weight.size(2))) 
        if m.bias is not None:
            m.bias.data.uniform_(-0.1,0.1)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-1/math.sqrt(m.weight.size(0)), 1/math.sqrt(m.weight.size(0))) 
        if m.bias is not None:
            m.bias.data.uniform_(-0.1,0.1)
            