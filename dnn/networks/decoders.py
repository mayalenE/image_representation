from abc import ABCMeta, abstractmethod
from  goalrepresent.helper.nnmodulehelper import Channelize, conv2d_output_sizes, convtranspose2d_get_output_padding
import math
import torch
from torch import nn

class BaseDNNDecoder (nn.Module, metaclass=ABCMeta):
    """
    Base Decoder class
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 3 layers for 32*32 images3 layers for 32*32 images, 6 layers for 256*256 images)
    
    Parameters
    ----------
    n_channels: number of channels
    input_size: size of input image
    n_conv_layers: desired number of convolutional layers
    n_latents: dimensionality of the infered latent output.
    """
    
    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 4, n_latents = 10,  **kwargs):
        super().__init__()
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents
    
    @abstractmethod
    def forward(self, z):
        pass
    
    def push_variable_to_device(self, x):
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
        return x


def get_decoder(model_architecture):
    '''
    model_architecture: string such that the class decoder called is <model_architecture>Decoder
    '''
    return eval("{}Decoder".format(model_architecture))


""" ========================================================================================================================
Decoder Modules 
========================================================================================================================="""

class BurgessDecoder (BaseDNNDecoder):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # network architecture
        hidden_channels = 32
        hidden_dim = 256
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        feature_map_sizes = conv2d_output_sizes(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        output_pads = [None]*self.n_conv_layers
        for conv_layer_id in range(self.n_conv_layers-1):
            output_pads[conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[-(conv_layer_id+1)], feature_map_sizes[-(conv_layer_id+2)], kernels_size[-(conv_layer_id+1)], strides[-(conv_layer_id+1)], pads[-(conv_layer_id+1)])
        output_pads[-1] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.input_size, kernels_size[0], strides[0], pads[0])
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        
        self.decoder = nn.Sequential()
        
        # linear layers
        self.decoder.add_module("lin_0", nn.Sequential(nn.Linear(self.n_latents, hidden_dim), nn.ReLU()))
        self.decoder.add_module("lin_1", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.decoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs), nn.ReLU()))
        self.decoder.add_module("channelize", Channelize(hidden_channels,  h_after_convs, w_after_convs))
                
        # convolution layers
        for conv_layer_id in range(0, self.n_conv_layers-1):
            # For convTranspose2d the padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
            self.decoder.add_module("convT_{}".format(conv_layer_id), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[-(conv_layer_id+1)], strides[-(conv_layer_id+1)], pads[-(conv_layer_id+1)], output_padding=output_pads[conv_layer_id]), nn.ReLU()))
        self.decoder.add_module("convT_{}".format(self.n_conv_layers-1), nn.ConvTranspose2d(hidden_channels, self.n_channels, kernels_size[0], strides[0], pads[0], output_padding=output_pads[self.n_conv_layers-1]))
        
    def forward(self, z):
        return self.decoder(z)
    

    
class HjelmDecoder (BaseDNNDecoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # network architecture
        hidden_channels = 64
        hidden_dim = 1024
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        feature_map_sizes = conv2d_output_sizes(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        output_pads = [None]*self.n_conv_layers
        for conv_layer_id in range(self.n_conv_layers-1):
            output_pads[conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[-(conv_layer_id+1)], feature_map_sizes[-(conv_layer_id+2)], kernels_size[-(conv_layer_id+1)], strides[-(conv_layer_id+1)], pads[-(conv_layer_id+1)])
        output_pads[-1] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.input_size, kernels_size[0], strides[0], pads[0])
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        
        self.decoder = nn.Sequential()
        
        hidden_channels = int(hidden_channels * math.pow(2, self.n_conv_layers))
        
        # linear layers
        self.decoder.add_module("lin_0", nn.Sequential(nn.Linear(self.n_latents, hidden_dim), nn.ReLU()))
        self.decoder.add_module("lin_1", nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs), nn.ReLU()))
        self.decoder.add_module("channelize", Channelize(hidden_channels,  h_after_convs, w_after_convs))
                
        # convolution layers
        for conv_layer_id in range(0, self.n_conv_layers-1):
            # For convTranspose2d the padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
            self.decoder.add_module("convT_{}".format(conv_layer_id), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels//2, kernels_size[-(conv_layer_id+1)], strides[-(conv_layer_id+1)], pads[-(conv_layer_id+1)], output_padding=output_pads[conv_layer_id]), nn.BatchNorm2d(hidden_channels //2), nn.ReLU()))
            hidden_channels = hidden_channels // 2
        self.decoder.add_module("convT_{}".format(self.n_conv_layers-1), nn.ConvTranspose2d(hidden_channels, self.n_channels, kernels_size[0], strides[0], pads[0], output_padding=output_pads[self.n_conv_layers-1]))
    
    def forward(self, z):
        return self.decoder(z)
    
    
class DumoulinDecoder (BaseDNNDecoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # network architecture
        hidden_channels = 512 
        kernels_size=[4,4]*self.n_conv_layers
        strides=[1,2]*self.n_conv_layers
        pads=[0,1]*self.n_conv_layers
        dils=[1,1]*self.n_conv_layers

        feature_map_sizes = conv2d_output_sizes(self.input_size, 2*self.n_conv_layers, kernels_size, strides, pads, dils)
        output_pads = [None]*2*self.n_conv_layers
        for conv_layer_id in range(self.n_conv_layers-1):
            output_pads[2*conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[-(2*conv_layer_id+1)], feature_map_sizes[-(2*conv_layer_id+2)], kernels_size[-(2*conv_layer_id+1)], strides[-(2*conv_layer_id+1)], pads[-(2*conv_layer_id+1)])
            output_pads[2*conv_layer_id+1] = convtranspose2d_get_output_padding(feature_map_sizes[-(2*conv_layer_id+1+1)], feature_map_sizes[-(2*conv_layer_id+1+2)], kernels_size[-(2*conv_layer_id+1+1)], strides[-(2*conv_layer_id+1+1)], pads[-(2*conv_layer_id+1+1)])
        output_pads[-2] = convtranspose2d_get_output_padding(feature_map_sizes[1], feature_map_sizes[0], kernels_size[1], strides[1], pads[1])
        output_pads[-1] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.input_size, kernels_size[0], strides[0], pads[0])
        self.decoder = nn.Sequential()
        
        # deconvolution layers
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.decoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(self.n_latents, hidden_channels // 2, kernels_size[-1], strides[-1], pads[-1], output_padding=output_pads[2*conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 2, kernels_size[-2], strides[-2], pads[-2], output_padding=output_pads[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True)
                        ))
            else:
                self.decoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[-(2*conv_layer_id+1)], strides[-(2*conv_layer_id+1)], pads[-(2*conv_layer_id+1)], output_padding=output_pads[2*conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 2, kernels_size[-(2*conv_layer_id+1+1)], strides[-(2*conv_layer_id+1+1)], pads[-(2*conv_layer_id+1+1)], output_padding=output_pads[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True)
                        ))
            hidden_channels = hidden_channels // 2
            
        self.decoder.add_module("conv_{}".format(self.n_conv_layers), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=1, stride=1), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(hidden_channels, self.n_channels, kernel_size=1, stride=1)
                        ))

    
    def forward(self, z):
        if z.dim() == 2: #B*n_latents -> B*n_latents*1*1 
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1) 
        return torch.sigmoid(self.decoder(z))