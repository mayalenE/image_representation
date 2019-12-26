from abc import ABCMeta, abstractmethod
from  goalrepresent.helper.nnmodulehelper import Channelize, conv2d_output_sizes, convtranspose2d_get_output_padding
from torch import nn

EPS = 1e-12

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


def get_decoder(model_architecture):
    '''
    model_architecture: string such that the class encoder called is <model_architecture>Encoder
    '''
    model_architecture = model_architecture.lower().capitalize()
    return eval("{}Decoder".format(model_architecture))


""" ========================================================================================================================
Decoder Modules 
========================================================================================================================="""

class BurgessDecoder (BaseDNNDecoder):
    """ 
    Extended Decoder of the model proposed in Burgess, Christopher P., et al. "Understanding disentangling in $\beta$-VAE." 
    """
        
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
            output_pads[conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[-(conv_layer_id+1)], feature_map_sizes[-(conv_layer_id+2)], kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id])
        output_pads[-1] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.input_size, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id])
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        
        self.decoder = nn.Sequential()
        
        # linear layers
        self.decoder.add_module("lin_1", nn.Sequential(nn.Linear(self.n_latents, hidden_dim), nn.ReLU()))
        self.decoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.decoder.add_module("lin_3", nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs), nn.ReLU()))
        self.decoder.add_module("channelize", Channelize(hidden_channels,  h_after_convs, w_after_convs))
                
        # convolution layers
        for conv_layer_id in range(0, self.n_conv_layers-1):
            # For convTranspose2d the padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
            self.decoder.add_module("convT_{}".format(conv_layer_id+1), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], output_padding=output_pads[conv_layer_id]), nn.ReLU()))
        self.decoder.add_module("convT_{}".format(self.n_conv_layers), nn.ConvTranspose2d(hidden_channels, self.n_channels, kernels_size[self.n_conv_layers-1], strides[self.n_conv_layers-1], pads[self.n_conv_layers-1], output_padding=output_pads[self.n_conv_layers-1]))
    
    def forward(self, z):
        return self.decoder(z)
'''  
class DecoderBurgess2 (nn.Module):
    """ 
    DecoderBurgess but:
        - added BatchNorm and LeakyReLU after convolutions
        - added Tanh output
    """
        
    def __init__(self,  n_channels = 1, input_size = (64,64), n_conv_layers = 4, n_latents = 10, **kwargs):
        super(DecoderBurgess2, self).__init__()
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents
        
        # network architecture
        hidden_channels = 32
        hidden_dim = 256
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        h_after_convs, w_after_convs = helper.conv2d_output_size(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        
        self.decoder = nn.Sequential()
        
        # linear layers
        self.decoder.add_module("lin_1", nn.Sequential(nn.Linear(self.n_latents, hidden_dim), nn.LeakyReLU(1e-2, inplace=True)))
        self.decoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(1e-2, inplace=True)))
        self.decoder.add_module("lin_3", nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs), nn.LeakyReLU(1e-2, inplace=True)))
        self.decoder.add_module("channelize", helper.Channelize(hidden_channels,  h_after_convs, w_after_convs))
                
        # convolution layers
        for conv_layer_id in range(0, self.n_conv_layers-1):
            # For convTranspose2d the padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
            self.decoder.add_module("convT_{}".format(conv_layer_id+1), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id]), nn.BatchNorm2d(hidden_channels), nn.LeakyReLU(1e-2, inplace=True)))
        self.decoder.add_module("convT_{}".format(self.n_conv_layers), nn.Sequential(nn.ConvTranspose2d(hidden_channels, self.n_channels, kernels_size[self.n_conv_layers-1], strides[self.n_conv_layers-1], pads[self.n_conv_layers-1])))
    
    def forward(self, z):import torch

        if len(z.size()) == 4:
            z = z.squeeze(dim=-1).squeeze(dim=-1)
        output = self.decoder(z)
        output = F.tanh(output)   
        return output


class DecoderAli(nn.Module):

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 6, n_latents = 32,  **kwargs):
        super(DecoderAli, self).__init__()
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents

        # network architecture
        hidden_channels_final = 2
        kernels_size=[4,3]*self.n_conv_layers
        strides=[2,1]*self.n_conv_layers
        pads=[1,1]*self.n_conv_layers
        
        self.decoder = nn.Sequential()
        
        # deconvolution layers
        hidden_channels = int(hidden_channels_final * math.pow(2,n_conv_layers))
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.decoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(self.n_latents, hidden_channels, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True)
                        ))
            else:
                self.decoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1]), 
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
        if len(z.size()) == 2:
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        output = self.decoder(z)
        output = F.tanh(output)
        return output
    

class DecoderAli2(nn.Module):

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 6, n_latents = 128,  **kwargs):
        super(DecoderAli2, self).__init__()
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents

        # network architecture
        hidden_channels_final = 32
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        
        
        self.decoder = nn.Sequential()
        
        # deconvolution layers
        hidden_channels = int(hidden_channels_final)*8
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.decoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(self.n_latents, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.Dropout(0.5),
                        nn.LeakyReLU(0.02, inplace=True)
                        ))
            elif conv_layer_id < 4:
                self.decoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.Dropout(0.5),
                        nn.LeakyReLU(0.02, inplace=True)
                        ))
            else:
                self.decoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(0.02, inplace=True)
                        ))
                hidden_channels = hidden_channels // 2
            
        self.decoder.add_module("conv_{}".format(self.n_conv_layers), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=1, stride=1), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(0.02, inplace=True),
                        nn.ConvTranspose2d(hidden_channels, self.n_channels, kernel_size=1, stride=1)
                        ))

    def forward(self, z):
        if len(z.size()) == 2:
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        output = self.decoder(z)
        output = F.tanh(output)
        return output
    
    
class DecoderRadford(nn.Module):

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 4, n_latents = 64,  **kwargs):
        super(DecoderRadford, self).__init__()
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents

        # network architecture
        hidden_channels_final = 32
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        
        
        self.decoder = nn.Sequential()
        
        # linear project and reshape
        hidden_channels = int(hidden_channels_final * math.pow(2, n_conv_layers-1))
        h_after_convs, w_after_convs = helper.conv2d_output_size(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        self.decoder.add_module("linear_reshape", nn.Sequential(
                        nn.Linear(self.n_latents, hidden_channels*h_after_convs*w_after_convs), 
                        helper.Channelize(hidden_channels, h_after_convs, w_after_convs),
                        nn.BatchNorm2d(hidden_channels),
                        nn.LeakyReLU(0.2, inplace=True)
                        ))
        
        # deconvolution layers
        for conv_layer_id in range(self.n_conv_layers-1):
            self.decoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], bias=False), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(0.2, inplace=True)
                        ))
            hidden_channels = hidden_channels // 2
            
        self.decoder.add_module("conv_{}".format(self.n_conv_layers), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, self.n_channels, kernels_size[self.n_conv_layers-1], strides[self.n_conv_layers-1], pads[self.n_conv_layers-1], bias=False), 
                        nn.Tanh()
                        ))

    def forward(self, z):
        return self.decoder(z)
'''