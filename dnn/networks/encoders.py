from abc import ABCMeta, abstractmethod
import goalrepresent as gr
from  goalrepresent.helper.nnmodulehelper import Flatten, conv2d_output_sizes
import torch
from torch import nn

EPS = 1e-12

class BaseDNNEncoder (nn.Module, gr.BaseEncoder, metaclass=ABCMeta):
    """
    Base Encoder class
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 3 layers for 32*32, 6 layers for 256*256 images)
    
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
    def forward(self, x):
        pass
    
    def push_variable_to_device(self, x):
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
        return x


def get_encoder(model_architecture):
    """
    model_architecture: string such that the class encoder called is <model_architecture>Encoder
    """
    return eval("{}Encoder".format(model_architecture))


""" ========================================================================================================================
Encoder Modules 
========================================================================================================================="""

class BurgessEncoder (BaseDNNEncoder):
    """ 
    Extended Encoder of the model proposed in Burgess, Christopher P., et al. "Understanding disentangling in $\beta$-VAE."
    
    Model Architecture (transposed for decoder)
    ------------
    - Convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
    - 2 fully connected layers (each of 256 units)
    - Latent distribution:
        - 1 fully connected layer of 2*n_latents units (log variance and mean for Gaussians distributions)
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
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        self.encoder = nn.Sequential()
        
        # convolution layers
        self.encoder.add_module("conv_{}".format(1), nn.Sequential(nn.Conv2d(self.n_channels, hidden_channels, kernels_size[0], strides[0], pads[0], dils[0]), nn.ReLU()))
        for conv_layer_id in range(1, self.n_conv_layers):
            self.encoder.add_module("conv_{}".format(conv_layer_id+1), nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), nn.ReLU()))
        self.encoder.add_module("flatten", Flatten())
        
        # linear layers
        self.encoder.add_module("lin_1", nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim), nn.ReLU()))
        self.encoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.encoder.add_module("mu_logvar_gen", nn.Linear(hidden_dim, 2 * self.n_latents))

        
    def forward(self, x):
        return torch.chunk(self.encoder(x), 2, dim=1)
    
    def calc_embedding(self, x):
        mu, logvar = self.forward(x)
        return mu
    
    
class CIFAREncoder (BaseDNNEncoder):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # network architecture
        hidden_channels = 32
        hidden_dim = 128
        kernels_size=[4,4]*self.n_conv_layers
        strides=[1,2]*self.n_conv_layers
        pads=[1,1]*self.n_conv_layers
        dils=[1,1]*self.n_conv_layers
        feature_map_sizes = conv2d_output_sizes(self.input_size, 2*self.n_conv_layers, kernels_size, strides, pads, dils)
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        self.encoder = nn.Sequential()
        
        # convolution layers
        self.encoder.add_module("conv_{}".format(1), nn.Sequential(nn.Conv2d(self.n_channels, self.n_channels, kernels_size[0], strides[0], pads[0], dils[0]), nn.ReLU(), nn.Conv2d(self.n_channels, hidden_channels, kernels_size[1], strides[1], pads[1], dils[1]), nn.ReLU()))
        for conv_layer_id in range(1, self.n_conv_layers):
            self.encoder.add_module("conv_{}".format(conv_layer_id+1), nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id], dils[2*conv_layer_id]), nn.ReLU(), nn.Conv2d(hidden_channels, hidden_channels, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1], dils[2*conv_layer_id+1]), nn.ReLU()))
        self.encoder.add_module("flatten", Flatten())
        
        # linear layers
        self.encoder.add_module("lin_1", nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim), nn.ReLU()))
        self.encoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.encoder.add_module("mu_logvar_gen", nn.Linear(hidden_dim, 2 * self.n_latents))

        
    def forward(self, x):
        return torch.chunk(self.encoder(x), 2, dim=1)
    
    def calc_embedding(self, x):
        mu, logvar = self.forward(x)
        return mu
    
class HjlemCIFAREncoder (BaseDNNEncoder):
        
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
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        self.encoder = nn.Sequential()
        
        # convolution layers
        self.encoder.add_module("conv_{}".format(1), nn.Sequential(nn.Conv2d(self.n_channels, hidden_channels, kernels_size[0], strides[0], pads[0], dils[0]), nn.ReLU()))
        for conv_layer_id in range(1, self.n_conv_layers):
            self.encoder.add_module("conv_{}".format(conv_layer_id+1), nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels*2, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), nn.BatchNorm2d(hidden_channels*2), nn.ReLU()))
            hidden_channels *= 2
        self.encoder.add_module("flatten", Flatten())
        
        # linear layers
        self.encoder.add_module("lin_1", nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim), nn.ReLU()))
        self.encoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.encoder.add_module("mu_logvar_gen", nn.Linear(hidden_dim, 2 * self.n_latents))

        
    def forward(self, x):
        return torch.chunk(self.encoder(x), 2, dim=1)
    
    def calc_embedding(self, x):
        mu, logvar = self.forward(x)
        return mu

'''
class EncoderBurgess2 (BaseDNNEncoder):
    """ 
    EncoderBurgess but:
        - added BatchNorm and LeakyReLU after convolutions
        - deterministic (i.e outputs directly z instead of (mu,logvar))
    """
    
    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 4, n_latents = 10,  **kwargs):
        super(EncoderBurgess2, self).__init__()
        
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
        
        self.encoder = nn.Sequential()
        
        # convolution layers
        self.encoder.add_module("conv_{}".format(1), nn.Sequential(nn.Conv2d(self.n_channels, hidden_channels, kernels_size[0], strides[0], pads[0], dils[0]), nn.BatchNorm2d(hidden_channels), nn.LeakyReLU(1e-2, inplace=True)))
        for conv_layer_id in range(1, self.n_conv_layers):
            self.encoder.add_module("conv_{}".format(conv_layer_id+1), nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), nn.BatchNorm2d(hidden_channels), nn.LeakyReLU(1e-2, inplace=True)))
        self.encoder.add_module("flatten", helper.Flatten())
        
        # linear layers
        self.encoder.add_module("lin_1", nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim), nn.LeakyReLU(1e-2, inplace=True)))
        self.encoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(1e-2, inplace=True)))
        #self.encoder.add_module("mu_logvar_gen", nn.Linear(hidden_dim, 2 * self.n_latents))
        self.encoder.add_module("z_gen", nn.Linear(hidden_dim, self.n_latents))
        
    def forward(self, x):
        #return torch.chunk(self.encoder(x), 2, dim=1)
        return self.encoder(x)


class EncoderAli(BaseDNNEncoder):
    """ 
    Some UNET-inspired encoder with BatchNorm and LeakyReLU as proposed in Dumoulin et al. "ADVERSARIALLY LEARNED INFERENCE"
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 8 layers for 256*256 images, 7 layers for 128*128 images)
    
    Parameters
    ----------
    n_channels: number of channels
    input_size: size of input image
    n_conv_layers: desired number of convolutional layers
    n_latents: dimensionality of the infered latent output.
    
    Model Architecture (transposed for decoder)
    ------------
    - Convolutional blocks composed of:
        - 1 convolutional layer (2*2^(conv_layer_id) channels), (3 x 3 kernel), (stride of 1), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
        - 1 convolutional layer (2*2^(conv_layer_id+1) channels), (4 x 4 kernel), (stride of 2), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
    - 1 Convolutional blocks composed of:
        - 1 convolutional layer (2*2^(n_conv_layers) channels), (1 x 1 kernel), (stride of 1), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
        - 1 convolutional layer (n_latents channels), (1 x 1 kernel), (stride of 1), (padding of 1)
    """

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 6, n_latents = 32, **kwargs ):
        super(EncoderAli, self).__init__()
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents

        # network architecture
        hidden_channels = 2
        kernels_size=[3,4]*self.n_conv_layers
        strides=[1,2]*self.n_conv_layers
        pads=[1,1]*self.n_conv_layers
        dils=[1,1]*self.n_conv_layers
        
        self.encoder = nn.Sequential()
        
        # convolution layers
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.encoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(self.n_channels, hidden_channels, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id], dils[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(hidden_channels, 2*hidden_channels, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1], dils[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(2*hidden_channels), 
                        nn.LeakyReLU(inplace=True)
                        ))
            else:
                self.encoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(hidden_channels, hidden_channels, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id], dils[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(hidden_channels, 2*hidden_channels, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1], dils[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(2*hidden_channels), 
                        nn.LeakyReLU(inplace=True)
                        ))
            hidden_channels *= 2
            
        self.encoder.add_module("conv_{}".format(self.n_conv_layers), nn.Sequential(
                        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(hidden_channels, self.n_latents, kernel_size=1, stride=1)
                        ))

    def forward(self, x):
        return self.encoder(x)
    

class EncoderAli2(BaseDNNEncoder):
    """ 
    Some encodery inspired from Dumoulin et al. "ADVERSARIALLY LEARNED INFERENCE"
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 8 layers for 256*256 images, 6 layers for 128*128 images)
    
    Parameters
    ----------
    n_channels: number of channels
    input_size: size of input image
    n_conv_layers: desired number of convolutional layers
    n_latents: dimensionality of the infered latent output.
    
    Model Architecture (transposed for decoder)
    ------------
    - Convolutional blocks composed of:
        - 1 convolutional layer (4 x 4 kernel), (stride of 2), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
    --> starts from 32 channels to 256 and then stays at 256
    - 1 Convolutional blocks composed of:
        - 1 convolutional layer (1 x 1 kernel), (stride of 1), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
        - 1 convolutional layer (n_latents channels), (1 x 1 kernel), (stride of 1), (padding of 1)
    """

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 6, n_latents = 128, **kwargs ):
        super(EncoderAli2, self).__init__()
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents

        # network architecture
        hidden_channels = 32
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        
        self.encoder = nn.Sequential()
        
        # convolution layers
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.encoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(self.n_channels, hidden_channels, kernels_size[conv_layer_id+1], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), 
                        nn.LeakyReLU(0.02, inplace=True)
                        ))
            elif conv_layer_id < 4:
                self.encoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(hidden_channels, 2*hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), 
                        nn.BatchNorm2d(2*hidden_channels), 
                        nn.LeakyReLU(0.02, inplace=True)
                        ))   
                hidden_channels *= 2
            else:
                self.encoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(0.02, inplace=True)
                        ))
            

        self.encoder.add_module("conv_{}".format(self.n_conv_layers), nn.Sequential(
                        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(0.02, inplace=True),
                        nn.Conv2d(hidden_channels, self.n_latents, kernel_size=1, stride=1)
                        ))



    def forward(self, x):
        return self.encoder(x)
    
class EncoderRadford(BaseDNNEncoder):

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 6, n_latents = 128, **kwargs):
        super(EncoderRadford, self).__init__()
         # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents

        # network architecture
        hidden_channels = 32
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        
        
        self.encoder = nn.Sequential()
        h_after_convs, w_after_convs = helper.conv2d_output_size(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        
        # convolution layers
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.encoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                            nn.Conv2d(self.n_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], bias=False), 
                            nn.LeakyReLU(0.2, inplace=True)
                            ))
                
            else:            
                self.encoder.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                            nn.Conv2d(hidden_channels, hidden_channels*2, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], bias=False), 
                            nn.BatchNorm2d(hidden_channels*2),
                            nn.LeakyReLU(0.2, inplace=True)
                            ))
                hidden_channels = hidden_channels * 2

        self.encoder.add_module("linear_reshape", nn.Sequential(
                        helper.Flatten(),
                        nn.Linear(hidden_channels*h_after_convs*w_after_convs, 2*self.n_latents)
                        #nn.Sigmoid()
                        ))

    def forward(self, x):
        return torch.chunk(self.encoder(x), 2, dim=1)
'''