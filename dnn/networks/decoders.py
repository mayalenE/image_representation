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
    
    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 4, n_latents = 10, feature_layer = 2, hidden_channels=None, hidden_dim=None, **kwargs):
        super().__init__()
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents
        self.feature_layer = feature_layer
        self.hidden_channels = hidden_channels
        self.hidden_dim = hidden_dim
        self.output_keys_list = ["z", "gfi", "lfi", "recon_x"]
    
    def forward(self, z):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(z)
        
        if z.dim() == 2 and type(self).__name__=="DumoulinDecoder": #B*n_latents -> B*n_latents*1*1 
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1) 
        
        # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        # decoder output
        decoder_outputs = {"z": z, "gfi": gfi, "lfi": lfi, "recon_x": recon_x}
        
        return decoder_outputs
    
    def forward_for_graph_tracing(self, z):
        if z.dim() == 2: #B*n_latents -> B*n_latents*1*1 
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1) 
            
        # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        return recon_x
    
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
        # WARNING: incrementation order follow the encoder top-down order
        if self.hidden_channels is None:
            self.hidden_channels = 32
        hidden_channels = self.hidden_channels
        if self.hidden_dim is None:
            self.hidden_dim = 256
        hidden_dim = self.hidden_dim
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        feature_map_sizes = conv2d_output_sizes(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        output_pads = [None]*self.n_conv_layers
        output_pads[0] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.input_size, kernels_size[0], strides[0], pads[0])
        for conv_layer_id in range(1, self.n_conv_layers):
            output_pads[conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[conv_layer_id], feature_map_sizes[conv_layer_id-1], kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id])
                
        # encoder feature inverse
        self.efi = nn.Sequential(nn.Linear(self.n_latents, hidden_dim), nn.ReLU())
        self.efi.out_connection_type = ("lin", hidden_dim)
        
        # global feature inverse
        ## linear layers
        self.gfi = nn.Sequential()
        self.gfi .add_module("lin_1_i", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.gfi.add_module("lin_0_i", nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs), nn.ReLU()))
        self.gfi.add_module("channelize", Channelize(hidden_channels,  h_after_convs, w_after_convs))
        ## convolutional layers
        for conv_layer_id in range(self.n_conv_layers-1, self.feature_layer+1-1,-1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], output_padding=output_pads[conv_layer_id]), nn.ReLU()))
        self.gfi.out_connection_type = ("conv", hidden_channels)
        
        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.feature_layer+1-1, 0,-1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], output_padding=output_pads[conv_layer_id]), nn.ReLU()))
        self.lfi.add_module("conv_0_i", nn.ConvTranspose2d(hidden_channels, self.n_channels, kernels_size[0], strides[0], pads[0], output_padding=output_pads[0]))
        self.lfi.out_connection_type = ("conv", self.n_channels)
    
class HjelmDecoder (BaseDNNDecoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # network architecture
        # WARNING: incrementation order follow the encoder top-down order
        if self.hidden_channels is None:
            self.hidden_channels = int (math.pow(2, 9 - int(math.log(self.input_size[0],2)) + 3))
        hidden_channels = self.hidden_channels
        if self.hidden_dim is None:
            self.hidden_dim = 1024
        hidden_dim = self.hidden_dim
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        feature_map_sizes = conv2d_output_sizes(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        output_pads = [None]*self.n_conv_layers
        output_pads[0] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.input_size, kernels_size[0], strides[0], pads[0])
        for conv_layer_id in range(1, self.n_conv_layers):
            output_pads[conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[conv_layer_id], feature_map_sizes[conv_layer_id-1], kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id])
            
        # encoder feature inverse
        self.efi = nn.Sequential(nn.Linear(self.n_latents, hidden_dim), nn.ReLU())
        self.efi.out_connection_type = ("lin", hidden_dim)
        
        # global feature inverse
        hidden_channels = int(hidden_channels * math.pow(2, self.n_conv_layers-1))
        ## linear layers
        self.gfi = nn.Sequential()
        self.gfi.add_module("lin_0_i", nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs), nn.ReLU()))
        self.gfi.add_module("channelize", Channelize(hidden_channels,  h_after_convs, w_after_convs))
        ## convolutional layers
        for conv_layer_id in range(self.n_conv_layers-1, self.feature_layer+1-1,-1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels//2, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], output_padding=output_pads[conv_layer_id]), nn.BatchNorm2d(hidden_channels //2), nn.ReLU()))
            hidden_channels = hidden_channels // 2
        self.gfi.out_connection_type = ("conv", hidden_channels)
            
        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.feature_layer+1-1, 0,-1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels//2, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], output_padding=output_pads[conv_layer_id]), nn.BatchNorm2d(hidden_channels //2), nn.ReLU()))
            hidden_channels = hidden_channels // 2
        self.lfi.add_module("conv_0_i", nn.ConvTranspose2d(hidden_channels, self.n_channels, kernels_size[0], strides[0], pads[0], output_padding=output_pads[0]))
        self.lfi.out_connection_type = ("conv", self.n_channels)
    
    
class DumoulinDecoder (BaseDNNDecoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

         # need square and power of 2 image size input
        power = math.log(self.input_size[0], 2)
        assert  (power % 1 == 0.0)  and (power > 3), "Dumoulin Encoder needs a power of 2 as image input size (>=16)"
        assert self.input_size[0] == self.input_size[1], "Dumoulin Encoder needs a square image input size"
        
        assert self.n_conv_layers == power - 2 , "The number of convolutional layers in DumoulinEncoder must be log(input_size, 2) - 2 "

        # network architecture
        if self.hidden_channels is None:
            self.hidden_channels = int(512 // math.pow(2, self.n_conv_layers))
        hidden_channels = self.hidden_channels
        kernels_size=[4,4]*self.n_conv_layers
        strides=[1,2]*self.n_conv_layers
        pads=[0,1]*self.n_conv_layers
        dils=[1,1]*self.n_conv_layers

        feature_map_sizes = conv2d_output_sizes(self.input_size, 2*self.n_conv_layers, kernels_size, strides, pads, dils)
        output_pads = [None]*2*self.n_conv_layers
        output_pads[0] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.input_size, kernels_size[0], strides[0], pads[0])
        output_pads[1] = convtranspose2d_get_output_padding(feature_map_sizes[1], feature_map_sizes[0], kernels_size[1], strides[1], pads[1])
        for conv_layer_id in range(1, self.n_conv_layers):
            output_pads[2*conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[2*conv_layer_id], feature_map_sizes[2*conv_layer_id-1], kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id])
            output_pads[2*conv_layer_id+1] = convtranspose2d_get_output_padding(feature_map_sizes[2*conv_layer_id+1], feature_map_sizes[2*conv_layer_id+1-1], kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1])
        
        # encoder feature inverse
        hidden_channels = int(hidden_channels * math.pow(2, self.n_conv_layers))
        self.efi = nn.Sequential(
                        nn.ConvTranspose2d(self.n_latents, hidden_channels, kernel_size=1, stride=1), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=1, stride=1), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True)
                        )
        self.efi.out_connection_type = ("conv", hidden_channels)
        
        # global feature inverse
        self.gfi = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.n_conv_layers-1, self.feature_layer+1-1,-1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1], output_padding=output_pads[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 2, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id], output_padding=output_pads[2*conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True),
                        ))
            hidden_channels = hidden_channels // 2
        self.gfi.out_connection_type = ("conv", hidden_channels)
            
        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.feature_layer+1-1, 0,-1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                        nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1], output_padding=output_pads[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True),
                        nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 2, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id], output_padding=output_pads[2*conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels // 2), 
                        nn.LeakyReLU(inplace=True),
                        ))
            hidden_channels = hidden_channels // 2
        self.lfi.add_module("conv_0_i", nn.Sequential(
                                nn.ConvTranspose2d(hidden_channels, hidden_channels//2, kernels_size[1], strides[1], pads[1], output_padding=output_pads[1]), 
                                nn.BatchNorm2d(hidden_channels//2), 
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(hidden_channels//2, self.n_channels, kernels_size[0], strides[0], pads[0], output_padding=output_pads[0]),
                                nn.Sigmoid()
                                ))
        self.lfi.out_connection_type = ("conv", self.n_channels)