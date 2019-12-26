from abc import ABCMeta, abstractmethod
from  goalrepresent.helper import nnmodulehelper
import torch
from torch import nn

EPS = 1e-12

class BaseDNNDiscriminator (nn.Module, metaclass=ABCMeta):
    '''
    Base Encoder class
    '''
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


def get_discriminator(model_architecture):
    '''
    model_architecture: string such that the class encoder called is <model_architecture>Encoder
    '''
    model_architecture = model_architecture.lower().capitalize()
    return eval("{}Discriminator".format(model_architecture))


""" ========================================================================================================================
Discriminator Modules 
========================================================================================================================="""

class BurgessDiscriminator (nn.Module):
    
    def __init__(self, config=None,  **kwargs):
        super().__init__(config, **kwargs)

        # network architecture
        hidden_channels = 32
        hidden_dim_0 = 256
        hidden_dim_1 = 512
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        h_after_convs, w_after_convs = nnmodulehelper.conv2d_output_size(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        
        self.inference_x = nn.Sequential()
        
        # convolution layers
        self.inference_x.add_module("conv_{}".format(1), nn.Sequential(nn.Conv2d(self.n_channels, hidden_channels, kernels_size[0], strides[0], pads[0], dils[0]), nn.BatchNorm2d(hidden_channels), nn.LeakyReLU(1e-2, inplace=True)))
        for conv_layer_id in range(1, self.n_conv_layers):
            self.inference_x.add_module("conv_{}".format(conv_layer_id+1), nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), nn.BatchNorm2d(hidden_channels), nn.LeakyReLU(1e-2, inplace=True)))
        self.inference_x.add_module("flatten", nnmodulehelper.Flatten())
        
        # linear layers
        self.inference_x.add_module("lin_1", nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim_0), nn.LeakyReLU(1e-2, inplace=True)))
        self.inference_x.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim_0, hidden_dim_0), nn.LeakyReLU(1e-2, inplace=True)))
        
        self.inference_joint = nn.Sequential (
                nn.Linear(hidden_dim_0 + self.n_latents, hidden_dim_1),
                nn.LeakyReLU(1e-2, inplace=True),
                nn.Linear(hidden_dim_1, hidden_dim_1),
                nn.LeakyReLU(1e-2, inplace=True),
                nn.Linear(hidden_dim_1, 1),
                )

        
    def forward(self, x, z):
        output_x = self.inference_x(x)
        output = self.inference_joint(torch.cat((output_x, z), 1))
        return output.squeeze()
    
    
'''    
class DiscriminatorBurgess2 (nn.Module):
    
    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 4, n_latents = 10,  **kwargs):
        super(DiscriminatorBurgess2, self).__init__()
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents

        # network architecture
        hidden_channels = 32
        hidden_dim_0 = 256
        hidden_dim_1 = 512
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        h_after_convs, w_after_convs = helper.conv2d_output_size(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        
        self.inference_x = nn.Sequential()
        
        # convolution layers
        self.inference_x.add_module("conv_{}".format(1), nn.Sequential(nn.Conv2d(self.n_channels, hidden_channels, kernels_size[0], strides[0], pads[0], dils[0]), nn.BatchNorm2d(hidden_channels), nn.LeakyReLU(1e-2, inplace=True)))
        for conv_layer_id in range(1, self.n_conv_layers):
            self.inference_x.add_module("conv_{}".format(conv_layer_id+1), nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), nn.BatchNorm2d(hidden_channels), nn.LeakyReLU(1e-2, inplace=True)))
        self.inference_x.add_module("flatten", helper.Flatten())
        
        # linear layers
        self.inference_x.add_module("lin_1", nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim_0), nn.LeakyReLU(1e-2, inplace=True)))
        self.inference_x.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim_0, hidden_dim_0), nn.LeakyReLU(1e-2, inplace=True)))
        
        self.inference_joint = nn.Sequential (
                nn.Linear(hidden_dim_0 + self.n_latents, hidden_dim_1),
                nn.LeakyReLU(1e-2, inplace=True),
                nn.Linear(hidden_dim_1, hidden_dim_1),
                nn.LeakyReLU(1e-2, inplace=True),
                nn.Linear(hidden_dim_1, 1),
                )

        
    def forward(self, x, z):
        output_x = self.inference_x(x)
        output = self.inference_joint(torch.cat((output_x, z), 1))
        return output.squeeze()
    

class DiscriminatorAli(nn.Module):

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 6, n_latents = 32, output_size=1, **kwargs):
        super(DiscriminatorAli, self).__init__()
        # network parameters
        self.output_size = output_size
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
        
        self.infer_x = nn.Sequential()
        
        # convolution layers
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.infer_x.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(self.n_channels, hidden_channels, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id], dils[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.Dropout2d(0.5),
                        nn.Conv2d(hidden_channels, 2*hidden_channels, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1], dils[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(2*hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.Dropout2d(0.5)
                        ))
            else:
                self.infer_x.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(hidden_channels, hidden_channels, kernels_size[2*conv_layer_id], strides[2*conv_layer_id], pads[2*conv_layer_id], dils[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.Dropout2d(0.5),
                        nn.Conv2d(hidden_channels, 2*hidden_channels, kernels_size[2*conv_layer_id+1], strides[2*conv_layer_id+1], pads[2*conv_layer_id+1], dils[2*conv_layer_id+1]), 
                        nn.BatchNorm2d(2*hidden_channels), 
                        nn.LeakyReLU(inplace=True),
                        nn.Dropout2d(0.5),
                        ))
            hidden_channels *= 2
            
        self.infer_z = nn.Sequential(
                    nn.Conv2d(self.n_latents, hidden_channels, kernel_size=1, stride=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout2d(0.5),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout2d(0.5)
                )
            
        self.infer_joint = nn.Sequential(
                    nn.Conv2d(2*hidden_channels, 2*hidden_channels, kernel_size=1, stride=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout2d(0.5),
                    nn.Conv2d(2*hidden_channels, 2*hidden_channels, kernel_size=1, stride=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout2d(0.5)
                )

        self.final = nn.Conv2d(2*hidden_channels, self.output_size, kernel_size=1, stride=1)


    def forward(self, x, z):
        if len(z.size()) == 2:
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        return output.squeeze().squeeze()


class DiscriminatorAli2(nn.Module):

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 6, n_latents = 128, output_size = 1, **kwargs):
        super(DiscriminatorAli2, self).__init__()
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents
        self.output_size = output_size

        # network architecture
        hidden_channels = 32
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        
        self.infer_x = nn.Sequential()
        
        # convolution layers
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.infer_x.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(self.n_channels, hidden_channels, kernels_size[conv_layer_id+1], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), 
                        nn.LeakyReLU(0.02, inplace=True)
                        ))
            elif conv_layer_id < 4:
                self.infer_x.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(hidden_channels, 2*hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), 
                        nn.BatchNorm2d(2*hidden_channels), 
                        nn.LeakyReLU(0.02, inplace=True)
                        ))   
                hidden_channels *= 2
            else:
                self.infer_x.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                        nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), 
                        nn.BatchNorm2d(hidden_channels), 
                        nn.LeakyReLU(0.02, inplace=True)
                        ))
            
        self.infer_z = nn.Sequential(
                    nn.Conv2d(self.n_latents, hidden_channels, kernel_size=1, stride=1),
                    nn.BatchNorm2d(hidden_channels), 
                    nn.LeakyReLU(0.02, inplace=True),
        
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                    nn.BatchNorm2d(hidden_channels), 
                    nn.LeakyReLU(0.02, inplace=True)
                )
            
        self.infer_joint = nn.Sequential(
                    nn.Conv2d(2*hidden_channels, 2*hidden_channels, kernel_size=1, stride=1),
                    nn.BatchNorm2d(2*hidden_channels), 
                    nn.LeakyReLU(0.02, inplace=True),
        
                    nn.Conv2d(2*hidden_channels, 2*hidden_channels, kernel_size=1, stride=1),
                    nn.BatchNorm2d(2*hidden_channels), 
                    nn.LeakyReLU(0.02, inplace=True)
                )

        self.final = nn.Conv2d(2*hidden_channels, self.output_size, kernel_size=1, stride=1)


    def forward(self, x, z):
        if len(z.size()) == 2:
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        return output.squeeze().squeeze()
    
    
    
class DiscriminatorRadford(nn.Module):

    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 6, output_size = 1, **kwargs):
        super(DiscriminatorRadford, self).__init__()
         # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.output_size = output_size

        # network architecture
        hidden_channels = 32
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        
        
        self.discriminator = nn.Sequential()
        h_after_convs, w_after_convs = helper.conv2d_output_size(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        
        # convolution layers
        for conv_layer_id in range(self.n_conv_layers):
            if conv_layer_id == 0:
                self.discriminator.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                            nn.Conv2d(self.n_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], bias=False), 
                            nn.LeakyReLU(0.2, inplace=True)
                            ))
                
            else:            
                self.discriminator.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                            nn.Conv2d(hidden_channels, hidden_channels*2, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], bias=False), 
                            nn.BatchNorm2d(hidden_channels*2),
                            nn.LeakyReLU(0.2, inplace=True)
                            ))
                hidden_channels = hidden_channels * 2

        self.discriminator.add_module("linear_reshape", nn.Sequential(
                        helper.Flatten(),
                        nn.Linear(hidden_channels*h_after_convs*w_after_convs, self.output_size)
                        #nn.Sigmoid()
                        ))

    def forward(self, x):
        if self.output_size == 1:
            return self.discriminator(x).squeeze()
        else:
            return self.discriminator(x)
'''