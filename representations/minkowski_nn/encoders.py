from image_representation.representations.torch_nn.encoders import Encoder
from image_representation.utils.torch_nn_module import conv_output_sizes
from image_representation.utils.minkowski_nn_module import MEFlatten
import math
import torch
from torch import nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as F


class MEEncoder(Encoder):
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

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        assert 2 <= len(self.config.input_size) <= 3, "Image must be 2D or 3D"
        self.spatial_dims = len(self.config.input_size)

        self.output_keys_list = ["x", "lf", "gf", "z"]
        if self.config.encoder_conditional_type == "gaussian":
            self.output_keys_list += ["mu", "logvar"]

    def forward(self, x):
        # local feature map
        lf = self.lf(x)
        # global feature map
        gf = self.gf(lf)

        encoder_outputs = {"x": x, "lf": lf, "gf": gf}

        # encoding
        if self.config.encoder_conditional_type == "gaussian":
            ef = self.global_pool(self.ef(gf))
            mu, logvar = torch.chunk(ef.F, 2, dim=1)
            zf = self.reparameterize(mu, logvar)
            z = ME.SparseTensor(
                features=zf,
                coordinates=ef.C,
                tensor_stride=torch.tensor([2**self.config.n_conv_layers]*self.spatial_dims, device=zf.device),
                coordinate_manager=ef.coordinate_manager,
            )
            encoder_outputs.update({"z": z, "mu": mu, "logvar": logvar})
        elif self.config.encoder_conditional_type == "deterministic":
            ef = self.global_pool(self.ef(gf))
            z = ME.SparseTensor(
                features=ef.F,
                coordinates=ef.C,
                tensor_stride=torch.tensor([2 ** self.config.n_conv_layers] * self.spatial_dims, device=ef.device),
                coordinate_manager=ef.coordinate_manager,
            )
            encoder_outputs.update({"z": z})

        # attention features
        if self.config.use_attention:
            af = self.af(gf)
            af = F.normalize(af, p=2)
            encoder_outputs.update({"af": af})

        return encoder_outputs


    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def calc_embedding(self, x):
        encoder_outputs = self.forward(x)
        return encoder_outputs["z"]


def get_encoder(model_architecture):
    """
    model_architecture: string such that the class encoder called is <model_architecture>Encoder
    """
    return eval("ME{}Encoder".format(model_architecture))

""" ========================================================================================================================
ME Encoder Modules 
========================================================================================================================="""

class MEDumoulinEncoder(MEEncoder):
    """
    Some Alexnet-inspired encoder with BatchNorm and LeakyReLU as proposed in Dumoulin et al. "Adversarially learned inference"

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

    def __init__(self, config=None, **kwargs):
        MEEncoder.__init__(self, config=config, **kwargs)

        # need square and power of 2 image size input
        power = math.log(self.config.input_size[0], 2)
        assert (power % 1 == 0.0) and (power > 3), "Dumoulin Encoder needs a power of 2 as image input size (>=16)"
        # need square image input
        assert torch.all(torch.tensor(
            [self.config.input_size[i] == self.config.input_size[0] for i in
             range(1, len(self.config.input_size))])), "Dumoulin Encoder needs a square image input size"

        assert self.config.n_conv_layers == power, "The number of convolutional layers in DumoulinEncoder must be log2(input_size) "

        # network architecture
        if self.config.hidden_channel is None:
            self.config.hidden_channel = 8

        hidden_channels = self.config.hidden_channels
        kernels_size = [4, 4] * self.config.n_conv_layers
        strides = [1, 2] * self.config.n_conv_layers
        pads = [0, 1] * self.config.n_conv_layers
        dils = [1, 1] * self.config.n_conv_layers

        # feature map size
        feature_map_sizes = conv_output_sizes(self.config.input_size, 2 * self.config.n_conv_layers, kernels_size,
                                              strides, pads,
                                              dils)

        # local feature
        ## convolutional layers
        self.local_feature_shape = (
            int(hidden_channels * math.pow(2, self.config.feature_layer + 1)),
            feature_map_sizes[2 * self.config.feature_layer + 1][0],
            feature_map_sizes[2 * self.config.feature_layer + 1][1])
        self.lf = nn.Sequential()

        for conv_layer_id in range(self.config.feature_layer + 1):
            if conv_layer_id == 0:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    ME.MinkowskiConvolution(self.config.n_channels, hidden_channels, kernel_size=kernels_size[2 * conv_layer_id],
                                            stride=strides[2 * conv_layer_id], dilation=dils[2 * conv_layer_id],
                                            dimension=self.spatial_dims, bias=True),
                    ME.MinkowskiBatchNorm(hidden_channels),
                    ME.MinkowskiELU(inplace=True),
                    ME.MinkowskiConvolution(hidden_channels, 2 * hidden_channels,
                                            kernel_size=kernels_size[2 * conv_layer_id + 1],
                                            stride=strides[2 * conv_layer_id + 1], dilation=dils[2 * conv_layer_id + 1],
                                            dimension=self.spatial_dims, bias=True),
                    ME.MinkowskiBatchNorm(2 * hidden_channels),
                    ME.MinkowskiELU(inplace=True),
                ))
            else:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    ME.MinkowskiConvolution(hidden_channels, hidden_channels,
                                            kernel_size=kernels_size[2 * conv_layer_id],
                                            stride=strides[2 * conv_layer_id], dilation=dils[2 * conv_layer_id],
                                            dimension=self.spatial_dims, bias=True),
                    ME.MinkowskiBatchNorm(hidden_channels),
                    ME.MinkowskiELU(inplace=True),
                    ME.MinkowskiConvolution(hidden_channels, 2 * hidden_channels,
                                            kernel_size=kernels_size[2 * conv_layer_id + 1],
                                            stride=strides[2 * conv_layer_id + 1], dilation=dils[2 * conv_layer_id + 1],
                                            dimension=self.spatial_dims, bias=True),
                    ME.MinkowskiBatchNorm(2 * hidden_channels),
                    ME.MinkowskiELU(inplace=True),
                ))
            hidden_channels *= 2
        self.lf.out_connection_type = ("conv", hidden_channels)

        # global feature
        self.gf = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.config.feature_layer + 1, self.config.n_conv_layers):
            self.gf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                ME.MinkowskiConvolution(hidden_channels, hidden_channels,
                                        kernel_size=kernels_size[2 * conv_layer_id],
                                        stride=strides[2 * conv_layer_id], dilation=dils[2 * conv_layer_id],
                                        dimension=self.spatial_dims, bias=True),
                ME.MinkowskiBatchNorm(hidden_channels),
                ME.MinkowskiELU(inplace=True),
                ME.MinkowskiConvolution(hidden_channels, 2 * hidden_channels,
                                        kernel_size=kernels_size[2 * conv_layer_id + 1],
                                        stride=strides[2 * conv_layer_id + 1], dilation=dils[2 * conv_layer_id + 1],
                                        dimension=self.spatial_dims, bias=True),
                ME.MinkowskiBatchNorm(2 * hidden_channels),
                ME.MinkowskiELU(inplace=True),
            ))
            hidden_channels *= 2
        self.gf.out_connection_type = ("conv", hidden_channels)

        # encoding feature
        if self.config.encoder_conditional_type == "gaussian":
            self.add_module("ef", nn.Sequential(
                ME.MinkowskiConvolution(hidden_channels, hidden_channels,
                                        kernel_size=1, stride=1, dilation=1,
                                        dimension=self.spatial_dims, bias=True),
                ME.MinkowskiBatchNorm(hidden_channels),
                ME.MinkowskiELU(inplace=True),
                ME.MinkowskiConvolution(hidden_channels, 2 * self.config.n_latents,
                                        kernel_size=1, stride=1, dilation=1,
                                        dimension=self.spatial_dims, bias=True),
            ))
        elif self.config.encoder_conditional_type == "deterministic":
            self.add_module("ef", nn.Sequential(
                ME.MinkowskiConvolution(hidden_channels, hidden_channels,
                                        kernel_size=1, stride=1, dilation=1,
                                        dimension=self.spatial_dims, bias=True),
                ME.MinkowskiBatchNorm(hidden_channels),
                ME.MinkowskiELU(inplace=True),
                ME.MinkowskiConvolution(hidden_channels, self.config.n_latents,
                                        kernel_size=1, stride=1, dilation=1,
                                        dimension=self.spatial_dims, bias=True),
            ))

        # global pool
        self.global_pool = ME.MinkowskiGlobalPooling()

        # attention feature
        if self.config.use_attention:
            self.add_module("af", ME.MinkowskiConvolution(hidden_channels, 4 * self.config.n_latents,
                                                     kernel_size=1, stride=1, dilation=1,
                                                     dimension=self.spatial_dims, bias=True
                                                     ))

    def forward(self, x):
        # batch norm cannot deal with batch_size 1 in train mode
        if self.training and len(x._batchwise_row_indices) == 1:
            self.eval()
            encoder_outputs = MEEncoder.forward(self, x)
            self.train()
        else:
            encoder_outputs = MEEncoder.forward(self, x)
        return encoder_outputs


class MEFDumoulinEncoder(MEDumoulinEncoder):
    pass