from image_representation.representations.torch_nn.decoders import Decoder
from image_representation.utils.torch_nn_module import conv_output_sizes, convtranspose_get_output_padding
from image_representation.utils.minkowski_nn_module import MEChannelize
import torch
from torch import nn
import MinkowskiEngine as ME
import math


class Decoder(Decoder):

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        assert 2 <= len(self.config.input_size) <= 3, "Image must be 2D or 3D"
        self.spatial_dims = len(self.config.input_size)

        self.output_keys_list = ["z", "gfi", "lfi", "recon_x"]

    def forward(self, z):
        # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        # decoder output
        decoder_outputs = {"z": z, "gfi": gfi, "lfi": lfi, "recon_x": recon_x}

        return decoder_outputs

def get_decoder(model_architecture):
    """
    model_architecture: string such that the class encoder called is <model_architecture>Encoder
    """
    return eval("ME{}Decoder".format(model_architecture))

""" ========================================================================================================================
ME Decoder Modules 
========================================================================================================================="""

class MEDumoulinDecoder(Decoder):

    def __init__(self, config=None, **kwargs):
        Decoder.__init__(self, config=config, **kwargs)

        # need square and power of 2 image size input
        power = math.log(self.config.input_size[0], 2)
        assert (power % 1 == 0.0) and (power > 3), "Dumoulin Encoder needs a power of 2 as image input size (>=16)"
        assert self.config.input_size[0] == self.config.input_size[
            1], "Dumoulin Encoder needs a square image input size"

        assert self.config.n_conv_layers == power, "The number of convolutional layers in DumoulinEncoder must be log2(input_size) "

        # network architecture
        if self.config.hidden_channels is None:
            self.config.hidden_channels = int(512 // math.pow(2, self.config.n_conv_layers))
        hidden_channels = self.config.hidden_channels
        kernels_size = [4, 4] * self.config.n_conv_layers
        strides = [1, 2] * self.config.n_conv_layers
        pads = [0, 1] * self.config.n_conv_layers
        dils = [1, 1] * self.config.n_conv_layers

        feature_map_sizes = conv_output_sizes(self.config.input_size, 2 * self.config.n_conv_layers, kernels_size,
                                                strides, pads,
                                                dils)
        output_pads = [None] * 2 * self.config.n_conv_layers
        output_pads[0] = convtranspose_get_output_padding(feature_map_sizes[0], self.config.input_size,
                                                            kernels_size[0],
                                                            strides[0], pads[0])
        output_pads[1] = convtranspose_get_output_padding(feature_map_sizes[1], feature_map_sizes[0], kernels_size[1],
                                                            strides[1], pads[1])
        for conv_layer_id in range(1, self.config.n_conv_layers):
            output_pads[2 * conv_layer_id] = convtranspose_get_output_padding(feature_map_sizes[2 * conv_layer_id],
                                                                                feature_map_sizes[
                                                                                    2 * conv_layer_id - 1],
                                                                                kernels_size[2 * conv_layer_id],
                                                                                strides[2 * conv_layer_id],
                                                                                pads[2 * conv_layer_id])
            output_pads[2 * conv_layer_id + 1] = convtranspose_get_output_padding(
                feature_map_sizes[2 * conv_layer_id + 1], feature_map_sizes[2 * conv_layer_id + 1 - 1],
                kernels_size[2 * conv_layer_id + 1], strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1])

        # encoder feature inverse
        hidden_channels = int(hidden_channels * math.pow(2, self.config.n_conv_layers))
        self.efi = nn.Sequential(
            ME.MinkowskiConvolution(self.config.n_latents, hidden_channels,
                                                       kernel_size=1, stride=1,
                                                       dimension=self.spatial_dims, bias=True),
            ME.MinkowskiBatchNorm(hidden_channels),
            ME.MinkowskiELU(inplace=True),
            ME.MinkowskiConvolution(hidden_channels, hidden_channels,
                                                       kernel_size=1, stride=1,
                                                       dimension=self.spatial_dims, bias=True),
            ME.MinkowskiBatchNorm(hidden_channels),
            ME.MinkowskiELU(inplace=True),
        )
        self.efi.out_connection_type = ("conv", hidden_channels)
        self.efi_cls = ME.MinkowskiConvolution(hidden_channels, 1, kernel_size=1, bias=True, dimension=self.spatial_dims)

        # global feature inverse
        self.gfi = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.config.n_conv_layers - 1, self.config.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(hidden_channels, hidden_channels // 2,
                                                          kernel_size=kernels_size[2 * conv_layer_id + 1],
                                                          stride=strides[2 * conv_layer_id + 1],
                                                          dimension=self.spatial_dims, bias=True),
                ME.MinkowskiBatchNorm(hidden_channels // 2),
                ME.MinkowskiELU(inplace=True),
                ME.MinkowskiConvolution(hidden_channels // 2, hidden_channels // 2,
                                                           kernel_size=kernels_size[2 * conv_layer_id ],
                                                           stride=strides[2 * conv_layer_id],
                                                           dimension=self.spatial_dims, bias=True),
                ME.MinkowskiBatchNorm(hidden_channels // 2),
                ME.MinkowskiELU(inplace=True),
            ))
            hidden_channels = hidden_channels // 2
        self.gfi.out_connection_type = ("conv", hidden_channels)
        self.gfi_cls = ME.MinkowskiConvolution(hidden_channels, 1, kernel_size=1, bias=True, dimension=self.spatial_dims)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(hidden_channels, hidden_channels // 2,
                                                           kernel_size=kernels_size[2 * conv_layer_id + 1],
                                                           stride=strides[2 * conv_layer_id + 1],
                                                           dimension=self.spatial_dims, bias=True),
                ME.MinkowskiBatchNorm(hidden_channels // 2),
                ME.MinkowskiELU(inplace=True),
                ME.MinkowskiConvolution(hidden_channels // 2, hidden_channels // 2,
                                                           kernel_size=kernels_size[2 * conv_layer_id],
                                                           stride=strides[2 * conv_layer_id],
                                                           dimension=self.spatial_dims, bias=True),
                ME.MinkowskiBatchNorm(hidden_channels // 2),
                ME.MinkowskiELU(inplace=True),
            ))
            hidden_channels = hidden_channels // 2

        self.lfi.add_module("conv_0_i", nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(hidden_channels, hidden_channels // 2,
                                                       kernel_size=kernels_size[1],
                                                       stride=strides[1],
                                                       dimension=self.spatial_dims, bias=True),
            ME.MinkowskiBatchNorm(hidden_channels // 2),
            ME.MinkowskiELU(inplace=True),
            ME.MinkowskiConvolution(hidden_channels // 2,  self.config.n_channels,
                                                       kernel_size=kernels_size[0],
                                                       stride=strides[0],
                                                       dimension=self.spatial_dims, bias=True),
        ))

        self.lfi.out_connection_type = ("conv", self.config.n_channels)
        self.lfi_cls = ME.MinkowskiConvolution(self.config.n_channels, 1, kernel_size=1, bias=True, dimension=self.spatial_dims)

        # pruning
        self.pruning = ME.MinkowskiPruning()

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def forward(self, z, target_key):
        # global feature map
        gfi = self.efi(z)
        gfi_cls = self.efi_cls(gfi)
        gfi_target = self.get_target(gfi, target_key)
        gfi_keep = (gfi_cls.F > 0).squeeze()
        if self.training:
            gfi_keep += gfi_target
        gfi = self.pruning(gfi, gfi_keep)

        # global feature map
        lfi = self.gfi(gfi)
        lfi_cls = self.gfi_cls(lfi)
        lfi_target = self.get_target(lfi, target_key)
        lfi_keep = (lfi_cls.F > 0).squeeze()
        if self.training:
            lfi_keep += lfi_target
        lfi = self.pruning(lfi, lfi_keep)

        # recon_x
        recon_x = self.lfi(lfi)
        recon_x_cls = self.lfi_cls(recon_x)
        recon_x_target = self.get_target(recon_x, target_key)
        recon_x_keep = (recon_x_cls.F > 0).squeeze()
        if self.training:
            recon_x_keep += recon_x_target
        recon_x = self.pruning(recon_x, recon_x_keep)

        # decoder output
        decoder_outputs = {"z": z, "gfi": gfi, "gfi_cls": gfi_cls, "gfi_target": gfi_target,
                           "lfi": lfi, "lfi_cls": lfi_cls, "lfi_target": lfi_target,
                           "recon_x": recon_x, "recon_x_cls": recon_x_cls, "recon_x_target": recon_x_target}

        return decoder_outputs