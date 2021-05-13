from image_representation.representations.torch_nn.decoders import Decoder
from copy import deepcopy
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

        self.output_keys_list = ["gfi", "lfi", "recon_x"]

    def forward(self, z, *args):

        # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        # decoder output
        decoder_outputs = {"gfi": gfi, "lfi": lfi, "recon_x": recon_x}

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
        kernels_size = [3, 2] * self.config.n_conv_layers
        strides = [1, 2] * self.config.n_conv_layers
        dils = [1, 1] * self.config.n_conv_layers

        if self.config.hidden_channel is None:
            self.config.hidden_channel = 8


        # encoder feature inverse
        hidden_channels = int(self.config.hidden_channels * math.pow(2, self.config.n_conv_layers))
        self.efi = nn.Sequential(
            ME.MinkowskiConvolution(self.config.n_latents, hidden_channels,
                                                       kernel_size=1, stride=1, #dilation=1,
                                                       dimension=self.spatial_dims, bias=False),
            ME.MinkowskiBatchNorm(hidden_channels),
            ME.MinkowskiELU(inplace=True),
            ME.MinkowskiConvolution(hidden_channels, hidden_channels,
                                                       kernel_size=1, stride=1, #dilation=1,
                                                       dimension=self.spatial_dims, bias=False),
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
                                                          #dilation=dils[2 * conv_layer_id + 1],
                                                          dimension=self.spatial_dims, bias=False),
                ME.MinkowskiBatchNorm(hidden_channels // 2),
                ME.MinkowskiELU(inplace=True),
                ME.MinkowskiConvolution(hidden_channels // 2, hidden_channels // 2,
                                                           kernel_size=kernels_size[2 * conv_layer_id ],
                                                           stride=strides[2 * conv_layer_id],
                                                           #dilation=dils[2 * conv_layer_id],
                                                           dimension=self.spatial_dims, bias=False),
                ME.MinkowskiBatchNorm(hidden_channels // 2),
                ME.MinkowskiELU(inplace=True),
            ))
            hidden_channels = hidden_channels // 2

            self.gfi.add_module("cls_{}_i".format(conv_layer_id), ME.MinkowskiConvolution(hidden_channels, 1, kernel_size=1, bias=True, dimension=self.spatial_dims))

        self.gfi.out_connection_type = ("conv", hidden_channels)
        #self.gfi_cls = ME.MinkowskiConvolution(hidden_channels, 1, kernel_size=1, bias=True, dimension=self.spatial_dims)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(hidden_channels, hidden_channels // 2,
                                                           kernel_size=kernels_size[2 * conv_layer_id + 1],
                                                           stride=strides[2 * conv_layer_id + 1],
                                                           #dilation=dils[2 * conv_layer_id + 1],
                                                           dimension=self.spatial_dims, bias=False),
                ME.MinkowskiBatchNorm(hidden_channels // 2),
                ME.MinkowskiELU(inplace=True),
                ME.MinkowskiConvolution(hidden_channels // 2, hidden_channels // 2,
                                                           kernel_size=kernels_size[2 * conv_layer_id],
                                                           stride=strides[2 * conv_layer_id],
                                                           #dilation=dils[2 * conv_layer_id],
                                                           dimension=self.spatial_dims, bias=False),
                ME.MinkowskiBatchNorm(hidden_channels // 2),
                ME.MinkowskiELU(inplace=True),
            ))
            hidden_channels = hidden_channels // 2

            self.lfi.add_module("cls_{}_i".format(conv_layer_id), ME.MinkowskiConvolution(hidden_channels, 1, kernel_size=1, bias=True, dimension=self.spatial_dims))

        self.lfi.add_module("conv_0_i", nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(hidden_channels, hidden_channels // 2,
                                                       kernel_size=kernels_size[1],
                                                       stride=strides[1], #dilation=dils[1],
                                                       dimension=self.spatial_dims, bias=False),
            ME.MinkowskiBatchNorm(hidden_channels // 2),
            ME.MinkowskiELU(inplace=True),
            ME.MinkowskiConvolution(hidden_channels // 2,  self.config.n_channels,
                                                       kernel_size=kernels_size[0],
                                                       stride=strides[0], #dilation=dils[0],
                                                       dimension=self.spatial_dims, bias=False),
        ))
        self.lfi.add_module("cls_0_i", ME.MinkowskiConvolution(self.config.n_channels, 1, kernel_size=1, bias=False, dimension=self.spatial_dims))

        self.lfi.out_connection_type = ("conv", self.config.n_channels)
        #self.lfi_cls = ME.MinkowskiConvolution(self.config.n_channels, 1, kernel_size=1, bias=True, dimension=self.spatial_dims)

        # pruning
        self.pruning = ME.MinkowskiPruning()

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(
            target_key,
            out.tensor_stride[0],
        )
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    @torch.no_grad()
    def empty_return(self):
        device = self.efi_cls.kernel.device
        dtype = self.efi_cls.kernel.dtype
        gfi = ME.SparseTensor(coordinates=torch.empty((0, self.spatial_dims+1), device=device),
                              features=torch.empty((0, self.efi.out_connection_type[1]), device=device,
                                                   dtype=dtype),
                              tensor_stride=[2 ** (self.config.n_conv_layers)] * self.spatial_dims)

        lfi = ME.SparseTensor(coordinates=torch.empty((0, self.spatial_dims+1), device=device),
                              features=torch.empty((0, self.gfi.out_connection_type[1]), device=device, dtype=dtype),
                              tensor_stride=[2 ** (self.config.feature_layer + 1)] * self.spatial_dims)

        recon_x = ME.SparseTensor(coordinates=torch.empty((0, self.spatial_dims+1), device=device),
                                  features=torch.empty((0, self.lfi.out_connection_type[1]), device=device,
                                                       dtype=dtype),
                                  tensor_stride=[1] * self.spatial_dims)

        empty_outputs = {"gfi": gfi, "lfi": lfi, "recon_x": recon_x}
        return empty_outputs



    def forward(self, z, target_key):

        out_cls, out_targets = [], []

        # global feature map
        gfi = self.efi(z)
        gfi_cls = self.efi_cls(gfi)
        out_cls.append(gfi_cls)
        gfi_target = self.get_target(gfi, target_key)
        out_targets.append(gfi_target)
        gfi_keep = (gfi_cls.F > 0).squeeze(-1)
        if self.training:
            gfi_keep += gfi_target
        if gfi_keep.sum() == 0:
            empty_outputs = self.empty_return()
            empty_outputs.update({"out_cls": out_cls, "out_targets": out_targets})
            try:
                print(empty_outputs)
            except:
                pass
            return empty_outputs
        elif gfi_keep.sum() > 0:
            gfi = self.pruning(gfi, gfi_keep)


        # local feature map
        gfi_out = gfi
        for conv_layer_id in range(self.config.n_conv_layers - 1, self.config.feature_layer + 1 - 1, -1):
            gfi_out = eval(f"self.gfi.conv_{conv_layer_id}_i")(gfi_out)
            gfi_out_cls = eval(f"self.gfi.cls_{conv_layer_id}_i")(gfi_out)
            out_cls.append(gfi_out_cls)
            gfi_out_target = self.get_target(gfi_out, target_key)
            out_targets.append(gfi_out_target)
            gfi_out_keep = (gfi_out_cls.F > 0).squeeze(-1)
            if self.training:
                gfi_out_keep += gfi_out_target
            if gfi_out_keep.sum() == 0:
                empty_outputs = self.empty_return()
                empty_outputs.update({"gfi": gfi, "out_cls": out_cls, "out_targets": out_targets})
                try:
                    print(empty_outputs)
                except:
                    pass
                return empty_outputs
            elif gfi_out_keep.sum() > 0:
                gfi_out = self.pruning(gfi_out, gfi_out_keep)
        lfi = gfi_out

        # recon_x
        lfi_out = lfi
        for conv_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            lfi_out = eval(f"self.lfi.conv_{conv_layer_id}_i")(lfi_out)
            lfi_out_cls = eval(f"self.lfi.cls_{conv_layer_id}_i")(lfi_out)
            out_cls.append(lfi_out_cls)
            lfi_out_target = self.get_target(lfi_out, target_key)
            out_targets.append(lfi_out_target)
            lfi_out_keep = (lfi_out_cls.F > 0).squeeze(-1)
            if self.training:
                lfi_out_keep += lfi_out_target
            if lfi_out_keep.sum() == 0:
                empty_outputs = self.empty_return()
                empty_outputs.update({"gfi": gfi, "lfi": lfi, "out_cls": out_cls, "out_targets": out_targets})
                try:
                    print(empty_outputs)
                except:
                    pass
                return empty_outputs
            elif lfi_out_keep.sum() > 0:
                lfi_out = self.pruning(lfi_out, lfi_out_keep)

        recon_x = self.lfi.conv_0_i(lfi_out)
        recon_x_cls = self.lfi.cls_0_i(recon_x)
        out_cls.append(recon_x_cls)
        recon_x_target = self.get_target(recon_x, target_key)
        out_targets.append(recon_x_target)
        recon_x_keep = (recon_x_cls.F > 0).squeeze(-1)
        recon_x = self.pruning(recon_x, recon_x_keep)

        # decoder output
        decoder_outputs = {"gfi": gfi, "lfi": lfi, "recon_x": recon_x, "out_cls": out_cls, "out_targets": out_targets}

        return decoder_outputs

class MEFDumoulinDecoder(Decoder):

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

        # global feature inverse
        self.gfi = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.config.n_conv_layers - 1, self.config.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                ME.MinkowskiConvolutionTranspose(hidden_channels, hidden_channels // 2,
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
        self.gfi.out_connection_type = ("conv", hidden_channels)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                ME.MinkowskiConvolutionTranspose(hidden_channels, hidden_channels // 2,
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
            ME.MinkowskiConvolutionTranspose(hidden_channels, hidden_channels // 2,
                                             kernel_size=kernels_size[1],
                                             stride=strides[1],
                                             dimension=self.spatial_dims, bias=True),
            ME.MinkowskiBatchNorm(hidden_channels // 2),
            ME.MinkowskiELU(inplace=True),
            ME.MinkowskiConvolution(hidden_channels // 2, self.config.n_channels,
                                    kernel_size=kernels_size[0],
                                    stride=strides[0],
                                    dimension=self.spatial_dims, bias=True),
        ))

        self.lfi.out_connection_type = ("conv", self.config.n_channels)
