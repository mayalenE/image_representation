from math import floor
import torch
from torch import nn

''' ---------------------------------------------
               NN MODULES HELPERS
-------------------------------------------------'''


class Flatten(nn.Module):
    """Flatten the input """

    def forward(self, input):
        return input.view(input.size(0), -1)


'''
class LinearFromFlatten(nn.Module):
    """Flatten the input and then apply a linear module """
    def __init__(self, output_flat_size):
        super(LinearFromFlatten, self).__init__()
        self.output_flat_size = output_flat_size
        
    def forward(self, input):
        input =  input.view(input.size(0), -1) # Batch_size * flatenned_size
        input_flatten_size = input.size(1) 
        Linear = nn.Linear(input_flatten_size, self.output_flat_size)
        return Linear(input)
 '''


class Channelize(nn.Module):
    """Channelize a flatten input to the given (C,D,H,W) or (C,H,W) output """

    def __init__(self, n_channels, out_size):
        nn.Module.__init__(self)
        self.n_channels = n_channels
        self.out_size = out_size

    def forward(self, input):
        out_size = (input.size(0), self.n_channels, ) + self.out_size
        return input.view(out_size)


class SphericPad(nn.Module):
    """Pads spherically the input on all sides with the given padding size."""

    def __init__(self, padding_size):
        nn.Module.__init__(self)
        if isinstance(padding_size, int):
            self.pad_left = self.pad_right = self.pad_top = self.pad_bottom = padding_size
        elif isinstance(padding_size, tuple) and len(padding_size) == 2:
            self.pad_left = self.pad_right = padding_size[0]
            self.pad_top = self.pad_bottom = padding_size[1]
        elif isinstance(padding_size, tuple) and len(padding_size) == 4:
            self.pad_left = padding_size[0]
            self.pad_top = padding_size[1]
            self.pad_right = padding_size[2]
            self.pad_bottom = padding_size[3]
        else:
            raise ValueError('The padding size shoud be: int, tuple of size 2 or tuple of size 4')

    def forward(self, input):

        output = torch.cat([input, input[:, :, :self.pad_bottom, :]], dim=2)
        output = torch.cat([output, output[:, :, :, :self.pad_right]], dim=3)
        output = torch.cat([output[:, :, -(self.pad_bottom + self.pad_top):-self.pad_bottom, :], output], dim=2)
        output = torch.cat([output[:, :, :, -(self.pad_right + self.pad_left):-self.pad_right], output], dim=3)

        return output


class Roll(nn.Module):
    """Rolls spherically the input with the given padding shit on the given dimension."""

    def __init__(self, shift, dim):
        nn.Module.__init__(self)
        self.shift = shift
        self.dim = dim

    def forward(self, input):
        """ Shifts an image by rolling it"""
        if self.shift == 0:
            return input

        elif self.shift < 0:
            self.shift = -self.shift
            gap = input.index_select(self.dim, torch.arange(self.shift, dtype=torch.long))
            return torch.cat(
                [input.index_select(self.dim, torch.arange(self.shift, input.size(self.dim), dtype=torch.long)), gap],
                dim=self.dim)

        else:
            self.shift = input.size(self.dim) - self.shift
            gap = input.index_select(self.dim, torch.arange(self.shift, input.size(self.dim), dtype=torch.long))
            return torch.cat([gap, input.index_select(self.dim, torch.arange(self.shift, dtype=torch.long))],
                             dim=self.dim)


def conv_output_sizes(input_size, n_conv=0, kernels_size=1, strides=1, pads=0, dils=1):
    """Returns the size of a tensor after a sequence of convolutions"""
    assert n_conv == len(kernels_size) == len(strides) == len(pads) == len(dils), print(
        'The number of kernels ({}), strides({}), paddings({}) and dilatations({}) has to match the number of convolutions({})'.format(
            len(kernels_size), len(strides), len(pads), len(dils), n_conv))

    spatial_dims = len(input_size) #2D or 3D
    in_sizes = list(input_size)
    output_sizes = []

    for conv_id in range(n_conv):
        if type(kernels_size[conv_id]) is not tuple:
            kernel_size = tuple([kernels_size[conv_id]]*spatial_dims)
        if type(strides[conv_id]) is not tuple:
            stride = tuple([strides[conv_id]]*spatial_dims)
        if type(pads[conv_id]) is not tuple:
            pad = tuple([pads[conv_id]]*spatial_dims)
        if type(dils[conv_id]) is not tuple:
            dil = tuple([dils[conv_id]]*spatial_dims)

        for dim in range(spatial_dims):
            in_sizes[dim] = floor(((in_sizes[dim] + (2 * pad[dim]) - (dil[dim] * (kernel_size[dim] - 1)) - 1) / stride[dim]) + 1)

        output_sizes.append(tuple(in_sizes))

    return output_sizes


def convtranspose_get_output_padding(input_size, output_size, kernel_size=1, stride=1, pad=0):

    assert len(input_size)==len(output_size)
    spatial_dims = len(input_size)  # 2D or 3D
    out_padding = []

    if type(kernel_size) is not tuple:
        kernel_size = tuple([kernel_size]*spatial_dims)
    if type(stride) is not tuple:
        stride = tuple([stride]*spatial_dims)
    if type(pad) is not tuple:
        pad = tuple([pad]*spatial_dims)

    out_padding = []
    for dim in range(spatial_dims):
        out_padding.append(output_size[dim] + 2 * pad[dim] - kernel_size[dim] - (input_size[dim] - 1) * stride[dim])

    return tuple(out_padding)
