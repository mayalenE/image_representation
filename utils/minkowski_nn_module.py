from torch import nn
import torch
import MinkowskiEngine as ME

''' ---------------------------------------------
               ME MODULES HELPERS
-------------------------------------------------'''


class MEFlatten(nn.Module):
    """Flatten the input """

    def forward(self, input):
        #return input.view(input.size(0), -1)
        output = torch.stack([t.view(-1) for t in input.decomposed_features])
        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

class MEChannelize(nn.Module):
    """Channelize a flatten input to the given (C,D,H,W) or (C,H,W) output """

    def __init__(self, n_channels, out_size):
        nn.Module.__init__(self)
        self.n_channels = n_channels
        self.out_size = out_size

    def forward(self, input):
        # out_size = (input.size(0), self.n_channels, ) + self.out_size
        # return input.view(out_size)
        raise NotImplementedError