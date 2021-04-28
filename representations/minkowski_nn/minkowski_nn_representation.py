from image_representation import TorchNNRepresentation
from image_representation.representations.minkowski_nn import encoders, losses
from image_representation.utils.minkowski_nn_init import get_weights_init
from torch import nn


class MinkowskiNNRepresentation(TorchNNRepresentation):

    def set_network(self, network_name, network_parameters):
        """
        Define the network modules, 
        Here simple encoder but this function is overwritten to include generator/discriminator.
        """
        self.network = nn.Module()
        encoder_class = encoders.get_encoder(network_name)
        self.network.encoder = encoder_class(config=network_parameters)
        self.n_latents = self.network.encoder.config.n_latents

        # update config
        self.config.network.name = network_name
        self.config.network.parameters.update(network_parameters)

    def init_network_weights(self, weights_init_name, weights_init_parameters):
        weights_init_function = get_weights_init(weights_init_name)
        if weights_init_name == "pretrain":
            self.network = weights_init_function(self.network, **weights_init_parameters)
        else:
            self.network.apply(lambda m: weights_init_function(m, **weights_init_parameters))

        # update config
        self.config.network.weights_init.name = weights_init_name
        self.config.network.weights_init.parameters.update(weights_init_parameters)


    def set_loss(self, loss_name, loss_parameters):
        loss_class = losses.get_loss(loss_name)
        self.loss_f = loss_class(**loss_parameters)

        # update config
        self.config.loss.name = loss_name
        self.config.loss.parameters.update(loss_parameters)