import image_representation
from image_representation import Representation
from image_representation.representations.torch_nn import encoders, losses
from image_representation.utils.torch_nn_init import get_weights_init
from addict import Dict
import torch
from torch import nn
import warnings


class TorchNNRepresentation(Representation, nn.Module):
    """
    Base Torch NN Representation Class
    Squeleton to follow for each dnn model, here simple single encoder that is not trained.
    """

    @staticmethod
    def default_config():
        default_config = Dict()

        # network parameters
        default_config.network = Dict()
        default_config.network.name = "Burgess"
        default_config.network.parameters = Dict()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4

        # weights_init parameters
        default_config.network.weights_init = Dict()
        default_config.network.weights_init.name = "pytorch"
        default_config.network.weights_init.parameters = Dict()

        # device parameters
        default_config.device = 'cuda'

        # loss parameters
        default_config.loss = Dict()
        default_config.loss.name = "VAE"
        default_config.loss.parameters = Dict()

        # optimizer parameters
        default_config.optimizer = Dict()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = Dict()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5

        # In training folder:
        ## logging (will save every X epochs)
        default_config.logging = Dict()
        default_config.logging.record_loss_every = 1
        default_config.logging.record_valid_images_every = 1
        default_config.logging.record_embeddings_every = 1

        ## checkpoints (will save model every X epochs)
        default_config.checkpoint = Dict()
        default_config.checkpoint.folder = None
        default_config.checkpoint.save_model_every = 10
        default_config.checkpoint.save_model_at_epochs = []

        ## evaluation (when we do testing during training, save every X epochs)
        default_config.evaluation = Dict()
        default_config.evaluation.folder = None
        default_config.evaluation.save_results_every = 1

        return default_config

    def __init__(self, config=None, **kwargs):
        Representation.__init__(self, config=config, **kwargs)
        nn.Module.__init__(self)


        # define the device to use (gpu or cpu)
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            self.config.device = 'cpu'
            warnings.warn("Cannot set model device as GPU because not available, setting it to CPU")

        # network
        self.set_network(self.config.network.name, self.config.network.parameters)
        self.init_network_weights(self.config.network.weights_init.name, self.config.network.weights_init.parameters)
        self.set_device(self.config.device)

        # loss function
        self.set_loss(self.config.loss.name, self.config.loss.parameters)

        # optimizer
        self.set_optimizer(self.config.optimizer.name, self.config.optimizer.parameters)

        self.n_epochs = 0

    def set_network(self, network_name, network_parameters):
        """
        Define the network modules, 
        Here simple encoder but this function is overwritten to include generator/discriminator.
        """
        self.network = nn.Module()
        encoder_class = encoders.get_encoder(network_name)
        self.network.encoder = encoder_class(config=network_parameters)

        # update config
        self.config.network.name = network_name
        self.config.network.parameters.update(network_parameters)

    def init_network_weights(self, weights_init_name, weights_init_parameters):
        weights_init_function = get_weights_init(weights_init_name)
        if weights_init_name == "pretrain":
            self.network = weights_init_function(self.network, weights_init_parameters.checkpoint_filepath)
        else:
            self.network.apply(weights_init_function)

        # update config
        self.config.network.weights_init.name = weights_init_name
        self.config.network.weights_init.parameters.update(weights_init_parameters)

    def set_device(self, device):
        self.to(device) #device="cuda" or "cpu"
        self.config.device = device


    def set_loss(self, loss_name, loss_parameters):
        loss_class = losses.get_loss(loss_name)
        self.loss_f = loss_class(**loss_parameters)

        # update config
        self.config.loss.name = loss_name
        self.config.loss.parameters.update(loss_parameters)

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(self.network.parameters(),
                                         **optimizer_parameters)  # the optimizer acts on all the network nn.parameters by default

        # update config
        self.config.optimizer.name = optimizer_name
        self.config.update(optimizer_parameters)

    def run_training(self, train_loader, n_epochs, valid_loader=None, training_logger=None):
        raise NotImplementedError

    def train_epoch(self, train_loader, logger=None):
        self.train()
        raise NotImplementedError

    def valid_epoch(self, valid_loader, logger=None):
        self.eval()
        raise NotImplementedError

    def save(self, filepath='representation.pickle'):
        # save current epoch weight file with optimizer if we want to relaunch training from that point
        network = {
            "epoch": self.n_epochs,
            "type": self.__class__.__name__,
            "config": self.config,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        torch.save(network, filepath)

    @staticmethod
    def load(filepath='representation.pickle', map_location='cpu'):
        saved_representation = torch.load(filepath, map_location=map_location)
        representation_type = saved_representation['type']
        representation_cls = getattr(image_representation.representations.torch_nn, representation_type)
        representation_config = saved_representation['config']
        representation_config.device = map_location
        representation = representation_cls(config=representation_config)
        representation.n_epochs = saved_representation["epoch"]
        representation.set_device(map_location)
        representation.network.load_state_dict(saved_representation['network_state_dict'])
        representation.optimizer.load_state_dict(saved_representation['optimizer_state_dict'])

        # TODO: ADD IN SUBCLASSES
        '''
        if "ProgressiveTree" in model_type:
            split_history = saved_model['split_history']

            for split_node_path, split_node_attr in split_history.items():
                model.split_node(split_node_path)
                node = model.network.get_child_node(split_node_path)
                node.boundary = split_node_attr["boundary"]
                node.feature_range = split_node_attr["feature_range"]

        model.network.load_state_dict(saved_model['network_state_dict'])

        if "GAN" in model_type:
            model.optimizer_discriminator.load_state_dict(saved_model['optimizer_discriminator_state_dict'])
            model.optimizer_generator.load_state_dict(saved_model['optimizer_generator_state_dict'])
        '''