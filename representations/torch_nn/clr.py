from copy import deepcopy
from addict import Dict
from image_representation import TorchNNRepresentation
from image_representation.utils.tensorboard_utils import resize_embeddings
import os
import sys
import time
import torch
from torch import nn

""" ========================================================================================================================
Base SimCLR architecture
========================================================================================================================="""

class ProjectionHead(nn.Module):
    """
    nn module
    """
    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        out_dim = self.config.n_latents
        self.network = nn.Sequential(
            nn.Linear(self.config.n_latents, self.config.n_latents),
            nn.ReLU(),
            nn.Linear(self.config.n_latents, out_dim),
        )
    def forward(self, z):
        proj_z = self.network(z)
        return proj_z

class SimCLR(TorchNNRepresentation):
    '''
    Base SimCLR Class
    '''

    @staticmethod
    def default_config():
        default_config = TorchNNRepresentation.default_config()

        # network parameters
        default_config.network = Dict()
        default_config.network.name = "Burgess"
        default_config.network.parameters = Dict()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4
        default_config.network.parameters.feature_layer = 2
        default_config.network.parameters.encoder_conditional_type = "gaussian"

        # initialization parameters
        default_config.network.weights_init = Dict()
        default_config.network.weights_init.name = "pytorch"
        default_config.network.weights_init.parameters = Dict()

        # loss parameters
        default_config.loss = Dict()
        default_config.loss.name = "SimCLR"
        default_config.loss.parameters = Dict()
        default_config.loss.parameters.temperature = 0.5
        default_config.loss.parameters.distance = 'cosine'

        # optimizer parameters
        default_config.optimizer = Dict()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = Dict()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config

    def __init__(self, config=None, **kwargs):
        TorchNNRepresentation.__init__(self, config=config, **kwargs)  # calls all constructors up to BaseDNN (MRO)

        self.output_keys_list = self.network.encoder.output_keys_list + ["recon_x"]

    def set_network(self, network_name, network_parameters):
        TorchNNRepresentation.set_network(self, network_name, network_parameters)
        # add a decoder to the network for the SimCLR
        self.network.projection_head = ProjectionHead(config=network_parameters)

    def forward_from_encoder(self, encoder_outputs):
        z = encoder_outputs["z"]
        proj_z = self.network.projection_head(z)
        model_outputs = encoder_outputs
        model_outputs["proj_z"] = proj_z
        return model_outputs

    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        x = x.to(self.config.device)
        encoder_outputs = self.network.encoder(x)
        return self.forward_from_encoder(encoder_outputs)

    def forward_for_graph_tracing(self, x):
        x = x.to(self.config.device)
        z, feature_map = self.network.encoder.forward_for_graph_tracing(x)
        proj_z = self.network.projection_head(z)
        return proj_z

    def calc_embedding(self, x, **kwargs):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        x = x.to(self.config.device)
        self.eval()
        with torch.no_grad():
            z = self.network.encoder.calc_embedding(x)
        return z

    def run_training(self, train_loader, training_config, valid_loader=None):

        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

        # Save the graph in the logger
        if self.logger is not None:
            dummy_input = torch.FloatTensor(1, self.config.network.parameters.n_channels,
                                            self.config.network.parameters.input_size[0],
                                            self.config.network.parameters.input_size[1]).uniform_(0, 1)
            dummy_input = dummy_input.to(self.config.device)
            self.eval()
            with torch.no_grad():
                self.logger.add_graph(self, dummy_input, verbose=False)

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        for epoch in range(training_config.n_epochs):
            t0 = time.time()
            train_losses = self.train_epoch(train_loader)
            t1 = time.time()

            if self.logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                for k, v in train_losses.items():
                    self.logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                self.logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                self.n_epochs)

            if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                self.save(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))
            if self.n_epochs in self.config.checkpoint.save_model_at_epochs:
                self.save(os.path.join(self.config.checkpoint.folder, "epoch_{}_weight_model.pth".format(self.n_epochs)))

            if do_validation:
                t2 = time.time()
                valid_losses = self.valid_epoch(valid_loader)
                t3 = time.time()
                if self.logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in valid_losses.items():
                        self.logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    self.logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                    self.n_epochs)

                valid_loss = valid_losses['total']
                if valid_loss < best_valid_loss and self.config.checkpoint.save_best_model:
                    best_valid_loss = valid_loss
                    self.save(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))

    def train_epoch(self, train_loader):
        self.train()
        losses = {}
        for data in train_loader:
            x = data['obs'].to(self.config.device)
            x.requires_grad = True
            x_aug = train_loader.dataset.get_augmented_batch(data['index'], augment=True).to(self.config.device)
            # forward
            model_outputs = self.forward(x)
            model_outputs_aug = self.forward(x_aug)
            model_outputs.update({k+'_aug': v for k, v in model_outputs_aug.items()})
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs)
            # backward
            loss = batch_losses['total']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())

        for k, v in losses.items():
            losses[k] = torch.mean(torch.tensor(v))

        self.n_epochs += 1

        return losses

    def valid_epoch(self, valid_loader):
        self.eval()
        losses = {}

        record_embeddings = False
        if self.logger is not None:
            if self.n_epochs % self.config.logging.record_embeddings_every == 0:
                record_embeddings = True
                embedding_samples = []
                embedding_metadata = []
                embedding_images = []

        with torch.no_grad():
            for data in valid_loader:
                x = data['obs'].to(self.config.device)
                x_aug = valid_loader.dataset.get_augmented_batch(data['index'], augment=True).to(self.config.device)
                # forward
                model_outputs = self.forward(x)
                model_outputs_aug = self.forward(x_aug)
                model_outputs.update({k + '_aug': v for k, v in model_outputs_aug.items()})
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())
                # record embeddings
                if record_embeddings:
                    embedding_samples.append(model_outputs["z"])
                    embedding_metadata.append(data['label'])
                    embedding_images.append(x)

        for k, v in losses.items():
            losses[k] = torch.mean(torch.tensor(v)).item()

        if record_embeddings:
            embedding_samples = torch.cat(embedding_samples)
            embedding_metadata = torch.cat(embedding_metadata)
            embedding_images = torch.cat(embedding_images)
            embedding_images = resize_embeddings(embedding_images)
            self.logger.add_embedding(
                embedding_samples,
                metadata=embedding_metadata,
                label_img=embedding_images,
                global_step=self.n_epochs)

        return losses

class TripletCLR(SimCLR):
    '''
    TripletCLR Class
    '''

    @staticmethod
    def default_config():
        default_config = SimCLR.default_config()

        # loss parameters
        default_config.loss.name = "TripletCLR"
        default_config.loss.parameters.distance = "cosine"
        default_config.loss.parameters.margin = 1.0

        return default_config

    def __init__(self, config=None, **kwargs):
        SimCLR.__init__(self, config, **kwargs)

    def set_network(self, network_name, network_parameters):
        TorchNNRepresentation.set_network(self, network_name, network_parameters)
        # no projection head here!

    def forward_from_encoder(self, encoder_outputs):
        return encoder_outputs

    def forward_for_graph_tracing(self, x):
        x = x.to(self.config.device)
        z, feature_map = self.network.encoder.forward_for_graph_tracing(x)
        return z
