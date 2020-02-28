from copy import deepcopy
import goalrepresent as gr
from goalrepresent import dnn
import numpy as np
import os
import random
import sys
import time
import torch
from torch import nn
from torch.autograd import Variable

""" ========================================================================================================================
DIM architecture
Based on https://github.com/jenkspt/deepinfomax
========================================================================================================================="""


class GlobalDiscriminator(nn.Module):
    """
    nn module with network
    """

    @staticmethod
    def default_config():
        default_config = gr.Config()
        return default_config

    def __init__(self, n_latents=64, local_feature_shape=(256, 4, 4), config=None, **kwargs):
        super().__init__()
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        in_features = n_latents + np.product(local_feature_shape)
        hidden_dim = 512

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, feature_map):
        # Flatten local features and concat with global encoding
        x = torch.cat([feature_map.view(feature_map.size(0), -1), z], dim=-1)
        return self.network(x)


class BlockLayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        return x.permute(0, 3, 1, 2)


class LocalEncodeAndDotDiscriminator(nn.Module):
    """
    nn module with network
    """

    @staticmethod
    def default_config():
        default_config = gr.Config()
        return default_config

    def __init__(self, n_latents=64, local_feature_shape=(256, 4, 4), config=None, **kwargs):
        super().__init__()
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        hidden_dim = 2048

        lC, lH, lW = local_feature_shape

        # Global encoder
        self.G1 = nn.Sequential(
            nn.Linear(n_latents, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.G2 = nn.Sequential(
            nn.Linear(n_latents, hidden_dim),
            nn.ReLU()
        )

        # Local encoder
        self.block_layer_norm = BlockLayerNorm(hidden_dim)
        self.L1 = nn.Sequential(
            nn.Conv2d(lC, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True)
        )

        self.L2 = nn.Sequential(
            nn.Conv2d(lC, hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, z, feature_map):
        if self.training and z.size(0) == 1:
            self.eval()
            g = self.G1(z) + self.G2(z)
            l = self.block_layer_norm(self.L1(feature_map) + self.L2(feature_map))
            self.train()
        else:
            g = self.G1(z) + self.G2(z)
            l = self.block_layer_norm(self.L1(feature_map) + self.L2(feature_map))
        # broadcast over channel dimension
        g = g.view(g.size(0), g.size(1), 1, 1)
        return (g * l).sum(1)


class PriorDiscriminator(nn.Module):
    def __init__(self, n_latents):
        super().__init__()
        hidden_dim_1 = 1000
        hidden_dim_2 = 200
        self.network = nn.Sequential(
            nn.Linear(n_latents, hidden_dim_1, bias=False),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2, bias=False),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1)
        )

    def forward(self, x):
        if self.training and x.size(0) == 1:
            self.eval()
            output = self.network(x)
            self.train()
        else:
            output = self.network(x)
        return output


class DIMModel(dnn.BaseDNN, gr.BaseModel):
    '''
    DIM Class
    '''

    @staticmethod
    def default_config():
        default_config = dnn.BaseDNN.default_config()

        # number of negative samples for discriminators
        default_config.num_negative = 2

        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Hjelm"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4
        default_config.network.parameters.feature_layer = 2
        default_config.network.parameters.conditional_type = "deterministic"

        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()

        # loss parameters
        default_config.loss = gr.Config()
        default_config.loss.name = "DIM"
        default_config.loss.parameters = gr.Config()
        default_config.loss.parameters.alpha = 1.0
        default_config.loss.parameters.beta = 0.0
        default_config.loss.parameters.gamma = 1.0

        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)  # calls all constructors up to BaseDNN (MRO)

        self.output_keys_list = self.network.encoder.output_keys_list + ["global_pos", "global_neg", "local_pos",
                                                                         "local_neg", "prior_pos", "prior_neg"]

    def set_network(self, network_name, network_parameters):
        # defines the encoder
        super().set_network(network_name, network_parameters)
        n_latents = self.config.network.parameters.n_latents
        local_feature_shape = self.network.encoder.local_feature_shape
        # add a global discriminator
        self.network.global_discrim = GlobalDiscriminator(n_latents, local_feature_shape)
        # add a local discriminator
        self.network.local_discrim = LocalEncodeAndDotDiscriminator(n_latents, local_feature_shape)
        # add a prior discriminator
        self.network.prior_discrim = PriorDiscriminator(n_latents)

    def forward_from_encoder(self, encoder_outputs):
        z = encoder_outputs["z"]
        feature_map = encoder_outputs["feature_map"]

        # outputs
        model_outputs = dict()
        model_outputs = encoder_outputs
        ## postive example
        model_outputs['global_pos'] = self.network.global_discrim(z, feature_map)
        model_outputs['local_pos'] = self.network.local_discrim(z, feature_map)
        prior_sample = torch.rand_like(z)  # uniform prior on [0,1]
        model_outputs['prior_pos'] = self.network.prior_discrim(prior_sample)

        ## negative outputs
        model_outputs['prior_neg'] = self.network.prior_discrim(torch.sigmoid(z))

        batch_size = z.detach().size(0)
        indices = list(range(batch_size))
        neg_indices = []
        num_negatives = min(batch_size - 1, self.config.num_negative)
        if num_negatives > 0:
            for i in range(batch_size):
                all_cur_neg_indices = indices[i + 1:] + indices[:i]
                cur_neg_indices = random.sample(all_cur_neg_indices, num_negatives)
                neg_indices.append(cur_neg_indices)

            feature_map_prime = feature_map[neg_indices, ...]  # Shape is [batch_size, batch_size-1, C, H, W]
            model_outputs['global_neg'] = torch.stack(
                [self.network.global_discrim(z, feature_map_prime[:, i]) for i in range(num_negatives)]).mean(dim=0)
            model_outputs['local_neg'] = torch.stack(
                [self.network.local_discrim(z, feature_map_prime[:, i]) for i in range(num_negatives)]).mean(dim=0)
        else:
            # if there is no negative example we add noise to the positive feature map
            noise = torch.empty_like(feature_map).normal_(0.0, 1.0)
            feature_map_prime = feature_map + noise
            model_outputs['global_neg'] = self.network.global_discrim(z, feature_map_prime)
            model_outputs['local_neg'] = self.network.local_discrim(z, feature_map_prime)

        return model_outputs

    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        x = self.push_variable_to_device(x)
        encoder_outputs = self.network.encoder(x)
        return self.forward_from_encoder(encoder_outputs)

    def forward_for_graph_tracing(self, x):
        x = self.push_variable_to_device(x)
        z, feature_map = self.network.encoder.forward_for_graph_tracing(x)
        global_pred = self.network.global_discrim(z, feature_map)
        local_pred = self.network.local_discrim(z, feature_map)
        prior_pred = self.network.prior_discrim(torch.sigmoid(z))
        return global_pred, local_pred, prior_pred

    def calc_embedding(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        x = self.push_variable_to_device(x)
        return self.network.encoder.calc_embedding(x)

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

        # Save the graph in the logger
        if logger is not None:
            dummy_input = torch.FloatTensor(1, self.config.network.parameters.n_channels,
                                            self.config.network.parameters.input_size[0],
                                            self.config.network.parameters.input_size[1]).uniform_(0, 1)
            dummy_input = self.push_variable_to_device(dummy_input)
            self.eval()
            with torch.no_grad():
                logger.add_graph(self, dummy_input, verbose=False)

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        for epoch in range(training_config.n_epochs):
            t0 = time.time()
            train_losses = self.train_epoch(train_loader, logger=logger)
            t1 = time.time()

            if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                for k, v in train_losses.items():
                    logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                self.n_epochs)

            if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))

            if do_validation:
                t2 = time.time()
                valid_losses = self.valid_epoch(valid_loader, logger=logger)
                t3 = time.time()
                if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in valid_losses.items():
                        logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                    self.n_epochs)

                valid_loss = valid_losses['total']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))

    def train_epoch(self, train_loader, logger=None):
        self.train()
        losses = {}
        for data in train_loader:
            x = Variable(data['obs'])
            x = self.push_variable_to_device(x)
            # forward
            model_outputs = self.forward(x)
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
            losses[k] = np.mean(v)

        self.n_epochs += 1

        return losses

    def valid_epoch(self, valid_loader, logger=None):
        self.eval()
        losses = {}

        record_embeddings = False
        if logger is not None:
            if self.n_epochs % self.config.logging.record_embeddings_every == 0:
                record_embeddings = True
                embedding_samples = []
                embedding_metadata = []
                embedding_images = []

        with torch.no_grad():
            for data in valid_loader:
                x = Variable(data['obs'])
                x = self.push_variable_to_device(x)
                # forward
                model_outputs = self.forward(x)
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
            losses[k] = np.mean(v)

        if record_embeddings:
            embedding_samples = torch.cat(embedding_samples)
            embedding_metadata = torch.cat(embedding_metadata)
            embedding_images = torch.cat(embedding_images)
            logger.add_embedding(
                embedding_samples,
                metadata=embedding_metadata,
                label_img=embedding_images,
                global_step=self.n_epochs)

        return losses

    def get_encoder(self):
        return deepcopy(self.network.encoder)

    def get_decoder(self):
        return None
