from copy import deepcopy
import goalrepresent as gr
from goalrepresent import dnn
from goalrepresent.dnn.networks import decoders, discriminators
from itertools import chain
import numpy as np
import os
import sys
import time
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid

""" ========================================================================================================================
Base BiGAN architecture
========================================================================================================================="""


# TODO: implement FID and early stopped on it

class BiGANModel(dnn.BaseDNN):
    '''
    BiGAN Class
    '''

    @staticmethod
    def default_config():
        default_config = dnn.BaseDNN.default_config()

        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Dumoulin"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4
        default_config.network.parameters.feature_layer = 2
        default_config.network.parameters.conditional_type = "gaussian"

        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()

        # loss parameters
        default_config.loss = gr.Config()
        default_config.loss.name = "BiGAN"
        default_config.loss.parameters = gr.Config()

        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)  # calls all constructors up to BaseDNN (MRO)

        self.output_keys_list = self.network.encoder.output_keys_list + ["prob_pos", "prob_neg"]

    def set_network(self, network_name, network_parameters):
        super().set_network(network_name, network_parameters)
        # add a decoder to the network for the BiGAN
        decoder_class = decoders.get_decoder(network_name)
        self.network.decoder = decoder_class(**network_parameters)
        # add a discriminator to the network for the BiGAN
        discriminator_class = discriminators.get_discriminator(network_name)
        self.network.discriminator = discriminator_class(**network_parameters)

    def set_optimizer(self, optimizer_name, optimizer_hyperparameters):
        optimizer = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer_generator = optimizer(
            chain(self.network.encoder.parameters(), self.network.decoder.parameters()), **optimizer_hyperparameters)
        self.optimizer_discriminator = optimizer(self.network.discriminator.parameters(), **optimizer_hyperparameters)

    def forward_from_encoder(self, encoder_outputs):
        x_real = encoder_outputs["x"]
        z_real = encoder_outputs["z"]
        model_outputs = encoder_outputs

        z_fake = Variable(torch.randn_like(z_real))
        x_fake = self.network.decoder(z_fake)

        if self.training:
            noise1 = Variable(torch.zeros_like(x_real.detach()).normal_(0, 0.1 * (1000 - self.n_epochs) / 1000))
            noise2 = Variable(torch.zeros_like(x_fake.detach()).normal_(0, 0.1 * (1000 - self.n_epochs) / 1000))
            x_real = x_real + noise1
            x_fake = x_fake + noise2

        prob_pos = self.network.discriminator(x_real, z_real)
        prob_neg = self.network.discriminator(x_fake, z_fake)

        model_outputs["prob_pos"] = prob_pos
        model_outputs["prob_neg"] = prob_neg

        # we reconstruct images during validation for visual evaluations
        if not self.training:
            with torch.no_grad():
                model_outputs["recon_x"] = self.network.decoder(z_real)

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
        prob_pos = self.network.discriminator(x, z)
        recon_x = self.network.decoder(z)
        return prob_pos, recon_x

    def calc_embedding(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        x = self.push_variable_to_device(x)
        return self.network.encoder.calc_embedding(x)

    def run_training(self, train_loader, n_epochs, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
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

        for epoch in range(n_epochs):
            t0 = time.time()
            train_losses = self.train_epoch(train_loader, logger=logger)
            t1 = time.time()
            print("Epoch {}: {:.2f} secs".format(self.n_epochs, t1 - t0))

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
            loss_d = batch_losses['discriminator']
            loss_g = batch_losses['generator']

            self.optimizer_discriminator.zero_grad()
            loss_d.backward(retain_graph=True)
            self.optimizer_discriminator.step()

            self.optimizer_generator.zero_grad()
            loss_g.backward()
            self.optimizer_generator.step()

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

        record_valid_images = False
        record_embeddings = False
        if logger is not None:
            if self.n_epochs % self.config.logging.record_valid_images_every == 0:
                record_valid_images = True
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

        if record_valid_images:
            input_images = x.cpu().data
            output_images = model_outputs['recon_x'].cpu().data
            n_images = data['obs'].size()[0]
            vizu_tensor_list = [None] * (2 * n_images)
            vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
            img = make_grid(vizu_tensor_list, nrow=2, padding=0)
            logger.add_image('reconstructions', img, self.n_epochs)

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
        return deepcopy(self.network.decoder)

    def save_checkpoint(self, checkpoint_filepath):
        # save current epoch weight file with optimizer if we want to relaunch training from that point
        network = {
            "epoch": self.n_epochs,
            "type": self.__class__.__name__,
            "config": self.config,
            "network_state_dict": self.network.state_dict(),
            "optimizer_discriminator_state_dict": self.optimizer_discriminator.state_dict(),
            "optimizer_generator_state_dict": self.optimizer_generator.state_dict(),
        }

        torch.save(network, checkpoint_filepath)
