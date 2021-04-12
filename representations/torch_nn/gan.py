from addict import Dict
import image_representation
from image_representation import TorchNNRepresentation
from image_representation.representations.torch_nn import decoders, discriminators
from image_representation.utils.tensorboard_utils import resize_embeddings
from itertools import chain
import os
import sys
import time
import torch
from torchvision.utils import make_grid

""" ========================================================================================================================
Base BiGAN architecture
========================================================================================================================="""
# TODO: implement FID and early stopped on it
class BiGAN(TorchNNRepresentation):
    '''
    BiGAN Class
    '''

    @staticmethod
    def default_config():
        default_config = TorchNNRepresentation.default_config()

        # network parameters
        default_config.network = Dict()
        default_config.network.name = "Dumoulin"
        default_config.network.parameters = Dict()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4
        default_config.network.parameters.feature_layer = 2
        default_config.network.parameters.encoder_conditional_type = "gaussian"

        # initialization parameters
        default_config.network.initialization = Dict()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = Dict()

        # loss parameters
        default_config.loss = Dict()
        default_config.loss.name = "BiGAN"
        default_config.loss.parameters = Dict()

        # optimizer parameters
        default_config.optimizer = Dict()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = Dict()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config

    def __init__(self, config=None, **kwargs):
        TorchNNRepresentation.__init__(self, config=config, **kwargs)  # calls all constructors up to BaseDNN (MRO)

        self.output_keys_list = self.network.encoder.output_keys_list + ["prob_pos", "prob_neg"]

    def set_network(self, network_name, network_parameters):
        TorchNNRepresentation.set_network(self, network_name, network_parameters)
        # add a decoder to the network for the BiGAN
        decoder_class = decoders.get_decoder(network_name)
        self.network.decoder = decoder_class(config=network_parameters)
        # add a discriminator to the network for the BiGAN
        discriminator_class = discriminators.get_discriminator(network_name)
        self.network.discriminator = discriminator_class(config=network_parameters)

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer_generator = optimizer_class(
            chain(self.network.encoder.parameters(), self.network.decoder.parameters()), **optimizer_parameters)
        self.optimizer_discriminator = optimizer_class(self.network.discriminator.parameters(), **optimizer_parameters)

        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters.update(optimizer_parameters)

    def forward_from_encoder(self, encoder_outputs):
        x_real = encoder_outputs["x"]
        z_real = encoder_outputs["z"]
        model_outputs = encoder_outputs

        z_fake = torch.randn_like(z_real)
        x_fake = self.network.decoder(z_fake)["recon_x"]

        if self.training and self.n_epochs < 5000:
            noise1 = torch.zeros_like(x_real.detach()).normal_(0, 0.1 * (5000 - self.n_epochs) / 5000)
            noise2 = torch.zeros_like(x_fake.detach()).normal_(0, 0.1 * (5000 - self.n_epochs) / 5000)
            x_real = x_real + noise1
            x_fake = x_fake + noise2

        prob_pos = self.network.discriminator(x_real, z_real)
        prob_neg = self.network.discriminator(x_fake, z_fake)

        model_outputs["prob_pos"] = prob_pos
        model_outputs["prob_neg"] = prob_neg

        # we reconstruct images during validation for visual evaluations
        if not self.training:
            with torch.no_grad():
                model_outputs["recon_x"] = self.network.decoder(z_real)["recon_x"].detach()

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
        prob_pos = self.network.discriminator(x, z)
        recon_x = self.network.decoder(z)
        return prob_pos, recon_x

    def calc_embedding(self, x, **kwargs):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        x = x.to(self.config.device)
        self.eval()
        with torch.no_grad():
            z = self.network.encoder.calc_embedding(x)
        return z

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
            dummy_input = dummy_input.to(self.config.device)
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
                self.save(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))
            if self.n_epochs in self.config.checkpoint.save_model_at_epochs:
                self.save(os.path.join(self.config.checkpoint.folder, "epoch_{}_weight_model.pth".format(self.n_epochs)))

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
                if valid_loss < best_valid_loss and self.config.checkpoint.save_best_model:
                    best_valid_loss = valid_loss
                    self.save(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))


    def train_epoch(self, train_loader, logger=None):
        self.train()
        losses = {}
        for data in train_loader:
            x = data['obs']
            x.requires_grad = True
            x = x.to(self.config.device)


            # forward/backward discriminator
            model_outputs = self.forward(x)
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs)
            loss_d = batch_losses['discriminator']
            self.optimizer_discriminator.zero_grad()
            loss_d.backward(retain_graph=True)
            self.optimizer_discriminator.step()

            # forward/backward generator
            # TODO: mixed forward for discriminator and generator
            model_outputs = self.forward(x)
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs)
            loss_g = batch_losses['generator']
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
            losses[k] = torch.mean(torch.tensor(v)).item()

        self.n_epochs += 1

        return losses


    def valid_epoch(self, valid_loader):
        self.eval()
        losses = {}

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        if self.logger is not None:
            if self.n_epochs % self.config.logging.record_valid_images_every == 0:
                record_valid_images = True
                images = []
                recon_images = []
            if self.n_epochs % self.config.logging.record_embeddings_every == 0:
                record_embeddings = True
                embeddings = []
                labels = []
                if images is None:
                    images = []
        with torch.no_grad():
            for data in valid_loader:
                x = data['obs']
                x = x.to(self.config.device)
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = v.detach().cpu().unsqueeze(-1)
                    else:
                        losses[k] = torch.vstack([losses[k], v.detach().cpu().unsqueeze(-1)])

                if record_valid_images:
                    recon_x = model_outputs["recon_x"]
                    images.append(x)
                    recon_images.append(recon_x)

                if record_embeddings:
                    embeddings.append(model_outputs["z"])
                    labels.append(data["label"])
                    if not record_valid_images:
                        images.append(x)

        if record_valid_images:
            recon_images = torch.cat(recon_images)
            images = torch.cat(images)
        if record_embeddings:
            embeddings = torch.cat(embeddings)
            labels = torch.cat(labels)
            if not record_valid_images:
                images = torch.cat(images)

        # log results
        if record_valid_images:
            n_images = min(len(images), 40)
            sampled_ids = torch.randperm(len(images))[:n_images]
            input_images = images[sampled_ids].detach().cpu()
            output_images = recon_images[sampled_ids].detach().cpu()
            if self.config.loss.parameters.reconstruction_dist == "bernoulli":
                output_images = torch.sigmoid(output_images)
            vizu_tensor_list = [None] * (2 * n_images)
            vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
            img = make_grid(vizu_tensor_list, nrow=2, padding=0)
            self.logger.add_image("reconstructions", img, self.n_epochs)

        if record_embeddings:
            images = resize_embeddings(images)
            self.logger.add_embedding(
                embeddings,
                metadata=labels,
                label_img=images,
                global_step=self.n_epochs)

        # average loss and return
        for k, v in losses.items():
            losses[k] = torch.mean(torch.tensor(v)).item()

        return losses

    def save(self, filepath='representation.pickle'):
        # save current epoch weight file with optimizer if we want to relaunch training from that point
        network = {
            "epoch": self.n_epochs,
            "type": self.__class__.__name__,
            "config": self.config,
            "network_state_dict": self.network.state_dict(),
            "optimizer_discriminator_state_dict": self.optimizer_discriminator.state_dict(),
            "optimizer_generator_state_dict": self.optimizer_generator.state_dict(),
        }

        torch.save(network, filepath)

    @staticmethod
    def load(filepath='representation.pickle', map_location='cpu'):
        saved_representation = torch.load(filepath, map_location=map_location)
        representation_type = saved_representation['type']
        representation_cls = getattr(image_representation, representation_type)
        representation_config = saved_representation['config']
        representation_config.device = map_location
        representation = representation_cls(config=representation_config)
        representation.n_epochs = saved_representation["epoch"]
        representation.set_device(map_location)
        representation.network.load_state_dict(saved_representation['network_state_dict'])
        representation.optimizer_discriminator.load_state_dict(saved_representation['optimizer_discriminator_state_dict'])
        representation.optimizer_generator.load_state_dict(saved_representation['optimizer_generator_state_dict'])


""" ========================================================================================================================
Modified versions of BiGAN architecture
========================================================================================================================="""


class VAEGAN(BiGAN):
    '''
    VAEGAN Class, see https://github.com/seangal/dcgan_vae_pytorch/blob/master/main.py
    '''

    @staticmethod
    def default_config():
        default_config = BiGAN.default_config()

        # loss parameters
        default_config.loss.name = "VAEGAN"
        default_config.loss.parameters.reconstruction_dist = "bernoulli"
        default_config.loss.parameters.beta = 5.0

        return default_config

    def __init__(self, config=None, **kwargs):
        BiGAN.__init__(self, config, **kwargs)

    def forward_from_encoder(self, encoder_outputs):
        x_real = encoder_outputs["x"]
        z_real = encoder_outputs["z"]
        model_outputs = encoder_outputs

        # decoder/generator reconstruction
        decoder_outputs_real = self.network.decoder(z_real)
        model_outputs.update(decoder_outputs_real)

        # discriminator probabilities

        z_fake = torch.randn_like(z_real, requires_grad=True)
        x_fake = self.network.decoder(z_fake)["recon_x"]

        if self.training and self.n_epochs < 5000:
            noise1 = torch.zeros_like(x_real.detach(), requires_grad=True).normal_(0, 0.1 * (5000 - self.n_epochs) / 5000)
            noise2 = torch.zeros_like(x_fake.detach(), requires_grad=True).normal_(0, 0.1 * (5000 - self.n_epochs) / 5000)
            x_real = x_real + noise1
            x_fake = x_fake + noise2

        prob_pos = self.network.discriminator(x_real, z_real)
        prob_neg = self.network.discriminator(x_fake, z_fake)

        model_outputs["prob_pos"] = prob_pos
        model_outputs["prob_neg"] = prob_neg

        return model_outputs

    def train_epoch(self, train_loader):
        self.train()
        losses = {}
        for data in train_loader:
            x = data['obs']
            x.requires_grad = True
            x = x.to(self.config.device)
            ## forward
            model_outputs = self.forward(x)
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs)

            # (1) Train the discriminator
            loss_d = batch_losses['discriminator']
            self.optimizer_discriminator.zero_grad()
            loss_d.backward(retain_graph=True)
            self.optimizer_discriminator.step()

            # (2) Train the generator
            ## (2) (a) with reconstruction loss
            loss_vae = batch_losses['vae']
            self.optimizer_generator.zero_grad()
            loss_vae.backward()
            self.optimizer_generator.step()

            ## (2) (b) with generator loss
            ### redo forward as graph has been freed
            model_outputs = self.forward(x)
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs)
            ### backward
            loss_g = batch_losses['generator']
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
            losses[k] = torch.mean(torch.tensor(v))

        self.n_epochs += 1

        return losses


'''
def adjust_learning_rate(optimizer, decay):
for param_group in optimizer.param_groups:
    param_group['lr'] *= decay
'''
