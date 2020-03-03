from copy import deepcopy
from itertools import chain
import goalrepresent as gr
from goalrepresent import dnn
from goalrepresent.dnn import losses
from goalrepresent.dnn.networks import decoders, discriminators
from goalrepresent.helper import mathhelper, tensorboardhelper
import numpy as np
import math
import os
import sys
import time
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import make_grid

""" ========================================================================================================================
Base VAE architecture
========================================================================================================================="""


class VAEModel(dnn.BaseDNN, gr.BaseModel):
    '''
    Base VAE Class
    '''

    @staticmethod
    def default_config():
        default_config = dnn.BaseDNN.default_config()

        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Burgess"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4
        default_config.network.parameters.feature_layer = 2
        default_config.network.parameters.encoder_conditional_type = "gaussian"

        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()

        # loss parameters
        default_config.loss = gr.Config()
        default_config.loss.name = "VAE"
        default_config.loss.parameters = gr.Config()
        default_config.loss.parameters.reconstruction_dist = "bernouilli"

        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)  # calls all constructors up to BaseDNN (MRO)

        self.output_keys_list = self.network.encoder.output_keys_list + ["recon_x"]

    def set_network(self, network_name, network_parameters):
        super().set_network(network_name, network_parameters)
        # add a decoder to the network for the VAE
        decoder_class = decoders.get_decoder(network_name)
        self.network.decoder = decoder_class(**network_parameters)

    def forward_from_encoder(self, encoder_outputs):
        decoder_outputs = self.network.decoder(encoder_outputs["z"])
        model_outputs = encoder_outputs
        model_outputs.update(decoder_outputs)
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
        recon_x = self.network.decoder(z)
        return recon_x

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
            if self.config.loss.parameters.reconstruction_dist == "bernouilli":
                output_images = torch.sigmoid(model_outputs['recon_x']).cpu().data
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
            embedding_images = tensorboardhelper.resize_embeddings(embedding_images)
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


""" ========================================================================================================================
State-of-the-art modifications of the basic VAE
========================================================================================================================="""


class BetaVAEModel(VAEModel):
    '''
    BetaVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = super().default_config()

        # hyperparameters
        default_config.hyperparameters.beta = 5.0

        return default_config

    def __init__(self, config=None, **kwargs):
        super(BetaVAEModel, self).__init__(config, **kwargs)

    def train_loss(self, outputs, inputs):
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']

        if self.use_gpu and not x.is_cuda:
            x = x.cuda()

        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()

        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()

        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()

        recon_loss = self.recon_loss(recon_x, x)
        KLD_loss, KLD_per_latent_dim, KLD_var = losses.KLD_loss(mu, logvar)
        total_loss = recon_loss + self.config.hyperparameters.beta * KLD_loss
        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}

    def valid_losses(self, outputs, inputs):
        return self.train_loss(outputs, inputs)


class AnnealedVAEModel(VAEModel):
    '''
    AnnealedVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = super().default_config()

        # hyperparameters
        default_config.hyperparameters.gamma = 1000.0
        default_config.hyperparameters.c_min = 0.0
        default_config.hyperparameters.c_max = 5.0
        default_config.hyperparameters.c_change_duration = 100000

        return default_config

    def __init__(self, config=None, **kwargs):
        super(AnnealedVAEModel, self).__init__(config, **kwargs)

        self.n_iters = 0

    def train_loss(self, outputs, inputs):
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']

        if self.use_gpu and not x.is_cuda:
            x = x.cuda()

        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()

        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()

        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()

        recon_loss = self.recon_loss(recon_x, x)
        KLD_loss, KLD_per_latent_dim, KLD_var = losses.KLD_loss(mu, logvar)
        total_loss = recon_loss + self.config.hyperparameters.gamma * (KLD_loss - self.C).abs()
        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}

    def valid_losses(self, outputs, inputs):
        return self.train_loss(outputs, inputs)

    def update_encoding_capacity(self):
        if self.n_iters > self.config.hyperparameters.c_change_duration:
            self.C = self.config.hyperparameters.c_max
        else:
            self.C = min(self.config.hyperparameters.c_min + (
                    self.config.hyperparameters.c_max - self.config.hyperparameters.c_min) * self.n_iters / self.config.hyperparameters.c_change_duration,
                         self.config.hyperparameters.c_max)

    def train_epoch(self, train_loader):
        self.train()
        losses = {}
        for data in train_loader:
            input_img = Variable(data['image'])
            # update capacity
            self.n_iters += 1
            self.update_encoding_capacity()
            # forward
            outputs = self.forward(input_img)
            batch_losses = self.train_loss(outputs, data)
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


class BetaTCVAEModel(VAEModel):
    '''
    $\beta$-TCVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = super().default_config()

        # hyperparameters
        default_config.hyperparameters.alpha = 1.0
        default_config.hyperparameters.beta = 10.0
        default_config.hyperparameters.gamma = 1.0
        default_config.hyperparameters.tc_approximate = 'mss'

        return default_config

    def __init__(self, dataset_size=0, config=None, **kwargs):
        super(BetaTCVAEModel, self).__init__(config, **kwargs)

        self.dataset_size = dataset_size

    def train_loss(self, outputs, inputs):
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        sampled_z = outputs['sampled_z']

        if self.use_gpu and not x.is_cuda:
            x = x.cuda()

        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()

        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()

        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()

        if self.use_gpu and not sampled_z.is_cuda:
            sampled_z = sampled_z.cuda()

        # RECON LOSS
        recon_loss = self.recon_loss(recon_x, x)

        # KL LOSS MODIFIED
        ## calculate log q(z|x) (log density of gaussian(mu,sigma2))
        log_q_zCx = (-0.5 * (math.log(2.0 * np.pi) + logvar) - (sampled_z - mu).pow(2) / (2 * logvar.exp())).sum(
            1)  # sum on the latent dimensions (factorized distribution so log of prod is sum of logs)

        ## calculate log p(z) (log density of gaussian(0,1))
        log_pz = (-0.5 * math.log(2.0 * np.pi) - sampled_z.pow(2) / 2).sum(1)

        ## calculate log_qz ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m)) and log_prod_qzi
        batch_size = sampled_z.size(0)
        _logqz = -0.5 * (math.log(2.0 * np.pi) + logvar.view(1, batch_size, self.n_latents)) - (
                sampled_z.view(batch_size, 1, self.n_latents) - mu.view(1, batch_size, self.n_latents)).pow(2) / (
                         2 * logvar.view(1, batch_size, self.n_latents).exp())
        if self.tc_approximate == 'mws':
            # minibatch weighted sampling
            log_prod_qzi = (mathhelper.logsumexp(_logqz, dim=1, keepdim=False) - math.log(
                batch_size * self.dataset_size)).sum(1)
            log_qz = (mathhelper.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(
                batch_size * self.dataset_size))
        elif self.tc_approximate == 'mss':
            # minibatch stratified sampling
            N = self.dataset_size
            M = batch_size - 1
            strat_weight = (N - M) / (N * M)
            W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
            W.view(-1)[::M + 1] = 1 / N
            W.view(-1)[1::M + 1] = strat_weight
            W[M - 1, 0] = strat_weight
            logiw_matrix = Variable(W.log().type_as(_logqz.data))
            log_qz = mathhelper.logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            log_prod_qzi = mathhelper.logsumexp(logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1,
                                                keepdim=False).sum(1)
        else:
            raise ValueError(
                'The minibatch approximation of the total correlation "{}" is not defined'.format(self.tc_approximate))

        ## I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        ## TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        ## dw_kl_loss is KL[q(z)||p(z)] (dimension-wise KL term)
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        # TOTAL LOSS
        total_loss = recon_loss + self.config.hyperparameters.alpha * mi_loss + self.config.hyperparameters.beta * tc_loss + self.config.hyperparameters.gamma * dw_kl_loss
        return {'total': total_loss, 'recon': recon_loss, 'mi': mi_loss, 'tc': tc_loss, 'dw_kl': dw_kl_loss}

    def valid_losses(self, outputs, inputs):
        return self.train_loss(outputs, inputs)

    def train_epoch(self, train_loader):
        self.train()
        losses = {}
        self.dataset_size = len(train_loader.dataset)

        for data in train_loader:
            input_img = Variable(data['image'])
            # forward
            outputs = self.forward(input_img)
            batch_losses = self.train_loss(outputs, data)
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
            # break

        for k, v in losses.items():
            losses[k] = np.mean(v)

        self.n_epochs += 1
        return losses


class VAEGANModel(VAEModel):
    # https://github.com/seangal/dcgan_vae_pytorch/blob/master/main.py
    '''
    def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
    '''
    '''
    VAEGAN Class
    '''

    def __init__(self, model_architecture="Radford", **kwargs):
        super(VAEGANModel, self).__init__(model_architecture=model_architecture, **kwargs)
        discriminator = discriminators.get_discriminator(model_architecture)
        self.discriminator = discriminator(self.n_channels, self.input_size, self.n_conv_layers, output_size=1)

    def discriminate(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        return self.discriminator(x)

    def set_optimizer(self, optimizer_name, optimizer_hyperparameters):
        optimizer = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer_generator = optimizer(
            chain(self.network.encoder.parameters(), self.network.decoder.parameters()), **optimizer_hyperparameters)
        self.optimizer_discriminator = optimizer(self.discriminator.parameters(), **optimizer_hyperparameters)
        self.criterion = nn.BCEWithLogitsLoss()

    def valid_losses(self, outputs, inputs):
        return self.train_loss(outputs, inputs)

    def train_epoch(self, train_loader):
        self.train()
        losses = {}
        for data in train_loader:
            x_real = Variable(data['image'], requires_grad=True)
            outputs_real = self.forward(x_real)

            z_fake = Variable(torch.randn(x_real.size(0), self.n_latents))
            x_fake = self.decode(z_fake)

            real_label = Variable(torch.ones(x_real.size(0)))
            fake_label = Variable(torch.zeros(x_real.size(0)))

            # (1) Train the discriminator
            output_prob_real = self.discriminate(x_real)
            output_prob_fake = self.discriminate(x_fake.detach())

            if self.use_gpu:
                output_prob_real = output_prob_real.cuda()
                output_prob_fake = output_prob_fake.cuda()
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()

            loss_discriminator = (self.criterion(output_prob_real, real_label) + self.criterion(output_prob_fake,
                                                                                                fake_label)) / 2.0
            self.optimizer_discriminator.zero_grad()
            loss_discriminator.backward()
            self.optimizer_discriminator.step()

            # (2) Train the generator
            self.optimizer_generator.zero_grad()
            ## (2.A) With VAE loss
            vae_losses = self.train_loss(outputs_real, data)
            loss_vae = vae_losses['total']
            loss_vae.backward()
            self.optimizer_generator.step()

            ## (2.A) With Discriminator loss
            outputs_real = self.forward(x_real)  # redo with trained generator as it has been freed from memory
            recon_x = outputs_real['recon_x']
            output_prob_fake = self.discriminate(recon_x)
            if self.use_gpu:
                output_prob_fake = output_prob_fake.cuda()
            loss_generator = self.criterion(output_prob_fake, real_label)
            self.optimizer_generator.zero_grad()
            loss_generator.backward()
            self.optimizer_generator.step()
            #            for name, param in self.network.decoder.named_parameters():
            #                if param.requires_grad:
            #                    print(name, '{:0.2f}'.format(param.data.sum().item()))

            # save losses
            final_losses = {'discriminator': loss_discriminator, 'vae': loss_vae, 'generator': loss_generator,
                            'total': loss_discriminator + loss_vae + loss_generator}
            for k, v in final_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())
            # debug only on first batch:        
            # break

        for k, v in losses.items():
            losses[k] = np.mean(v)

        self.n_epochs += 1
        return losses
