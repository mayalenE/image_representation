from addict import Dict
from image_representation.utils.minkowski_nn_init import ME_weights_init_kaiming_normal
from image_representation.representations.minkowski_nn.minkowski_nn_representation import MinkowskiNNRepresentation
from image_representation.representations.minkowski_nn import decoders
from image_representation.utils.tensorboard_utils import resize_embeddings, logger_add_image_list
from image_representation.utils.minkowski_utils import ME_sparse_to_dense
import os
import sys
import time
import torch
import MinkowskiEngine as ME

""" ========================================================================================================================
Base VAE architecture
========================================================================================================================="""
class MEVAE(MinkowskiNNRepresentation):
    '''
    Base Minkowski VAE Class
    '''

    @staticmethod
    def default_config():
        default_config = MinkowskiNNRepresentation.default_config()

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

        # weights_init parameters
        default_config.network.weights_init = Dict()
        default_config.network.weights_init.name = "pytorch"
        default_config.network.weights_init.parameters = Dict()

        # loss parameters
        default_config.loss = Dict()
        default_config.loss.name = "VAE"
        default_config.loss.parameters = Dict()
        default_config.loss.parameters.reconstruction_dist = "bernoulli"

        # optimizer parameters
        default_config.optimizer = Dict()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = Dict()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config

    def __init__(self, config=None, **kwargs):
        MinkowskiNNRepresentation.__init__(self, config=config, **kwargs)  # calls all constructors up to BaseDNN (MRO)

        self.output_keys_list = self.network.encoder.output_keys_list + ["recon_x"]

    def set_network(self, network_name, network_parameters):
        MinkowskiNNRepresentation.set_network(self, network_name, network_parameters)
        # add a decoder to the network for the VAE
        decoder_class = decoders.get_decoder(network_name)
        self.network.decoder = decoder_class(config=network_parameters)

    def init_network_weights(self, weights_init_name, weights_init_parameters):
        #MinkowskiNNRepresentation.init_network_weights(self, weights_init_name, weights_init_parameters)
        # TODO: trick below to encourage sparsity following ME's examples
        self.network.encoder.apply(lambda m: ME_weights_init_kaiming_normal(m, a=0, mode="fan_out", nonlinearity="relu"))


    def set_logger(self, logger_config):
        MinkowskiNNRepresentation.set_logger(self, logger_config)
        # add_graph in the logger is impossible for sparse tensor so far

    def forward_from_encoder(self, encoder_outputs):
        z = encoder_outputs["z"]
        target_key = encoder_outputs["x"].coordinate_map_key
        decoder_outputs = self.network.decoder(z, target_key)
        encoder_outputs.update(decoder_outputs)
        return encoder_outputs

    def forward(self, x):
        encoder_outputs = self.network.encoder(x)
        return self.forward_from_encoder(encoder_outputs)


    def calc_embedding(self, x, **kwargs):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        z = self.network.encoder.calc_embedding(x)
        return z

    def run_training(self, train_loader, training_config, valid_loader=None):

        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

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
            if 'cuda' in self.config.device:
                torch.cuda.empty_cache()

            coords = data['coords'].to(self.config.device)
            feats = data['feats'].to(self.config.device).type(self.config.dtype)
            x = ME.SparseTensor(feats, coords)
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
                    losses[k] = [v.cpu().data.item()]
                else:
                    losses[k].append(v.cpu().data.item())

        for k, v in losses.items():
            losses[k] = torch.mean(torch.tensor(v))

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
                if 'cuda' in self.config.device:
                    torch.cuda.empty_cache()

                coords = data['coords'].to(self.config.device)
                feats = data['feats'].to(self.config.device).type(self.config.dtype)
                x = ME.SparseTensor(feats, coords)
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
                batch_size = len(x._batchwise_row_indices)
                if record_valid_images:
                    recon_x = model_outputs["recon_x"]
                    shape = torch.Size([batch_size, self.config.network.parameters.n_channels] + list(self.config.network.parameters.input_size))
                    min_coordinate = torch.zeros(len(self.config.network.parameters.input_size), device=self.config.device, dtype=torch.int32)
                    x = ME_sparse_to_dense(x, shape=shape, min_coordinate=min_coordinate)[0].cpu().detach()
                    recon_x = ME_sparse_to_dense(recon_x, shape=shape, min_coordinate=min_coordinate)[0].cpu().detach()
                    images.append(x)
                    recon_images.append(recon_x)
                if record_embeddings:
                    shape = torch.Size([batch_size, self.config.network.parameters.n_latents] + [1]*len(self.config.network.parameters.input_size))
                    min_coordinate = torch.zeros(len(self.config.network.parameters.input_size), device=self.config.device, dtype=torch.int32)
                    z = ME_sparse_to_dense(model_outputs["z"], shape=shape, min_coordinate=min_coordinate)[0].cpu().detach().view(batch_size, self.config.network.parameters.n_latents)
                    embeddings.append(z)
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
                output_images[output_images != 0.0] = torch.sigmoid(output_images[output_images != 0.0])
            vizu_tensor_list = [None] * (2 * n_images)
            vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
            logger_add_image_list(self.logger, vizu_tensor_list, "reconstructions",
                                  global_step=self.n_epochs, n_channels=self.config.network.parameters.n_channels,
                                  spatial_dims=len(self.config.network.parameters.input_size))


        if record_embeddings:
            if len(images.shape) == 5:
                images = images[:, :, self.config.network.parameters.input_size[0] // 2, :, :] #we take slice at middle depth only
            if (images.shape[1] != 1) or (images.shape[1] != 3):
                images = images[:, :3, ...]
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


""" ========================================================================================================================
State-of-the-art modifications of the basic VAE
========================================================================================================================="""


class MEBetaVAE(MEVAE):
    '''
    BetaVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = MEVAE.default_config()

        # loss parameters
        default_config.loss.name = "BetaVAE"
        default_config.loss.parameters.reconstruction_dist = "bernoulli"
        default_config.loss.parameters.beta = 5.0

        return default_config

    def __init__(self, config=None, **kwargs):
        MEVAE.__init__(self, config, **kwargs)


class MEAnnealedVAE(MEVAE):
    '''
    AnnealedVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = MEVAE.default_config()

        # loss parameters
        default_config.loss.name = "AnnealedVAE"
        default_config.loss.parameters.reconstruction_dist = "bernoulli"
        default_config.loss.parameters.gamma = 1000.0
        default_config.loss.parameters.c_min = 0.0
        default_config.loss.parameters.c_max = 5.0
        default_config.loss.parameters.c_change_duration = 100000

        return default_config

    def __init__(self, config=None, **kwargs):
        MEVAE.__init__(self, config, **kwargs)

class MEBetaTCVAE(MEVAE):
    '''
    BetaTCVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = MEVAE.default_config()

        # loss parameters
        default_config.loss.name = "BetaTCVAE"
        default_config.loss.parameters.reconstruction_dist = "bernoulli"
        default_config.loss.parameters.alpha = 1.0
        default_config.loss.parameters.beta = 10.0
        default_config.loss.parameters.gamma = 1.0
        default_config.loss.parameters.tc_approximate = 'mss'
        default_config.loss.parameters.dataset_size = 0

        return default_config

    def __init__(self, config=None, **kwargs):
        MEVAE.__init__(self, config, **kwargs)


    def train_epoch(self, train_loader):
        self.train()
        losses = {}

        # update dataset size
        self.loss_f.dataset_size = len(train_loader.dataset)

        for data in train_loader:
            if 'cuda' in self.config.device:
                torch.cuda.empty_cache()

            coords = data['coords'].to(self.config.device)
            feats = data['feats'].to(self.config.device).type(self.config.dtype)
            x = ME.SparseTensor(feats, coords)
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
            losses[k] = torch.mean(torch.tensor(v))

        self.n_epochs += 1

        return losses