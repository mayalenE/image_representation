from image_representation import TorchNNRepresentation, VAE, BetaVAE, AnnealedVAE, BetaTCVAE
from image_representation.utils.tensorboard_utils import resize_embeddings, logger_add_image_list
import numpy as np
import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F


class QuadrupletNet(nn.Module):

    def forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        x_pos_a_outputs = self.forward(x_pos_a)
        x_pos_b_outputs = self.forward(x_pos_b)
        x_neg_a_outputs = self.forward(x_neg_a)
        x_neg_b_outputs = self.forward(x_neg_b)
        model_outputs = {"x_pos_a_outputs": x_pos_a_outputs, "x_pos_b_outputs": x_pos_b_outputs, "x_neg_a_outputs": x_neg_a_outputs,
                         "x_neg_b_outputs": x_neg_b_outputs}

        if self.loss_f.use_attention:
            att_outputs = [x_pos_a_outputs["af"], x_pos_b_outputs["af"], x_neg_a_outputs["af"], x_neg_b_outputs["af"]]
            sum_att = torch.stack(att_outputs, dim=0).sum(dim=0)
            attention = F.softmax(self.network.fc_cast(sum_att))
            model_outputs.update({"attention": attention})
        return model_outputs

    def forward_for_graph_tracing(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b):
        z_pos_a = self.calc_embedding(x_pos_a)
        z_pos_b = self.calc_embedding(x_pos_b)
        z_neg_a = self.calc_embedding(x_neg_a)
        z_neg_b = self.calc_embedding(x_neg_b)

        return z_pos_a, z_pos_b, z_neg_a, z_neg_b

    def run_training(self, train_loader, training_config, valid_loader=None):

        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

        # Save the graph in the logger
        if self.logger is not None:
            dummy_size = (1, self.config.network.parameters.n_channels,) + self.config.network.parameters.input_size
            dummy_input = torch.FloatTensor(size=dummy_size).uniform_(0, 1)
            dummy_input = dummy_input.to(self.config.device)
            self.eval()
            with torch.no_grad():
                self.logger.add_graph(self, dummy_input, verbose=False)

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        for epoch in range(training_config.n_epochs):
            # start with evaluation/test epoch
            if (self.config.evaluation.folder is not None) and (self.n_epochs % self.config.evaluation.save_results_every == 0):
                train_acc, test_acc = self.evaluation_epoch(train_loader, valid_loader)
                if self.logger is not None:
                    self.logger.add_scalars('pred_acc', {'train': train_acc}, self.n_epochs)
                    self.logger.add_scalars('pred_acc', {'test': test_acc}, self.n_epochs)

            # train epoch
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

            # validation epoch
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
            data_pos_a, data_pos_b, data_neg_a, data_neg_b = data
            x_pos_a = data_pos_a["obs"].to(self.config.device)
            x_pos_b = data_pos_b["obs"].to(self.config.device)
            x_neg_a = data_neg_a["obs"].to(self.config.device)
            x_neg_b = data_neg_b["obs"].to(self.config.device)
            # forward
            model_outputs = self.forward(x_pos_a, x_pos_b, x_neg_a, x_neg_b)
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
                data_pos_a, data_pos_b, data_neg_a, data_neg_b = data
                x_pos_a = data_pos_a["obs"].to(self.config.device)
                x_pos_b = data_pos_b["obs"].to(self.config.device)
                x_neg_a = data_neg_a["obs"].to(self.config.device)
                x_neg_b = data_neg_b["obs"].to(self.config.device)
                # forward
                model_outputs = self.forward(x_pos_a, x_pos_b, x_neg_a, x_neg_b)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs, reduction="none")
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis=-1)
                    else:
                        losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis=-1)])

                if record_valid_images:
                    recon_x_pos_a = self.forward(x_pos_a)["recon_x"]
                    recon_x_pos_b = self.forward(x_pos_b)["recon_x"]
                    recon_x_neg_a = self.forward(x_neg_a)["recon_x"]
                    recon_x_neg_b = self.forward(x_neg_b)["recon_x"]
                    if len(images) < self.config.logging.record_embeddings_max:
                        images += [x_pos_a, x_pos_b, x_neg_a, x_neg_b]
                    if len(recon_images) < self.config.logging.record_valid_images_max:
                        recon_images += [recon_x_pos_a, recon_x_pos_b, recon_x_neg_a, recon_x_neg_b]

                if record_embeddings:
                    if len(embeddings) < self.config.logging.record_embeddings_max:
                        embeddings += [model_outputs["z_pos_a"], model_outputs["z_pos_b"], model_outputs["z_neg_a"],
                                   model_outputs["z_neg_b"]]
                        labels += [data_pos_a["label"], data_pos_b["label"], data_neg_a["label"], data_neg_b["label"]]
                    if not record_valid_images:
                        if len(images) < self.config.logging.record_embeddings_max:
                            images += [x_pos_a, x_pos_b, x_neg_a, x_neg_b]

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
            sampled_ids = np.random.choice(len(images), n_images, replace=False)
            input_images = images[sampled_ids].detach().cpu()
            output_images = recon_images[sampled_ids].detach().cpu()
            if self.config.loss.parameters.reconstruction_dist == "bernoulli":
                output_images = torch.sigmoid(output_images)
            vizu_tensor_list = [None] * (2 * n_images)
            vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
            logger_add_image_list(self.logger, vizu_tensor_list, "reconstructions",
                                  global_step=self.n_epochs, n_channels=self.config.network.parameters.n_channels,
                                  spatial_dims=len(self.config.network.parameters.input_size))

        if record_embeddings:
            if len(images.shape) == 5:
                images = images[:, :, self.config.network.parameters.input_size[0] // 2, :,
                         :]  # we take slice at middle depth only
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
            losses[k] = np.mean(v)

        return losses

    def evaluation_epoch(self, train_loader, test_loader):
        self.eval()
        with torch.no_grad():
            # train prediction accuracy
            train_quadruplets = train_loader.dataset.annotated_quadruplets
            train_acc = 0
            for quadruplet in train_quadruplets:
                x_ref = train_loader.dataset.get_image(quadruplet[0]).unsqueeze(0).to(self.config.device)
                x_a = train_loader.dataset.get_image(quadruplet[1]).unsqueeze(0).to(self.config.device)
                x_b = train_loader.dataset.get_image(quadruplet[2]).unsqueeze(0).to(self.config.device)
                x_c = train_loader.dataset.get_image(quadruplet[3]).unsqueeze(0).to(self.config.device)
                model_outputs = self.forward(x_ref, x_a, x_b, x_c)
                quadruplet_z = [model_outputs['x_ref_outputs']['z'], model_outputs['x_a_outputs']['z'], model_outputs['x_b_outputs']['z'], model_outputs['x_c_outputs']['z']]
                if "attention" in model_outputs:
                    attention = model_outputs["attention"]
                else:
                    attention = torch.ones_like(quadruplet_z[0])
                pairs = []
                # 6 possible pairs
                for i in range(4):
                    for j in range(i + 1, 4):
                        pairs.append([i, j])
                dist_per_pair = []
                for pair in pairs:
                    z_a = quadruplet_z[pair[0]]
                    z_b = quadruplet_z[pair[1]]
                    dist = self.calc_distance(z_a, z_b, attention=attention)
                    dist_per_pair.append(dist)
                if np.isnan(dist_per_pair).all() or np.isinf(dist_per_pair).all():
                    continue
                else:
                    closest_pair = np.nanargmin(dist_per_pair)
                    if closest_pair == 0:
                        train_acc += 1

            train_acc /= len(train_quadruplets)

            # test prediction accuracy
            test_quadruplets = test_loader.dataset.annotated_quadruplets
            test_acc = 0
            for quadruplet in test_quadruplets:
                x_ref = test_loader.dataset.get_image(quadruplet[0]).unsqueeze(0).to(self.config.device)
                x_a = test_loader.dataset.get_image(quadruplet[1]).unsqueeze(0).to(self.config.device)
                x_b = test_loader.dataset.get_image(quadruplet[2]).unsqueeze(0).to(self.config.device)
                x_c = test_loader.dataset.get_image(quadruplet[3]).unsqueeze(0).to(self.config.device)
                model_outputs = self.forward(x_ref, x_a, x_b, x_c)
                quadruplet_z = [model_outputs['x_ref_outputs']['z'], model_outputs['x_a_outputs']['z'],
                                model_outputs['x_b_outputs']['z'], model_outputs['x_c_outputs']['z']]
                if "attention" in model_outputs:
                    attention = model_outputs["attention"]
                else:
                    attention = torch.ones_like(quadruplet_z[0])
                pairs = []
                # 6 possible pairs
                for i in range(4):
                    for j in range(i + 1, 4):
                        pairs.append([i, j])
                dist_per_pair = []
                for pair in pairs:
                    z_a = quadruplet_z[pair[0]]
                    z_b = quadruplet_z[pair[1]]
                    dist = self.calc_distance(z_a, z_b, attention=attention)
                    dist_per_pair.append(dist)
                if np.isnan(dist_per_pair).all() or np.isinf(dist_per_pair).all():
                    continue
                else:
                    closest_pair = np.nanargmin(dist_per_pair)
                    if closest_pair == 0:
                        test_acc += 1

            test_acc /= len(test_quadruplets)

        return train_acc, test_acc


class VAEQuadruplet(VAE, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = VAE.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0
        default_config.loss.parameters.distance = 'euclidean'
        default_config.loss.combined_loss_name = None

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        VAE.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = TorchNNRepresentation.load(config.pretrained_model_filepath, map_location=self.config.device)
            if hasattr(model, "config"):
                self.config = model.config
                self.config.update(self.config)
                self.config.update(kwargs)
                if self.network.encoder.config.use_attention:
                    self.network.encoder.lf.load_state_dict(model.network.encoder.lf.state_dict())
                    self.network.encoder.gf.load_state_dict(model.network.encoder.gf.state_dict())
                    self.network.encoder.ef.load_state_dict(model.network.encoder.ef.state_dict())
                    self.network.decoder.load_state_dict(model.network.decoder.state_dict())
                else:
                    self.network.load_state_dict(model.network.state_dict())

    def set_network(self, network_name, network_parameters):
        VAE.set_network(self, network_name, network_parameters)
        if self.network.encoder.config.use_attention:
            self.network.fc_cast = nn.Linear(self.config.network.parameters.n_latents * 4,
                                             self.config.network.parameters.n_latents)


    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return VAE.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "VAEQuadruplet can take either one input image (VAE) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def run_training(self, train_loader, training_config, valid_loader=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader)

    def train_epoch(self, train_loader):
        return QuadrupletNet.train_epoch(self, train_loader)

    def valid_epoch(self, valid_loader):
        return QuadrupletNet.valid_epoch(self, valid_loader)


class BetaVAEQuadruplet(BetaVAE, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = BetaVAE.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0
        default_config.loss.parameters.distance = 'euclidean'
        default_config.loss.combined_loss_name = None

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        BetaVAE.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = TorchNNRepresentation.load(config.pretrained_model_filepath, map_location=self.config.device)
            if hasattr(model, "config"):
                self.config = model.config
                self.config.update(self.config)
                self.config.update(kwargs)
                if self.network.encoder.config.use_attention:
                    self.network.encoder.lf.load_state_dict(model.network.encoder.lf.state_dict())
                    self.network.encoder.gf.load_state_dict(model.network.encoder.gf.state_dict())
                    self.network.encoder.ef.load_state_dict(model.network.encoder.ef.state_dict())
                    self.network.decoder.load_state_dict(model.network.decoder.state_dict())
                else:
                    self.network.load_state_dict(model.network.state_dict())

    def set_network(self, network_name, network_parameters):
        BetaVAE.set_network(self, network_name, network_parameters)
        if self.network.encoder.config.use_attention:
            self.network.fc_cast = nn.Linear(self.config.network.parameters.n_latents * 4,
                                             self.config.network.parameters.n_latents)


    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return BetaVAE.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "BetaVAEQuadruplet can take either one input image (BetaVAE) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def run_training(self, train_loader, training_config, valid_loader=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader)

    def train_epoch(self, train_loader):
        return QuadrupletNet.train_epoch(self, train_loader)

    def valid_epoch(self, valid_loader):
        return QuadrupletNet.valid_epoch(self, valid_loader)


class AnnealedVAEQuadruplet(AnnealedVAE, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = AnnealedVAE.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0
        default_config.loss.parameters.distance = 'euclidean'
        default_config.loss.combined_loss_name = None

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        AnnealedVAE.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = TorchNNRepresentation.load(config.pretrained_model_filepath, map_location=self.config.device)
            if hasattr(model, "config"):
                self.config = model.config
                self.config.update(self.config)
                self.config.update(kwargs)
                if self.network.encoder.config.use_attention:
                    self.network.encoder.lf.load_state_dict(model.network.encoder.lf.state_dict())
                    self.network.encoder.gf.load_state_dict(model.network.encoder.gf.state_dict())
                    self.network.encoder.ef.load_state_dict(model.network.encoder.ef.state_dict())
                    self.network.decoder.load_state_dict(model.network.decoder.state_dict())
                else:
                    self.network.load_state_dict(model.network.state_dict())

    def set_network(self, network_name, network_parameters):
        AnnealedVAE.set_network(self, network_name, network_parameters)
        if self.network.encoder.config.use_attention:
            self.network.fc_cast = nn.Linear(self.config.network.parameters.n_latents * 4,
                                             self.config.network.parameters.n_latents)


    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return AnnealedVAE.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "AnnealedVAEQuadruplet can take either one input image (AnnealedVAE) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def run_training(self, train_loader, training_config, valid_loader=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader)

    def train_epoch(self, train_loader):
        return QuadrupletNet.train_epoch(self, train_loader)

    def valid_epoch(self, valid_loader):
        return QuadrupletNet.valid_epoch(self, valid_loader)


class BetaTCVAEQuadruplet(BetaTCVAE, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = BetaTCVAE.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0
        default_config.loss.parameters.distance = 'euclidean'
        default_config.loss.combined_loss_name = None

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        BetaTCVAE.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = TorchNNRepresentation.load(config.pretrained_model_filepath, map_location=self.config.device)
            if hasattr(model, "config"):
                self.config = model.config
                self.config.update(self.config)
                self.config.update(kwargs)
                if self.network.encoder.config.use_attention:
                    self.network.encoder.lf.load_state_dict(model.network.encoder.lf.state_dict())
                    self.network.encoder.gf.load_state_dict(model.network.encoder.gf.state_dict())
                    self.network.encoder.ef.load_state_dict(model.network.encoder.ef.state_dict())
                    self.network.decoder.load_state_dict(model.network.decoder.state_dict())
                else:
                    self.network.load_state_dict(model.network.state_dict())

    def set_network(self, network_name, network_parameters):
        BetaTCVAE.set_network(self, network_name, network_parameters)
        if self.network.encoder.config.use_attention:
            self.network.fc_cast = nn.Linear(self.config.network.parameters.n_latents * 4,
                                             self.config.network.parameters.n_latents)


    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return BetaTCVAE.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "BetaTCVAEQuadruplet can take either one input image (BetaTCVAE) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def run_training(self, train_loader, training_config, valid_loader=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader)

    def train_epoch(self, train_loader):
        return QuadrupletNet.train_epoch(self, train_loader)

    def valid_epoch(self, valid_loader):
        return QuadrupletNet.valid_epoch(self, valid_loader)