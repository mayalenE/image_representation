import goalrepresent as gr
from goalrepresent import models
from goalrepresent.helper import tensorboardhelper
from itertools import chain
import numpy as np
import os
import sys
import time
import torch
from torch import nn
from torchvision.utils import make_grid


class QuadrupletNet(nn.Module):

    def forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        z_pos_a = self.calc_embedding(x_pos_a)
        z_pos_b = self.calc_embedding(x_pos_b)
        z_neg_a = self.calc_embedding(x_neg_a)
        z_neg_b = self.calc_embedding(x_neg_b)

        model_outputs = {"z_pos_a": z_pos_a, "z_pos_b": z_pos_b, "z_neg_a": z_neg_a, "z_neg_b": z_neg_b}
        return model_outputs

    def forward_for_graph_tracing(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b):
        z_pos_a = self.calc_embedding(x_pos_a)
        z_pos_b = self.calc_embedding(x_pos_b)
        z_neg_a = self.calc_embedding(x_neg_a)
        z_neg_b = self.calc_embedding(x_neg_b)

        return z_pos_a, z_pos_b, z_neg_a,

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
            # start with evaluation/test epoch
            if self.n_epochs % self.config.evaluation.save_results_every == 0:
                train_acc, test_acc = self.evaluation_epoch(train_loader, valid_loader)
                if logger is not None:
                    logger.add_scalars('pred_acc', {'train': train_acc}, self.n_epochs)
                    logger.add_scalars('pred_acc', {'test': test_acc}, self.n_epochs)

            # train epoch
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

            # validation epoch
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
            data_pos_a, data_pos_b, data_neg_a, data_neg_b = data
            x_pos_a = self.push_variable_to_device(data_pos_a["obs"])
            x_pos_b = self.push_variable_to_device(data_pos_b["obs"])
            x_neg_a = self.push_variable_to_device(data_neg_a["obs"])
            x_neg_b = self.push_variable_to_device(data_neg_b["obs"])
            # forward
            model_outputs = self.forward(x_pos_a, x_pos_b, x_neg_a, x_neg_b)
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs)
            # backward
            loss = batch_losses['total']
            self.optimizer_encoder.zero_grad()
            loss.backward()
            self.optimizer_encoder.step()
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

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        if logger is not None:
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
                x_pos_a = self.push_variable_to_device(data_pos_a["obs"])
                x_pos_b = self.push_variable_to_device(data_pos_b["obs"])
                x_neg_a = self.push_variable_to_device(data_neg_a["obs"])
                x_neg_b = self.push_variable_to_device(data_neg_b["obs"])
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
                    images += [x_pos_a, x_pos_b, x_neg_a, x_neg_b]
                    recon_images += [recon_x_pos_a, recon_x_pos_b, recon_x_neg_a, recon_x_neg_b]

                if record_embeddings:
                    embeddings += [model_outputs["z_pos_a"], model_outputs["z_pos_b"], model_outputs["z_neg_a"],
                                   model_outputs["z_neg_b"]]
                    labels += [data_pos_a["label"], data_pos_b["label"], data_neg_a["label"], data_neg_b["label"]]
                    if not record_valid_images:
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
            img = make_grid(vizu_tensor_list, nrow=2, padding=0)
            logger.add_image("reconstructions", img, self.n_epochs)

        if record_embeddings:
            images = tensorboardhelper.resize_embeddings(images)
            logger.add_embedding(
                embeddings,
                metadata=labels,
                label_img=images,
                global_step=self.n_epochs)

        # average loss and return
        for k, v in losses.items():
            losses[k] = np.mean(v)

        return losses

    def evaluation_epoch(self, train_loader, test_loader):
        # train prediction accuracy
        train_quadruplets = train_loader.dataset.annotated_quadruplets
        train_acc = 0
        for quadruplet in train_quadruplets:
            pairs = []
            # 6 possible pairs
            for i in range(4):
                for j in range(i + 1, 4):
                    pairs.append([quadruplet[i], quadruplet[j]])
            dist_per_pair = []
            for pair in pairs:
                x_a = self.push_variable_to_device(train_loader.dataset.images[pair[0]].unsqueeze(0))
                x_b = self.push_variable_to_device(train_loader.dataset.images[pair[1]].unsqueeze(0))
                z_a = self.calc_embedding(x_a)
                z_b = self.calc_embedding(x_b)
                dist = (z_a - z_b).pow(2).sum(1).item()
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
            pairs = []
            # 6 possible pairs
            for i in range(4):
                for j in range(i + 1, 4):
                    pairs.append([quadruplet[i], quadruplet[j]])
            dist_per_pair = []
            for pair in pairs:
                x_a = self.push_variable_to_device(test_loader.dataset.images[pair[0]].unsqueeze(0))
                x_b = self.push_variable_to_device(test_loader.dataset.images[pair[1]].unsqueeze(0))
                z_a = self.calc_embedding(x_a)
                z_b = self.calc_embedding(x_b)
                dist = (z_a - z_b).pow(2).sum(1).item()
                dist_per_pair.append(dist)
            if np.isnan(dist_per_pair).all() or np.isinf(dist_per_pair).all():
                continue
            else:
                closest_pair = np.nanargmin(dist_per_pair)
                if closest_pair == 0:
                    test_acc += 1

        test_acc /= len(test_quadruplets)

        return train_acc, test_acc


class VAEQuadrupletModel(models.VAEModel, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = models.VAEModel.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        models.VAEModel.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = gr.dnn.BaseDNN.load_checkpoint(config.pretrained_model_filepath)
            if hasattr(model, "config"):
                self.config = gr.config.update_config(kwargs, self.config, model.config)
            self.network.load_state_dict(model.network.state_dict())

    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return models.VAEModel.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "VAEQuadrupletModel can take either one input image (VAE) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # vae optimizer
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(self.network.parameters(),
                                         **optimizer_parameters)

        # triplet optimizer on the encoder
        self.optimizer_encoder = optimizer_class(self.network.encoder.parameters(),
                                                 **optimizer_parameters)

        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader, logger=logger)

    def train_epoch(self, train_loader, logger=None):
        return QuadrupletNet.train_epoch(self, train_loader, logger=logger)

    def valid_epoch(self, valid_loader, logger=None):
        return QuadrupletNet.valid_epoch(self, valid_loader, logger=logger)


class BetaVAEQuadrupletModel(models.BetaVAEModel, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = models.BetaVAEModel.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        models.BetaVAEModel.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = gr.dnn.BaseDNN.load_checkpoint(config.pretrained_model_filepath)
            if hasattr(model, "config"):
                self.config = gr.config.update_config(kwargs, self.config, model.config)
            self.network.load_state_dict(model.network.state_dict())

    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return models.BetaVAEModel.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "BetaVAEQuadrupletModel can take either one input image (BetaVAE) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # vae optimizer
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(self.network.parameters(),
                                         **optimizer_parameters)

        # quadruplet optimizer on the encoder
        self.optimizer_encoder = optimizer_class(self.network.encoder.parameters(),
                                                 **optimizer_parameters)

        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader,
                                          logger=logger)

    def train_epoch(self, train_loader, logger=None):
        return QuadrupletNet.train_epoch(self, train_loader, logger=logger)

    def valid_epoch(self, valid_loader, logger=None):
        return QuadrupletNet.valid_epoch(self, valid_loader, logger=logger)


class AnnealedVAEQuadrupletModel(models.AnnealedVAEModel, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = models.AnnealedVAEModel.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        models.AnnealedVAEModel.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = gr.dnn.BaseDNN.load_checkpoint(config.pretrained_model_filepath)
            if hasattr(model, "config"):
                self.config = gr.config.update_config(kwargs, self.config, model.config)
            self.network.load_state_dict(model.network.state_dict())

    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return models.AnnealedVAEModel.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "AnnealedVAEQuadrupletModel can take either one input image (AnnealedVAE) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # vae optimizer
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(self.network.parameters(),
                                         **optimizer_parameters)

        # quadruplet optimizer on the encoder
        self.optimizer_encoder = optimizer_class(self.network.encoder.parameters(),
                                                 **optimizer_parameters)

        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader,
                                          logger=logger)

    def train_epoch(self, train_loader, logger=None):
        return QuadrupletNet.train_epoch(self, train_loader, logger=logger)

    def valid_epoch(self, valid_loader, logger=None):
        return QuadrupletNet.valid_epoch(self, valid_loader, logger=logger)


class BetaTCVAEQuadrupletModel(models.BetaTCVAEModel, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = models.BetaTCVAEModel.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        models.BetaTCVAEModel.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = gr.dnn.BaseDNN.load_checkpoint(config.pretrained_model_filepath)
            if hasattr(model, "config"):
                self.config = gr.config.update_config(kwargs, self.config, model.config)
            self.network.load_state_dict(model.network.state_dict())

    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return models.BetaTCVAEModel.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "BetaTCVAEQuadrupletModel can take either one input image (BetaTCVAE) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # vae optimizer
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(self.network.parameters(),
                                         **optimizer_parameters)

        # quadruplet optimizer on the encoder
        self.optimizer_encoder = optimizer_class(self.network.encoder.parameters(),
                                                 **optimizer_parameters)

        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader,
                                          logger=logger)

    def train_epoch(self, train_loader, logger=None):
        return QuadrupletNet.train_epoch(self, train_loader, logger=logger)

    def valid_epoch(self, valid_loader, logger=None):
        return QuadrupletNet.valid_epoch(self, valid_loader, logger=logger)


class BiGANQuadrupletModel(models.BiGANModel, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = models.BiGANModel.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        models.BiGANModel.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = gr.dnn.BaseDNN.load_checkpoint(config.pretrained_model_filepath)
            if hasattr(model, "config"):
                self.config = gr.config.update_config(kwargs, self.config, model.config)
            self.network.load_state_dict(model.network.state_dict())

    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return models.BiGANModel.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "BiGANQuadrupletModel can take either one input image (BiGAN) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # bigans optimizer
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer_generator = optimizer_class(
            chain(self.network.encoder.parameters(), self.network.decoder.parameters()), **optimizer_parameters)
        self.optimizer_discriminator = optimizer_class(self.network.discriminator.parameters(), **optimizer_parameters)

        # quadruplet optimizer on the encoder
        self.optimizer_encoder = optimizer_class(self.network.encoder.parameters(),
                                                 **optimizer_parameters)
        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader,
                                          logger=logger)

    def train_epoch(self, train_loader, logger=None):
        return QuadrupletNet.train_epoch(self, train_loader, logger=logger)

    def valid_epoch(self, valid_loader, logger=None):
        return QuadrupletNet.valid_epoch(self, valid_loader, logger=logger)


class VAEGANQuadrupletModel(models.VAEGANModel, QuadrupletNet):
    @staticmethod
    def default_config():
        default_config = models.VAEGANModel.default_config()

        # loss parameters
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters.margin = 1.0

        # load pretrained model
        default_config.load_pretrained_model = False

        return default_config

    def __init__(self, config=None, **kwargs):
        models.VAEGANModel.__init__(self, config=config, **kwargs)

        if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
            model = gr.dnn.BaseDNN.load_checkpoint(config.pretrained_model_filepath)
            if hasattr(model, "config"):
                self.config = gr.config.update_config(kwargs, self.config, model.config)
            self.network.load_state_dict(model.network.state_dict())


    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            return models.VAEGANModel.forward(self, x)
        elif len(args) == 4:
            x_pos_a = args[0]
            x_pos_b = args[1]
            x_neg_a = args[2]
            x_neg_b = args[3]
            return QuadrupletNet.forward(self, x_pos_a, x_pos_b, x_neg_a, x_neg_b)
        else:
            raise ValueError(
                "VAEGANQuadrupletModel can take either one input image (VAEGAN) or four input images (Quadruplet), not {}".format(
                    len(args)))

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # bigans optimizer
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer_generator = optimizer_class(
            chain(self.network.encoder.parameters(), self.network.decoder.parameters()), **optimizer_parameters)
        self.optimizer_discriminator = optimizer_class(self.network.discriminator.parameters(), **optimizer_parameters)

        # quadruplet optimizer on the encoder
        self.optimizer_encoder = optimizer_class(self.network.encoder.parameters(),
                                                 **optimizer_parameters)
        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        return QuadrupletNet.run_training(self, train_loader, training_config, valid_loader=valid_loader,
                                          logger=logger)

    def train_epoch(self, train_loader, logger=None):
        return QuadrupletNet.train_epoch(self, train_loader, logger=logger)

    def valid_epoch(self, valid_loader, logger=None):
        return QuadrupletNet.valid_epoch(self, valid_loader, logger=logger)
