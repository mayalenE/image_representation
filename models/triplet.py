from copy import deepcopy
import goalrepresent as gr
from goalrepresent import dnn
from goalrepresent import models
from goalrepresent.helper import tensorboardhelper
import numpy as np
import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid


def get_combined_triplet_class(base_class):

    class CombinedTripletModel(TripletModel, base_class):
        @staticmethod
        def default_config():
            default_config = gr.config.update_config(TripletModel.default_config(), base_class.default_config())
            return default_config

        def __init__(self, config=None, **kwargs):
            base_class.__init__(self, config=config, **kwargs)

            if (config.load_pretrained_model) and os.path.exists(config.pretrained_model_filepath):
                model = gr.dnn.BaseDNN.load_checkpoint(config.pretrained_model_filepath)
                if hasattr(model, "config"):
                    self.config = gr.config.update_config(kwargs, self.config, model.config)
                if self.network.encoder.config.use_attention:
                    self.network.encoder.lf.load_state_dict(model.network.encoder.lf.state_dict())
                    self.network.encoder.gf.load_state_dict(model.network.encoder.gf.state_dict())
                    self.network.encoder.ef.load_state_dict(model.network.encoder.ef.state_dict())
                    self.network.decoder.load_state_dict(model.network.decoder.state_dict())
                else:
                    self.network.load_state_dict(model.network.state_dict())

        def set_network(self, network_name, network_parameters):
            base_class.set_network(self, network_name, network_parameters)
            if self.network.encoder.config.use_attention:
                self.network.fc_cast = nn.Linear(self.config.network.parameters.n_latents * 4,
                                                 self.config.network.parameters.n_latents)

        def forward_combined(self, x_ref, x_a, x_b, x_c):
            x_ref_outputs = self.forward(x_ref)
            x_a_outputs = self.forward(x_a)
            x_b_outputs = self.forward(x_b)
            x_c_outputs = self.forward(x_c)
            model_outputs = {"x_ref_outputs": x_ref_outputs, "x_a_outputs": x_a_outputs, "x_b_outputs": x_b_outputs,
                             "x_c_outputs": x_c_outputs}
            if self.loss_f.use_attention:
                att_outputs = [x_ref_outputs["af"], x_a_outputs["af"], x_b_outputs["af"], x_c_outputs["af"]]
                sum_att = torch.stack(att_outputs, dim=0).sum(dim=0)
                attention = F.softmax(self.network.fc_cast(sum_att))
                model_outputs.update({"attention": attention})
            return model_outputs

        def forward(self, *args):
            if torch._C._get_tracing_state():
                return self.forward_for_graph_tracing(*args)
            if len(args) == 1:
                x = args[0]
                return base_class.forward(self, x)
            elif len(args) == 4:
                x_ref = args[0]
                x_a = args[1]
                x_b = args[2]
                x_c = args[3]
                return self.forward_combined(x_ref, x_a, x_b, x_c)
            else:
                raise ValueError(
                    "CombinedTripletModel can take either one input image or four input images (Triplet), not {}".format(
                        len(args)))

        def forward_for_graph_tracing(self, *args):
            return base_class.forward_for_graph_tracing(self, args[0])

        def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
            return TripletModel.run_training(self, train_loader, training_config, valid_loader=valid_loader,
                                           logger=logger)

        def train_epoch(self, train_loader, logger=None):
            return TripletModel.train_epoch(self, train_loader, logger=logger)

        def valid_epoch(self, valid_loader, logger=None):
            return TripletModel.valid_epoch(self, valid_loader, logger=logger)

        def __reduce__(self):
            base_class_str = base_class.__name__[:-5]
            cur_class = eval("gr.models.{}TripletModel".format(base_class_str))
            self.__class__ = cur_class
            return (self.__class__, (self.config,))

    return CombinedTripletModel

class TripletModel(dnn.BaseDNN, gr.BaseModel):
    '''
    Base Triplet Class
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
        default_config.loss.name = "Triplet"
        default_config.loss.parameters = gr.Config()
        default_config.loss.parameters.distance = "squared_euclidean"
        default_config.loss.parameters.margin = 1.0
        default_config.loss.parameters.use_attention = True

        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config

    def __init__(self, config=None, **kwargs):
        dnn.BaseDNN.__init__(self, config=config, **kwargs)


    def set_network(self, network_name, network_parameters):
        dnn.BaseDNN.set_network(self, network_name, network_parameters)
        # add attention head
        if self.network.encoder.config.use_attention:
            self.network.fc_cast = nn.Linear(self.config.network.parameters.n_latents * 4,
                                             self.config.network.parameters.n_latents)
    def calc_embedding(self, x):
        x = self.push_variable_to_device(x)
        self.eval()
        with torch.no_grad():
            z = self.network.encoder.calc_embedding(x)
        return z

    def forward(self, x_ref, x_a, x_b, x_c):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x_ref, x_a, x_b, x_c)
        x_ref_outputs = self.network.encoder(x_ref)
        x_a_outputs = self.network.encoder(x_a)
        x_b_outputs = self.network.encoder(x_b)
        x_c_outputs = self.network.encoder(x_c)
        model_outputs = {"x_ref_outputs": x_ref_outputs, "x_a_outputs": x_a_outputs, "x_b_outputs": x_b_outputs,
                         "x_c_outputs": x_c_outputs}
        if self.loss_f.use_attention:
            att_outputs = [x_ref_outputs["af"], x_a_outputs["af"], x_b_outputs["af"], x_c_outputs["af"]]
            sum_att = torch.stack(att_outputs, dim=0).sum(dim=0)
            attention = F.softmax(self.network.fc_cast(sum_att))
            model_outputs.update({"attention": attention})
        return model_outputs

    def forward_for_graph_tracing(self, x_ref, x_a, x_b, x_c):
        z_ref, _ = self.network.encoder(x_ref)
        z_a, _ = self.network.encoder(x_ref)
        z_b, _ = self.network.encoder(x_ref)
        z_c, _ = self.network.encoder(x_ref)

        return z_ref, z_a, z_b, z_c

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

        # Save the graph in the logger
        if logger is not None:
            dummy_x_ref = torch.FloatTensor(1, self.config.network.parameters.n_channels,
                                            self.config.network.parameters.input_size[0],
                                            self.config.network.parameters.input_size[1]).uniform_(0, 1)
            dummy_x_a = dummy_x_ref
            dummy_x_b = dummy_x_ref
            dummy_x_c = dummy_x_ref
            dummy_x_ref = self.push_variable_to_device(dummy_x_ref)
            dummy_x_a = self.push_variable_to_device(dummy_x_a)
            dummy_x_b = self.push_variable_to_device(dummy_x_b)
            dummy_x_c = self.push_variable_to_device(dummy_x_c)
            self.eval()
            with torch.no_grad():
                logger.add_graph(self, (dummy_x_ref, dummy_x_a, dummy_x_b, dummy_x_c), verbose=False)

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
            if self.n_epochs in self.config.checkpoint.save_model_at_epochs:
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, "epoch_{}_weight_model.pth".format(self.n_epochs)))

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
            data_ref, data_a, data_b, data_c = data
            x_ref = self.push_variable_to_device(data_ref["obs"])
            x_a = self.push_variable_to_device(data_a["obs"])
            x_b = self.push_variable_to_device(data_b["obs"])
            x_c = self.push_variable_to_device(data_c["obs"])
            # forward
            model_outputs = self.forward(x_ref, x_a, x_b, x_c)
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

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        if logger is not None:
            if self.n_epochs % self.config.logging.record_valid_images_every == 0 and hasattr(self.network, "decoder"):
                record_valid_images = True
                images = []
                recon_images = []
            if self.n_epochs % self.config.logging.record_embeddings_every == 0:
                record_embeddings = True
                embeddings = []
                labels = []
                if not record_valid_images:
                    images = []

        with torch.no_grad():
            for data in valid_loader:
                data_ref, data_a, data_b, data_c = data
                x_ref = self.push_variable_to_device(data_ref["obs"])
                x_a = self.push_variable_to_device(data_a["obs"])
                x_b = self.push_variable_to_device(data_b["obs"])
                x_c = self.push_variable_to_device(data_c["obs"])
                # forward
                model_outputs = self.forward(x_ref, x_a, x_b, x_c)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs, reduction="none")
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis=-1)
                    else:
                        losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis=-1)])

                if record_valid_images:
                    recon_x_ref = model_outputs["x_ref_outputs"]["recon_x"]
                    recon_x_a = model_outputs["x_a_outputs"]["recon_x"]
                    recon_x_b = model_outputs["x_b_outputs"]["recon_x"]
                    recon_x_c = model_outputs["x_c_outputs"]["recon_x"]
                    images += [x_ref, x_a, x_b, x_c]
                    recon_images += [recon_x_ref, recon_x_a, recon_x_b, recon_x_c]

                if record_embeddings:
                    embeddings += [model_outputs["x_ref_outputs"]["z"], model_outputs["x_a_outputs"]["z"],
                                   model_outputs["x_b_outputs"]["z"], model_outputs["x_c_outputs"]["z"]]
                    labels += [data_ref["label"], data_a["label"], data_b["label"], data_c["label"]]
                    if not record_valid_images:
                        images += [x_ref, x_a, x_b, x_c]

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
        self.eval()
        with torch.no_grad():
            # train prediction accuracy
            train_quadruplets = train_loader.dataset.annotated_quadruplets
            train_acc = 0
            for quadruplet in train_quadruplets:
                x_ref = self.push_variable_to_device(train_loader.dataset.get_image(quadruplet[0])).unsqueeze(0)
                x_a = self.push_variable_to_device(train_loader.dataset.get_image(quadruplet[1])).unsqueeze(0)
                x_b = self.push_variable_to_device(train_loader.dataset.get_image(quadruplet[2])).unsqueeze(0)
                x_c = self.push_variable_to_device(train_loader.dataset.get_image(quadruplet[3])).unsqueeze(0)
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
                        train_acc += 1

            train_acc /= len(train_quadruplets)

            # test prediction accuracy
            test_quadruplets = test_loader.dataset.annotated_quadruplets
            test_acc = 0
            for quadruplet in test_quadruplets:
                x_ref = self.push_variable_to_device(test_loader.dataset.get_image(quadruplet[0])).unsqueeze(0)
                x_a = self.push_variable_to_device(test_loader.dataset.get_image(quadruplet[1])).unsqueeze(0)
                x_b = self.push_variable_to_device(test_loader.dataset.get_image(quadruplet[2])).unsqueeze(0)
                x_c = self.push_variable_to_device(test_loader.dataset.get_image(quadruplet[3])).unsqueeze(0)
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

    def calc_distance(self, z_a, z_b, attention=None):
        if attention is None:
            attention = torch.ones_like(z_a)
        if self.loss_f.distance == 'euclidean':
            dist = ((z_a - z_b).pow(2) * attention).sum(1).sqrt().item()
        elif self.loss_f.distance == 'cosine':
            dist = (1.0 - torch.nn.functional.cosine_similarity(z_a, z_b)).item()
        elif self.loss_f.distance == 'chebyshev':
            dist = ((z_a - z_b).abs() * attention).max(dim=1)[0].item()
        elif self.loss_f.distance == 'squared_euclidean':
            dist = ((z_a - z_b).pow(2) * attention).sum(1).item()
        elif self.loss_f.distance == 'squared_cosine':
            dist = (1.0 - torch.nn.functional.cosine_similarity(z_a, z_b)).pow(2).item()
        elif self.loss_f.distance == 'squared_chebyshev':
            dist = ((z_a - z_b).abs() * attention).max(dim=1)[0].pow(2).item()

        return dist

    def get_encoder(self):
        return deepcopy(self.network.encoder)

    def get_decoder(self):
        return None


VAETripletModel_local = get_combined_triplet_class(models.VAEModel)
VAETripletModel = type('VAETripletModel', (TripletModel, models.VAEModel), dict(VAETripletModel_local.__dict__))

BetaVAETripletModel_local = get_combined_triplet_class(models.BetaVAEModel)
BetaVAETripletModel = type('BetaVAETripletModel', (TripletModel, models.BetaVAEModel), dict(BetaVAETripletModel_local.__dict__))

AnnealedVAETripletModel_local = get_combined_triplet_class(models.AnnealedVAEModel)
AnnealedVAETripletModel = type('AnnealedVAETripletModel', (TripletModel, models.AnnealedVAEModel), dict(AnnealedVAETripletModel_local.__dict__))

BetaTCVAETripletModel_local = get_combined_triplet_class(models.BetaTCVAEModel)
BetaTCVAETripletModel = type('BetaTCVAETripletModel', (TripletModel, models.BetaTCVAEModel), dict(BetaTCVAETripletModel_local.__dict__))