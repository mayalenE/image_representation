import goalrepresent as gr
from goalrepresent import models
from goalrepresent.helper import tensorboardhelper
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
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

        return z_pos_a, z_pos_b, z_neg_a, z_neg_b

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

    def train_epoch(self, train_loader, logger=None):
        # train loader must have a "quadruplet" dataset
        assert "Quadruplet" in train_loader.dataset.__class__.__name__, "Quadruplet model needs a Quadruplet dataset"

        self.train()
        losses = {}
        for data in train_loader:
            x_pos_a, x_pos_b, x_neg_a, x_neg_b = data
            x_pos_a = self.push_variable_to_device(x_pos_a)
            x_pos_b = self.push_variable_to_device(x_pos_b)
            x_neg_a = self.push_variable_to_device(x_neg_a)
            x_neg_b = self.push_variable_to_device(x_neg_b)
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
                data_pos_a, data_pos_b, data_neg_a, data_neg_b = data
                x_pos_a = self.push_variable_to_device(data_pos_a["obs"])
                x_pos_b = self.push_variable_to_device(data_pos_b["obs"])
                x_neg_a = self.push_variable_to_device(data_neg_a["obs"])
                x_neg_b = self.push_variable_to_device(data_neg_b["obs"])
                # forward
                model_outputs = self.forward(x_pos_a, x_pos_b, x_neg_a, x_neg_b)
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
                    embedding_samples += [model_outputs["z_pos_a"], model_outputs["z_pos_b"], model_outputs["z_neg_a"], model_outputs["z_neg_b"]]
                    embedding_metadata += [data_pos_a["label"], data_pos_b["label"], data_neg_a["label"], data_neg_b["label"]]
                    embedding_images += [x_pos_a, x_pos_b, x_neg_a, x_neg_b]

        for k, v in losses.items():
            losses[k] = np.mean(v)

        if record_valid_images:
            input_images = torch.cat([x_pos_a.cpu().data, x_pos_b.cpu().data, x_neg_a.cpu().data, x_neg_b.cpu().data], dim=0)
            with torch.no_grad():
                recon_x_pos_a = self.forward(x_pos_a)["recon_x"]
                recon_x_pos_b = self.forward(x_pos_b)["recon_x"]
                recon_x_neg_a = self.forward(x_neg_a)["recon_x"]
                recon_x_neg_b = self.forward(x_neg_b)["recon_x"]
            output_images = torch.cat([recon_x_pos_a.cpu().data, recon_x_pos_b.cpu().data, recon_x_neg_a.cpu().data, recon_x_neg_b.cpu().data], dim=0)
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



class VAEQuadrupletModel(models.VAEModel):
    @staticmethod
    def default_config():
        default_config = models.VAEModel

        # loss parameters
        default_config.loss = gr.Config()
        default_config.loss.name = "Quadruplet"
        default_config.loss.parameters = gr.Config()
        default_config.loss.parameters.margin = 1.0
        default_config.loss.parameters.reconstruction_dist = "bernouilli"

        return default_config

    def __init__(self, vae_model, config=None, **kwargs):
        # fuse the quadruplet config with the one of the vae model
        if hasattr("vae_model", config):
            config = gr.config.update_config(kwargs, vae_model.config, config)
        # call VAE initializer with corresponding parameters (including Quadruplet loss)
        models.VAEModel.__init__(self, config=config, loss_margin = 0, **kwargs)
        # load the pretrained state dict
        self.network.load_state_dict(vae_model.state_dict())
        # uncomment to load the pretrained optimizer and number of epochs
        """
        if hasattr(vae_model, "optimizer"):
            self.optimizer.load_state_dict(vae_model.optimizer.state_dict())
        if hasattr(vae_model, "n_epochs");
            self.n_epochs = vae_model.n_epochs
        """

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
            raise ValueError("VAEQuadrupletModel can take either one input image (VAE) or four input images (Quadruplet), not {}".format(len(args)))

    def train_epoch(self, train_loader, logger=None):
        return QuadrupletNet.train_epoch(self, train_loader, logger=logger)

    def valid_epoch(self, valid_loader, logger=None):
        return QuadrupletNet.valid_epoch(self, valid_loader, logger=logger)