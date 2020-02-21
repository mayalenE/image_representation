import goalrepresent as gr
from goalrepresent import dnn, models
from goalrepresent.dnn.losses import losses
from goalrepresent.dnn.networks import encoders, decoders
from goalrepresent.helper.nnmodulehelper import Flatten
import numpy as np
import os
import sys
import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torchvision.utils import make_grid


class InnerNode(nn.Module):

    @staticmethod
    def default_config():
        default_config = gr.Config()

        # network parameters
        default_config.n_channels = 1
        default_config.input_size = (64, 64)
        default_config.lmbda = 0.1

        return default_config

    def __init__(self, depth, config=None, **kwargs):
        super().__init__()
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        input_flatten_size = self.config.n_channels * self.config.input_size[0] * self.config.input_size[1]
        self.fc = nn.Sequential(Flatten(), nn.Linear(input_flatten_size, 1))

        beta = torch.randn(1)
        self.beta = nn.Parameter(beta)

        self.depth = depth
        self.leaf = False

    def forward(self, x):
        return torch.sigmoid(self.beta * self.fc(x))

    def get_child_node(self, path):
        node = self
        for d in range(1, len(path)):
            if path[d] == "0":
                node = node.left
            else:
                node = node.right

        return node


class LeafNode(nn.Module):
    @staticmethod
    def default_config():
        default_config = gr.Config()

        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Burgess"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4

        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        self.set_network(self.config.network.name, self.config.network.parameters)

        self.leaf = True

    def set_network(self, network_name, network_parameters):
        encoder_class = encoders.get_encoder(network_name)
        self.encoder = encoder_class(**network_parameters)
        decoder_class = decoders.get_decoder(network_name)
        self.decoder = decoder_class(**network_parameters)
        # update config
        self.config.network.name = network_name
        self.config.network.parameters = gr.config.update_config(network_parameters, self.config.network.parameters)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


class SoftDecisionTreeModel(dnn.BaseDNN, gr.BaseModel):
    @staticmethod
    def default_config():
        default_config = dnn.BaseDNN.default_config()

        # inner node config
        default_config.inner_node = InnerNode.default_config()

        # leaf node config
        default_config.leaf_node = LeafNode.default_config()

        # tree config
        default_config.tree = gr.Config()
        default_config.tree.max_depth = 2

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
        super().__init__(config=config, **kwargs)

    def set_network(self, network_name, network_parameters):
        depth = 0
        self.network = InnerNode(depth, self.config.inner_node)  # root node that links to child nodes
        curr_depth_nodes = [self.network]
        while depth < self.config.tree.max_depth - 1:
            next_depth_nodes = []
            for node in curr_depth_nodes:
                node.left = InnerNode(depth + 1, self.config.inner_node)
                next_depth_nodes.append(node.left)
                node.right = InnerNode(depth + 1, self.config.inner_node)
                next_depth_nodes.append(node.right)

            depth += 1
            curr_depth_nodes = next_depth_nodes

        else:
            for node in curr_depth_nodes:
                node.left = LeafNode(self.config.leaf_node)
                node.right = LeafNode(self.config.leaf_node)

    def set_loss(self, loss_name, loss_parameters):
        self.loss_f = SoftDecisionTreeLoss(leaf_loss_name=loss_name, leaf_loss_parameters=loss_parameters)

        # update config
        self.config.loss.name = loss_name
        self.config.loss.parameters = gr.config.update_config(loss_parameters, self.config.loss.parameters)

    def forward(self, x):
        x = self.push_variable_to_device(x)
        batch_size = x.size(0)
        curr_paths = ["0"] * batch_size
        path_prob = Variable(torch.ones(batch_size, 1))
        path_prob = self.push_variable_to_device(path_prob)

        recon_x = Variable(torch.empty(x.size()))
        mu = Variable(torch.empty(batch_size, self.config.leaf_node.network.parameters.n_latents))
        logvar = Variable(torch.empty(batch_size, self.config.leaf_node.network.parameters.n_latents))
        z = Variable(torch.empty(batch_size, self.config.leaf_node.network.parameters.n_latents))
        recon_x = self.push_variable_to_device(recon_x)
        mu = self.push_variable_to_device(mu)
        logvar = self.push_variable_to_device(logvar)
        z = self.push_variable_to_device(z)

        depth = 0

        # inner nodes
        while depth < self.config.tree.max_depth:

            pathes = sorted(set(curr_paths))
            for path in pathes:
                curr_node = self.network.get_child_node(path)
                node_x_ids = [i for i, x_path in enumerate(curr_paths) if x_path == path]
                prob_right = curr_node.forward(x[node_x_ids])  # probability of selecting right child node
                # torch bernouilli distribution allow to do "hard" samples and still be differentiable
                go_right = Bernoulli(prob_right).sample()
                prob = go_right * prob_right + (1 - go_right) * (1 - prob_right)
                for i in range(len(node_x_ids)):
                    path_prob[node_x_ids[i]] *= prob[i]
                    curr_paths[node_x_ids[i]] += str(int(go_right[i]))

            depth += 1

        # leaf nodes
        pathes = sorted(set(curr_paths))
        for path in pathes:
            curr_node = self.network.get_child_node(path)
            node_x_ids = [i for i, x_path in enumerate(curr_paths) if x_path == path]
            curr_recon_x, curr_mu, curr_logvar, curr_z = curr_node.forward(x[node_x_ids])
            for i in range(len(node_x_ids)):
                recon_x[node_x_ids[i]] = curr_recon_x[i]
                mu[node_x_ids[i]] = curr_mu[i]
                logvar[node_x_ids[i]] = curr_logvar[i]
                z[node_x_ids[i]] = curr_z[i]

        leaf_outputs = recon_x, mu, logvar, z

        return leaf_outputs, path_prob

    def calc_embedding(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        recon_x, mu, logvar, z, path_prob = self.forward(x)
        return mu

    def run_training(self, train_loader, n_epochs, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
        # Save the graph in the logger
        if logger is not None:
            dummy_input = torch.FloatTensor(1, self.config.inner_node.n_channels, self.config.inner_node.input_size[0],
                                            self.config.inner_node.input_size[1]).uniform_(0, 1)
            dummy_input = self.push_variable_to_device(dummy_input)
            try:
                logger.add_graph(self, dummy_input)
            except:
                pass

        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        for epoch in range(n_epochs):
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
            y = Variable(data['label']).squeeze()
            y = self.push_variable_to_device(y)
            # forward
            leaf_outputs, path_prob = self.forward(x)
            batch_losses = self.loss_f(x, y, leaf_outputs, path_prob, logger=logger)
            # backward
            loss = batch_losses['total']
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
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
                y = Variable(data['label']).squeeze()
                y = self.push_variable_to_device(y)
                # forward
                leaf_outputs, path_prob = self.forward(x)
                batch_losses = self.loss_f(x, y, leaf_outputs, path_prob, logger=logger)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())
                # record embeddings
                if record_embeddings:
                    embedding_samples.append(leaf_outputs[1])
                    embedding_metadata.append(data['label'])
                    embedding_images.append(x)

        for k, v in losses.items():
            losses[k] = np.mean(v)

        if record_valid_images:
            input_images = x.cpu().data
            output_images = leaf_outputs[0].cpu().data
            if self.config.loss.parameters.reconstruction_dist == "bernouilli":
                output_images = torch.sigmoid(leaf_outputs[0]).cpu().data
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
        return SoftDecisionTreeEncoder(self.network, self.config.leaf_node.network.parameters.n_latents,
                                       self.config.tree.max_depth)

    def get_decoder(self):
        return SoftDecisionTreeDecoder(self.network)


"""
PLUGGINS FOR SOFT DECISION TREE
"""


class SoftDecisionTreeLoss(losses.BaseLoss):
    def __init__(self, leaf_loss_name="VAE", leaf_loss_parameters=models.VAEModel.default_config().loss.parameters,
                 **kwargs):
        self.leaf_loss_name = leaf_loss_name
        loss_class = losses.get_loss(leaf_loss_name)
        self.loss_f = loss_class(**leaf_loss_parameters)

    def __call__(self, x, y, leaf_outputs, path_prob, reduction=True, logger=None, **kwargs):
        if self.leaf_loss_name == "VAE":
            recon_x, mu, logvar, z = leaf_outputs
            leaf_losses = self.loss_f(x, recon_x, mu, logvar, reduction=False, logger=logger)
            for k in leaf_losses.keys():
                leaf_losses[k] = leaf_losses[
                                     k] * path_prob.squeeze()  # we weight the gradient by path_prob to lower the influence of unsure results

            if reduction:
                for k in leaf_losses.keys():
                    leaf_losses[k] = leaf_losses[k].sum() / x.size()[0]

            return {'total': leaf_losses['total'], 'recon': leaf_losses['recon'], 'KLD': leaf_losses['KLD']}
        else:
            raise ValueError("Soft Decision Tree with this leaf loss is not implemented yet")


class SoftDecisionTreeEncoder(encoders.BaseDNNEncoder):
    """
    input: x
    output: z, path 
    """

    def __init__(self, root, n_latents, max_depth, **kwargs):
        super().__init__(**kwargs)

        self.root = root
        self.n_latents = n_latents
        self.max_depth = max_depth

        # remove the decoders from the tree if present
        def remove_decoder(node):
            if hasattr(node, "decoder"):
                del node.decoder
            if hasattr(node, "left"):
                remove_decoder(node.left)
            if hasattr(node, "right"):
                remove_decoder(node.right)

        remove_decoder(self.root)

    def forward(self, x):
        batch_size = x.size(0)
        curr_paths = ["0"] * batch_size
        path_prob = Variable(torch.ones(batch_size, 1))

        mu = Variable(torch.empty(batch_size, self.n_latents))
        logvar = Variable(torch.empty(batch_size, self.n_latents))

        depth = 0

        # inner nodes
        while depth < self.max_depth:

            pathes = sorted(set(curr_paths))
            for path in pathes:
                curr_node = self.root.get_child_node(path)
                node_x_ids = [i for i, x_path in enumerate(curr_paths) if x_path == path]
                prob_right = curr_node.forward(x[node_x_ids])  # probability of selecting right child node
                # torch bernouilli distribution allow to do "hard" samples and still be differentiable
                go_right = Bernoulli(prob_right).sample()
                prob = go_right * prob_right + (1 - go_right) * (1 - prob_right)
                for i in range(len(node_x_ids)):
                    path_prob[node_x_ids[i]] *= prob[i]
                    curr_paths[node_x_ids[i]] += str(int(go_right[i]))

            depth += 1

        # leaf nodes
        pathes = sorted(set(curr_paths))
        for path in pathes:
            curr_node = self.root.get_child_node(path)
            node_x_ids = [i for i, x_path in enumerate(curr_paths) if x_path == path]
            curr_mu, curr_logvar = curr_node.encoder(x[node_x_ids])
            for i in range(len(node_x_ids)):
                mu[node_x_ids[i]] = curr_mu[i]
                logvar[node_x_ids[i]] = curr_logvar[i]

        return mu, logvar, curr_paths

    def calc_embedding(self, x):
        mu, logvar, path = self.forward(x)
        return mu


class SoftDecisionTreeDecoder(decoders.BaseDNNDecoder):
    """
    input: z, path 
    output: recon_x
    """

    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)

        # get the decoders from the tree if present
        def get_decoder_list(node):
            decoder_list = nn.ModuleList()
            if hasattr(node, "decoder"):
                decoder_list.append(node.decoder)

        self.decoder_list = get_decoder_list(root)

    def forward(self, z, path):
        decoder_idx = int(path,
                          2)  # the path is given in base 2 and we convert it to integer (respect the order of decoder list)
        recon_x = self.decoder_list[decoder_idx].forward(z)
        return recon_x

    def calc_embedding(self, x):
        mu, logvar, path = self.forward(x)
        return mu
