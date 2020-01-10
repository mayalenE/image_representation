from copy import deepcopy
import goalrepresent as gr
from goalrepresent import dnn, models
from goalrepresent.dnn.losses.losses import BaseLoss
from goalrepresent.dnn.networks import encoders, decoders
from goalrepresent.helper.datahelper import tensor2string
import math
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
    """
    nn module with encoder+fc network that returns z, prob_right
    """
    @staticmethod
    def default_config():
        default_config = dnn.BaseDNN.default_config()
           
        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Burgess"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64,64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4
        
        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()
        
        return default_config
    
    
    def __init__(self, depth, config = None, **kwargs):
        super().__init__()
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())
        self.depth = depth
        self.leaf = False
        
        self.set_network(self.config.network.name, self.config.network.parameters)
        
        
    def set_network(self, network_name, network_parameters):
        self.network = nn.Module()
        #encoder
        encoder_class = encoders.get_encoder(network_name)
        self.network.encoder = encoder_class(**network_parameters)
        # add a fc to the network predicting probability of going right
        self.network.fc = nn.Linear(network_parameters.n_latents, 1)
        # add beta parameter
        beta = torch.randn(1)
        self.network.beta = nn.Parameter(beta)
        # update config
        self.config.network.name = network_name
        self.config.network.parameters = gr.config.update_config(network_parameters, self.config.network.parameters)
        
        
    def forward(self, x):
        mu, logvar = self.network.encoder(x)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        prob_right = torch.sigmoid(self.network.beta*self.network.fc(z))
        return mu, logvar, z, prob_right
    
    
    def get_child_node(self, path):
        node = self
        for d in range(1, path.size(0)):
            if path[d] == 0:
                node = node.left
            else:
                node = node.right
        
        return node
   


class LeafNode(nn.Module):
    """
    nn module with network + loss which is a copy of an existing model's network and loss
    """
    @staticmethod
    def default_config():
        default_config = gr.Config()
        
        # network and loss are taken from an existing model (VAE by default)
        default_config.model = gr.Config()
        default_config.model.name = "VAE"
        default_config.model.config = models.VAEModel.default_config()
        
        return default_config
    
    
    def __init__(self, depth, config = None, **kwargs):
        super().__init__()
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())
        self.depth = depth
        self.leaf = True
        
        # we set the network and the loss by copying an existing model
        model_class = gr.BaseModel.get_model(self.config.model.name)
        model = model_class(config = self.config.model.config)
        
        self.network = nn.Module()
        self.network.encoder = model.get_encoder()
        
        if self.config.model.name == "VAE":
            decoder_class = decoders.get_decoder(model.config.network.name)
            new_network_parameters = model.config.network.parameters
            new_network_parameters.n_latents *= (self.depth + 1) 
            self.network.decoder = decoder_class(**new_network_parameters)
        else:
            raise NotImplementedError
        
        self.loss_f = model.loss_f
        
        
    def forward(self, x, prefix_z):
        mu, logvar = self.network.encoder(x)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            curr_z = eps.mul(std).add_(mu)
        else:
            curr_z = mu
            
        z = torch.cat([prefix_z, curr_z], dim = -1)
        
        if self.config.model.name == "VAE":
            recon_x = self.network.decoder(z)
        return recon_x, mu, logvar, z
    
    
    

class HierarchicalTreeModel(dnn.BaseDNN, gr.BaseModel):
    """
    dnn with tree network, loss which is based on leaf's losses, optimizer from that loss
    """
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
        
        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config
        
    
    def __init__(self, config = None, **kwargs):
        super().__init__(config=config, **kwargs)
        
    def set_network(self, network_name, network_parameters):
        depth = 0
        self.network = InnerNode(depth, self.config.inner_node) #root node that links to child nodes
        curr_depth_nodes = [self.network]
        while depth < self.config.tree.max_depth - 1:
            depth += 1
            next_depth_nodes = []
            for node in curr_depth_nodes:
                node.left = InnerNode(depth, self.config.inner_node)
                next_depth_nodes.append(node.left)
                node.right = InnerNode(depth, self.config.inner_node)
                next_depth_nodes.append(node.right)
            
            curr_depth_nodes = next_depth_nodes

        else :
            for node in curr_depth_nodes:
                node.left = LeafNode(depth+1, self.config.leaf_node)
                node.right = LeafNode(depth+1, self.config.leaf_node)
                
                
    def set_loss(self, loss_name, loss_parameters):
        self.loss_f = HierarchicalTreeLoss(tree_network = self.network, tree_depth = self.config.tree.max_depth)
        
        # update config
        self.config.loss.name = loss_name
        self.config.loss.parameters = gr.config.update_config(loss_parameters, self.config.loss.parameters)
        

    
    def forward(self, x):
        x = self.push_variable_to_device(x)
        batch_size = x.size(0)
        path_taken =  torch.zeros((batch_size, self.config.tree.max_depth+1), dtype=int)
        
        path_prob = Variable(torch.ones(batch_size, 1))
        path_prob = self.push_variable_to_device(path_prob)
        n_latent = self.network.config.network.parameters.n_latents
        n_tot_latent = int ((self.config.tree.max_depth + 1) * n_latent)
        mu = Variable(torch.empty(batch_size, n_tot_latent))
        mu = self.push_variable_to_device(mu)
        logvar = Variable(torch.empty(batch_size, n_tot_latent))
        logvar = self.push_variable_to_device(logvar)
        z = Variable(torch.empty(batch_size, n_tot_latent))
        z = self.push_variable_to_device(z)
        recon_x = Variable(torch.empty(x.size()))
        recon_x = self.push_variable_to_device(recon_x)

        depth = 0
        
        # inner nodes
        while depth < self.config.tree.max_depth:
            
            pathes = torch.unique(path_taken[:, :depth+1], dim=0)
            for path in pathes:
                curr_node = self.network.get_child_node(path)
                curr_node_x_ids = [i for i, x_path in enumerate(path_taken[:, :depth+1]) if torch.all(torch.eq(x_path, path))]
                curr_mu, curr_logvar, curr_z, curr_prob_right = curr_node.forward(x[curr_node_x_ids]) #probability of selecting right child node
                # torch bernouilli distribution allow to do "hard" samples and still be differentiable
                curr_go_right = Bernoulli(curr_prob_right).sample()
                curr_prob = curr_go_right * curr_prob_right + (1.0 - curr_go_right) * (1.0 - curr_prob_right)
                for i in range(len(curr_node_x_ids)):
                    mu[curr_node_x_ids[i], depth*n_latent : (depth+1)*n_latent] = curr_mu[i]
                    logvar[curr_node_x_ids[i], depth*n_latent : (depth+1)*n_latent] = curr_logvar[i]
                    z[curr_node_x_ids[i], depth*n_latent : (depth+1)*n_latent] = curr_z[i]
                    path_prob[curr_node_x_ids[i]] *= curr_prob[i]
                    path_taken[curr_node_x_ids[i], depth+1] = int(curr_go_right[i])

            depth += 1
        
        # leaf nodes
        pathes = torch.unique(path_taken, dim=0)
        for path in pathes:
            curr_node = self.network.get_child_node(path)
            curr_node_x_ids = [i for i, x_path in enumerate(path_taken) if torch.all(torch.eq(x_path, path))]
            if curr_node.config.model.name == "VAE":
                curr_recon_x, curr_mu, curr_logvar, curr_z = curr_node.forward(x[curr_node_x_ids], z[curr_node_x_ids, :-n_latent])
            else:
                raise NotImplementedError
            for i in range(len(curr_node_x_ids)):
                mu[curr_node_x_ids[i], -n_latent:] = curr_mu[i]
                logvar[curr_node_x_ids[i], -n_latent:] = curr_logvar[i]
                z[curr_node_x_ids[i], :] = curr_z[i]
                recon_x[curr_node_x_ids[i]] = curr_recon_x[i]
                
        return recon_x, mu, logvar, z, path_taken, path_prob
    
    
    def calc_embedding(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        recon_x, mu, logvar, z, path_taken, path_prob = self.forward(x)
        return mu
    
    
    def run_training (self, train_loader, n_epochs, valid_loader = None, logger=None):
        """
        logger: tensorboard X summary writer
        """            
        # Save the graph in the logger
        if logger is not None:
            root_network_config = self.config.inner_node.network.parameters
            dummy_input = torch.FloatTensor(10, root_network_config.n_channels, root_network_config.input_size[0], root_network_config.input_size[1]).uniform_(0,1)
            dummy_input = self.push_variable_to_device(dummy_input)
            try:
                logger.add_graph(self, dummy_input, verbose = False)
            except:
                pass
            
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True
        
        for epoch in range(n_epochs):
            t0 = time.time()
            train_losses = self.train_epoch (train_loader, logger=logger)
            t1 = time.time()
            
            if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                for k, v in train_losses.items():
                    logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1-t0), self.n_epochs)
            
            if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))
            
            if do_validation:
                t2 = time.time()
                valid_losses = self.valid_epoch (valid_loader, logger=logger)
                t3 = time.time()
                if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in valid_losses.items():
                        logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3-t2), self.n_epochs)
                
                valid_loss = valid_losses['total']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))
                    
                    
    def train_epoch (self, train_loader, logger=None):
        self.train()
        losses = {}
        for data in train_loader:
            x =  Variable(data['obs'])
            x = self.push_variable_to_device(x)
            y = Variable(data['label']).squeeze()
            y = self.push_variable_to_device(y)
            # forward
            recon_x, mu, logvar, z, path_taken, path_prob = self.forward(x)
            loss_inputs = {'recon_x': recon_x, 'mu': mu, 'logvar': logvar}
            loss_targets = {'x': x}
            batch_losses = self.loss_f(loss_inputs, loss_targets, path_taken, path_prob, logger=logger)
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
            losses [k] = np.mean (v)
        
        self.n_epochs += 1
        
        return losses
    
    
    def valid_epoch (self, valid_loader, logger=None):
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
                x =  Variable(data['obs'])
                x = self.push_variable_to_device(x)
                y = Variable(data['label']).squeeze()
                y = self.push_variable_to_device(y)
                # forward
                recon_x, mu, logvar, z, path_taken, path_prob = self.forward(x)
                loss_inputs = {'recon_x': recon_x, 'mu': mu, 'logvar': logvar}
                loss_targets = {'x': x}
                batch_losses = self.loss_f(loss_inputs, loss_targets, path_taken, path_prob, logger=logger)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())
                # record embeddings
                if record_embeddings:
                     embedding_samples.append(mu)
                     embedding_metadata.append(data['label'])
                     embedding_images.append(x)
                    
        for k, v in losses.items():
            losses [k] = np.mean (v)
            
        if record_valid_images:
            input_images = x.cpu().data
            output_images = recon_x.cpu().data
            if self.config.loss.parameters.reconstruction_dist == "bernouilli":
                output_images = torch.sigmoid(recon_x).cpu().data
            n_images = data['obs'].size()[0]
            vizu_tensor_list = [None] * (2*n_images)
            vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
            img = make_grid(vizu_tensor_list, nrow = 2, padding=0)
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
        return HierarchicalTreeEncoder(self.network, self.network.config.network.parameters.n_latents, self.config.tree.max_depth)
    
    def get_decoder(self):
        return HierarchicalTreeDecoder(self.network)
    
    
"""
PLUGGINS FOR SOFT DECISION TREE
"""
class HierarchicalTreeLoss(BaseLoss):
    def __init__(self, tree_network, tree_depth, **kwargs):
        pathes = generate_binary_pathes(tree_depth)
        self.path2loss = {}
        for path in pathes:
            node = tree_network.get_child_node(path)
            self.path2loss[tensor2string(path)]= node.loss_f
        
    def __call__(self, loss_inputs, loss_targets, path_taken, path_prob, reduction = True, logger=None, **kwargs):
        pathes = torch.unique(path_taken, dim=0)
        
        losses = {}
        batch_size = list(loss_inputs.values())[0].size(0)
        #TODO: here work with VAE because KLD (tot_mu) = sum(KLD(small mu)), change for rest
        for path in pathes:
            node_x_ids = [i for i, x_path in enumerate(path_taken) if torch.all(torch.eq(x_path, path))]
            leaf_loss_f = self.path2loss[tensor2string(path)]
            leaf_loss_inputs = {}
            leaf_loss_targets = {}
            for k, v in loss_inputs.items():
                leaf_loss_inputs[k] = v[node_x_ids]
            for k, v in loss_targets.items():
                leaf_loss_targets[k] = v[node_x_ids]
            leaf_losses = leaf_loss_f(leaf_loss_inputs, leaf_loss_targets, reduction = False, logger=logger)
            for k in leaf_losses.keys():
                #TODO: HARD OR SOFT
                leaf_losses[k] = leaf_losses[k] # path_prob[node_x_ids].squeeze() # we weight the gradient by path_prob to lower the influence of unsure results
                if k not in losses.keys():
                    losses[k] = torch.empty(batch_size)
                losses[k][node_x_ids] = leaf_losses[k]
        
        if reduction: 
            for k in losses.keys():
                losses[k] = losses[k].sum() / float(batch_size)
                    
        return losses
        

class HierarchicalTreeEncoder(encoders.BaseDNNEncoder):
    """
    input: x
    output: z, path 
    """
    def __init__(self, network, n_leaf_latents, max_depth, **kwargs):
        super().__init__(**kwargs)
        

        self.network = deepcopy(network)
        self.max_depth = max_depth
        self.n_latents = int ((self.max_depth + 1) * n_leaf_latents)
        
        # remove the decoders from the tree if present
        def remove_decoder(node):
            if hasattr(node, "decoder"):
                del node.decoder
            if hasattr(node, "left"):
                remove_decoder(node.left)
            if hasattr(node, "right"):
                remove_decoder(node.right)
        
        remove_decoder(self.network)

        
    def forward(self, x):
        x = self.push_variable_to_device(x)
        batch_size = x.size(0)
        path_taken =  torch.zeros((batch_size, self.max_depth+1), dtype=int)
        
        path_prob = Variable(torch.ones(batch_size, 1))
        path_prob = self.push_variable_to_device(path_prob)
        mu = Variable(torch.empty(batch_size, self.n_latents))
        mu = self.push_variable_to_device(mu)
        logvar = Variable(torch.empty(batch_size, self.n_latents))
        logvar = self.push_variable_to_device(logvar)
        z = Variable(torch.empty(batch_size, self.n_latents))
        z = self.push_variable_to_device(z)
        recon_x = Variable(torch.empty(x.size()))
        recon_x = self.push_variable_to_device(recon_x)

        depth = 0
        n_latent = int(self.n_latents / (self.max_depth+1))
        # inner nodes
        while depth < self.max_depth:
            
            pathes = torch.unique(path_taken[:, :depth+1], dim=0)
            for path in pathes:
                curr_node = self.network.get_child_node(path)
                curr_node_x_ids = [i for i, x_path in enumerate(path_taken[:, :depth+1]) if torch.all(torch.eq(x_path, path))]
                curr_mu, curr_logvar, curr_z, curr_prob_right = curr_node.forward(x[curr_node_x_ids]) #probability of selecting right child node
                # torch bernouilli distribution allow to do "hard" samples and still be differentiable
                curr_go_right = Bernoulli(curr_prob_right).sample()
                curr_prob = curr_go_right * curr_prob_right + (1.0 - curr_go_right) * (1.0 - curr_prob_right)
                for i in range(len(curr_node_x_ids)):
                    mu[curr_node_x_ids[i], depth*n_latent : (depth+1)*n_latent] = curr_mu[i]
                    logvar[curr_node_x_ids[i], depth*n_latent : (depth+1)*n_latent] = curr_logvar[i]
                    z[curr_node_x_ids[i], depth*n_latent : (depth+1)*n_latent] = curr_z[i]
                    path_prob[curr_node_x_ids[i]] *= curr_prob[i]
                    path_taken[curr_node_x_ids[i], depth+1] = int(curr_go_right[i])

            depth += 1
        
        # leaf nodes
        pathes = torch.unique(path_taken, dim=0)
        for path in pathes:
            curr_node = self.network.get_child_node(path)
            curr_node_x_ids = [i for i, x_path in enumerate(path_taken) if torch.all(torch.eq(x_path, path))]
            if curr_node.config.model.name == "VAE":
                curr_recon_x, curr_mu, curr_logvar, curr_z = curr_node.forward(x[curr_node_x_ids], z[curr_node_x_ids, :-n_latent])
            else:
                raise NotImplementedError
            for i in range(len(curr_node_x_ids)):
                mu[curr_node_x_ids[i], -n_latent:] = curr_mu[i]
                logvar[curr_node_x_ids[i], -n_latent:] = curr_logvar[i]
                z[curr_node_x_ids[i], :] = curr_z[i]
                
        return mu, logvar, z, path_taken, path_prob
    
    def calc_embedding(self, x):
        mu, logvar, z, path_taken, path_prob = self.forward(x)
        return mu
    
    
class HierarchicalTreeDecoder(decoders.BaseDNNDecoder):
    """
    input: z, path 
    output: recon_x
    """
    def __init__(self, network, **kwargs):
        super().__init__(**kwargs)
        
        # get the decoders from the tree if present
        def get_decoder_list(node, decoder_list):
            if hasattr(node.network, "decoder"):
                decoder_list.append(node.network.decoder)
            else:
                get_decoder_list(node.left, decoder_list)
                get_decoder_list(node.right, decoder_list)
                
            return decoder_list
            

        self.decoder_list = get_decoder_list(deepcopy(network), nn.ModuleList())
        
    def forward(self, z, path_taken):
        batch_size = z.size(0)
        image_size = (self.decoder_list[0].n_channels,
                      self.decoder_list[0].input_size[0],
                      self.decoder_list[0].input_size[1])
        recon_x = Variable(torch.empty(batch_size, image_size[0], image_size[1], image_size[2]))
        pathes = torch.unique(path_taken, dim=0)
        for path in pathes:
            curr_decoder_idx = int (tensor2string(path), 2)  # the path is given in base 2 and we convert it to integer (respect the order of decoder list)
            curr_decoder = self.decoder_list[curr_decoder_idx]
            curr_decoder_z_ids = [i for i, z_path in enumerate(path_taken) if torch.all(torch.eq(z_path, path))]
            curr_recon_x = curr_decoder.forward(z[curr_decoder_z_ids])
            for i in range(len(curr_decoder_z_ids)):
                recon_x[curr_decoder_z_ids[i]] = curr_recon_x[i]
        return recon_x
    
    def calc_embedding(self, x):
        mu, logvar, path = self.forward(x)
        return mu
    
    
def generate_binary_pathes(tree_depth):
    n_path = int(math.pow(2, tree_depth))
    pathes = torch.zeros((n_path, tree_depth+1), dtype=int)
    for i in range(n_path):
        suffix = bin(i)[2:]
        for c_idx, c in enumerate(reversed(suffix)):
            pathes[i, -(c_idx+1)] = int(c)
    return pathes
        
        