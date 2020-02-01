from copy import deepcopy
import goalrepresent as gr
from goalrepresent import dnn, models
from goalrepresent.dnn.losses.losses import BaseLoss
from goalrepresent.dnn.networks import encoders, decoders, discriminators
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
        default_config.network.parameters.feature_layer = 2
        default_config.network.parameters.conditional_type = "gaussian"
        
        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()
        
        return default_config
    
    
    def __init__(self, depth, lf_layers = None, config = None, **kwargs):
        super().__init__()
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())
        self.depth = depth
        self.leaf = False
        
        self.set_network(self.config.network.name, self.config.network.parameters)
        
        if lf_layers is not None:
            self.network.encoder.lf = lf_layers
        
        
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
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)
        inner_outputs = dict()
        encoder_outputs = self.network.encoder(x)
        inner_outputs = encoder_outputs
        prob_right = torch.sigmoid(self.network.beta*self.network.fc(encoder_outputs["z"]))
        inner_outputs['prob_right'] = prob_right
        return inner_outputs
    
    def forward_for_graph_tracing(self, x):
        z, feature_map = self.network.encoder(x)
        prob_right = torch.sigmoid(self.network.beta*self.network.fc(z))
        return z, prob_right
    
    
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
    Leaf Node base class
    """
    def __init__(self, depth, lf_layers = None, **kwargs):
            self.depth = depth
            self.leaf = True
            
            #share the local feature layers
            self.network.encoder.lf = lf_layers
                
    def forward(self, x, prefix_z):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x, prefix_z)
        encoder_outputs = self.network.encoder(x)
        
        cur_z = encoder_outputs["z"]
        z = torch.cat([prefix_z, cur_z], dim = -1)
        encoder_outputs["z"] = z
        
        leaf_outputs = self.forward_from_encoder(encoder_outputs)
        return leaf_outputs 
    
    
class VAELeafNode(LeafNode, models.VAEModel):
    def __init__(self, depth, lf_layers = None, config = None, **kwargs):
        super(LeafNode, self).__init__(config = config, **kwargs)
        super().__init__(depth, lf_layers = lf_layers, **kwargs)
        # modify the decoder :
        decoder_class = decoders.get_decoder(self.config.network.name)
        new_network_parameters = self.config.network.parameters
        new_network_parameters.n_latents *= (self.depth + 1) 
        self.network.decoder = decoder_class(**new_network_parameters)
        
        # update config
        self.config.network.parameters = gr.config.update_config(new_network_parameters, self.config.network.parameters)
            
    def forward_for_graph_tracing(self, x, prefix_z):
        z, feature_map = self.network.encoder.forward_for_graph_tracing(x)
        z = torch.cat([prefix_z, z], dim = -1)
        recon_x = self.network.decoder(z)
        return recon_x
            
    
class DIMLeafNode(LeafNode, models.DIMModel):
    def __init__(self, depth, lf_layers = None, config = None, **kwargs):
        super(LeafNode, self).__init__(config = config, **kwargs)
        super().__init__(depth, lf_layers = lf_layers, **kwargs)
        # modify the discriminators:
        new_network_parameters = self.config.network.parameters
        new_network_parameters.n_latents *= (self.depth + 1) 
        self.network.global_discrim = models.dim.GlobalDiscriminator(new_network_parameters.n_latents, self.network.encoder.local_feature_shape)
        self.network.local_discrim = models.dim.LocalEncodeAndDotDiscriminator(new_network_parameters.n_latents, self.network.encoder.local_feature_shape)
        self.network.prior_discrim = models.dim.PriorDiscriminator(new_network_parameters.n_latents)
        
        # update config
        self.config.network.parameters = gr.config.update_config(new_network_parameters, self.config.network.parameters)

    def forward_for_graph_tracing(self, x, prefix_z):
        z, feature_map = self.network.encoder.forward_for_graph_tracing(x)
        z = torch.cat([prefix_z, z], dim = -1)
        global_pred = self.network.global_discrim(z, feature_map)
        local_pred = self.network.local_discrim(z, feature_map)
        prior_pred = self.network.prior_discrim(torch.sigmoid(z))
        return global_pred, local_pred, prior_pred
    
    
class BiGANLeafNode(LeafNode, models.BiGANModel):
    def __init__(self, depth, lf_layers = None, config = None, **kwargs):
        super(LeafNode, self).__init__(config = config, **kwargs)
        super().__init__(depth, lf_layers = lf_layers, **kwargs)
        # modify the decoder and discriminator:
        decoder_class = decoders.get_decoder(self.config.network.name)
        discriminator_class = discriminators.get_discriminator(self.config.network.name)
        new_network_parameters = self.config.network.parameters
        new_network_parameters.n_latents *= (self.depth + 1) 
        self.network.decoder = decoder_class(**new_network_parameters)
        self.network.discriminator = discriminator_class(**new_network_parameters)
        
        # update config
        self.config.network.parameters = gr.config.update_config(new_network_parameters, self.config.network.parameters)

    def forward_for_graph_tracing(self, x, prefix_z):
        x = self.push_variable_to_device(x)
        z, feature_map = self.network.encoder.forward_for_graph_tracing(x)
        z = torch.cat([prefix_z, z], dim = -1)
        prob_pos = self.network.discriminator(x, z)
        recon_x = self.network.decoder(z)
        return prob_pos, recon_x
            
            

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
        default_config.leaf_node_classname = "VAE"
        default_config.leaf_node = eval("gr.models.{}Model.default_config()".format(default_config.leaf_node_classname))
        
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
        
        self.LeafNodeClass = eval("{}LeafNode".format(config.leaf_node_classname))
        
        super().__init__(config=config, **kwargs)
        
        
    def set_network(self, network_name, network_parameters):
        depth = 0
        self.network = InnerNode(depth, config=self.config.inner_node) #root node that links to child nodes
        lf_layers = self.network.network.encoder.lf # we gonna share the local feature layers between all rencoder
        
        cur_depth_nodes = [self.network]
        while depth < self.config.tree.max_depth - 1:
            depth += 1
            next_depth_nodes = []
            for node in cur_depth_nodes:
                node.left = InnerNode(depth, lf_layers=lf_layers, config=self.config.inner_node)
                next_depth_nodes.append(node.left)
                node.right = InnerNode(depth, lf_layers=lf_layers, config=self.config.inner_node)
                next_depth_nodes.append(node.right)
            
            cur_depth_nodes = next_depth_nodes

        else :
            for node in cur_depth_nodes:
                node.left = self.LeafNodeClass(depth+1, lf_layers=lf_layers, config=self.config.leaf_node)
                node.right = self.LeafNodeClass(depth+1, lf_layers=lf_layers, config=self.config.leaf_node)
        
        self.output_keys_list = node.left.output_keys_list + ["path_taken"] #node.left is a leaf node
                
                
    def set_loss(self, loss_name, loss_parameters):
        self.loss_f = HierarchicalTreeLoss(tree_network = self.network, tree_depth = self.config.tree.max_depth)
        
        # update config
        self.config.loss.name = loss_name
        self.config.loss.parameters = gr.config.update_config(loss_parameters, self.config.loss.parameters)
        

    
    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)
        
        x = self.push_variable_to_device(x)
        batch_size = x.detach().size(0)
        
        encoder_cond_type = self.network.network.encoder.conditional_type
        
        depth = 0
        tree_path_taken = torch.zeros((batch_size, 1)).int() # all the paths start with "O"
        tree_path_taken = self.push_variable_to_device(tree_path_taken)
        # inner nodes
        while depth < self.config.tree.max_depth:
            z = None
            x_ids = []
            pathes = torch.unique(tree_path_taken[:, :depth+1], dim=0)
            for path in pathes:
                # go to node
                cur_node = self.network.get_child_node(path)
                cur_node_x_ids = [i for i, x_path in enumerate(tree_path_taken[:, :depth+1]) if torch.all(torch.eq(x_path, path))]
                
                # forward
                cur_inner_outputs = cur_node.forward(x[cur_node_x_ids])
                    
                cur_prob_right = cur_inner_outputs["prob_right"]
                # torch bernouilli distribution allow to do "hard" samples and still be differentiable
                cur_go_right = Bernoulli(cur_prob_right).sample()
                #cur_prob = cur_go_right * cur_prob_right + (1.0 - cur_go_right) * (1.0 - cur_prob_right)
                #path_prob *= cur_prob

                if z is None:
                    z = cur_inner_outputs["z"]
                    if encoder_cond_type == "gaussian":
                        mu = cur_inner_outputs["mu"]
                        logvar = cur_inner_outputs["logvar"]
                    #prob_right = cur_inner_outputs["prob_right"]
                    path_taken = cur_go_right.int()
                else:
                    z = torch.cat([z, cur_inner_outputs["z"]])
                    if encoder_cond_type == "gaussian":
                        mu = torch.cat([mu, cur_inner_outputs["mu"]])
                        logvar = torch.cat([logvar, cur_inner_outputs["logvar"]])
                    #prob_right = torch.cat([prob_right, cur_inner_outputs["prob_right"]])
                    path_taken = torch.cat([path_taken, cur_go_right.int()])

                # save data points that went through that node
                x_ids += cur_node_x_ids
                
            # reorder points
            sort_order = tuple(np.argsort(x_ids))
            z = z[sort_order, :]
            if encoder_cond_type == "gaussian":
                mu = mu[sort_order, :]
                logvar = logvar[sort_order, :]
            #prob_right = prob_right[sort_order, :]
            path_taken = path_taken[sort_order, :]
            # append to the tree results
            if depth == 0:
                prefix_z = z
                if encoder_cond_type == "gaussian":
                    tree_mu = mu
                    tree_logvar = logvar
                #tree_prob_right = prob_right
                tree_path_taken = torch.cat([tree_path_taken, path_taken], dim=1)
            else:
                prefix_z = torch.cat([prefix_z, z], dim=1)
                if encoder_cond_type == "gaussian":
                    tree_mu = torch.cat([tree_mu, mu], dim=1)
                    tree_logvar = torch.cat([tree_logvar, logvar], dim=1)
                #tree_prob_right = torch.cat([tree_prob_right, prob_right], dim=1)
                tree_path_taken = torch.cat([tree_path_taken, path_taken], dim=1)
                
            # go to next level of the tree
            depth += 1
        
        # leaf nodes
        z = None
        x_ids = []
        pathes = torch.unique(tree_path_taken, dim=0)
        for path in pathes:
            # go to leaf
            cur_leaf = self.network.get_child_node(path)
            cur_leaf_x_ids = [i for i, x_path in enumerate(tree_path_taken) if torch.all(torch.eq(x_path, path))]
            
            # forward for that leaf
            cur_leaf_outputs = cur_leaf.forward(x[cur_leaf_x_ids], prefix_z[cur_leaf_x_ids])
            
            if z is None:
                z = cur_leaf_outputs["z"]
                if encoder_cond_type == "gaussian":
                    mu = cur_leaf_outputs["mu"]
                    logvar = cur_leaf_outputs["logvar"]
                if self.config.leaf_node_classname == "VAE":
                    recon_x = cur_leaf_outputs["recon_x"]
                elif self.config.leaf_node_classname == "DIM":
                    global_pos = cur_leaf_outputs["global_pos"]
                    global_neg = cur_leaf_outputs["global_neg"]
                    local_pos = cur_leaf_outputs["local_pos"]
                    local_neg = cur_leaf_outputs["local_neg"]
                    prior_pos = cur_leaf_outputs["prior_pos"]
                    prior_neg = cur_leaf_outputs["prior_neg"]
                elif self.config.leaf_node_classname == "BiGAN":
                    prob_pos = cur_leaf_outputs["prob_pos"]
                    prob_neg = cur_leaf_outputs["prob_neg"]
                    
            else:
                z = torch.cat([z, cur_leaf_outputs["z"]])
                if encoder_cond_type == "gaussian":
                    mu = torch.cat([mu, cur_leaf_outputs["mu"]])
                    logvar = torch.cat([logvar, cur_leaf_outputs["logvar"]])
                if self.config.leaf_node_classname == "VAE":
                    recon_x = torch.cat([recon_x, cur_leaf_outputs["recon_x"]])
                elif self.config.leaf_node_classname == "DIM":
                    global_pos = torch.cat([global_pos, cur_leaf_outputs["global_pos"]])
                    global_neg = torch.cat([global_neg, cur_leaf_outputs["global_neg"]])
                    local_pos = torch.cat([local_pos, cur_leaf_outputs["local_pos"]])
                    local_neg = torch.cat([local_neg, cur_leaf_outputs["local_neg"]])
                    prior_pos = torch.cat([prior_pos, cur_leaf_outputs["prior_pos"]])
                    prior_neg = torch.cat([prior_neg, cur_leaf_outputs["prior_neg"]])
                elif self.config.leaf_node_classname == "BiGAN":
                     prob_pos = torch.cat([prob_pos, cur_leaf_outputs["prob_pos"]])
                     prob_neg = torch.cat([prob_neg, cur_leaf_outputs["prob_neg"]])
            
            # save data points that went through that node
            x_ids += cur_leaf_x_ids
        # reorder points
        sort_order = tuple(np.argsort(x_ids))
        z = z[sort_order, :]
        if encoder_cond_type == "gaussian":
            mu = mu[sort_order, :]
            logvar = logvar[sort_order, :]
        if self.config.leaf_node_classname == "VAE":
            recon_x = recon_x[sort_order, :]
        elif self.config.leaf_node_classname == "DIM":
            global_pos = global_pos[sort_order, :]
            global_neg = global_neg[sort_order, :]
            local_pos = local_pos[sort_order, :]
            local_neg = local_neg[sort_order, :]
            prior_pos = prior_pos[sort_order, :]
            prior_neg = prior_neg[sort_order, :]
        elif self.config.leaf_node_classname == "BiGAN":
            prob_pos = prob_pos[sort_order, :]
            prob_neg = prob_neg[sort_order, :]
            
        # append to the tree results
        tree_z = z # the concatenation already happens in the leaf encoders
        if encoder_cond_type == "gaussian":
            tree_mu = torch.cat([tree_mu, mu], dim=1)
            tree_logvar = torch.cat([tree_logvar, logvar], dim=1)

        model_outputs = {"x": x, "z": tree_z, "path_taken": tree_path_taken}
        if encoder_cond_type == "gaussian":
            model_outputs.update({"mu": tree_mu, "logvar": tree_logvar})
        if self.config.leaf_node_classname == "VAE":
            model_outputs.update({"recon_x": recon_x})
        elif self.config.leaf_node_classname == "DIM":
            model_outputs.update({"global_pos": global_pos, "global_neg": global_neg, "local_pos": local_pos, "local_neg": local_neg, "prior_pos": prior_pos, "prior_neg": prior_neg})
        elif self.config.leaf_node_classname == "BiGAN":
            model_outputs.update({"prob_pos": prob_pos, "prob_neg": prob_neg})
       
        return model_outputs
        
    
    def forward_for_graph_tracing(self, x):
        x = self.push_variable_to_device(x)
        batch_size = x.detach().size(0)
        depth = 0
        tree_path_taken = torch.zeros((batch_size, 1)).int() # all the paths start with "O"
        tree_path_taken = self.push_variable_to_device(tree_path_taken)
        # inner nodes
        while depth < self.config.tree.max_depth:
            path_taken = None
            x_ids = []
            pathes = torch.unique(tree_path_taken[:, :depth+1], dim=0)
            for path in pathes:
                cur_node = self.network.get_child_node(path)
                cur_node_x_ids = [i for i, x_path in enumerate(tree_path_taken[:, :depth+1]) if torch.all(torch.eq(x_path, path))]
                cur_z, cur_prob_right = cur_node.forward_for_graph_tracing(x[cur_node_x_ids])
                cur_go_right = cur_prob_right > 0.5
                if path_taken is None:
                    z = cur_z
                    path_taken = cur_go_right.int()
                else:
                    z = torch.cat([z, cur_z])
                    path_taken = torch.cat([path_taken, cur_go_right.int()])
    
                # save data points that went through that node
                x_ids += cur_node_x_ids
                
            # reorder points
            sort_order = tuple(np.argsort(x_ids))
            path_taken = path_taken[sort_order, :]
            # append to the tree results
            tree_path_taken = torch.cat([tree_path_taken, path_taken], dim=1)
            if depth == 0:
                tree_z = z
            else:
                tree_z = torch.cat([tree_z, z], dim=1)
            # go to next level of the tree
            depth += 1
        
        # leaf nodes
        leaf_output = None
        pathes = torch.unique(tree_path_taken, dim=0)
        for path in pathes:
            # go to leaf
            cur_leaf = self.network.get_child_node(path)
            cur_leaf_x_ids = [i for i, x_path in enumerate(tree_path_taken) if torch.all(torch.eq(x_path, path))]
            if self.config.leaf_node_classname == "VAE":
                cur_recon_x = cur_leaf.forward_for_graph_tracing(x[cur_leaf_x_ids], tree_z[cur_leaf_x_ids])
                if leaf_output is None:
                    recon_x = cur_recon_x
                    leaf_output = True
                else:
                    recon_x = torch.cat([recon_x, cur_recon_x])
            elif self.config.leaf_node_classname == "DIM":
                cur_global_pred, cur_local_pred, cur_prior_pred = cur_leaf.forward_for_graph_tracing(x[cur_leaf_x_ids], tree_z[cur_leaf_x_ids])
                if leaf_output is None:
                    global_pred = cur_global_pred
                    local_pred = cur_local_pred
                    prior_pred = cur_prior_pred
                    leaf_output = True
                else:
                    global_pred = torch.cat([global_pred, cur_global_pred])
                    local_pred = torch.cat([local_pred, cur_local_pred])
                    prior_pred = torch.cat([prior_pred, cur_prior_pred])
            elif self.config.leaf_node_classname == "BiGAN":
                cur_prob_pos, cur_recon_x = cur_leaf.forward_for_graph_tracing(x[cur_leaf_x_ids], tree_z[cur_leaf_x_ids])
                if leaf_output is None:
                    prob_pos = cur_prob_pos
                    recon_x = cur_recon_x
                    leaf_output = True
                else:
                    prob_pos = torch.cat([prob_pos, cur_prob_pos])
                    recon_x = torch.cat([recon_x, cur_recon_x])
                    
        if self.config.leaf_node_classname == "VAE":
            return recon_x
        elif self.config.leaf_node_classname == "DIM":
            return global_pred, local_pred, prior_pred
        elif self.config.leaf_node_classname == "BiGAN":
            return prob_pos, recon_x
        
        
    
    def calc_embedding(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        encoder = self.get_encoder()
        return encoder.calc_embedding(x)
    
    
    def run_training (self, train_loader, n_epochs, valid_loader = None, logger=None):
        """
        logger: tensorboard X summary writer
        """            
        # Save the graph in the logger
        if logger is not None:
            root_network_config = self.config.inner_node.network.parameters
            dummy_input = torch.FloatTensor(4, root_network_config.n_channels, root_network_config.input_size[0], root_network_config.input_size[1]).uniform_(0,1)
            dummy_input = self.push_variable_to_device(dummy_input)
            self.eval()
            #with torch.no_grad():
            #    logger.add_graph(self, dummy_input, verbose = False)
            
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
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'curent_weight_model.pth'))
            
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
            x =  data['obs']
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
                    losses[k] = [v.item()]
                else:
                    losses[k].append(v.item())
          
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
                x =  data['obs']
                x = self.push_variable_to_device(x)
                y = data['label'].squeeze()
                y = self.push_variable_to_device(y)
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = [v.item()]
                    else:
                        losses[k].append(v.item())
                # record embeddings
                if record_embeddings:
                     embedding_samples.append(model_outputs["z"])
                     embedding_metadata.append(data['label'])
                     embedding_images.append(x)
                    
        for k, v in losses.items():
            losses [k] = np.mean (v)
            
        if record_valid_images and "recon_x" in model_outputs:
            input_images = x.cpu()
            output_images = model_outputs["recon_x"].cpu()
            if self.config.loss.parameters.reconstruction_dist == "bernouilli":
                output_images = torch.sigmoid(model_outputs["recon_x"]).cpu()
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
            
        self.input_keys_list = node.loss_f.input_keys_list + ["path_taken"]
        
    def __call__(self, loss_inputs, reduction = True, logger=None, **kwargs):
        try:
            path_taken = loss_inputs["path_taken"] # add path_prob
        except:
            raise ValueError("HierarchicalTreeLoss needs path_taken inputs")
        
        pathes = torch.unique(path_taken, dim=0)
        
        losses = {}
        batch_size = list(loss_inputs.values())[0].size(0)
        
        #TODO: here work with VAE because KLD (tot_mu) = sum(KLD(small mu)), change for rest
        for path in pathes:
            node_x_ids = [i for i, x_path in enumerate(path_taken) if torch.all(torch.eq(x_path, path))]
            leaf_loss_f = self.path2loss[tensor2string(path)]
            leaf_loss_inputs = {}
            for k, v in loss_inputs.items():
                leaf_loss_inputs[k] = v[node_x_ids]

            leaf_losses = leaf_loss_f(leaf_loss_inputs, reduction = False)
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
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)
        
        x = self.push_variable_to_device(x)
        batch_size = x.detach().size(0)
        
        encoder_cond_type = self.network.network.encoder.conditional_type
        
        depth = 0
        tree_path_taken = torch.zeros((batch_size, 1)).int() # all the paths start with "O"
        tree_path_taken = self.push_variable_to_device(tree_path_taken)
        # inner nodes
        while depth < self.max_depth:
            z = None
            x_ids = []
            pathes = torch.unique(tree_path_taken[:, :depth+1], dim=0)
            for path in pathes:
                # go to node
                cur_node = self.network.get_child_node(path)
                cur_node_x_ids = [i for i, x_path in enumerate(tree_path_taken[:, :depth+1]) if torch.all(torch.eq(x_path, path))]
                
                # forward
                cur_inner_outputs = cur_node.forward(x[cur_node_x_ids])
                    
                cur_prob_right = cur_inner_outputs["prob_right"]
                # torch bernouilli distribution allow to do "hard" samples and still be differentiable
                cur_go_right = Bernoulli(cur_prob_right).sample()
                #cur_prob = cur_go_right * cur_prob_right + (1.0 - cur_go_right) * (1.0 - cur_prob_right)
                #path_prob *= cur_prob

                if z is None:
                    z = cur_inner_outputs["z"]
                    if encoder_cond_type == "gaussian":
                        mu = cur_inner_outputs["mu"]
                        logvar = cur_inner_outputs["logvar"]
                    #prob_right = cur_inner_outputs["prob_right"]
                    path_taken = cur_go_right.int()
                else:
                    z = torch.cat([z, cur_inner_outputs["z"]])
                    if encoder_cond_type == "gaussian":
                        mu = torch.cat([mu, cur_inner_outputs["mu"]])
                        logvar = torch.cat([logvar, cur_inner_outputs["logvar"]])
                    #prob_right = torch.cat([prob_right, cur_inner_outputs["prob_right"]])
                    path_taken = torch.cat([path_taken, cur_go_right.int()])

                # save data points that went through that node
                x_ids += cur_node_x_ids
                
            # reorder points
            sort_order = tuple(np.argsort(x_ids))
            z = z[sort_order, :]
            if encoder_cond_type == "gaussian":
                mu = mu[sort_order, :]
                logvar = logvar[sort_order, :]
            #prob_right = prob_right[sort_order, :]
            path_taken = path_taken[sort_order, :]
            # append to the tree results
            if depth == 0:
                prefix_z = z
                if encoder_cond_type == "gaussian":
                    tree_mu = mu
                    tree_logvar = logvar
                #tree_prob_right = prob_right
                tree_path_taken = torch.cat([tree_path_taken, path_taken], dim=1)
            else:
                prefix_z = torch.cat([prefix_z, z], dim=1)
                if encoder_cond_type == "gaussian":
                    tree_mu = torch.cat([tree_mu, mu], dim=1)
                    tree_logvar = torch.cat([tree_logvar, logvar], dim=1)
                #tree_prob_right = torch.cat([tree_prob_right, prob_right], dim=1)
                tree_path_taken = torch.cat([tree_path_taken, path_taken], dim=1)
                
            # go to next level of the tree
            depth += 1
        
        # leaf nodes
        z = None
        x_ids = []
        pathes = torch.unique(tree_path_taken, dim=0)
        for path in pathes:
            # go to leaf
            cur_leaf = self.network.get_child_node(path)
            cur_leaf_x_ids = [i for i, x_path in enumerate(tree_path_taken) if torch.all(torch.eq(x_path, path))]
            
            # forward for that leaf
            cur_leaf_encoder_outputs = cur_leaf.network.encoder.forward(x[cur_leaf_x_ids])
            
            if z is None:
                z = cur_leaf_encoder_outputs["z"]
                if encoder_cond_type == "gaussian":
                    mu = cur_leaf_encoder_outputs["mu"]
                    logvar = cur_leaf_encoder_outputs["logvar"]       
            else:
                z = torch.cat([z, cur_leaf_encoder_outputs["z"]])
                if encoder_cond_type == "gaussian":
                    mu = torch.cat([mu, cur_leaf_encoder_outputs["mu"]])
                    logvar = torch.cat([logvar, cur_leaf_encoder_outputs["logvar"]])
            
            # save data points that went through that node
            x_ids += cur_leaf_x_ids
        # reorder points
        sort_order = tuple(np.argsort(x_ids))
        z = z[sort_order, :]
        if encoder_cond_type == "gaussian":
            mu = mu[sort_order, :]
            logvar = logvar[sort_order, :]
            
        # append to the tree results
        tree_z = torch.cat([prefix_z, z], dim=1)
        if encoder_cond_type == "gaussian":
            tree_mu = torch.cat([tree_mu, mu], dim=1)
            tree_logvar = torch.cat([tree_logvar, logvar], dim=1)

        encoder_outputs = {"x": x, "z": tree_z, "path_taken": tree_path_taken}
        if encoder_cond_type == "gaussian":
            encoder_outputs.update({"mu": tree_mu, "logvar": tree_logvar})
       
        return encoder_outputs
    
    def calc_embedding(self, x):
        encoder_outputs = self.forward(x)
        return encoder_outputs["z"]
    
    
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
        recon_x = torch.empty(batch_size, image_size[0], image_size[1], image_size[2])
        recon_x = self.push_variable_to_device(recon_x)
        pathes = torch.unique(path_taken, dim=0)
        for path in pathes:
            cur_decoder_idx = int (tensor2string(path), 2)  # the path is given in base 2 and we convert it to integer (respect the order of decoder list)
            cur_decoder = self.decoder_list[cur_decoder_idx]
            cur_decoder_z_ids = [i for i, z_path in enumerate(path_taken) if torch.all(torch.eq(z_path, path))]
            cur_recon_x = cur_decoder.forward(z[cur_decoder_z_ids])
            for i in range(len(cur_decoder_z_ids)):
                recon_x[cur_decoder_z_ids[i]] = cur_recon_x[i]
        return recon_x
    
    
def generate_binary_pathes(tree_depth):
    n_path = int(math.pow(2, tree_depth))
    pathes = torch.zeros((n_path, tree_depth+1), dtype=int)
    for i in range(n_path):
        suffix = bin(i)[2:]
        for c_idx, c in enumerate(reversed(suffix)):
            pathes[i, -(c_idx+1)] = int(c)
    return pathes
        
        