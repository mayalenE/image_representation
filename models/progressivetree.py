from copy import deepcopy
import goalrepresent as gr
from goalrepresent import dnn, models
from goalrepresent.dnn.solvers import initialization
import numpy as np
import os
from sklearn import cluster, svm
import sys
import time
import torch
from torch import nn
from torchvision.utils import make_grid
import warnings
#from torchviz import make_dot

class Node(nn.Module):
    """
    Leaf Node base class
    """
    def __init__(self, depth, **kwargs):
        self.depth = depth
        self.leaf = True # set to Fale when node is split
        self.boundary = None
        self.feature_range = None
        self.leaf_accumulator = []
        
    def reset_accumulator(self):
        self.leaf_accumulator = []
        if not self.leaf:
            self.left.reset_accumulator()
            self.right.reset_accumulator()
    
    def get_child_node(self, path):
        node = self
        for d in range(1, len(path)):
            if path[d] == "0":
                node = node.left
            else:
                node = node.right
        
        return node
    
    def get_leaf_pathes(self, path_taken=[]):
        if self.depth == 0:
            path_taken = "0"
        if self.leaf:
            return ([path_taken])
  
        else:
            left_leaf_accumulator = self.left.get_leaf_pathes(path_taken=path_taken+"0")
            self.leaf_accumulator.extend(left_leaf_accumulator)
            right_leaf_accumulator = self.right.get_leaf_pathes(path_taken=path_taken+"1")
            self.leaf_accumulator.extend(right_leaf_accumulator)
            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()
            
        return leaf_accumulator
    
        
    def get_node_pathes(self, path_taken=[]):
        if self.depth == 0:
            path_taken = "0"
        if self.leaf:
            return ([path_taken])
  
        else:
            self.leaf_accumulator.extend([path_taken])
            left_leaf_accumulator = self.left.get_node_pathes(path_taken=path_taken+"0")
            self.leaf_accumulator.extend(left_leaf_accumulator)
            right_leaf_accumulator = self.right.get_node_pathes(path_taken=path_taken+"1")
            self.leaf_accumulator.extend(right_leaf_accumulator)
            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()
            
        return leaf_accumulator
        
        
    def split_node(self, z_library=None, z_fitness=None):
        """
        z_library: n_samples * n_latents
        z_fitness: n_samples * 1 (eg: reconstruction loss)
        """
        
        # create childrens
        self.NodeClass = type(self)
        self.left = self.NodeClass(self.depth+1, parent_network = deepcopy(self.network), config=self.config)
        self.right = self.NodeClass(self.depth+1, parent_network = deepcopy(self.network), config=self.config)
        
        # create boundary based on database history
        if z_library is not None:
            self.create_boundary(z_library, z_fitness)
        
        # freeze parameters
        for param in self.network.parameters():
            param.requires_grad = False
            
        # node becomes inner node
        self.leaf = False
        
        return
    
    def create_boundary(self, z_library, z_fitness=None):
        # normalize z points
        self.feature_range = (z_library.min(axis=0), z_library.max(axis=0))
        X =  z_library - self.feature_range[0]
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(scale == 0)] = 1.0 # trick when some some latents are the same for every point (no scale and divide by 1)
        X = X / scale
        
        if z_fitness is None:
            # fit cluster boundary
            self.boundary = cluster.KMeans(n_clusters=2).fit(X)
        else:
            # fit the linear boundary
            if z_fitness.dtype == np.bool:
                y = z_fitness
            else:
                y = z_fitness > np.median(z_fitness)
            #center_0 = np.mean(X[y], axis=0)
            #center_1 = np.mean(X[1 - y], axis=0)
            #self.boundary = cluster.KMeans(n_clusters=2, init=np.stack([center_0,center_1])).fit(X)
            self.boundary = svm.SVC(kernel='rbf', C=100).fit(X, y)
        return
    
    def depth_first_forward(self, x, tree_path_taken=None, x_ids=None, ancestors_lf=None, ancestors_gf=None, ancestors_gfi=None, ancestors_lfi=None, ancestors_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"]*len(x) # all the paths start with "O"
            x_ids = list(range(len(x)))
            ancestors_lf = []
            ancestors_gf = []
            ancestors_gfi = []
            ancestors_lfi = []
            ancestors_recon_x = []
        
        node_outputs = self.node_forward(x,ancestors_lf, ancestors_gf, ancestors_gfi, ancestors_lfi, ancestors_recon_x)
        
        ancestors_lf.append(node_outputs["lf"].detach())
        ancestors_gf.append(node_outputs["gf"].detach())
        ancestors_gfi.append(node_outputs["gfi"].detach())
        ancestors_lfi.append(node_outputs["lfi"].detach())
        ancestors_recon_x.append(node_outputs["recon_x"].detach())
    
        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, x_ids, node_outputs]])
  
        else:
            z = node_outputs["z"]
            x_side = self.get_children_node(z)
            x_ids_left = np.where(x_side==0)[0]
            if len(x_ids_left) > 0:
                left_leaf_accumulator = self.left.depth_first_forward(x[x_ids_left], [path + "0" for path in [tree_path_taken[x_idx] for x_idx in x_ids_left]], [x_ids[x_idx] for x_idx in x_ids_left], 
                                                                      [ancestors_lf[i][x_ids_left] for i in range(self.depth+1)], [ancestors_gf[i][x_ids_left] for i in range(self.depth+1)], 
                                                                        [ancestors_gfi[i][x_ids_left] for i in range(self.depth+1)], [ancestors_lfi[i][x_ids_left] for i in range(self.depth+1)], 
                                                                        [ancestors_recon_x[i][x_ids_left] for i in range(self.depth+1)])
                self.leaf_accumulator.extend(left_leaf_accumulator)
            
            x_ids_right = np.where(x_side==1)[0]
            if len(x_ids_right) > 0:
                right_leaf_accumulator = self.right.depth_first_forward(x[x_ids_right], [path + "1" for path in [tree_path_taken[x_idx] for x_idx in x_ids_right]],[x_ids[x_idx] for x_idx in x_ids_right], 
                                                                        [ancestors_lf[i][x_ids_right] for i in range(self.depth+1)], [ancestors_gf[i][x_ids_right] for i in range(self.depth+1)], 
                                                                        [ancestors_gfi[i][x_ids_right] for i in range(self.depth+1)], [ancestors_lfi[i][x_ids_right] for i in range(self.depth+1)], 
                                                                        [ancestors_recon_x[i][x_ids_right] for i in range(self.depth+1)])
                self.leaf_accumulator.extend(right_leaf_accumulator)
            
            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()
            
        return leaf_accumulator
    
    
    def depth_first_forward_all_nodes_preorder(self, x, tree_path_taken=None, x_ids=None, ancestors_lf=None, ancestors_gf=None, ancestors_gfi=None, ancestors_lfi=None, ancestors_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"]*len(x) # all the paths start with "O"
            x_ids = list(range(len(x)))
            ancestors_lf = []
            ancestors_gf = []
            ancestors_gfi = []
            ancestors_lfi = []
            ancestors_recon_x = []
        
        node_outputs = self.node_forward(x,ancestors_lf, ancestors_gf, ancestors_gfi, ancestors_lfi, ancestors_recon_x)
        
        ancestors_lf.append(node_outputs["lf"].detach())
        ancestors_gf.append(node_outputs["gf"].detach())
        ancestors_gfi.append(node_outputs["gfi"].detach())
        ancestors_lfi.append(node_outputs["lfi"].detach())
        ancestors_recon_x.append(node_outputs["recon_x"].detach())
    
        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, x_ids, node_outputs]])
  
        else:
            self.leaf_accumulator.extend([[tree_path_taken, x_ids, node_outputs]])
            
            z = node_outputs["z"]
            x_side = self.get_children_node(z)
            x_ids_left = np.where(x_side==0)[0]
            if len(x_ids_left) > 0:
                left_leaf_accumulator = self.left.depth_first_forward_all_nodes_preorder(x[x_ids_left], [path + "0" for path in [tree_path_taken[x_idx] for x_idx in x_ids_left]], [x_ids[x_idx] for x_idx in x_ids_left], 
                                                                      [ancestors_lf[i][x_ids_left] for i in range(self.depth+1)], [ancestors_gf[i][x_ids_left] for i in range(self.depth+1)], 
                                                                        [ancestors_gfi[i][x_ids_left] for i in range(self.depth+1)], [ancestors_lfi[i][x_ids_left] for i in range(self.depth+1)], 
                                                                        [ancestors_recon_x[i][x_ids_left] for i in range(self.depth+1)])
                self.leaf_accumulator.extend(left_leaf_accumulator)
            
            x_ids_right = np.where(x_side==1)[0]
            if len(x_ids_right) > 0:
                right_leaf_accumulator = self.right.depth_first_forward_all_nodes_preorder(x[x_ids_right], [path + "1" for path in [tree_path_taken[x_idx] for x_idx in x_ids_right]],[x_ids[x_idx] for x_idx in x_ids_right], 
                                                                        [ancestors_lf[i][x_ids_right] for i in range(self.depth+1)], [ancestors_gf[i][x_ids_right] for i in range(self.depth+1)], 
                                                                        [ancestors_gfi[i][x_ids_right] for i in range(self.depth+1)], [ancestors_lfi[i][x_ids_right] for i in range(self.depth+1)], 
                                                                        [ancestors_recon_x[i][x_ids_right] for i in range(self.depth+1)])
                self.leaf_accumulator.extend(right_leaf_accumulator)
            
            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()
            
        return leaf_accumulator
    
    def node_forward(self, x, ancestors_lf=None, ancestors_gf=None, ancestors_gfi=None, ancestors_lfi=None, ancestors_recon_x=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x, ancestors_lf, ancestors_gf, ancestors_gfi, ancestors_lfi, ancestors_recon_x)
        if self.depth == 0:
            encoder_outputs = self.network.encoder(x)
        else:
            encoder_outputs = self.network.encoder(x, ancestors_lf, ancestors_gf)
        model_outputs = self.node_forward_from_encoder(encoder_outputs, ancestors_gfi, ancestors_lfi, ancestors_recon_x)
        return model_outputs 
    
    def get_boundary_side(self, z):
        if self.boundary is None:
            raise ValueError("Boundary computation is required before calling this function")
        else:
            # normalize
            if isinstance(z, torch.Tensor):
                z = z.detach().cpu().numpy()
            z = z - self.feature_range[0]
            scale = self.feature_range[1] - self.feature_range[0]
            scale[np.where(scale==0)] = 1.0 # trick when some some latents are the same for every point (no scale and divide by 1)
            z = z / scale
            # compute boundary side
            side = self.boundary.predict(z) # returns 0: left, 1: right
        return side
        
    
    def get_children_node(self, z):
        """
        Return code: -1 for leaf node,  0 to send left, 1 to send right
        """
        if self.leaf:
            return -1*torch.ones_like(z)
        else:
            return self.get_boundary_side(z)
        
    
class VAENode(Node, models.VAEModel):
    def __init__(self, depth, parent_network=None, config = None, **kwargs):
        super(Node, self).__init__(config = config, **kwargs)
        super().__init__(depth, **kwargs)
        
        # connect encoder and decoder
        if self.depth > 0:
            self.network.encoder = ConnectedEncoder(parent_network.encoder, depth, connect_lf = True, connect_gf = False)
            self.network.decoder = ConnectedDecoder(parent_network.decoder, depth, connect_gfi = True, connect_lfi = True, connect_recon = True)
            
        self.set_device(self.config.device.use_gpu)
            
    
    def node_forward_from_encoder(self, encoder_outputs, ancestors_gfi=None, ancestors_lfi=None, ancestors_recon_x=None):
        if self.depth == 0:
            decoder_outputs = self.network.decoder(encoder_outputs["z"])
        else:
            decoder_outputs = self.network.decoder(encoder_outputs["z"], ancestors_gfi, ancestors_lfi, ancestors_recon_x)
        model_outputs = encoder_outputs
        model_outputs.update(decoder_outputs)
        return model_outputs
            
    


class ProgressiveTreeModel(dnn.BaseDNN, gr.BaseModel):
    """
    dnn with tree network, loss which is based on leaf's losses, optimizer from that loss
    """
    @staticmethod
    def default_config():
        default_config = dnn.BaseDNN.default_config()
        
        # tree config
        default_config.tree = gr.Config()
        
        # node config
        default_config.node_classname = "VAE"
        default_config.node = eval("gr.models.{}Model.default_config()".format(default_config.node_classname))
        
        # loss parameters
        default_config.loss.name = "VAE"   
        
        # optimizer
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5

        return default_config
        
    
    def __init__(self, config = None, **kwargs):
        
        self.NodeClass = eval("{}Node".format(config.node_classname))
        self.split_history = {} # dictionary with node path keys and boundary values
        
        super().__init__(config=config, **kwargs)
        
        
    def set_network(self, network_name, network_parameters):
        depth = 0
        self.network = self.NodeClass(depth, config=self.config.node) #root node that links to child nodes
        self.network.optimizer_group_id = 0
        self.output_keys_list = self.network.output_keys_list + ["path_taken"] #node.left is a leaf node
        
        
    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # give only trainable parameters
        trainable_parameters = [p for p in self.network.parameters() if p.requires_grad]
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(trainable_parameters, **optimizer_parameters) 
        self.network.optimizer_group_id = 0
        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters, self.config.optimizer.parameters)
                
    def forward_for_graph_tracing(self, x):
        pass
    
    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)
        x = self.push_variable_to_device(x)
        is_train = self.network.training
        if len(x) == 1 and is_train:
            self.network.eval()
            depth_first_traversal_outputs = self.network.depth_first_forward(x)
            self.network.train()
        else:
            depth_first_traversal_outputs = self.network.depth_first_forward(x)
        
        model_outputs = {}
        x_order_ids = []
        for leaf_idx in range(len(depth_first_traversal_outputs)):
            cur_node_path = depth_first_traversal_outputs[leaf_idx][0]
            cur_node_x_ids = depth_first_traversal_outputs[leaf_idx][1]
            cur_node_outputs = depth_first_traversal_outputs[leaf_idx][2]
            # stack results
            if not model_outputs:
                model_outputs["path_taken"] = cur_node_path
                for k,v in cur_node_outputs.items():
                    model_outputs[k] = v
            else:
                model_outputs["path_taken"] += cur_node_path
                for k,v in cur_node_outputs.items():
                    model_outputs[k] =  torch.cat([model_outputs[k], v], dim=0)
            # save the sampled ids to reorder as in the input batch at the end
            x_order_ids += list(cur_node_x_ids)
            
        # reorder points
        sort_order = tuple(np.argsort(x_order_ids))
        for k,v in model_outputs.items():
            if isinstance(v, list):
                model_outputs[k] = [v[i] for i in sort_order]
            else:
                model_outputs[k] = v[sort_order,:]
        
        return model_outputs
             
    
    def calc_embedding(self, x, node_path = None, **kwargs):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        if node_path is None:
            warnings.warn("WARNING: computing the embedding in root node of progressive tree as no path specified")
            node_path = "0"
        n_latents = self.network.config.network.parameters.n_latents
        z = torch.Tensor().new_full((len(x), n_latents), float("nan"))
        x = self.push_variable_to_device(x)
        self.eval()
        with torch.no_grad():
            all_nodes_outputs = self.network.depth_first_forward_all_nodes_preorder(x)
            for node_idx in range(len(all_nodes_outputs)):
                cur_node_path = all_nodes_outputs[node_idx][0][0]
                if cur_node_path != node_path:
                    continue;
                else:
                    cur_node_x_ids = all_nodes_outputs[node_idx][1]
                    cur_node_outputs = all_nodes_outputs[node_idx][2]
                    for idx in range(len(cur_node_x_ids)):
                        z[cur_node_x_ids[idx]] = cur_node_outputs["z"][idx] 
                    break;
        return z
    
    def split_node(self, node_path, z_library=None, z_fitness=None):
        self.eval()
        node = self.network.get_child_node(node_path)
        node.split_node(z_library=z_library, z_fitness=z_fitness)
        self.split_history[node_path] = {"depth": node.depth, "leaf": node.leaf, "boundary": node.boundary, "feature_range": node.feature_range}
        
        # remove group from optimizer and add one group per children
        del self.optimizer.param_groups[node.optimizer_group_id]
        node.optimizer_group_id = None
        n_groups = len(self.optimizer.param_groups)
        self.optimizer.add_param_group({"params": node.left.parameters()})
        node.left.optimizer_group_id = n_groups
        self.optimizer.add_param_group({"params": node.right.parameters()})
        node.right.optimizer_group_id = n_groups + 1
    
    
    def run_training (self, train_loader, n_epochs, valid_loader = None, logger=None):
        """
        logger: tensorboard X summary writer
        """            
        # Save the graph in the logger
        if logger is not None:
            root_network_config = self.config.node.network.parameters
            dummy_input = torch.FloatTensor(4, root_network_config.n_channels, root_network_config.input_size[0], root_network_config.input_size[1]).uniform_(0,1)
            dummy_input = self.push_variable_to_device(dummy_input)
            self.eval()
            #with torch.no_grad():
            #    logger.add_graph(self, dummy_input, verbose = False)
        
        do_validation = False
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
                
                if valid_losses:
                    valid_loss = valid_losses['total']
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))
        
                    
                    
    def train_epoch (self, train_loader, logger=None):
        self.train()
        losses = {}
        taken_pathes = []
        for data in train_loader:
            x =  data['obs']
            x = self.push_variable_to_device(x)
            # forward
            model_outputs = self.forward(x)
            #g = make_dot(model_outputs, params=dict(self.network.named_parameters()))
            #g.view()
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs, reduction=False)
            # backward
            loss = batch_losses['total'].mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis = -1)
                else:
                    losses[k]= np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis = -1)])
            taken_pathes += model_outputs["path_taken"]
        
        self.n_epochs += 1
        
        # LOGGER SAVE RESULT PER LEAF
        for leaf_path in list(set(taken_pathes)):
            if len(leaf_path) > 1:
                leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]
                for k, v in losses.items():
                    leaf_v = v[leaf_x_ids,:]
                    logger.add_scalars('loss/{}'.format(k), {'train-{}'.format(leaf_path): np.mean(leaf_v)}, self.n_epochs)

            
        # AVERAGE LOSS ON WHOLE TREE
        for k, v in losses.items():
            losses [k] = np.mean (v)
        
        return losses
    
    
    def valid_epoch (self, valid_loader, logger=None):
        self.eval()
        losses = {}
        
        record_valid_images = False
        record_embeddings = False
        if logger is not None:
            if self.n_epochs % self.config.logging.record_valid_images_every == 0:
                record_valid_images = True
                images = []
                recon_images = []
            if self.n_epochs % self.config.logging.record_embeddings_every == 0:
                record_embeddings = True
                embedding_samples = []
                embedding_metadata = []
                embedding_images = []
        taken_pathes = []
                
            
        with torch.no_grad():
            for data in valid_loader:
                x =  data['obs']
                x = self.push_variable_to_device(x)
                y = data['label'].squeeze()
                y = self.push_variable_to_device(y)
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs, reduction=False)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis = -1)
                    else:
                        losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis = -1)])
                # record embeddings
                if record_embeddings:
                    samples = model_outputs["z"]
                    metadata = data['label']
                    images = x
                    for i in range(len(x)):
                        embedding_samples.append(samples[i].unsqueeze(0))
                        embedding_metadata.append(metadata[i].unsqueeze(0))
                        embedding_images.append(images[i].unsqueeze(0))
                if record_valid_images:
                    if not record_embeddings:
                        for i in range(len(x)):
                            images.append(x[i].unsqueeze(0))
                    for i in range(len(x)):
                        recon_images.append(model_outputs["recon_x"][i].unsqueeze(0)) 
                        
                taken_pathes += model_outputs["path_taken"]
                    
        # LOGGER SAVE RESULT PER LEAF
        for leaf_path in list(set(taken_pathes)):
            leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]
            
            if record_embeddings:
                leaf_embedding_samples = torch.cat([embedding_samples[i] for i in leaf_x_ids])
                leaf_embedding_metadata = torch.cat([embedding_metadata[i] for i in leaf_x_ids])
                leaf_embedding_images = torch.cat([embedding_images[i] for i in leaf_x_ids])
                try:
                    logger.add_embedding(
                            leaf_embedding_samples,
                            metadata=leaf_embedding_metadata,
                            label_img=leaf_embedding_images,
                            global_step=self.n_epochs,
                            tag = "leaf_{}".format(leaf_path))
                except:
                    pass
            
            if record_valid_images:
                n_images = min(len(leaf_x_ids), 40)
                sampled_ids = np.random.choice(len(leaf_x_ids), n_images, replace=False)
                if not record_embeddings:
                    input_images = torch.cat([images[i] for i in leaf_x_ids[sampled_ids]]).cpu()
                else:
                    input_images = leaf_embedding_images[sampled_ids,:].cpu()
                output_images = torch.cat([recon_images[i] for i in leaf_x_ids[sampled_ids]]).cpu()
                if self.config.loss.parameters.reconstruction_dist == "bernouilli":
                    output_images = torch.sigmoid(output_images)
                vizu_tensor_list = [None] * (2*n_images)
                vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
                vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
                img = make_grid(vizu_tensor_list, nrow = 2, padding=0)
                try:
                    logger.add_image("leaf_{}".format(leaf_path), img, self.n_epochs)
                except:
                    pass
                    
            
                
        
        # AVERAGE LOSS ON WHOLE TREE
        for k, v in losses.items():
            losses [k] = np.mean (v)

        return losses
    
    def get_encoder(self):
        pass
    
    def get_decoder(self):
        pass
    
    def save_checkpoint(self, checkpoint_filepath):
        # save current epoch weight file with optimizer if we want to relaunch training from that point
        network = {
        "epoch": self.n_epochs,
        "type": self.__class__.__name__,
        "config": self.config,
        "network_state_dict": self.network.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "split_history": self.split_history
        }
        
        torch.save(network, checkpoint_filepath)

""" =========================================================================================
CONNECTED MODULES 
==========================================================================================="""
class ConnectedEncoder(gr.dnn.networks.encoders.BaseDNNEncoder):
    def __init__(self, encoder_instance, depth, connect_lf = False, connect_gf = False, **kwargs):
        super().__init__(n_channels = encoder_instance.n_channels, input_size = encoder_instance.input_size, 
                         n_conv_layers = encoder_instance.n_conv_layers, n_latents = encoder_instance.n_latents,
                         encoder_conditional_type = encoder_instance.conditional_type, feature_layer = encoder_instance.feature_layer,
                         hidden_channels = encoder_instance.hidden_channels, hidden_dim = encoder_instance.hidden_dim)
        # connections and depth in the tree (number of connections)
        self.connect_lf = connect_lf
        self.connect_gf = connect_gf
        self.depth = depth
    
        
        # copy parent network layers
        self.lf = encoder_instance.lf
        self.gf = encoder_instance.gf
        self.ef = encoder_instance.ef
        
        # add lateral connections
        ## lf
        self.lf_c = nn.ModuleList()
        if self.connect_lf:
            if self.lf.out_connection_type[0] == "conv":
                connection_channels = self.lf.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.lf_c.append(nn.Sequential(nn.Conv2d(connection_channels,connection_channels, kernel_size=1, stride=1), nn.ReLU()))
            elif self.lf.out_connection_type[0] == "lin":
                connection_dim = self.lf.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.lf_c.append(nn.Sequential(nn.Linear(connection_dim,connection_dim), nn.ReLU()))
            
        ## gf
        self.gf_c = nn.ModuleList()
        if self.connect_gf:
            if self.gf.out_connection_type[0] == "conv":
                connection_channels = self.gf.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.gf_c.append(nn.Sequential(nn.Conv2d(connection_channels,connection_channels, kernel_size=1, stride=1), nn.ReLU()))
            elif self.gf.out_connection_type[0] == "lin":
                connection_dim = self.gf.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.gf_c.append(nn.Sequential(nn.Linear(connection_dim,connection_dim), nn.ReLU()))
                
        # lf is initialized with parent weights, gf and ef with kaiming, connections with small random weights
        initialization_net = initialization.get_initialization("kaiming_uniform")
        self.gf.apply(initialization_net)
        self.ef.apply(initialization_net)
        initialization_c = initialization.get_initialization("uniform")
        self.lf_c.apply(initialization_c)
        self.gf_c.apply(initialization_c)
        
            
            
    def forward(self, x, ancestors_lf=None, ancestors_gf=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)
        
        # local feature map
        lf = self.lf(x)
        # loop over the connections
        for lf_c_idx in range(len(self.lf_c)): 
            lf += self.lf_c[lf_c_idx](ancestors_lf[lf_c_idx])
            
        # global feature map
        gf = self.gf(lf)
        for gf_c_idx in range(len(self.gf_c)): 
            gf += self.gf_c[gf_c_idx](ancestors_gf[gf_c_idx])
        # encoding
        if self.conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
            if z.ndim > 2:
                mu = mu.squeeze(dim=-1).squeeze(dim=-1)
                logvar = logvar.squeeze(dim=-1).squeeze(dim=-1)
                z = z.squeeze(dim=-1).squeeze(dim=-1)
            encoder_outputs = {"x": x, "lf": lf, "gf": gf, "z": z, "mu": mu, "logvar": logvar}
        elif self.conditional_type == "deterministic":
            z = self.ef(gf)
            if z.ndim > 2:
                z = z.squeeze(dim=-1).squeeze(dim=-1)
            encoder_outputs =  {"x": x, "lf": lf, "gf": gf, "z": z}
        
        return encoder_outputs
    
    def forward_for_graph_tracing(self, x, ancestors_lf=None, ancestors_gf=None):
        # local feature map
        lf = self.lf(x)
        # loop over the connections
        for lf_c_idx in range(len(self.lf_c)): 
            lf += self.lf_c[lf_c_idx](ancestors_lf[lf_c_idx])
            
        # global feature map
        gf = self.gf(lf)
        for gf_c_idx in range(len(self.gf_c)): 
            gf += self.gf_c[gf_c_idx](ancestors_gf[gf_c_idx])
        if self.conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
        else:
            z = self.ef(gf)
        return z, lf
    
    
class ConnectedDecoder(gr.dnn.networks.decoders.BaseDNNDecoder):
    def __init__(self, decoder_instance, depth, connect_gfi = False, connect_lfi = False, connect_recon = False, **kwargs):
        super().__init__(n_channels = decoder_instance.n_channels, input_size = decoder_instance.input_size, 
                         n_conv_layers = decoder_instance.n_conv_layers, n_latents = decoder_instance.n_latents,
                         feature_layer = decoder_instance.feature_layer, hidden_channels = decoder_instance.hidden_channels, 
                         hidden_dim = decoder_instance.hidden_dim)
        
        # connections and depth in the tree (number of connections)
        self.connect_gfi = connect_gfi
        self.connect_lfi = connect_lfi
        self.connect_recon = connect_recon
        self.depth = depth
        
        # copy parent network layers
        self.efi = decoder_instance.efi
        self.gfi = decoder_instance.gfi
        self.lfi = decoder_instance.lfi
        
        # add lateral connections
        ## gfi
        self.gfi_c = nn.ModuleList()
        if self.connect_gfi:
            if self.efi.out_connection_type[0] == "conv":
                connection_channels = self.efi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.gfi_c.append(nn.Sequential(nn.Conv2d(connection_channels,connection_channels, kernel_size=1, stride=1), nn.ReLU()))
            elif self.efi.out_connection_type[0] == "lin":
                connection_dim = self.efi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.gfi_c.append(nn.Sequential(nn.Linear(connection_dim,connection_dim), nn.ReLU()))
        ## lfi
        self.lfi_c = nn.ModuleList()
        if self.connect_lfi:
            if self.gfi.out_connection_type[0] == "conv":
                connection_channels = self.gfi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.lfi_c.append(nn.Sequential(nn.Conv2d(connection_channels,connection_channels, kernel_size=1, stride=1), nn.ReLU()))
            elif self.gfi.out_connection_type[0] == "lin":
                connection_dim = self.gfi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.lfi_c.append(nn.Sequential(nn.Linear(connection_dim,connection_dim), nn.ReLU()))
                    
        ## lfi
        self.recon_c = nn.ModuleList()
        if self.connect_recon:
            if self.lfi.out_connection_type[0] == "conv":
                connection_channels = self.lfi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.recon_c.append(nn.Sequential(nn.Conv2d(connection_channels,connection_channels, kernel_size=1, stride=1), nn.ReLU()))
            elif self.lfi.out_connection_type[0] == "lin":
                connection_dim = self.lfi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.recon_c.append(nn.Sequential(nn.Linear(connection_dim,connection_dim), nn.ReLU()))

                
        # lf is initialized with parent weights, gf and ef with kaiming, connections with small random weights
        initialization_net = initialization.get_initialization("kaiming_uniform")
        self.efi.apply(initialization_net)
        self.gfi.apply(initialization_net)
        self.lfi.apply(initialization_net)
        initialization_c = initialization.get_initialization("uniform")
        self.gfi_c.apply(initialization_c)
        self.lfi_c.apply(initialization_c)
        self.recon_c.apply(initialization_c)
        
        
    def forward(self, z, ancestors_gfi=None, ancestors_lfi=None, ancestors_recon_x=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(z)
        
        if z.dim() == 2 and type(self).__name__=="DumoulinDecoder": #B*n_latents -> B*n_latents*1*1 
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1) 
        
        # global feature map
        gfi = self.efi(z)
        # loop over the connections
        for gfi_c_idx in range(len(self.gfi_c)): 
            gfi += self.gfi_c[gfi_c_idx](ancestors_gfi[gfi_c_idx])
            
        # local feature map
        lfi = self.gfi(gfi)
        # loop over the connections
        for lfi_c_idx in range(len(self.lfi_c)): 
            lfi += self.lfi_c[lfi_c_idx](ancestors_lfi[lfi_c_idx])
        
        # recon_x
        recon_x = self.lfi(lfi)
        # loop over the connections
        for recon_c_idx in range(len(self.recon_c)): 
            recon_x += self.recon_c[recon_c_idx](ancestors_recon_x[recon_c_idx])
        
        # decoder output
        decoder_outputs = {"z": z, "gfi": gfi, "lfi": lfi, "recon_x": recon_x}
        
        return decoder_outputs
    
    
    def forward_for_graph_tracing(self, z, ancestors_gfi=None, ancestors_lfi=None, ancestors_recon_x=None):
        if z.dim() == 2: #B*n_latents -> B*n_latents*1*1 
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1) 
            
        # global feature map
        gfi = self.efi(z)
        # loop over the connections
        for gfi_c_idx in range(len(self.gfi_c)): 
            gfi += self.lf_c[gfi_c_idx](ancestors_gfi[gfi_c_idx])
            
        # local feature map
        lfi = self.gfi(gfi)
        # loop over the connections
        for lfi_c_idx in range(len(self.lfi_c)): 
            lfi += self.lf_c[lfi_c_idx](ancestors_lfi[lfi_c_idx])
        
        # recon_x
        recon_x = self.lfi(lfi)
        # loop over the connections
        for recon_c_idx in range(len(self.recon_c)): 
            recon_x += self.lf_c[recon_c_idx](ancestors_recon_x[recon_c_idx])
        return recon_x

        
        