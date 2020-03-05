from copy import deepcopy
import goalrepresent as gr
from goalrepresent import dnn, models
from goalrepresent.dnn.solvers import initialization
from goalrepresent.helper import tensorboardhelper
from goalrepresent.helper.misc import do_filter_boolean
import numpy as np
import os
from sklearn import cluster, svm
import sys
import time
import torch
from torch import nn
from torchvision.utils import make_grid
import warnings


# from torchviz import make_dot

class Node(nn.Module):
    """
    Leaf Node base class
    """

    def __init__(self, depth, **kwargs):
        self.depth = depth
        self.leaf = True  # set to Fale when node is split
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
            left_leaf_accumulator = self.left.get_leaf_pathes(path_taken=path_taken + "0")
            self.leaf_accumulator.extend(left_leaf_accumulator)
            right_leaf_accumulator = self.right.get_leaf_pathes(path_taken=path_taken + "1")
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
            left_leaf_accumulator = self.left.get_node_pathes(path_taken=path_taken + "0")
            self.leaf_accumulator.extend(left_leaf_accumulator)
            right_leaf_accumulator = self.right.get_node_pathes(path_taken=path_taken + "1")
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
        self.left = self.NodeClass(self.depth + 1, parent_network=deepcopy(self.network), config=self.config)
        self.right = self.NodeClass(self.depth + 1, parent_network=deepcopy(self.network), config=self.config)

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
        X = z_library - self.feature_range[0]
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(
            scale == 0)] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
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
            # center_0 = np.mean(X[y], axis=0)
            # center_1 = np.mean(X[1 - y], axis=0)
            # self.boundary = cluster.KMeans(n_clusters=2, init=np.stack([center_0,center_1])).fit(X)
            self.boundary = svm.SVC(kernel='linear', C=1.0, class_weight='balanced').fit(X, y)

        #TODO: assert fair partitioning
        return

    def depth_first_forward(self, x, tree_path_taken=None, x_ids=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * len(x)  # all the paths start with "O"
            x_ids = list(range(len(x)))

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, x_ids, node_outputs]])

        else:
            z = node_outputs["z"]
            x_side = self.get_children_node(z)
            x_ids_left = np.where(x_side == 0)[0]
            if len(x_ids_left) > 0:
                left_leaf_accumulator = self.left.depth_first_forward(x[x_ids_left], [path + "0" for path in
                                                                                      [tree_path_taken[x_idx] for x_idx
                                                                                       in x_ids_left]],
                                                                      [x_ids[x_idx] for x_idx in x_ids_left],
                                                                      parent_lf[x_ids_left],
                                                                      parent_gf[x_ids_left],
                                                                      parent_gfi[x_ids_left],
                                                                      parent_lfi[x_ids_left],
                                                                      parent_recon_x[x_ids_left])
                self.leaf_accumulator.extend(left_leaf_accumulator)

            x_ids_right = np.where(x_side == 1)[0]
            if len(x_ids_right) > 0:
                right_leaf_accumulator = self.right.depth_first_forward(x[x_ids_right], [path + "1" for path in
                                                                                         [tree_path_taken[x_idx] for
                                                                                          x_idx in x_ids_right]],
                                                                        [x_ids[x_idx] for x_idx in x_ids_right],
                                                                        parent_lf[x_ids_right],
                                                                        parent_gf[x_ids_right],
                                                                        parent_gfi[x_ids_right],
                                                                        parent_lfi[x_ids_right],
                                                                        parent_recon_x[x_ids_right])
                self.leaf_accumulator.extend(right_leaf_accumulator)

            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def depth_first_forward_all_nodes_preorder(self, x, tree_path_taken=None, x_ids=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * len(x)  # all the paths start with "O"
            x_ids = list(range(len(x)))

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, x_ids, node_outputs]])

        else:
            self.leaf_accumulator.extend([[tree_path_taken, x_ids, node_outputs]])

            z = node_outputs["z"]
            x_side = self.get_children_node(z)
            x_ids_left = np.where(x_side == 0)[0]
            if len(x_ids_left) > 0:
                left_leaf_accumulator = self.left.depth_first_forward_all_nodes_preorder(x[x_ids_left], [path + "0" for path in
                                                                                      [tree_path_taken[x_idx] for x_idx
                                                                                       in x_ids_left]],
                                                                      [x_ids[x_idx] for x_idx in x_ids_left],
                                                                      parent_lf[x_ids_left],
                                                                      parent_gf[x_ids_left],
                                                                      parent_gfi[x_ids_left],
                                                                      parent_lfi[x_ids_left],
                                                                      parent_recon_x[x_ids_left])
                self.leaf_accumulator.extend(left_leaf_accumulator)

            x_ids_right = np.where(x_side == 1)[0]
            if len(x_ids_right) > 0:
                right_leaf_accumulator = self.right.depth_first_forward_all_nodes_preorder(x[x_ids_right], [path + "1" for path in
                                                                                         [tree_path_taken[x_idx] for
                                                                                          x_idx in x_ids_right]],
                                                                        [x_ids[x_idx] for x_idx in x_ids_right],
                                                                        parent_lf[x_ids_right],
                                                                        parent_gf[x_ids_right],
                                                                        parent_gfi[x_ids_right],
                                                                        parent_lfi[x_ids_right],
                                                                        parent_recon_x[x_ids_right])
                self.leaf_accumulator.extend(right_leaf_accumulator)

            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def node_forward(self, x, parent_lf=None, parent_gf=None, parent_gfi=None, parent_lfi=None,
                     parent_recon_x=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x, parent_lf, parent_gf, parent_gfi, parent_lfi,
                                                  parent_recon_x)
        if self.depth == 0:
            encoder_outputs = self.network.encoder(x)
        else:
            encoder_outputs = self.network.encoder(x, parent_lf, parent_gf)
        model_outputs = self.node_forward_from_encoder(encoder_outputs, parent_gfi, parent_lfi, parent_recon_x)
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
            scale[np.where(
                scale == 0)] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
            z = z / scale
            # compute boundary side
            side = self.boundary.predict(z)  # returns 0: left, 1: right
        return side

    def get_children_node(self, z):
        """
        Return code: -1 for leaf node,  0 to send left, 1 to send right
        """
        if self.leaf:
            return -1 * torch.ones_like(z)
        else:
            return self.get_boundary_side(z)


class VAENode(Node, models.VAEModel):
    def __init__(self, depth, parent_network=None, config=None, **kwargs):
        super(Node, self).__init__(config=config, **kwargs)
        super().__init__(depth, **kwargs)

        # connect encoder and decoder
        if self.depth > 0:
            self.network.encoder = ConnectedEncoder(parent_network.encoder, depth, connect_lf=True, connect_gf=False)
            self.network.decoder = ConnectedDecoder(parent_network.decoder, depth, connect_gfi=True, connect_lfi=True,
                                                    connect_recon=True)

        self.set_device(self.config.device.use_gpu)

    def node_forward_from_encoder(self, encoder_outputs, parent_gfi=None, parent_lfi=None,
                                  parent_recon_x=None):
        if self.depth == 0:
            decoder_outputs = self.network.decoder(encoder_outputs["z"])
        else:
            decoder_outputs = self.network.decoder(encoder_outputs["z"], parent_gfi, parent_lfi,
                                                   parent_recon_x)
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

    def __init__(self, config=None, **kwargs):

        self.NodeClass = eval("{}Node".format(config.node_classname))
        self.split_history = {}  # dictionary with node path keys and boundary values

        super().__init__(config=config, **kwargs)

    def set_network(self, network_name, network_parameters):
        depth = 0
        self.network = self.NodeClass(depth, config=self.config.node)  # root node that links to child nodes
        self.network.optimizer_group_id = 0
        self.output_keys_list = self.network.output_keys_list + ["path_taken"]  # node.left is a leaf node

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # give only trainable parameters
        trainable_parameters = [p for p in self.network.parameters() if p.requires_grad]
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(trainable_parameters, **optimizer_parameters)
        self.network.optimizer_group_id = 0
        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

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
                for k, v in cur_node_outputs.items():
                    model_outputs[k] = v
            else:
                model_outputs["path_taken"] += cur_node_path
                for k, v in cur_node_outputs.items():
                    model_outputs[k] = torch.cat([model_outputs[k], v], dim=0)
            # save the sampled ids to reorder as in the input batch at the end
            x_order_ids += list(cur_node_x_ids)

        # reorder points
        sort_order = tuple(np.argsort(x_order_ids))
        for k, v in model_outputs.items():
            if isinstance(v, list):
                model_outputs[k] = [v[i] for i in sort_order]
            else:
                model_outputs[k] = v[sort_order, :]

        return model_outputs

    def calc_embedding(self, x, node_path=None, **kwargs):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        if node_path is None:
            warnings.warn("WARNING: computing the embedding in root node of progressive tree as no path specified")
            node_path = "0"
        n_latents = self.config.network.parameters.n_latents
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
        self.split_history[node_path] = {"depth": node.depth, "leaf": node.leaf, "boundary": node.boundary,
                                         "feature_range": node.feature_range}

        # remove group from optimizer and add one group per children
        del self.optimizer.param_groups[node.optimizer_group_id]
        node.optimizer_group_id = None
        n_groups = len(self.optimizer.param_groups)
        self.optimizer.add_param_group({"params": node.left.parameters()})
        node.left.optimizer_group_id = n_groups
        self.optimizer.add_param_group({"params": node.right.parameters()})
        node.right.optimizer_group_id = n_groups + 1

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

        # Save the graph in the logger
        if logger is not None:
            root_network_config = self.config.node.network.parameters
            dummy_input = torch.FloatTensor(4, root_network_config.n_channels, root_network_config.input_size[0],
                                            root_network_config.input_size[1]).uniform_(0, 1)
            dummy_input = self.push_variable_to_device(dummy_input)
            self.eval()
            # with torch.no_grad():
            #    logger.add_graph(self, dummy_input, verbose = False)

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
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'curent_weight_model.pth'))

            if do_validation:
                t2 = time.time()
                valid_losses = self.valid_epoch(valid_loader, logger=logger)
                t3 = time.time()
                if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in valid_losses.items():
                        logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                    self.n_epochs)

                if valid_losses:
                    valid_loss = valid_losses['total']
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))

    def run_sequential_training(self, train_loader, training_config, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
        # Save the graph in the logger
        if logger is not None:
            root_network_config = self.config.node.network.parameters
            dummy_input = torch.FloatTensor(4, root_network_config.n_channels, root_network_config.input_size[0],
                                            root_network_config.input_size[1]).uniform_(0, 1)
            dummy_input = self.push_variable_to_device(dummy_input)
            self.eval()
            # with torch.no_grad():
            #    logger.add_graph(self, dummy_input, verbose = False)

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        # prepare setting for data distribution and splitting of the hierarchy
        n_episodes = training_config.n_episodes
        episodes_data_filter = training_config.episodes_data_filter
        n_epochs_per_episodes = training_config.n_epochs_per_episode
        blend_between_episodes = training_config.blend_between_episodes
        if "split_trigger" not in training_config:
            training_config.split_trigger = {"active": False}
        split_trigger = training_config.split_trigger
        if split_trigger["active"] and ("n_max_splits" not in split_trigger or split_trigger.n_max_splits > 1e8):
            split_trigger.n_max_splits = 1e8 # maximum splits
        if split_trigger["active"] and ("n_epochs_min_between_splits" not in split_trigger):
            split_trigger.n_epochs_min_between_splits = 10 # a node should at least be trained X epoch before being splitted again
        if split_trigger["active"] and ("min_init_n_epochs" not in split_trigger):
            split_trigger.min_init_n_epochs = 10 # the initial node should at least be trained X epoch before being splitted again
        n_epochs_total = np.asarray(n_epochs_per_episodes).sum()

        # prepare datasets per episodes
        images = [None] * n_episodes
        labels = [None] * n_episodes
        n_images = [None] * n_episodes
        weights = [None] * n_episodes
        for episode_idx in range(n_episodes):
            cur_episode_data_filter = episodes_data_filter[episode_idx]
            cur_episode_n_images = 0
            cum_ratio = 0
            cur_episode_images = None
            cur_episode_labels = None
            for data_filter_idx, data_filter in enumerate(cur_episode_data_filter):
                cur_dataset_name = data_filter["dataset"]
                cur_dataset = train_loader.dataset.datasets[cur_dataset_name]
                cur_filter = data_filter["filter"]
                cur_filtered_inds = do_filter_boolean(cur_dataset, cur_filter)
                cur_ratio = data_filter["ratio"]
                if data_filter_idx == len(cur_episode_data_filter) - 1:
                    if cur_ratio != 1.0 - cum_ratio:
                        raise ValueError("the sum of ratios per dataset in the episode must sum to one")

                cur_n_images = cur_filtered_inds.sum()
                if cur_episode_images is None:
                    cur_episode_images = cur_dataset.images[cur_filtered_inds]
                    cur_episode_labels = cur_dataset.labels[cur_filtered_inds]
                    cur_episode_weights = torch.tensor([cur_ratio / cur_n_images] * cur_n_images, dtype=torch.double)
                else:
                    cur_episode_images = torch.cat([cur_episode_images, cur_dataset.images[cur_filtered_inds]], dim=0)
                    cur_episode_labels = torch.cat([cur_episode_labels, cur_dataset.labels[cur_filtered_inds]], dim=0)
                    cur_episode_weights = torch.cat([cur_episode_weights, torch.tensor([cur_ratio / cur_n_images] * cur_n_images, dtype=torch.double)])
                cur_episode_n_images += cur_n_images
                cum_ratio += cur_ratio

            images[episode_idx] = cur_episode_images
            labels[episode_idx] = cur_episode_labels
            n_images[episode_idx] = cur_episode_n_images
            weights[episode_idx] = cur_episode_weights

        for episode_idx in range(n_episodes):
            train_loader.dataset.update(n_images[episode_idx], images[episode_idx], labels[episode_idx])
            train_loader.sampler.num_samples = len(weights[episode_idx])
            train_loader.sampler.weights = weights[episode_idx]

            for epoch in range(n_epochs_per_episodes[episode_idx]):
                #1) Prepare DataLoader by blending with prev/next episode
                if blend_between_episodes["active"]:
                    cur_epoch_images = deepcopy(images[episode_idx])
                    cur_epoch_labels = deepcopy(labels[episode_idx])
                    # blend images from previous experiment
                    blend_with_prev_episode = blend_between_episodes["blend_with_prev"]
                    if episode_idx > 0 and blend_with_prev_episode["active"]:
                        cur_episode_fraction = float(epoch / n_epochs_per_episodes[episode_idx])
                        if cur_episode_fraction < blend_with_prev_episode["time_fraction"]:
                            if blend_with_prev_episode["blend_type"] == "linear":
                                blend_prev_proportion = - 0.5 / blend_with_prev_episode[
                                    "time_fraction"] * cur_episode_fraction + 0.5
                                n_data_from_prev = int(blend_prev_proportion * n_images[episode_idx])
                                ids_to_take_from_prev_episode = torch.multinomial(weights[episode_idx - 1],
                                                                                  n_data_from_prev, True)
                                ids_to_replace_in_cur_episode = torch.multinomial(weights[episode_idx],
                                                                                  n_data_from_prev, False)
                                for data_idx in range(n_data_from_prev):
                                    cur_epoch_images[ids_to_replace_in_cur_episode[data_idx]] = images[episode_idx - 1][
                                        ids_to_take_from_prev_episode[data_idx]]
                                    cur_epoch_labels[ids_to_replace_in_cur_episode[data_idx]] = labels[episode_idx - 1][
                                        ids_to_take_from_prev_episode[data_idx]]
                            else:
                                raise NotImplementedError("only linear blending is implemented")
                    # blend images from next experiment
                    blend_with_next_episode = blend_between_episodes["blend_with_next"]
                    if episode_idx < n_episodes - 1 and blend_with_next_episode["active"]:
                        cur_episode_fraction = float(epoch / n_epochs_per_episodes[episode_idx])
                        if cur_episode_fraction > (1.0 - blend_with_next_episode["time_fraction"]):
                            if blend_with_next_episode["blend_type"] == "linear":
                                blend_next_proportion = 0.5 / blend_with_prev_episode[
                                    "time_fraction"] * cur_episode_fraction \
                                                        + 0.5 - (0.5 / blend_with_prev_episode["time_fraction"])
                                n_data_from_next = int(blend_next_proportion * n_images[episode_idx])
                                ids_to_take_from_next_episode = torch.multinomial(weights[episode_idx + 1],
                                                                                  n_data_from_next, True)
                                ids_to_replace_in_cur_episode = torch.multinomial(weights[episode_idx],
                                                                                  n_data_from_next, False)
                                for data_idx in range(n_data_from_next):
                                    cur_epoch_images[ids_to_replace_in_cur_episode[data_idx]] = images[episode_idx + 1][
                                        ids_to_take_from_next_episode[data_idx]]
                                    cur_epoch_labels[ids_to_replace_in_cur_episode[data_idx]] = labels[episode_idx + 1][
                                        ids_to_take_from_next_episode[data_idx]]
                            else:
                                raise NotImplementedError("only linear blending is implemented")

                    train_loader.dataset.update(n_images[episode_idx], cur_epoch_images, cur_epoch_labels)

                # 2) check if poor data buffer is filled and split if needed
                if split_trigger is not None and split_trigger["active"]:
                    self.trigger_split(train_loader, split_trigger)
                # Stop splitting when close to the end
                if epoch >= (n_epochs_total - split_trigger.n_epochs_min_between_splits):
                    split_trigger["active"] = False


                # 3) Perform train epoch
                t0 = time.time()
                train_losses = self.train_epoch(train_loader, logger=logger)
                t1 = time.time()
                if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in train_losses.items():
                        logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                    logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                    self.n_epochs)
                if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                    self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'curent_weight_model.pth'))

                # 4) Perform validation epoch
                if do_validation:
                    t2 = time.time()
                    valid_losses = self.valid_epoch(valid_loader, logger=logger)
                    t3 = time.time()
                    if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                        for k, v in valid_losses.items():
                            logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                        logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                        self.n_epochs)

                    if valid_losses:
                        valid_loss = valid_losses['total']
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))

                # 5) Perform evaluation/test epoch
                if self.n_epochs % self.config.evaluation.save_results_every == 0:
                    self.evaluation_epoch(valid_loader)

    def trigger_split(self, train_loader, split_trigger):
        self.eval()
        train_nll = None
        taken_pathes = []
        z_library = None
        x_ids = []

        with torch.no_grad():
            for data in train_loader:
                x = data["obs"]
                x = self.push_variable_to_device(x)
                x_ids.append(data["index"])
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs, reduction="none")
                cur_train_nll = batch_losses["total"]
                # save losses
                if train_nll is None:
                    train_nll = np.expand_dims(cur_train_nll.detach().cpu().numpy(), axis=-1)
                else:
                    train_nll = np.vstack(
                        [train_nll, np.expand_dims(cur_train_nll.detach().cpu().numpy(), axis=-1)])
                # save taken pathes
                taken_pathes += model_outputs["path_taken"]
                # save z library
                if z_library is None:
                    z_library = model_outputs["z"].detach()
                else:
                    z_library = torch.cat([z_library, model_outputs["z"].detach()], dim=0)

        x_ids = torch.cat(x_ids).tolist()

        for leaf_path in list(set(taken_pathes)):
            leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]
            leaf_losses = train_nll[leaf_x_ids, :]
            bad_points_ids = np.where(leaf_losses > split_trigger.loss_threshold)[0]
            n_bad_points = len(bad_points_ids)
            # if poor data buffer filled, check if elligible for split
            if n_bad_points > split_trigger.n_max_bad_points:

                leaf_node = self.network.get_child_node(leaf_path)

                # if elligible for split, trigger a split
                if (len(self.split_history) < split_trigger.n_max_splits) and (
                        self.n_epochs > split_trigger.min_init_n_epochs) and (
                        leaf_node.epochs_since_split > split_trigger.n_epochs_min_between_splits):

                    # dump poor data buffer and z_library for logging
                    if not os.path.exists(os.path.join(self.config.evaluation.folder, "poor_train_data_buffer")):
                        os.makedirs(os.path.join(self.config.evaluation.folder, "poor_train_data_buffer"))

                    # save z_library
                    split_z_library = z_library[leaf_x_ids, :].cpu().numpy()
                    # split_z_fitness = leaf_losses # v1
                    # split_z_fitness = None # v2
                    split_z_fitness = (leaf_losses > split_trigger.loss_threshold) # v3
                    np.savez(os.path.join(self.config.evaluation.folder, "poor_train_data_buffer",
                                          "z_library_when_splitting_{}.npz".format(leaf_path)),
                             **{'z_library': split_z_library, 'z_fitness': split_z_fitness})

                    # save poor data buffer
                    poor_data_buffer = np.empty(
                        (n_bad_points, self.network.config.network.parameters.n_channels,
                         self.network.config.network.parameters.input_size[0],
                         self.network.config.network.parameters.input_size[1]), dtype=np.float)
                    for idx in range(len(bad_points_ids)):
                        poor_data_buffer[idx] = \
                            train_loader.dataset.__getitem__(x_ids[leaf_x_ids[bad_points_ids[idx]]])["obs"]
                    np.save(os.path.join(self.config.evaluation.folder, "poor_train_data_buffer",
                                         "poor_buffer_when_splitting_{}.npy".format(leaf_path)), poor_data_buffer)


                    # Split Node
                    self.split_node(leaf_path, split_z_library, split_z_fitness)

                    # update counters
                    leaf_node.epochs_since_split = 0
                    # uncomment following line for allowing only one split at the time
                    # break;

        return


    def train_epoch(self, train_loader, logger=None):
        self.train()

        taken_pathes = []
        losses = {}

        for data in train_loader:
            x = data['obs']
            x = self.push_variable_to_device(x)
            # forward
            model_outputs = self.forward(x)
            # g = make_dot(model_outputs, params=dict(self.network.named_parameters()))
            # g.view()
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs, reduction="none")
            # backward
            loss = batch_losses['total'].mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis=-1)
                else:
                    losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis=-1)])
            # save taken path
            taken_pathes += model_outputs["path_taken"]


        # update epoch
        for leaf_path in list(set(taken_pathes)):
            leaf_node = self.network.get_child_node(leaf_path)
            if hasattr(leaf_node, "epochs_since_split"):
                leaf_node.epochs_since_split += 1
            else:
                leaf_node.epochs_since_split = 1
        self.n_epochs += 1

        # Logger save results per leaf
        for leaf_path in list(set(taken_pathes)):
            if len(leaf_path) > 1:
                leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]
                for k, v in losses.items():
                    leaf_v = v[leaf_x_ids, :]
                    logger.add_scalars('loss/{}'.format(k), {'train-{}'.format(leaf_path): np.mean(leaf_v)},
                                       self.n_epochs)

        # Average loss on all tree
        for k, v in losses.items():
            losses[k] = np.mean(v)

        return losses

    def valid_epoch(self, valid_loader, logger=None):
        self.eval()
        losses = {}

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        images = None
        recon_images = None
        embeddings = None
        labels = None
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


        taken_pathes = []

        with torch.no_grad():
            for data in valid_loader:
                x = data['obs']
                x = self.push_variable_to_device(x)
                y = data['label'].squeeze()
                y = self.push_variable_to_device(y)
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs, reduction="none")
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis=-1)
                    else:
                        losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis=-1)])
                # record embeddings
                if record_valid_images:
                    for i in range(len(x)):
                        images.append(x[i].unsqueeze(0))
                    for i in range(len(x)):
                        recon_images.append(model_outputs["recon_x"][i].unsqueeze(0))
                if record_embeddings:
                    for i in range(len(x)):
                        embeddings.append(model_outputs["z"][i].unsqueeze(0))
                        labels.append(data['label'][i].unsqueeze(0))
                    if not record_valid_images:
                        for i in range(len(x)):
                            images.append(x[i].unsqueeze(0))

                taken_pathes += model_outputs["path_taken"]


        # 2) LOGGER SAVE RESULT PER LEAF
        for leaf_path in list(set(taken_pathes)):
            leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]

            if record_embeddings:
                leaf_embeddings = torch.cat([embeddings[i] for i in leaf_x_ids])
                leaf_labels = torch.cat([labels[i] for i in leaf_x_ids])
                leaf_images = torch.cat([images[i] for i in leaf_x_ids])
                leaf_images = tensorboardhelper.resize_embeddings(leaf_images)
                try:
                    logger.add_embedding(
                        leaf_embeddings,
                        metadata=leaf_labels,
                        label_img=leaf_images,
                        global_step=self.n_epochs,
                        tag="leaf_{}".format(leaf_path))
                except:
                    pass

            if record_valid_images:
                n_images = min(len(leaf_x_ids), 40)
                sampled_ids = np.random.choice(len(leaf_x_ids), n_images, replace=False)
                input_images = torch.cat([images[i] for i in leaf_x_ids[sampled_ids]]).cpu()
                output_images = torch.cat([recon_images[i] for i in leaf_x_ids[sampled_ids]]).cpu()
                if self.config.loss.parameters.reconstruction_dist == "bernouilli":
                    output_images = torch.sigmoid(output_images)
                vizu_tensor_list = [None] * (2 * n_images)
                vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
                vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
                img = make_grid(vizu_tensor_list, nrow=2, padding=0)
                try:
                    logger.add_image("leaf_{}".format(leaf_path), img, self.n_epochs)
                except:
                    pass

        # 4) AVERAGE LOSS ON WHOLE TREE AND RETURN
        for k, v in losses.items():
            losses[k] = np.mean(v)

        return losses

    def evaluation_epoch(self, test_loader):
        self.eval()
        losses = {}

        n_images = len(test_loader.dataset)
        if "RandomSampler" in test_loader.sampler.__class__.__name__:
            warnings.warn("WARNING: evaluation is performed on shuffled test dataloader")
        tree_depth = 1
        for split_k, split in self.split_history.items():
            if (split["depth"]+2) > tree_depth:
                tree_depth = (split["depth"]+2)

        test_results = {}
        test_results["taken_pathes"] = [None] * n_images
        test_results["labels"] = np.empty(n_images, dtype=np.int)
        test_results["nll"] = np.empty(n_images, dtype=np.float)
        test_results["recon_loss"] = np.empty(n_images, dtype=np.float)
        test_results["cluster_classification_pred"] = np.empty(n_images, dtype=np.int)
        test_results["cluster_classification_acc"] = np.empty(n_images, dtype=np.bool)
        test_results["5NN_classification_pred"] = np.empty(n_images, dtype=np.int)
        test_results["5NN_classification_err"] = np.empty(n_images, dtype=np.bool)
        test_results["10NN_classification_pred"] = np.empty(n_images, dtype=np.int)
        test_results["10NN_classification_err"] = np.empty(n_images, dtype=np.bool)
        test_results["20NN_classification_pred"] = np.empty(n_images, dtype=np.int)
        test_results["20NN_classification_err"] = np.empty(n_images, dtype=np.bool)
        #TODO: "spread" kNN

        test_results_per_node = {}
        for node_path in self.network.get_node_pathes():
            test_results_per_node[node_path] = dict()
            test_results_per_node[node_path]["x_ids"] = []
            test_results_per_node[node_path]["z"] = []
            test_results_per_node[node_path]["recon_x"] = []
            test_results_per_node[node_path]["nll"] = []
            test_results_per_node[node_path]["recon_loss"] = []
            test_results_per_node[node_path]["cluster_classification_pred"] = []
            test_results_per_node[node_path]["cluster_classification_acc"] = []
            test_results_per_node[node_path]["5NN_classification_pred"] = []
            test_results_per_node[node_path]["5NN_classification_err"] = []
            test_results_per_node[node_path]["10NN_classification_pred"] = []
            test_results_per_node[node_path]["10NN_classification_err"] = []
            test_results_per_node[node_path]["20NN_classification_pred"] = []
            test_results_per_node[node_path]["20NN_classification_err"] = []

        with torch.no_grad():
            # Compute results per image
            idx_offset = 0
            for data in test_loader:
                x = data['obs']
                x = self.push_variable_to_device(x)
                cur_x_ids = np.asarray(data["index"], dtype=np.int)
                y = data['label'].squeeze()
                test_results["labels"][cur_x_ids] = y.detach().cpu().numpy()

                # forward
                all_nodes_outputs = self.network.depth_first_forward_all_nodes_preorder(x)
                for node_idx in range(len(all_nodes_outputs)):
                    cur_node_path = all_nodes_outputs[node_idx][0][0]
                    cur_node_x_ids = np.asarray(all_nodes_outputs[node_idx][1], dtype=np.int)
                    cur_node_outputs = all_nodes_outputs[node_idx][2]
                    loss_inputs = {key: cur_node_outputs[key] for key in self.loss_f.input_keys_list}
                    cur_losses = self.loss_f(loss_inputs, reduction="none")
                    test_results_per_node[cur_node_path]["x_ids"] += cur_x_ids[cur_node_x_ids].tolist()
                    test_results_per_node[cur_node_path]["z"] += cur_node_outputs["z"].detach().cpu().numpy().tolist()
                    test_results_per_node[cur_node_path]["recon_x"] += cur_node_outputs["recon_x"].detach().cpu().numpy().tolist()
                    test_results_per_node[cur_node_path]["nll"] += cur_losses["total"].detach().cpu().numpy().tolist()
                    test_results_per_node[cur_node_path]["recon_loss"] += cur_losses["recon"].detach().cpu().numpy().tolist()

                    cur_node = self.network.get_child_node(cur_node_path)
                    if cur_node.leaf:
                        for x_idx in cur_node_x_ids:
                            test_results["taken_pathes"][cur_x_ids[x_idx]] = cur_node_path
                        test_results["nll"][cur_x_ids[cur_node_x_ids]] = cur_losses["total"].detach().cpu().numpy()
                        test_results["recon_loss"][cur_x_ids[cur_node_x_ids]] = cur_losses["recon"].detach().cpu().numpy()

        # compute results for classification
        for node_path in self.network.get_node_pathes():
            # update lists to numpy
            test_results_per_node[node_path]["x_ids"] = np.asarray(test_results_per_node[node_path]["x_ids"], dtype = np.int)
            test_results_per_node[node_path]["z"] = np.asarray(test_results_per_node[node_path]["z"], dtype = np.float)

            # cluster classification accuracy
            if len(test_results_per_node[node_path]["x_ids"]) > 0:
                labels_in_node = test_results["labels"][test_results_per_node[node_path]["x_ids"]]
            else:
                print("break")
                continue
            majority_voted_class = -1
            max_n_votes = 0
            for label in list(set(labels_in_node)):
                label_count = (labels_in_node == label).sum()
                if label_count > max_n_votes:
                    max_n_votes = label_count
                    majority_voted_class = label
            cur_node_x_ids = test_results_per_node[node_path]["x_ids"]
            test_results_per_node[node_path]["cluster_classification_pred"] = [majority_voted_class] * len(cur_node_x_ids)
            test_results_per_node[node_path]["cluster_classification_acc"] = (test_results["labels"][cur_node_x_ids] == majority_voted_class)

            # k-NN classification accuracy
            for x_idx in range(len(cur_node_x_ids)):
                distances_to_point_in_node = np.linalg.norm(test_results_per_node[node_path]["z"][x_idx] - test_results_per_node[node_path]["z"], axis=1)
                closest_point_ids = np.argpartition(distances_to_point_in_node, min(20, len(distances_to_point_in_node)))
                for k_idx, k in enumerate([5,10,20]):
                    voting_labels = test_results["labels"][closest_point_ids[:k]]
                    majority_voted_class = -1
                    max_n_votes = 0
                    for label in list(set(voting_labels)):
                        label_count = (voting_labels == label).sum()
                        if label_count > max_n_votes:
                            max_n_votes = label_count
                            majority_voted_class = label
                    test_results_per_node[node_path]["{}NN_classification_pred".format(k)].append(majority_voted_class)
                    test_results_per_node[node_path]["{}NN_classification_err".format(k)].append(test_results["labels"][x_idx] != majority_voted_class)

            node = self.network.get_child_node(node_path)
            if node.leaf:
                test_results["cluster_classification_pred"][cur_node_x_ids] = test_results_per_node[node_path]["cluster_classification_pred"]
                test_results["cluster_classification_acc"][cur_node_x_ids] = test_results_per_node[node_path]["cluster_classification_acc"]
                test_results["5NN_classification_pred"][cur_node_x_ids] = test_results_per_node[node_path]["5NN_classification_pred"]
                test_results["5NN_classification_err"][cur_node_x_ids] = test_results_per_node[node_path]["5NN_classification_err"]
                test_results["10NN_classification_pred"][cur_node_x_ids] = test_results_per_node[node_path]["10NN_classification_pred"]
                test_results["10NN_classification_err"][cur_node_x_ids] = test_results_per_node[node_path]["10NN_classification_err"]
                test_results["20NN_classification_pred"][cur_node_x_ids] = test_results_per_node[node_path]["20NN_classification_pred"]
                test_results["20NN_classification_err"][cur_node_x_ids] = test_results_per_node[node_path]["20NN_classification_err"]



        # Save results
        if not os.path.exists(os.path.join(self.config.evaluation.folder, "test_results")):
            os.makedirs(os.path.join(self.config.evaluation.folder, "test_results"))
        np.savez(os.path.join(self.config.evaluation.folder, "test_results", "test_results_epoch_{}.npz".format((self.n_epochs))), **test_results)
        return

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
    def __init__(self, encoder_instance, depth, connect_lf=False, connect_gf=False, **kwargs):
        super().__init__(n_channels=encoder_instance.n_channels, input_size=encoder_instance.input_size,
                         n_conv_layers=encoder_instance.n_conv_layers, n_latents=encoder_instance.n_latents,
                         encoder_conditional_type=encoder_instance.conditional_type,
                         feature_layer=encoder_instance.feature_layer,
                         hidden_channels=encoder_instance.hidden_channels, hidden_dim=encoder_instance.hidden_dim)
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
        if self.connect_lf:
            if self.lf.out_connection_type[0] == "conv":
                connection_channels = self.lf.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.lf_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.lf.out_connection_type[0] == "lin":
                connection_dim = self.lf.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.lf_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        ## gf
        if self.connect_gf:
            if self.gf.out_connection_type[0] == "conv":
                connection_channels = self.gf.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.gf_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.gf.out_connection_type[0] == "lin":
                connection_dim = self.gf.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.gf_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        # lf is initialized with parent weights, gf and ef with kaiming, connections with small random weights
        # v1: uncomment the 4 following lines
        #initialization_net = initialization.get_initialization("null")
        #self.gf.apply(initialization_net)
        #self.ef.apply(initialization_net)
        #initialization_c = initialization.get_initialization("connections_identity")
        initialization_c = initialization.get_initialization("null")
        if self.connect_lf:
            self.lf_c.apply(initialization_c)
        if self.connect_gf:
            self.gf_c.apply(initialization_c)

    def forward(self, x, parent_lf=None, parent_gf=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        # local feature map
        lf = self.lf(x)
        # add the connections
        if self.connect_lf:
            lf += self.lf_c(parent_lf)

        # global feature map
        gf = self.gf(lf)
        # add the connections
        if self.connect_gf:
            gf += self.gf_c(parent_gf)

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
            encoder_outputs = {"x": x, "lf": lf, "gf": gf, "z": z}

        return encoder_outputs

    def forward_for_graph_tracing(self, x, parent_lf=None, parent_gf=None):
        # local feature map
        lf = self.lf(x)
        # add the connections
        if self.connect_lf:
            lf += self.lf_c(parent_lf)

        # global feature map
        gf = self.gf(lf)
        # add the connections
        if self.connect_gf:
            gf += self.gf_c(parent_gf)

        if self.conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
        else:
            z = self.ef(gf)
        return z, lf


class ConnectedDecoder(gr.dnn.networks.decoders.BaseDNNDecoder):
    def __init__(self, decoder_instance, depth, connect_gfi=False, connect_lfi=False, connect_recon=False, **kwargs):
        super().__init__(n_channels=decoder_instance.n_channels, input_size=decoder_instance.input_size,
                         n_conv_layers=decoder_instance.n_conv_layers, n_latents=decoder_instance.n_latents,
                         feature_layer=decoder_instance.feature_layer, hidden_channels=decoder_instance.hidden_channels,
                         hidden_dim=decoder_instance.hidden_dim)

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
        if self.connect_gfi:
            if self.efi.out_connection_type[0] == "conv":
                connection_channels = self.efi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.gfi_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.efi.out_connection_type[0] == "lin":
                connection_dim = self.efi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.gfi_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())
        ## lfi
        if self.connect_lfi:
            if self.gfi.out_connection_type[0] == "conv":
                connection_channels = self.gfi.out_connection_type[1]
                self.lfi_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.gfi.out_connection_type[0] == "lin":
                connection_dim = self.gfi.out_connection_type[1]
                self.lfi_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        ## lfi
        if self.connect_recon:
            if self.lfi.out_connection_type[0] == "conv":
                connection_channels = self.lfi.out_connection_type[1]
                for ancestor_depth in range(self.depth):
                    self.recon_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False))
            elif self.lfi.out_connection_type[0] == "lin":
                connection_dim = self.lfi.out_connection_type[1]
                self.recon_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        # lf is initialized with parent weights, gf and ef with kaiming, connections with small random weights
        # v1: uncomment the 5 following lines
        #initialization_net = initialization.get_initialization("null")
        #self.efi.apply(initialization_net)
        #self.gfi.apply(initialization_net)
        #self.lfi.apply(initialization_net)
        #initialization_c = initialization.get_initialization("connections_identity")
        initialization_c = initialization.get_initialization("null")
        if self.connect_gfi:
            self.gfi_c.apply(initialization_c)
        if self.connect_lfi:
            self.lfi_c.apply(initialization_c)
        if self.connect_recon:
            self.recon_c.apply(initialization_c)

    def forward(self, z, parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(z)

        if z.dim() == 2 and type(self).__name__ == "DumoulinDecoder":  # B*n_latents -> B*n_latents*1*1
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # global feature map
        gfi = self.efi(z)
        # add the connections
        if self.connect_gfi:
            gfi += self.gfi_c(parent_gfi)

        # local feature map
        lfi = self.gfi(gfi)
        # add the connections
        if self.connect_lfi:
            lfi += self.lfi_c(parent_lfi)

        # recon_x
        recon_x = self.lfi(lfi)
        # add the connections
        if self.connect_recon:
            recon_x += self.recon_c(parent_recon_x)

        # decoder output
        decoder_outputs = {"z": z, "gfi": gfi, "lfi": lfi, "recon_x": recon_x}

        return decoder_outputs

    def forward_for_graph_tracing(self, z,  parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if z.dim() == 2:  # B*n_latents -> B*n_latents*1*1
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # global feature map
        gfi = self.efi(z)
        # add the connections
        if self.connect_gfi:
            gfi += self.gfi_c(parent_gfi)

        # local feature map
        lfi = self.gfi(gfi)
        # add the connections
        if self.connect_lfi:
            lfi += self.lfi_c(parent_lfi)

        # recon_x
        recon_x = self.lfi(lfi)
        # add the connections
        if self.connect_recon:
            recon_x += self.recon_c(parent_recon_x)

        return recon_x
