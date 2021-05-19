from copy import deepcopy
from addict import Dict
import image_representation
from image_representation import TorchNNRepresentation, VAE, BetaVAE, AnnealedVAE, BetaTCVAE
from image_representation.representations.torch_nn import encoders, decoders
from image_representation.utils.tensorboard_utils import resize_embeddings, logger_add_image_list
from image_representation.utils.torch_nn_init import get_weights_init
import numpy as np
import os
from sklearn import cluster, svm, mixture
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import warnings
# from torchviz import make_dot

class TmpDataset(Dataset):
    @staticmethod
    def default_config():
        default_config = Dict()

        # data info
        default_config.obs_size = ()  # (N,D,H,W) or (N,H,W)
        default_config.obs_dtype = "float32"
        default_config.label_size = (1,)
        default_config.label_dtype = "long"

        return default_config

    def __init__(self, preprocess=None, transform=None, label_transform=None, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        self.preprocess = preprocess
        self.transform = transform
        self.label_transform = label_transform

        self.images = torch.FloatTensor([])  # list or torch tensor of size N*C*H*W
        self.labels = torch.LongTensor([])  # list or torch tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        obs = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            obs = self.transform(obs)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return {"obs": obs, "label": label, "index": idx}

    def update(self, images, labels=None):
        """ Update the current dataset lists """
        n_images = len(images)
        if labels is None:
            labels = torch.Tensor([-1] * n_images).type(self.config.label_dtype)

        if images.dtype != self.config.obs_dtype:
            obs = images.type(eval(f"torch.{self.config.obs_dtype}"))

        if self.preprocess is not None:
            images = self.preprocess(images)

        if labels.dtype != self.config.label_dtype:
            labels = labels.type(eval(f"torch.{self.config.label_dtype}"))

        assert (images.shape[1:] == self.config.obs_size)
        assert (labels.shape[1:] == self.config.label_size)

        self.images = images
        self.labels = labels



class Node(nn.Module):
    """
    Node base class
    """

    def __init__(self, depth, **kwargs):
        self.depth = depth
        self.leaf = True  # set to Fale when node is split
        self.boundary = None
        self.feature_range = None
        self.leaf_accumulator = []
        self.fitness_last_epochs = []

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

    def create_boundary(self, z_library, z_fitness=None, boundary_config=None):
        # normalize z points
        self.feature_range = (z_library.min(axis=0), z_library.max(axis=0))
        X = z_library - self.feature_range[0]
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(scale == 0)[0]] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
        X = X / scale

        default_boundary_config = Dict()
        default_boundary_config.algo = 'svm.SVC'
        default_boundary_config.kwargs = Dict()
        if boundary_config is not None:
            default_boundary_config.update(boundary_config)
        boundary_config = default_boundary_config
        boundary_algo = eval(boundary_config.algo)

        if z_fitness is None:
            if boundary_config.algo == 'cluster.KMeans':
                boundary_config.kwargs.n_clusters = 2
                self.boundary = boundary_algo(**boundary_config.kwargs).fit(X)
        else:
            y = z_fitness.squeeze()
            if boundary_config.algo == 'cluster.KMeans':
                center0 = np.median(X[y <= np.percentile(y, 20), :], axis=0)
                center1 = np.median(X[y > np.percentile(y, 80), :], axis=0)
                center = np.stack([center0, center1])
                center = np.nan_to_num(center)
                boundary_config.kwargs.init = center
                boundary_config.kwargs.n_clusters = 2
                self.boundary = boundary_algo(**boundary_config.kwargs).fit(X)
            elif boundary_config.algo == 'svm.SVC':
                y = y > np.percentile(y, 80)
                self.boundary = boundary_algo(**boundary_config.kwargs).fit(X, y)

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

    def depth_first_forward_whole_branch_preorder(self, x, tree_path_taken=None, x_ids=None, parent_lf=None, parent_gf=None,
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
                left_leaf_accumulator = self.left.depth_first_forward_whole_branch_preorder(x[x_ids_left], [path + "0" for path in
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
                right_leaf_accumulator = self.right.depth_first_forward_whole_branch_preorder(x[x_ids_right], [path + "1" for path in
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


    def depth_first_forward_whole_tree_preorder(self, x, tree_path_taken=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * len(x)  # all the paths start with "O"

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, node_outputs]])

        else:
            self.leaf_accumulator.extend([[tree_path_taken, node_outputs]])
            #send everything left
            left_leaf_accumulator = self.left.depth_first_forward_whole_tree_preorder(x, [path+"0" for path in tree_path_taken],
                                                                  parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)
            self.leaf_accumulator.extend(left_leaf_accumulator)

            #send everything right
            right_leaf_accumulator = self.right.depth_first_forward_whole_tree_preorder(x, [path+"1" for path in tree_path_taken],
                                                                    parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)
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
            # compute boundary side
            if isinstance(z, torch.Tensor):
                z = z.detach().cpu().numpy()

            # normalize
            z = z - self.feature_range[0]
            scale = self.feature_range[1] - self.feature_range[0]
            scale[np.where(scale == 0)[0]] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
            z = z / scale
            side = self.boundary.predict(z)  # returns 0: left, 1: right
        return side

    def get_children_node(self, z):
        """
        Return code: -1 for leaf node,  0 to send left, 1 to send right
        """
        z = z.to(self.config.device)

        if self.leaf:
            return -1 * torch.ones_like(z)
        else:
            return self.get_boundary_side(z)

    def generate_images(self, n_images, from_node_path='', from_side = None, parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        self.eval()
        with torch.no_grad():

            if len(from_node_path) == 0:
                if from_side is None or self.boundary is None:
                    z_gen = torch.randn((n_images, self.n_latents))
                else:
                    desired_side = int(from_side)
                    z_gen = []
                    remaining = n_images
                    max_trials = n_images
                    trials = 0
                    while remaining > 0:
                        cur_z_gen = torch.randn((remaining, self.n_latents))
                        if trials == max_trials:
                            z_gen.append(cur_z_gen)
                            remaining = 0
                            break
                        cur_z_gen_side = self.get_children_node(cur_z_gen)
                        if desired_side == 0:
                            left_side_ids = np.where(cur_z_gen_side == 0)[0]
                            if len(left_side_ids) > 0:
                                z_gen.append(cur_z_gen[left_side_ids[:remaining]])
                                remaining -= len(left_side_ids)
                        elif desired_side == 1:
                            right_side_ids = np.where(cur_z_gen_side == 1)[0]
                            if len(right_side_ids) > 0:
                                z_gen.append(cur_z_gen[right_side_ids[:remaining]])
                                remaining -= len(right_side_ids)
                        else:
                            raise ValueError("wrong path")
                        trials += 1
                    z_gen = torch.cat(z_gen)

            else:
                desired_side = int(from_node_path[0])
                z_gen = []
                remaining = n_images
                max_trials = n_images
                trials = 0
                while remaining > 0:
                    cur_z_gen = torch.randn((remaining, self.n_latents))
                    if trials == max_trials:
                        z_gen.append(cur_z_gen)
                        remaining = 0
                        break
                    cur_z_gen_side = self.get_children_node(cur_z_gen)
                    if desired_side == 0:
                        left_side_ids = np.where(cur_z_gen_side == 0)[0]
                        if len(left_side_ids) > 0:
                            z_gen.append(cur_z_gen[left_side_ids[:remaining]])
                            remaining -= len(left_side_ids)
                    elif desired_side == 1:
                        right_side_ids = np.where(cur_z_gen_side == 1)[0]
                        if len(right_side_ids) > 0:
                            z_gen.append(cur_z_gen[right_side_ids[:remaining]])
                            remaining -= len(right_side_ids)
                    else:
                        raise ValueError("wrong path")
                    trials += 1
                z_gen = torch.cat(z_gen)

            z_gen = z_gen.to(self.config.device).type(self.config.dtype)
            node_outputs = self.node_forward_from_encoder({'z': z_gen}, parent_gfi, parent_lfi, parent_recon_x)
            gfi = node_outputs["gfi"].detach()
            lfi = node_outputs["lfi"].detach()
            recon_x = node_outputs["recon_x"].detach()

            if len(from_node_path) == 0:
                if self.config.loss.parameters.reconstruction_dist == "bernoulli":
                    recon_x = torch.sigmoid(recon_x)
                recon_x = recon_x.detach()
                return recon_x

            else:
                if desired_side == 0:
                    return self.left.generate_images(n_images, from_node_path=from_node_path[1:], from_side = from_side, parent_gfi=gfi, parent_lfi=lfi, parent_recon_x=recon_x)
                elif desired_side == 1:
                    return self.right.generate_images(n_images, from_node_path=from_node_path[1:], from_side = from_side, parent_gfi=gfi, parent_lfi=lfi, parent_recon_x=recon_x)


def get_node_class(base_class):

    class NodeClass(Node, base_class):
        def __init__(self, depth, parent_network=None, config=None, **kwargs):
            base_class.__init__(self, config=config, **kwargs)
            Node.__init__(self, depth, **kwargs)

            # connect encoder and decoder
            if self.depth > 0:
                self.network.encoder = ConnectedEncoder(parent_network.encoder, depth, connect_lf=config.create_connections["lf"],
                                                        connect_gf=config.create_connections["gf"])
                self.network.decoder = ConnectedDecoder(parent_network.decoder, depth, connect_gfi=config.create_connections["gfi"],
                                                        connect_lfi=config.create_connections["lfi"],
                                                        connect_recon=config.create_connections["recon"])

            self.set_device(self.config.device)
            self.set_dtype(self.config.dtype)

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

    return NodeClass

# possible node classes:
VAENode_local = get_node_class(VAE)
VAENode = type('VAENode', (Node, VAE), dict(VAENode_local.__dict__))

BetaVAENode_local = get_node_class(BetaVAE)
BetaVAENode = type('BetaVAENode', (Node, BetaVAE), dict(BetaVAENode_local.__dict__))

AnnealedVAENode_local = get_node_class(AnnealedVAE)
AnnealedVAENode = type('AnnealedVAENode', (Node, AnnealedVAE), dict(AnnealedVAENode_local.__dict__))

BetaTCVAENode_local = get_node_class(BetaTCVAE)
BetaTCVAENode = type('BetaTCVAENode', (Node, BetaTCVAE), dict(BetaTCVAENode_local.__dict__))


class HOLMES_VAE(TorchNNRepresentation):
    """
    dnn with tree network, loss which is based on leaf's losses, optimizer from that loss
    """

    @staticmethod
    def default_config():
        default_config = TorchNNRepresentation.default_config()

        # tree config
        default_config.tree = Dict()

        # node config
        default_config.node_classname = "VAE"
        default_config.node = eval(f"image_representation.{default_config.node_classname}.default_config()")
        default_config.node.create_connections = {"lf": True, "gf": False, "gfi": True, "lfi": True, "recon": True}

        # loss parameters
        default_config.loss.name = "VAE"

        # optimizer
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        self.NodeClass = eval("{}Node".format(self.config.node_classname))
        self.split_history = {}  # dictionary with node path keys and boundary values

        TorchNNRepresentation.__init__(self, config=config, **kwargs)

    def set_network(self, network_name, network_parameters):
        depth = 0
        self.network = self.NodeClass(depth, config=self.config.node)  # root node that links to child nodes
        self.n_latents = self.network.network.encoder.config.n_latents
        self.network.optimizer_group_id = 0
        self.output_keys_list = self.network.output_keys_list + ["path_taken"]  # node.left is a leaf node

        # update config
        self.config.network.name = network_name
        self.config.network.parameters.update(network_parameters)

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # give only trainable parameters
        trainable_parameters = [p for p in self.network.parameters() if p.requires_grad]
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(trainable_parameters, **optimizer_parameters)
        self.network.optimizer_group_id = 0
        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters.update(optimizer_parameters)

    def forward_for_graph_tracing(self, x):
        pass

    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)
        x = x.to(self.config.device)
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

    def forward_through_given_path(self, x, tree_desired_path):
        x = x.to(self.config.device)
        is_train = self.network.training
        if len(x) == 1 and is_train:
            self.network.eval()
            depth_first_traversal_outputs = self.network.depth_first_forward_through_given_path(x, tree_desired_path)
            self.network.train()
        else:
            depth_first_traversal_outputs = self.network.depth_first_forward_through_given_path(x, tree_desired_path)

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

    def calc_embedding(self, x, mode="niche", node_path=None, **kwargs):
        """
        :param mode: either "exhaustif" or "niche"
        :param node_path: if "niche" must specify niche's node_path
        :return:
        """
        x = x.to(self.config.device).type(self.config.dtype)

        if mode == "niche":
            z = torch.Tensor().new_full((len(x), self.n_latents), float("nan")).to(self.config.device)
            all_nodes_outputs = self.network.depth_first_forward_whole_branch_preorder(x)
            if node_path is None:
                node_path = "0"
                warnings.warn("Node path not specified... setting it to root node 0.")

        elif mode == "exhaustif":
            z = Dict.fromkeys(self.network.get_node_pathes())
            all_nodes_outputs = self.network.depth_first_forward_whole_tree_preorder(x)

        for node_idx in range(len(all_nodes_outputs)):
            cur_node_path = all_nodes_outputs[node_idx][0][0]
            if mode == "niche" and cur_node_path != node_path:
                continue
            elif mode == "niche" and cur_node_path == node_path:
                cur_node_x_ids = all_nodes_outputs[node_idx][1]
                cur_node_outputs = all_nodes_outputs[node_idx][2]
                for idx in range(len(cur_node_x_ids)):
                    z[cur_node_x_ids[idx]] = cur_node_outputs["z"][idx]
                break
            else:
                cur_node_outputs = all_nodes_outputs[node_idx][1]
                z[cur_node_path] = cur_node_outputs["z"]
        return z




    def split_node(self, node_path, split_trigger=None, x_loader=None, x_fitness=None):
        """
        z_library: n_samples * n_latents
        z_fitness: n_samples * 1 (eg: reconstruction loss)
        """
        node = self.network.get_child_node(node_path)
        node.NodeClass = type(node)

        # save model
        if split_trigger is not None and (split_trigger.save_model_before_after or split_trigger.save_model_before_after == 'before'):
            self.save(os.path.join(self.config.checkpoint.folder,
                                              'weight_model_before_split_{}_node_{}_epoch_{}.pth'.format(
                                                  len(self.split_history)+1, node_path, self.n_epochs)))

        # (optional) Train for X epoch parent with new data (+ replay optional)
        if x_loader is not None and split_trigger.n_epochs_before_split > 0:
            for epoch_before_split in range(split_trigger.n_epochs_before_split):
                _ = self.train_epoch(x_loader)

        self.eval()
        # Create boundary
        if x_loader is not None:
            with torch.no_grad():
                z_library = self.calc_embedding(x_loader.dataset.images, node_path=node_path).detach()
                if split_trigger.boundary_config.z_fitness is None:
                    z_fitness = None
                else:
                    z_fitness = x_fitness
                if torch.isnan(z_library).any():
                    keep_ids = ~(torch.isnan(z_library.sum(1)))
                    z_library = z_library[keep_ids]
                    if z_fitness is not None:
                        x_fitness = x_fitness[keep_ids]
                        z_fitness = x_fitness
                if z_library.shape[0] == 0:
                    z_library = torch.zeros((2, self.n_latents))
                    if z_fitness is not None:
                        z_fitness = np.zeros(2)
                node.create_boundary(z_library.cpu().numpy(), z_fitness, boundary_config=split_trigger.boundary_config)

        # Instanciate childrens
        node.left = node.NodeClass(node.depth + 1, parent_network=deepcopy(node.network), config=node.config)
        node.right = node.NodeClass(node.depth + 1, parent_network=deepcopy(node.network), config=node.config)
        node.leaf = False

        # Freeze parent parameters
        for param in node.network.parameters():
            param.requires_grad = False

        # Update optimize
        cur_node_optimizer_group_id = node.optimizer_group_id
        # delete param groups and residuates in optimize.state
        del self.optimizer.param_groups[cur_node_optimizer_group_id]
        for n, p in node.named_parameters():
            if p in self.optimizer.state.keys():
                del self.optimizer.state[p]
        node.optimizer_group_id = None
        n_groups = len(self.optimizer.param_groups)
        # update optimizer_group ids in the tree and sanity check that there is no conflict
        sanity_check = np.asarray([False] * n_groups)
        for leaf_path in self.network.get_leaf_pathes():
            if leaf_path[:len(node_path)] != node_path:
                other_leaf = self.network.get_child_node(leaf_path)
                if other_leaf.optimizer_group_id > cur_node_optimizer_group_id:
                    other_leaf.optimizer_group_id -= 1
                if sanity_check[other_leaf.optimizer_group_id] == False:
                    sanity_check[other_leaf.optimizer_group_id] = True
                else:
                    raise ValueError("doublons in the optimizer group ids")
        if (n_groups > 0) and (~sanity_check).any():
            raise ValueError("optimizer group ids does not match the optimzer param groups length")
        self.optimizer.add_param_group({"params": node.left.parameters()})
        node.left.optimizer_group_id = n_groups
        self.optimizer.add_param_group({"params": node.right.parameters()})
        node.right.optimizer_group_id = n_groups + 1


        # save split history
        self.split_history[node_path] = {"depth": node.depth, "leaf": node.leaf, "boundary": node.boundary,
                                         "feature_range": node.feature_range, "epoch": self.n_epochs}
        if x_loader is not None:
            self.save_split_history(node_path, x_loader, z_library, x_fitness, split_trigger)

        # save model
        if split_trigger is not None and (split_trigger.save_model_before_after or split_trigger.save_model_before_after == 'after'):
            self.save(os.path.join(self.config.checkpoint.folder, 'weight_model_after_split_{}_node_{}_epoch_{}.pth'.format(len(self.split_history), node_path, self.n_epochs)))

        return


    def save_split_history(self, node_path, x_loader, z_library, z_fitness, split_trigger):
        # save results
        title = f"split_{len(self.split_history)}_node_{node_path}_epoch_{self.n_epochs}"

        ## poor data buffer
        if split_trigger.type == "threshold":
            poor_data_buffer = x_loader.dataset.images[np.where(z_fitness > split_trigger.parameters.threshold)[0]]
            logger_add_image_list(self.logger, poor_data_buffer, title + "/poor_data",
                                  n_channels=self.config.node.network.parameters.n_channels,
                                  spatial_dims=len(self.config.node.network.parameters.input_size))

        ## left/right samples from which boundary is fitted
        if split_trigger.boundary_config.z_fitness is not None:
            y_fit = z_fitness > np.percentile(z_fitness, 80)
            samples_left_fit_ids = np.where(y_fit == 0)[0]
            samples_left_fit_ids = np.random.choice(samples_left_fit_ids, min(100, len(samples_left_fit_ids)))
            samples_left_fit_buffer = x_loader.dataset.images[samples_left_fit_ids]
            logger_add_image_list(self.logger, samples_left_fit_buffer, title + "/samples_left_fit",
                                  n_channels=self.config.node.network.parameters.n_channels,
                                  spatial_dims=len(self.config.node.network.parameters.input_size))

            samples_right_fit_ids = np.where(y_fit == 1)[0]
            samples_right_fit_ids = np.random.choice(samples_right_fit_ids, min(100, len(samples_right_fit_ids)))
            samples_right_fit_buffer = x_loader.dataset.images[samples_right_fit_ids]
            logger_add_image_list(self.logger, samples_right_fit_buffer, title + "/samples_right_fit",
                                  n_channels=self.config.node.network.parameters.n_channels,
                                  spatial_dims=len(self.config.node.network.parameters.input_size))

            ## wrongly classified buffer make grid
            node = self.network.get_child_node(node_path)
            y_predicted = node.get_children_node(z_library)
            wrongly_sent_left_ids = np.where((y_predicted == 0) & (y_predicted != y_fit))[0]
            if len(wrongly_sent_left_ids) > 0:
                wrongly_sent_left_ids = np.random.choice(wrongly_sent_left_ids, min(100, len(wrongly_sent_left_ids)))
                wrongly_sent_left_buffer = x_loader.dataset.images[wrongly_sent_left_ids]
                logger_add_image_list(self.logger, wrongly_sent_left_buffer, title + "/wrongly_sent_left",
                                      n_channels=self.config.node.network.parameters.n_channels,
                                      spatial_dims=len(self.config.node.network.parameters.input_size))

            wrongly_sent_right_ids = np.where((y_predicted == 1) & (y_predicted != y_fit))[0]
            if len(wrongly_sent_right_ids) > 0:
                wrongly_sent_right_ids = np.random.choice(wrongly_sent_right_ids, min(100, len(wrongly_sent_right_ids)))
                wrongly_sent_right_buffer = x_loader.dataset.images[wrongly_sent_right_ids]
                logger_add_image_list(self.logger, wrongly_sent_right_buffer, title + "/wrongly_sent_right",
                                      n_channels=self.config.node.network.parameters.n_channels,
                                      spatial_dims=len(self.config.node.network.parameters.input_size))


        ## left and right side generated samples
        samples_left_gen_buffer = self.network.generate_images(100, from_node_path=node_path[1:], from_side='0')
        logger_add_image_list(self.logger, samples_left_gen_buffer, title + "/samples_gen_sent_left",
                              n_channels=self.config.node.network.parameters.n_channels,
                              spatial_dims=len(self.config.node.network.parameters.input_size))

        samples_right_gen_buffer = self.network.generate_images(100, from_node_path=node_path[1:], from_side='1')
        logger_add_image_list(self.logger, samples_right_gen_buffer, title + "/samples_gen_sent_right",
                              n_channels=self.config.node.network.parameters.n_channels,
                              spatial_dims=len(self.config.node.network.parameters.input_size))

        return



    def run_training(self, train_loader, training_config, valid_loader=None):

        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

        # Save the graph in the logger
        if self.logger is not None:
            dummy_size = (1, self.config.node.network.parameters.n_channels,) + self.config.node.network.parameters.input_size
            dummy_input = torch.FloatTensor(size=dummy_size).uniform_(0, 1)
            dummy_input = dummy_input.to(self.config.device)
            self.eval()
            # with torch.no_grad():
            #    self.logger.add_graph(self, dummy_input, verbose = False)

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        # Prepare settings for the training of HOLMES
        ## split trigger settings
        split_trigger = Dict()
        split_trigger.active = False
        split_trigger.update(training_config.split_trigger)
        if split_trigger["active"]:
            default_split_trigger = Dict()
            ## conditions
            default_split_trigger.conditions = Dict()
            default_split_trigger.conditions.min_init_n_epochs = 1
            default_split_trigger.conditions.n_epochs_min_between_splits = 1
            default_split_trigger.conditions.n_min_points = 0
            default_split_trigger.conditions.n_max_splits = 1e8
            ## fitness
            default_split_trigger.fitness_key = "total"
            ## type
            default_split_trigger.type = "plateau"
            if default_split_trigger.type == "plateau":
                default_split_trigger.parameters.epsilon = 1
                default_split_trigger.parameters.n_steps_average = 5
            elif default_split_trigger.type == "threshold":
                default_split_trigger.parameters.threshold = 200
                default_split_trigger.parameters.n_max_bad_points = 100
            ## train before split
            default_split_trigger.n_epochs_before_split = 0
            ## boundary config
            default_split_trigger.boundary_config = Dict()
            default_split_trigger.boundary_config.z_fitness = None
            default_split_trigger.boundary_config.algo = "cluster.KMeans"
            ## save
            default_split_trigger.save_model_before_after = False

            default_split_trigger.update(split_trigger)
            split_trigger = default_split_trigger


        ## alternated training
        alternated_backward = Dict()
        alternated_backward.active = False
        alternated_backward.update(training_config.alternated_backward)
        if alternated_backward["active"]:
            default_ratio_epochs = Dict({"connections": 1, "core": 9})
            default_ratio_epochs.update(alternated_backward.ratio_epochs)
            alternated_backward.ratio_epochs = default_ratio_epochs


        for epoch in range(training_config.n_epochs):
            # 1) check if elligible for split
            if split_trigger["active"]:
                self.trigger_split(train_loader, split_trigger)

            # 2) perform train epoch
            if (len(self.split_history) > 0) and (alternated_backward["active"]):
                if (self.n_epochs % int(alternated_backward["ratio_epochs"]["connections"]+alternated_backward["ratio_epochs"]["core"])) < alternated_backward["ratio_epochs"]["connections"]:
                    # train only connections
                    for leaf_path in self.network.get_leaf_pathes():
                        leaf_node = self.network.get_child_node(leaf_path)
                        for n, p in leaf_node.network.named_parameters():
                            if "_c" not in n:
                                p.requires_grad = False
                            else:
                                p.requires_grad = True
                else:
                    # train only children module without connections
                    for leaf_path in self.network.get_leaf_pathes():
                        leaf_node = self.network.get_child_node(leaf_path)
                        for n, p in leaf_node.network.named_parameters():
                            if "_c" not in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

            t0 = time.time()
            train_losses = self.train_epoch(train_loader)
            t1 = time.time()

            # update epoch counters
            self.n_epochs += 1
            # for leaf_path in self.network.get_leaf_pathes():
            #     leaf_node = self.network.get_child_node(leaf_path)
            #     if hasattr(leaf_node, "epochs_since_split"):
            #         leaf_node.epochs_since_split += 1
            #     else:
            #         leaf_node.epochs_since_split = 1

            # log
            if self.logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                for k, v in train_losses.items():
                    self.logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                self.logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                self.n_epochs)

            # save model
            if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                self.save(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))
            if self.n_epochs in self.config.checkpoint.save_model_at_epochs:
                self.save(os.path.join(self.config.checkpoint.folder, "epoch_{}_weight_model.pth".format(self.n_epochs)))

            if do_validation:
                # 3) Perform evaluation
                t2 = time.time()
                valid_losses = self.valid_epoch(valid_loader)
                t3 = time.time()
                if self.logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in valid_losses.items():
                        self.logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    self.logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                    self.n_epochs)

                if valid_losses:
                    valid_loss = valid_losses['total']
                    if valid_loss < best_valid_loss and self.config.checkpoint.save_best_model:
                        best_valid_loss = valid_loss
                        self.save(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))

            # 5) Stop splitting when close to the end
            # if split_trigger["active"] and (self.n_epochs >= (training_config.n_epochs - split_trigger.conditions.n_epochs_min_between_splits)):
            #     split_trigger["active"] = False


    def trigger_split(self, train_loader, split_trigger):

        splitted_leafs = []

        if (len(self.split_history) > split_trigger.conditions.n_max_splits) or (
                self.n_epochs < split_trigger.conditions.min_init_n_epochs):
            return

        self.eval()
        train_fitness = None
        taken_pathes = []
        x_ids = []
        labels = []

        old_transform_state = train_loader.dataset.transform
        train_loader.dataset.transform = None

        with torch.no_grad():
            for data in train_loader:
                x = data["obs"].to(self.config.device).type(self.config.dtype)
                x_ids.append(data["index"])
                labels.append(data["label"])
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs, reduction="none")
                cur_train_fitness = batch_losses[split_trigger.fitness_key]
                # save losses
                if train_fitness is None:
                    train_fitness = np.expand_dims(cur_train_fitness.detach().cpu().numpy(), axis=-1)
                else:
                    train_fitness = np.vstack(
                        [train_fitness, np.expand_dims(cur_train_fitness.detach().cpu().numpy(), axis=-1)])
                # save taken pathes
                taken_pathes += model_outputs["path_taken"]

            x_ids = torch.cat(x_ids)
            labels = torch.cat(labels)

        for leaf_path in list(set(taken_pathes)):
            leaf_node = self.network.get_child_node(leaf_path)
            leaf_x_ids = (np.array(taken_pathes, copy=False) == leaf_path)
            generated_ids_in_leaf_x_ids = (np.asarray(labels[leaf_x_ids]).squeeze() == -1)
            leaf_n_real_points = leaf_x_ids.sum() - generated_ids_in_leaf_x_ids.sum()
            split_x_fitness = train_fitness[leaf_x_ids, :]
            split_x_fitness[generated_ids_in_leaf_x_ids] = 0
            leaf_node.fitness_last_epochs.append(split_x_fitness[~generated_ids_in_leaf_x_ids].mean())

            if leaf_path == "0":
                n_epochs_since_split = self.n_epochs
            else:
                n_epochs_since_split = (self.n_epochs - self.split_history[leaf_path[:-1]]["epoch"])
            if (n_epochs_since_split < split_trigger.conditions.n_epochs_min_between_splits) or \
                    (leaf_n_real_points < split_trigger.conditions.n_min_points):
                continue

            trigger_split_in_leaf = False
            if split_trigger.type == "threshold":
                poor_buffer = (split_x_fitness > split_trigger.parameters.threshold).squeeze()
                if (poor_buffer.sum() > split_trigger.parameters.n_max_bad_points):
                    trigger_split_in_leaf = True

            elif split_trigger.type == "plateau":
                if len(leaf_node.fitness_last_epochs) > split_trigger.parameters.n_steps_average:
                    leaf_node.fitness_last_epochs.pop(0)
                fitness_vals = np.asarray(leaf_node.fitness_last_epochs)
                fitness_speed_last_epochs = fitness_vals[1:] - fitness_vals[:-1]
                running_average_speed = np.abs(fitness_speed_last_epochs.mean())
                if (running_average_speed < split_trigger.parameters.epsilon):
                    trigger_split_in_leaf = True

            if trigger_split_in_leaf:
                # Split Node
                split_dataset = TmpDataset(preprocess=train_loader.dataset.preprocess, transform=None, target_transform=None, config=train_loader.dataset.config)

                split_x_ids = x_ids[leaf_x_ids]
                n_seen_images = len(split_x_ids)
                split_seen_images = torch.empty((0, ) + train_loader.dataset.config.obs_size, dtype=eval(f"torch.{train_loader.dataset.config.obs_dtype}"))
                split_seen_labels = torch.empty((0, ) + train_loader.dataset.config.label_size, dtype=eval(f"torch.{train_loader.dataset.config.label_dtype}"))
                for x_idx in split_x_ids:
                    data = train_loader.dataset.get_data(int(x_idx.item()))
                    split_seen_images = torch.cat([split_seen_images, data["obs"].unsqueeze(0)])
                    split_seen_labels = torch.cat([split_seen_labels, data["label"].unsqueeze(0)])

                split_dataset.update(split_seen_images, split_seen_labels)
                split_loader = DataLoader(split_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=0)

                self.split_node(leaf_path,  split_trigger, x_loader=split_loader, x_fitness=split_x_fitness)

                del split_seen_images, split_seen_labels, split_dataset, split_loader
                # update counters
                #leaf_node.epochs_since_split = None
                #leaf_node.fitness_speed_last_epoches = []

                splitted_leafs.append(leaf_path)

                # uncomment following line for allowing only one split at the time
                # break

        train_loader.dataset.transform = old_transform_state
        return splitted_leafs


    def train_epoch(self, train_loader):
        self.train()

        taken_pathes = []
        losses = {}

        for data in train_loader:
            x = data['obs'].to(self.config.device).type(self.config.dtype)
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


        # Logger save results per leaf
        if self.logger is not None:
            for leaf_path in list(set(taken_pathes)):
                if len(leaf_path) > 1:
                    leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]
                    for k, v in losses.items():
                        leaf_v = v[leaf_x_ids, :]
                        self.logger.add_scalars('loss/{}'.format(k), {'train-{}'.format(leaf_path): np.mean(leaf_v)},
                                           self.n_epochs)

        # Average loss on all tree
        for k, v in losses.items():
            losses[k] = np.mean(v)

        return losses

    def valid_epoch(self, valid_loader):
        self.eval()
        losses = {}

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        images = None
        recon_images = None
        embeddings = None
        labels = None
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


        taken_pathes = []

        with torch.no_grad():
            for data in valid_loader:
                x = data['obs'].to(self.config.device).type(self.config.dtype)
                y = data['label'].squeeze().to(self.config.device)
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
                        #if len(images) < self.config.logging.record_embeddings_max:
                        images.append(x[i].unsqueeze(0))
                    for i in range(len(x)):
                        #if len(recon_images) < self.config.logging.record_valid_images_max:
                        recon_images.append(model_outputs["recon_x"][i].unsqueeze(0))
                if record_embeddings:
                    for i in range(len(x)):
                        #if len(embeddings) < self.config.logging.record_embeddings_max:
                        embeddings.append(model_outputs["z"][i].unsqueeze(0))
                        labels.append(data['label'][i].unsqueeze(0))
                    if not record_valid_images:
                        for i in range(len(x)):
                            #if len(images) < self.config.logging.record_embeddings_max:
                            images.append(x[i].unsqueeze(0))

                taken_pathes += model_outputs["path_taken"]


        # 2) LOGGER SAVE RESULT PER LEAF
        for leaf_path in list(set(taken_pathes)):
            leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]

            if record_embeddings:
                leaf_embeddings = torch.cat([embeddings[i] for i in leaf_x_ids])
                leaf_labels = torch.cat([labels[i] for i in leaf_x_ids])
                leaf_images = torch.cat([images[i] for i in leaf_x_ids])
                if len(leaf_images.shape) == 5:
                    leaf_images = leaf_images[:, :, self.config.node.network.parameters.input_size[0] // 2, :, :]  # we take slice at middle depth only
                if (leaf_images.shape[1] != 1) or (leaf_images.shape[1] != 3):
                    leaf_images = leaf_images[:, :3, ...]
                leaf_images = resize_embeddings(leaf_images)
                try:
                    self.logger.add_embedding(
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
                if self.config.loss.parameters.reconstruction_dist == "bernoulli":
                    output_images = torch.sigmoid(output_images)
                vizu_tensor_list = [None] * (2 * n_images)
                vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
                vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
                logger_add_image_list(self.logger, vizu_tensor_list, f"leaf_{leaf_path}/reconstructions",
                                      global_step=self.n_epochs, n_channels=self.config.node.network.parameters.n_channels,
                                      spatial_dims=len(self.config.node.network.parameters.input_size))

        # 4) AVERAGE LOSS ON WHOLE TREE AND RETURN
        for k, v in losses.items():
            losses[k] = np.mean(v)

        return losses

    def save(self, filepath='representation.pickle'):
        # save current epoch weight file with optimizer if we want to relaunch training from that point
        network = {
            "epoch": self.n_epochs,
            "type": self.__class__.__name__,
            "config": self.config,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "split_history": self.split_history
        }

        torch.save(network, filepath)

    def get_checkpoint(self):
        checkpoint = {
            "epoch": self.n_epochs,
            "type": self.__class__.__name__,
            "config": self.config,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "split_history": self.split_history
        }
        return checkpoint

    @staticmethod
    def load(filepath='representation.pickle', map_location='cpu'):
        saved_representation = torch.load(filepath, map_location=map_location)
        representation_type = saved_representation['type']
        representation_cls = getattr(image_representation, representation_type)
        representation_config = saved_representation['config']
        representation_config.device = map_location
        representation = representation_cls(config=representation_config)
        representation.n_epochs = saved_representation["epoch"]

        split_history = saved_representation['split_history']

        for split_node_path, split_node_attr in split_history.items():
            representation.split_node(split_node_path)
            node = representation.network.get_child_node(split_node_path)
            node.boundary = split_node_attr["boundary"]
            node.feature_range = split_node_attr["feature_range"]

        representation.set_device(map_location)
        representation.network.load_state_dict(saved_representation['network_state_dict'])
        representation.optimizer.load_state_dict(saved_representation['optimizer_state_dict'])




""" =========================================================================================
CONNECTED MODULES 
==========================================================================================="""


class ConnectedEncoder(encoders.Encoder):
    def __init__(self, encoder_instance, depth, connect_lf=False, connect_gf=False, **kwargs):
        encoders.Encoder.__init__(self, config=encoder_instance.config)
        # connections and depth in the tree (number of connections)
        self.connect_lf = connect_lf
        self.connect_gf = connect_gf
        self.depth = depth

        # copy parent network layers
        self.lf = encoder_instance.lf
        self.gf = encoder_instance.gf
        self.ef = encoder_instance.ef

        # add lateral connections
        self.spatial_dims = encoder_instance.spatial_dims
        if self.spatial_dims == 2:
            self.conv_module = nn.Conv2d
        elif self.spatial_dims == 3:
            self.conv_module = nn.Conv3d
        ## lf
        if self.connect_lf:
            if self.lf.out_connection_type[0] == "conv":
                connection_channels = self.lf.out_connection_type[1]
                self.lf_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.lf.out_connection_type[0] == "lin":
                connection_dim = self.lf.out_connection_type[1]
                self.lf_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        ## gf
        if self.connect_gf:
            if self.gf.out_connection_type[0] == "conv":
                connection_channels = self.gf.out_connection_type[1]
                self.gf_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.gf.out_connection_type[0] == "lin":
                connection_dim = self.gf.out_connection_type[1]
                self.gf_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        initialization_net = get_weights_init("kaiming_uniform")
        self.gf.apply(initialization_net)
        self.ef.apply(initialization_net)
        initialization_c = get_weights_init("uniform")
        if self.connect_lf:
            self.lf_c.apply(initialization_c)
        if self.connect_gf:
            self.gf_c.apply(initialization_c)

    """
        #initialization_net = initialization.get_initialization("null")
        #self.gf.apply(initialization_net)
        #self.ef.apply(initialization_net)
        #initialization_c = initialization.get_initialization("connections_identity")
        initialization_c = initialization.get_initialization("null")
        if self.connect_lf:
            self.lf_c.apply(initialization_c)
        if self.connect_gf:
            self.gf_c.apply(initialization_c)
        """

    def forward(self, x, parent_lf=None, parent_gf=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        # batch norm cannot deal with batch_size 1 in train mode
        was_training = None
        if self.training and x.size()[0] == 1:
            self.eval()
            was_training = True


        # local feature map
        lf = self.lf(x)
        # add the connections
        if self.connect_lf:
            lf = lf + self.lf_c(parent_lf)

        # global feature map
        gf = self.gf(lf)
        # add the connections
        if self.connect_gf:
            gf = gf + self.gf_c(parent_gf)

        # encoding
        if self.config.encoder_conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
            if z.ndim > 2:
                for _ in range(self.spatial_dims):
                    mu = mu.squeeze(-1)
                    logvar = logvar.squeeze(-1)
                    z = z.squeeze(-1)
            encoder_outputs = {"x": x, "lf": lf, "gf": gf, "z": z, "mu": mu, "logvar": logvar}
        elif self.config.encoder_conditional_type == "deterministic":
            z = self.ef(gf)
            if z.ndim > 2:
                for _ in range(self.spatial_dims):
                    z = z.squeeze(-1)
            encoder_outputs = {"x": x, "lf": lf, "gf": gf, "z": z}

        if was_training and x.size()[0] == 1:
            self.train()

        return encoder_outputs

    def forward_for_graph_tracing(self, x, parent_lf=None, parent_gf=None):
        # local feature map
        lf = self.lf(x)
        # add the connections
        if self.connect_lf:
            lf = lf + self.lf_c(parent_lf)

        # global feature map
        gf = self.gf(lf)
        # add the connections
        if self.connect_gf:
            gf = gf + self.gf_c(parent_gf)

        if self.config.encoder_conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
        else:
            z = self.ef(gf)
        return z, lf


class ConnectedDecoder(decoders.Decoder):
    def __init__(self, decoder_instance, depth, connect_gfi=False, connect_lfi=False, connect_recon=False, **kwargs):
        decoders.Decoder.__init__(self, config=decoder_instance.config)

        # connections and depth in the tree (number of connections)
        self.connect_gfi = connect_gfi
        self.connect_lfi = connect_lfi
        self.connect_recon = connect_recon
        self.depth = depth

        # copy parent network layers
        self.efi = decoder_instance.efi
        self.gfi = decoder_instance.gfi
        self.lfi = decoder_instance.lfi

        self.spatial_dims = decoder_instance.spatial_dims
        if self.spatial_dims == 2:
            self.conv_module = nn.Conv2d
        elif self.spatial_dims == 3:
            self.conv_module = nn.Conv3d

        # add lateral connections

        ## gfi
        if self.connect_gfi:
            if self.efi.out_connection_type[0] == "conv":
                connection_channels = self.efi.out_connection_type[1]
                self.gfi_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.efi.out_connection_type[0] == "lin":
                connection_dim = self.efi.out_connection_type[1]
                self.gfi_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())
        ## lfi
        if self.connect_lfi:
            if self.gfi.out_connection_type[0] == "conv":
                connection_channels = self.gfi.out_connection_type[1]
                self.lfi_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.gfi.out_connection_type[0] == "lin":
                connection_dim = self.gfi.out_connection_type[1]
                self.lfi_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        ## lfi
        if self.connect_recon:
            if self.lfi.out_connection_type[0] == "conv":
                connection_channels = self.lfi.out_connection_type[1]
                self.recon_c = nn.Sequential(self.conv_module(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False))
            elif self.lfi.out_connection_type[0] == "lin":
                connection_dim = self.lfi.out_connection_type[1]
                self.recon_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())


        initialization_net = get_weights_init("kaiming_uniform")
        self.efi.apply(initialization_net)
        self.gfi.apply(initialization_net)
        self.lfi.apply(initialization_net)
        initialization_c = get_weights_init("uniform")
        if self.connect_gfi:
            self.gfi_c.apply(initialization_c)
        if self.connect_lfi:
            self.lfi_c.apply(initialization_c)
        if self.connect_recon:
            self.recon_c.apply(initialization_c)

        """
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
        """

    def forward(self, z, parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(z)

        if z.dim() == 2 and self.efi.out_connection_type[0] == "conv":  # B*n_latents -> B*n_latents*1*1(*1)
            for _ in range(self.spatial_dims):
                z = z.unsqueeze(-1)

        # batch norm cannot deal with batch_size 1 in train mode
        was_training = None
        if self.training and z.size()[0] == 1:
            self.eval()
            was_training = True

        # global feature map
        gfi = self.efi(z)
        # add the connections
        if self.connect_gfi:
            gfi = gfi + self.gfi_c(parent_gfi)

        # local feature map
        lfi = self.gfi(gfi)
        # add the connections
        if self.connect_lfi:
            lfi = lfi + self.lfi_c(parent_lfi)

        # recon_x
        recon_x = self.lfi(lfi)
        # add the connections
        if self.connect_recon:
            recon_x = recon_x + self.recon_c(parent_recon_x)

        # decoder output
        decoder_outputs = {"gfi": gfi, "lfi": lfi, "recon_x": recon_x}

        if was_training and z.size()[0] == 1:
            self.train()

        return decoder_outputs

    def forward_for_graph_tracing(self, z,  parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if z.dim() == 2:  # B*n_latents -> B*n_latents*1*1
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # global feature map
        gfi = self.efi(z)
        # add the connections
        if self.connect_gfi:
            gfi = gfi + self.gfi_c(parent_gfi)

        # local feature map
        lfi = self.gfi(gfi)
        # add the connections
        if self.connect_lfi:
            lfi = lfi + self.lfi_c(parent_lfi)

        # recon_x
        recon_x = self.lfi(lfi)
        # add the connections
        if self.connect_recon:
            recon_x = recon_x + self.recon_c(parent_recon_x)

        return recon_x
