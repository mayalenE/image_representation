from image_representation.datasets.torch_dataset import MNISTDataset
from image_representation import HOLMES_VAE
from addict import Dict
import torch
from torch.utils.data import DataLoader
import os
import shutil

def run_training():

    print('Load dataset ...')
    dataset_config = Dict()
    dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
    dataset_config.download = False
    dataset_config.split = 'train'

    # process data
    dataset_config.preprocess = lambda x, f0=lambda x: (torch.rand_like(x) < x).to(torch.float) : f0(x)
    dataset_config.data_augmentation = False
    dataset_config.transform = None
    dataset_config.target_transform = None

    train_dataset = MNISTDataset(config=dataset_config)
    dataset_config.split = 'valid'
    valid_dataset = MNISTDataset(config=dataset_config)


    print('Load dataloader ...')
    train_loader = DataLoader(train_dataset, batch_size=64,
                                  shuffle=True,
                                  num_workers=0)
    valid_loader = DataLoader(train_dataset, batch_size=10,
                             shuffle=False,
                             num_workers=0)

    print('Load HOLMES_VAE ...')
    holmes_vae_config = Dict()
    holmes_vae_config.node_classname = "VAE"

    holmes_vae_config.node = Dict()
    holmes_vae_config.node.network.name = "Burgess"
    holmes_vae_config.node.network.parameters.input_size = (28, 28)
    holmes_vae_config.node.network.parameters.n_latents = 10
    holmes_vae_config.node.network.parameters.n_conv_layers = 3
    holmes_vae_config.node.network.parameters.feature_layer = 2
    holmes_vae_config.node.network.parameters.encoder_conditional_type = "gaussian"
    holmes_vae_config.node.network.weights_init = Dict()
    holmes_vae_config.node.network.weights_init.name = "pytorch"

    holmes_vae_config.node.create_connections = {"lf": True, "gf": False, "gfi": True, "lfi": True, "recon": True}

    holmes_vae_config.loss.name = "VAE"
    holmes_vae_config.loss.parameters.reconstruction_dist = "bernoulli"
    holmes_vae_config.optimizer.name = "Adam"
    holmes_vae_config.optimizer.parameters.lr = 1e-3
    holmes_vae_config.optimizer.parameters.weight_decay = 1e-5

    holmes_vae_config.checkpoint.folder = "./checkpoints/holmes_vae"
    holmes_vae_config.logging.folder = "./logs/holmes_vae"
    if os.path.exists(holmes_vae_config.logging.folder):
        shutil.rmtree(holmes_vae_config.logging.folder)
    holmes_vae_config.logging.record_loss_every = 1
    holmes_vae_config.logging.record_valid_images_every = 10
    holmes_vae_config.logging.record_embeddings_every = 10

    holmes_vae = HOLMES_VAE(config=holmes_vae_config)

    training_config = Dict()
    training_config.n_epochs = 20


    print('Run Training ...')
    holmes_vae.run_training(train_loader, training_config, valid_loader=valid_loader)

    print('Finished.')


if __name__ == "__main__":
    run_training()