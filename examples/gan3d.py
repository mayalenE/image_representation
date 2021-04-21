import os

import torch
torch.autograd.set_detect_anomaly(True)
from addict import Dict
from image_representation import BiGAN
from image_representation.datasets.torch_dataset import Mnist3dDataset
from torch.utils.data import DataLoader


def run_training():

    print('Load dataset ...')
    dataset_config = Dict()
    dataset_config.data_root = '/home/mayalen/data/kaggle_datasets/'
    dataset_config.download = False
    dataset_config.split = 'train'

    # process data
    dataset_config.preprocess = lambda x, f0=lambda x: (torch.rand_like(x) < x).to(torch.float) : f0(x)
    dataset_config.data_augmentation = False
    dataset_config.transform = None
    dataset_config.target_transform = None

    train_dataset = Mnist3dDataset(config=dataset_config)
    dataset_config.split = 'valid'
    valid_dataset = Mnist3dDataset(config=dataset_config)


    print('Load dataloader ...')
    train_loader = DataLoader(train_dataset, batch_size=64,
                                  shuffle=True,
                                  num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=10,
                             shuffle=False,
                             num_workers=0)

    print('Load BiGAN ...')
    bigan_config = Dict()
    bigan_config.network.name = "Burgess"
    bigan_config.network.parameters.input_size = (16, 16, 16)
    bigan_config.network.parameters.n_latents = 10
    bigan_config.network.parameters.n_conv_layers = 3
    bigan_config.network.parameters.feature_layer = 2
    bigan_config.network.parameters.hidden_dim = 10
    bigan_config.network.parameters.encoder_conditional_type = "gaussian"
    bigan_config.network.weights_init = Dict()
    bigan_config.network.weights_init.name = "pytorch"

    bigan_config.device = "cuda"

    bigan_config.loss.name = "BiGAN"
    bigan_config.loss.parameters.reconstruction_dist = "bernoulli"
    bigan_config.optimizer.name = "Adam"
    bigan_config.optimizer.parameters.lr = 1e-3
    bigan_config.optimizer.parameters.weight_decay = 1e-5

    bigan_config.checkpoint.folder = "./checkpoints/bigan3d"
    bigan_config.logging.folder = "./logs/bigan3d"
    bigan_config.logging.record_loss_every = 1
    bigan_config.logging.record_valid_images_every = 1
    bigan_config.logging.record_embeddings_every = 1

    bigan = BiGAN(config=bigan_config)
    training_config = Dict()
    training_config.n_epochs = 20

    print('Run Training ...')
    bigan.run_training(train_loader, training_config, valid_loader=valid_loader)

    print('Finished.')


if __name__ == "__main__":
    run_training()