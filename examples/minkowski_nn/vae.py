from image_representation.datasets.minkowski_dataset import RandomDataset, sparse_collation, ModelNet40Dataset, Mnist3dDataset
from image_representation.representations.minkowski_nn.vae import VAE
from addict import Dict
from torch.utils.data import DataLoader
import os
import shutil

def run_training():

    # input_size = (32, 32)
    # n_channels = 1
    # train_dataset = RandomDataset(img_size=input_size, n_channels=n_channels, n_images=1000)
    # valid_dataset = RandomDataset(img_size=input_size, n_channels=n_channels, n_images=10)

    # resolution = 128
    # input_size = (resolution, resolution, resolution)
    # n_channels = 1
    # dataset_config = ModelNet40Dataset.default_config()
    # dataset_config.resolution = resolution
    # dataset_config.data_root = '/home/mayalen/data/pc_datasets'
    # dataset_config.split = 'test'
    # train_dataset = ModelNet40Dataset(config=dataset_config)
    # dataset_config.split = 'test'
    # valid_dataset = ModelNet40Dataset(config=dataset_config)

    input_size = (16, 16, 16)
    n_channels = 1
    dataset_config = Mnist3dDataset.default_config()
    dataset_config.data_root = '/home/mayalen/data/kaggle_datasets'
    dataset_config.split = 'test'
    train_dataset = Mnist3dDataset(config=dataset_config)
    dataset_config.split = 'valid'
    valid_dataset = Mnist3dDataset(config=dataset_config)


    print('Load dataloader ...')
    train_loader = DataLoader(train_dataset, batch_size=2,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=sparse_collation)
    valid_loader = DataLoader(valid_dataset, batch_size=2,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=sparse_collation)

    print('Load VAE ...')
    vae_config = Dict()
    vae_config.network.name = "Dumoulin"
    vae_config.network.parameters.n_channels = n_channels
    vae_config.network.parameters.input_size = input_size
    vae_config.network.parameters.n_latents = 20
    vae_config.network.parameters.n_conv_layers = 4
    vae_config.network.parameters.feature_layer = 2
    vae_config.network.parameters.encoder_conditional_type = "gaussian"
    vae_config.network.weights_init = Dict()
    vae_config.network.weights_init.name = "kaiming_normal"
    vae_config.network.weights_init.parameters.mode = "fan_out"
    vae_config.network.weights_init.parameters.nonlinearity = "relu"
    vae_config.loss.name = "VAE"
    vae_config.loss.parameters.reconstruction_dist = "bernoulli"
    vae_config.optimizer.name = "Adam"
    vae_config.optimizer.parameters.lr = 1e-3
    vae_config.optimizer.parameters.weight_decay = 1e-5
    vae_config.checkpoint.folder = "./checkpoints/vae"
    vae_config.logging.folder = "./logs/vae"
    if os.path.exists(vae_config.logging.folder):
        shutil.rmtree(vae_config.logging.folder)
    vae_config.logging.record_loss_every = 1
    vae_config.logging.record_valid_images_every = 1
    vae_config.logging.record_embeddings_every = 1
    vae = VAE(config=vae_config)
    training_config = Dict()
    training_config.n_epochs = 50

    print('Run Training ...')
    vae.run_training(train_loader, training_config, valid_loader=valid_loader)

    print('Finished.')


if __name__ == "__main__":
    run_training()