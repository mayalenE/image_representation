from image_representation.datasets.torch_dataset import EvocraftDataset
from image_representation import VAE
from addict import Dict
import torch
from torch.utils.data import DataLoader

def run_training():

    print('Load dataset ...')
    dataset_config = Dict()
    dataset_config.data_root = '/home/mayalen/data/evocraft_datasets/data_000/'
    dataset_config.download = False
    dataset_config.split = 'train'

    # process data
    dataset_config.preprocess = lambda x, f0=lambda x: (torch.rand_like(x) < x).to(torch.float) : f0(x)
    dataset_config.data_augmentation = False
    dataset_config.transform = None
    dataset_config.target_transform = None

    train_dataset = EvocraftDataset(config=dataset_config)
    dataset_config.split = 'valid'
    valid_dataset = EvocraftDataset(config=dataset_config)


    print('Load dataloader ...')
    train_loader = DataLoader(train_dataset, batch_size=128,
                                  shuffle=True,
                                  num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=10,
                             shuffle=False,
                             num_workers=0)

    print('Load VAE ...')
    vae_config = Dict()
    vae_config.network.name = "Burgess"
    vae_config.network.parameters.input_size = (16, 16, 16)
    vae_config.network.parameters.n_channels = 6
    vae_config.network.parameters.n_latents = 100
    vae_config.network.parameters.n_conv_layers = 2
    vae_config.network.parameters.feature_layer = 1
    vae_config.network.parameters.encoder_conditional_type = "gaussian"
    vae_config.network.weights_init = Dict()
    vae_config.network.weights_init.name = "pytorch"

    vae_config.device = "cuda"

    vae_config.loss.name = "VAE"
    vae_config.loss.parameters.reconstruction_dist = "bernoulli"
    vae_config.optimizer.name = "Adam"
    vae_config.optimizer.parameters.lr = 1e-3
    vae_config.optimizer.parameters.weight_decay = 1e-5

    vae_config.checkpoint.folder = "./checkpoints/vae3d"
    vae_config.logging.folder = "./logs/vae3d"
    vae_config.logging.record_loss_every = 1
    vae_config.logging.record_valid_images_every = 1
    vae_config.logging.record_embeddings_every = 10

    vae = VAE(config=vae_config)
    training_config = Dict()
    training_config.n_epochs = 1000

    print('Run Training ...')
    vae.run_training(train_loader, training_config, valid_loader=valid_loader)

    print('Finished.')


if __name__ == "__main__":
    run_training()