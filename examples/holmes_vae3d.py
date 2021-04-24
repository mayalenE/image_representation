import os
import shutil
from image_representation.datasets.torch_dataset import EvocraftDataset
from image_representation import HOLMES_VAE
from addict import Dict
import torch
from torch.utils.data import DataLoader

def run_training():
    print('Load dataset ...')
    dataset_config = Dict()
    dataset_config.data_root = '/home/mayalen/data/evocraft_datasets/data_001/'
    dataset_config.download = False
    dataset_config.split = 'train'

    # process data

    dataset_config.preprocess = lambda x: x.argmax(1).unsqueeze(1).float()/6.0
    dataset_config.data_augmentation = False
    dataset_config.transform = None
    dataset_config.target_transform = None

    train_dataset = EvocraftDataset(config=dataset_config)
    dataset_config.split = 'test'
    valid_dataset = EvocraftDataset(config=dataset_config)

    print('Load dataloader ...')
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=10,
                              shuffle=False,
                              num_workers=0)

    print('Load HOLMES_VAE ...')
    holmes_vae_config = Dict()
    holmes_vae_config.node_classname = "VAE"

    holmes_vae_config.node = Dict()
    holmes_vae_config.node.network.name = "Burgess"
    holmes_vae_config.node.network.parameters.input_size = (16, 16, 16)
    holmes_vae_config.node.network.parameters.n_channels = 1
    holmes_vae_config.node.network.parameters.n_latents = 16
    holmes_vae_config.node.network.parameters.n_conv_layers = 2
    holmes_vae_config.node.network.parameters.feature_layer = 1
    holmes_vae_config.node.network.parameters.encoder_conditional_type = "gaussian"
    holmes_vae_config.node.network.weights_init = Dict()
    holmes_vae_config.node.network.weights_init.name = "pytorch"

    holmes_vae_config.device = "cuda"
    holmes_vae_config.dtype = torch.float32

    holmes_vae_config.node.create_connections = {"lf": True, "gf": False, "gfi": True, "lfi": True, "recon": True}

    holmes_vae_config.loss.name = "VAE"
    holmes_vae_config.loss.parameters.reconstruction_dist = "bernoulli"
    holmes_vae_config.optimizer.name = "Adam"
    holmes_vae_config.optimizer.parameters.lr = 1e-3
    holmes_vae_config.optimizer.parameters.weight_decay = 1e-5

    holmes_vae_config.checkpoint.folder = "./checkpoints/holmes_vae3d"
    holmes_vae_config.logging.folder = "./logs/holmes_vae3d"
    if os.path.exists(holmes_vae_config.logging.folder):
        shutil.rmtree(holmes_vae_config.logging.folder)
    holmes_vae_config.logging.record_loss_every = 1
    holmes_vae_config.logging.record_valid_images_every = 10
    holmes_vae_config.logging.record_embeddings_every = 10

    holmes_vae = HOLMES_VAE(config=holmes_vae_config)

    training_config = Dict()
    training_config.n_epochs = 5000
    training_config.split_trigger.active = True
    training_config.split_trigger.fitness_key = 'recon'
    training_config.split_trigger.type = 'plateau'
    training_config.split_trigger.parameters = Dict(epsilon=20, n_steps_average=50)
    training_config.split_trigger.conditions = Dict(min_init_n_epochs=200, n_min_points=500, n_max_splits=10, n_epochs_min_between_splits=100)
    training_config.split_trigger.save_model_before_after = True
    training_config.split_trigger.boundary_config.z_fitness = "recon_loss"
    training_config.split_trigger.boundary_config.algo = "cluster.KMeans"
    training_config.alternated_backward.active = True
    training_config.alternated_backward.ratio_epochs = {"connections": 2, "core": 8}


    print('Run Training ...')
    holmes_vae.run_training(train_loader, training_config, valid_loader=valid_loader)

    print('Finished.')


if __name__ == "__main__":
    run_training()