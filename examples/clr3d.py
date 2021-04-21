from image_representation.datasets.torch_dataset import Mnist3dDataset
from image_representation import SimCLR
from addict import Dict
import torch
from torch.utils.data import DataLoader

def run_training():

    print('Load dataset ...')
    dataset_config = Dict()
    dataset_config.data_root = '/home/mayalen/data/kaggle_datasets/'
    dataset_config.download = False
    dataset_config.split = 'train'

    # process data
    dataset_config.preprocess = lambda x, f0=lambda x: (torch.rand_like(x) < x).to(torch.float) : f0(x)
    dataset_config.data_augmentation = True
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

    print('Load SimCLR ...')
    simclr_config = Dict()
    simclr_config.network.name = "Burgess"
    simclr_config.network.parameters.input_size = (16, 16, 16)
    simclr_config.network.parameters.n_latents = 10
    simclr_config.network.parameters.n_conv_layers = 3
    simclr_config.network.parameters.feature_layer = 2
    simclr_config.network.parameters.encoder_conditional_type = "gaussian"
    simclr_config.network.weights_init = Dict()
    simclr_config.network.weights_init.name = "pytorch"
    simclr_config.loss.name = "SimCLR"
    simclr_config.loss.parameters.temperature = 0.5
    simclr_config.loss.parameters.distance = 'cosine'
    simclr_config.optimizer.name = "Adam"
    simclr_config.optimizer.parameters.lr = 1e-3
    simclr_config.optimizer.parameters.weight_decay = 1e-5
    simclr_config.checkpoint.folder = "./checkpoints/simclr3d"
    simclr_config.logging.folder = "./logs/simclr3d"
    simclr_config.logging.record_loss_every = 1
    simclr_config.logging.record_valid_images_every = 0
    simclr_config.logging.record_embeddings_every = 1
    simclr = SimCLR(config=simclr_config)
    training_config = Dict()
    training_config.n_epochs = 20


    print('Run Training ...')
    simclr.run_training(train_loader, training_config, valid_loader=valid_loader)

    print('Finished.')


if __name__ == "__main__":
    run_training()