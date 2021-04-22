from unittest import TestCase
from exputils.seeding import set_seed
from image_representation.datasets.torch_dataset import QuadrupletDataset, MNISTDataset, SVHNDataset, CIFAR10Dataset, CIFAR100Dataset, LENIADataset, Mnist3dDataset, EvocraftDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch

class TestDatasets(TestCase):

    def test_data_augmentation(self):
        set_seed(0)

        for name in ['Evocraft']:
            if name == 'MNIST':
                dataset_config = MNISTDataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
                dataset_config.download = False
                dataset_config.preprocess = lambda x, f0=lambda x: (torch.rand_like(x) < x).to(torch.float): f0(x)
                dataset_config.data_augmentation = True
                dataset = MNISTDataset(dataset_config)
            elif name == 'SVHN':
                dataset_config = SVHNDataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
                dataset_config.download = False
                dataset_config.data_augmentation = True
                dataset = SVHNDataset(dataset_config)
            elif name == 'CIFAR10':
                dataset_config = CIFAR10Dataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
                dataset_config.download = False
                dataset_config.data_augmentation = True
                dataset = CIFAR10Dataset(dataset_config)
            elif name == 'CIFAR100':
                dataset_config = CIFAR10Dataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
                dataset_config.download = False
                dataset_config.data_augmentation = True
                dataset = CIFAR100Dataset(dataset_config)
            elif name == 'LENIA':
                dataset_config = LENIADataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/lenia_datasets/data_005_sub_b'
                dataset_config.download = False
                dataset_config.data_augmentation = True
                dataset = LENIADataset(dataset_config)
            elif name == 'Mnist3d':
                dataset_config = Mnist3dDataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/kaggle_datasets/'
                dataset_config.download = False
                dataset_config.preprocess = lambda x, f0=lambda x: (torch.rand_like(x) < x).to(torch.float): f0(x)
                dataset_config.data_augmentation = True
                dataset = Mnist3dDataset(dataset_config)
            if name == 'Evocraft':
                dataset_config = EvocraftDataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/evocraft_datasets/data_000'
                dataset_config.download = False
                dataset_config.preprocess = lambda x, f0=lambda x: (torch.rand_like(x) < x).to(torch.float): f0(x)
                dataset_config.data_augmentation = True
                dataset = EvocraftDataset(dataset_config)

            data_ids = np.random.choice(dataset.n_images, 20)
            fig = plt.figure(figsize=(10, 20))
            tensor_list = []
            for data_idx in data_ids:
                data = dataset.get_image(data_idx)
                data_augs = [data]
                for aug in range(9):
                    data_augs.append(dataset.__getitem__(data_idx)['obs'])
                tensor_list.append(torch.stack(data_augs))
            tensor_vizu = torch.cat(tensor_list)
            if tensor_vizu.ndim == 5:
                for z_slice in range(tensor_vizu.shape[2]):
                    img = np.transpose(make_grid(tensor_vizu[:, :3, z_slice, :, :], nrow=10).numpy(), (1, 2, 0)) # only show first three channels
                    plt.axis('off')
                    plt.imshow(img)
                    plt.savefig(f'example_augmentations/augmentation_{name}_zslice_{z_slice}.png')
            else:
                img = np.transpose(make_grid(tensor_vizu, nrow=10).numpy(), (1, 2, 0))
                plt.axis('off')
                plt.imshow(img)
                plt.savefig(f'example_augmentations/augmentation_{name}.png')

    def test_data_quadruplets(self):
        set_seed(0)

        for name in ['MNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'LENIA']:
            if name == 'MNIST':
                dataset_config = MNISTDataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
                dataset_config.download = True
                dataset_config.preprocess = lambda x, f0=lambda x: (torch.rand_like(x) < x).to(torch.float): f0(x)
                dataset_config.data_augmentation = True

                dataset = QuadrupletDataset("mnist", dataset_config)

            elif name == 'SVHN':
                dataset_config = SVHNDataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
                dataset_config.download = False
                dataset_config.data_augmentation = True

                dataset = QuadrupletDataset("svhn", dataset_config)

            elif name == 'CIFAR10':
                dataset_config = CIFAR10Dataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
                dataset_config.download = False
                dataset_config.data_augmentation = True

                dataset = QuadrupletDataset("cifar10", dataset_config)

            elif name == 'CIFAR100':
                dataset_config = CIFAR10Dataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/pytorch_datasets/'
                dataset_config.download = False
                dataset_config.data_augmentation = True

                dataset = QuadrupletDataset("cifar100", dataset_config)

            elif name == 'LENIA':
                dataset_config = LENIADataset.default_config()
                dataset_config.split = 'test'
                dataset_config.data_root = '/home/mayalen/data/lenia_datasets/data_005_sub_b'
                dataset_config.download = False
                dataset_config.data_augmentation = True

                dataset = QuadrupletDataset("lenia", dataset_config)

            quad_ids = np.random.choice(dataset.n_images, 9)
            fig = plt.figure(figsize=(5, 10))
            tensor_list = []
            for quad_idx in quad_ids:
                data_ref, data_a, data_b, data_c = dataset.__getitem__(quad_idx)
                tensor_list.append(torch.stack([data_ref['obs'], data_a['obs'], data_b['obs'], data_c['obs']]))
            tensor_vizu = torch.cat(tensor_list)
            img = np.transpose(make_grid(tensor_vizu, nrow=4).numpy(), (1, 2, 0))
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('example_quadruplets/quadruplets_{}.png'.format(name))
