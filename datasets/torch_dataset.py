import h5py
from addict import Dict
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision
from image_representation.datasets.preprocess import TensorRandomResizedCrop, TensorRandomCenterCrop, TensorRandomRoll,  TensorRandomSphericalRotation, TensorRandomGaussianBlur
from torchvision.transforms import CenterCrop, Compose, ToTensor, ToPILImage, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, Pad, RandomApply, RandomResizedCrop
import warnings
from PIL import Image

to_tensor = ToTensor()
to_PIL_image = ToPILImage()

# ===========================
# get dataset function
# ===========================

def get_dataset(dataset_name):
    """
    dataset_name: string such that the model called is <dataset_name>Dataset
    """
    return eval("image_representation.datasets.{}Dataset".format(dataset_name.upper()))

# ===========================
# Mixed Datasets
# ===========================

class MIXEDDataset(Dataset):

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None
        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        # initially dataset lists are empty
        self.n_images = 0
        self.images = torch.FloatTensor([]) # list or torch tensor of size N*C*H*W
        self.labels = torch.LongTensor([]) # list or torch tensor
        self.datasets_ids = [] # list of the dataset idx each image is coming from

        self.datasets = {}
        for dataset in self.config.datasets:
            dataset_class = get_dataset(dataset["name"])
            self.datasets[dataset["name"]] = dataset_class(config=dataset.config)

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def update(self, n_images, images, labels=None, datasets_ids=None):
        """ Update the current dataset lists """
        if labels is None:
            labels = torch.LongTensor([-1] * n_images)
        assert n_images == images.shape[0] == labels.shape[0], print(
            'ERROR: the given dataset size ({0}) mismatch with observations size ({1}) and labels size ({2})'.format(
                n_images, images.shape[0], labels.shape[0]))

        self.n_images = int(n_images)
        self.images = images
        self.labels = labels

        if datasets_ids is not None:
            self.datasets_ids = datasets_ids

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
            datasets_names = list(self.datasets.keys())
            if len(datasets_names) == 1:
                img_tensor = self.datasets[datasets_names[0]].augment(img_tensor)
            elif len(self.datasets_ids) > 0:
                dataset_id = self.datasets_ids[idx]
                if dataset_id is not None:  # if generated data (None) we do not augment
                    img_tensor = self.datasets[dataset_id].augment(img_tensor)
            else:
                raise ValueError("Cannot augment data if dataset_ids is not given")

        if self.transform is not None:
            datasets_names = list(self.datasets.keys())
            if len(datasets_names) == 1:
                img_tensor = self.datasets[datasets_names[0]].transform(img_tensor)
            elif len(self.datasets_ids) > 0:
                dataset_id = self.datasets_ids[idx]
                if dataset_id is not None:
                    img_tensor = self.datasets[dataset_id].transform(img_tensor)
            else:
                raise ValueError("Cannot augment data if dataset_ids is not given")

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            datasets_names = list(self.datasets.keys())
            if len(datasets_names) == 1:
                label = self.datasets[datasets_names[0]].target_transform(label)
            elif len(self.datasets_ids) > 0:
                dataset_id = self.datasets_ids[idx]
                if dataset_id is not None:
                    label = self.datasets[dataset_id].target_transform(label)
            else:
                raise ValueError("Cannot augment data if dataset_ids is not given")

        return {'obs': img_tensor, 'label': label, 'index': idx}


    def save(self, output_npz_filepath):
        np.savez(output_npz_filepath, n_images=self.n_images, images=np.stack(self.images),
                 labels=np.asarray(self.labels))




# ==================================
# Torchvision Datasets
# ===============================

class MNISTDataset(torchvision.datasets.MNIST):

    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = 'dataset'
        default_config.download = True
        default_config.split = "train"

        # process data
        default_config.preprocess = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.split == "train":
            train = True
        elif self.config.split == "valid":
            train = False
            warnings.warn("WARNING: MNIST does not have a valid dataset so the test set is given, should NOT BE USED for model selection")
        elif self.config.split == "test":
            train = False
        else:
            raise ValueError("MNIST dataset does not have a {} split".format((self.config.split)))
        torchvision.datasets.MNIST.__init__(
            self, root=os.path.join(self.config['data_root'], "mnist"),
            train=train, download=self.config.download)

        # self.data transform to N*C*H*W, float type between 0 and 1, preprocess
        self.images = self.data.unsqueeze(1)
        self.images = self.images.float() / 255.0
        if self.config.preprocess is not None:
            self.images = self.config.preprocess(self.images)
        del self.data

        self.n_channels = self.images.shape[1]
        self.img_size = (self.images.shape[2], self.images.shape[3])
        self.n_images = len(self.images)

        self.labels = self.targets
        del self.targets

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            # MNIST AUGMENT
            ## rotation
            if self.n_channels == 1:
                fill = (0,)
            else:
                fill = 0
            self.random_rotation = RandomApply([RandomRotation(30, resample=Image.BILINEAR, fill=fill)], p=0.6)
            ## resized crop
            self.random_resized_crop = RandomApply(
                [RandomResizedCrop(self.img_size, scale=(0.9, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)],
                p=0.6)
            ## composition
            self.augment = Compose([to_PIL_image, self.random_rotation, self.random_resized_crop, to_tensor])


        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def get_image(self, image_idx, augment=False, transform=True):
        image = self.images[image_idx]
        if augment and self.data_augmentation:
            image = self.augment(image)
        if transform and self.transform is not None:
            image = self.transform(image)
        return image

    def get_augmented_batch(self, image_ids, augment=True, transform=True):
        images_aug = []
        for img_idx in image_ids:
            image_aug = self.get_image(img_idx, augment=augment, transform=transform)
            images_aug.append(image_aug)
        images_aug = torch.stack(images_aug, dim=0)
        return images_aug

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
            img_tensor = self.augment(img_tensor)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'obs': img_tensor, 'label': label, 'index': idx}


class SVHNDataset(torchvision.datasets.SVHN):
    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = 'dataset'
        default_config.download = True
        default_config.split = "train"

        # process data
        default_config.preprocess = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        if self.config.split == "valid":
            self.config.split = "test"
            warnings.warn(
                "WARNING: SVHN does not have a valid dataset so the test set is given, should NOT BE USED for model selection")
        torchvision.datasets.SVHN.__init__(
            self, root=os.path.join(self.config['data_root'], "svhn"),
            split=self.config.split, download=self.config.download)

        # self.data transform to N*C*H*W, float type between 0 and 1, preprocess
        self.images = torch.from_numpy(self.data)
        self.images = self.images.float() / 255.0
        if self.config.preprocess is not None:
            self.images = self.config.preprocess(self.images)
        del self.data

        self.n_channels = self.images.shape[1]
        self.img_size = (self.images.shape[2], self.images.shape[3])
        self.n_images = len(self.images)

        self.labels = torch.from_numpy(self.labels)

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            # SVHN AUGMENT
            ## rotation
            if self.n_channels == 1:
                fill = (0,)
            else:
                fill = 0
            radius = max(self.img_size[0], self.img_size[1]) / 2
            padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
            self.pad = Pad(padding_size, padding_mode='edge')
            self.center_crop = CenterCrop(self.img_size)
            self.random_rotation = RandomApply([self.pad, RandomRotation(30, resample=Image.BILINEAR, fill=fill), self.center_crop], p=0.6)
            ## resized crop
            self.random_resized_crop = RandomApply(
                [RandomResizedCrop(self.img_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)],
                p=0.6)
            ## color
            self.random_color_jitter = RandomApply([ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.1)], p=0.8)
            ## gaussian blur
            #self.random_gaussian_blur = RandomGaussianBlur(p=0.5, kernel_radius=1, max_sigma=2.0, n_channels=self.n_channels)
            ## composition
            self.augment = Compose([to_PIL_image, self.random_rotation, self.random_resized_crop, self.random_color_jitter, to_tensor])

        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def get_image(self, image_idx, augment=False, transform=True):
        image = self.images[image_idx]
        if augment and self.data_augmentation:
            image = self.augment(image)
        if transform and self.transform is not None:
            image = self.transform(image)
        return image


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
            img_tensor = self.augment(img_tensor)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'obs': img_tensor, 'label': label, 'index': idx}

class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = 'dataset'
        default_config.download = True
        default_config.split = "train"

        # process data
        default_config.preprocess = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        if self.config.split == "train":
            train = True
        elif self.config.split == "valid":
            train = False
            warnings.warn(
                "WARNING: CIFAR10 does not have a valid dataset so the test set is given, should NOT BE USED for model selection")
        elif self.config.split == "test":
            train = False
        else:
            raise ValueError("CIFAR10 dataset does not have a {} split".format((self.config.split)))
        torchvision.datasets.CIFAR10.__init__(
            self, root=os.path.join(self.config['data_root'], "cifar10"),
            train=train, download=self.config.download)

        # self.data transform to N*C*H*W, float type between 0 and 1, preprocess
        self.images = torch.from_numpy(np.transpose(self.data, (0, 3, 1, 2)))
        self.images = self.images.float() / 255.0
        if self.config.preprocess is not None:
            self.images = self.config.preprocess(self.images)
        del self.data

        self.n_channels = self.images.shape[1]
        self.img_size = (self.images.shape[2], self.images.shape[3])
        self.n_images = len(self.images)

        self.labels = torch.LongTensor(self.targets)

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            # CIFAR10 AUGMENT
            ## rotation
            if self.n_channels == 1:
                fill = (0,)
            else:
                fill = 0
            radius = max(self.img_size[0], self.img_size[1]) / 2
            padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
            self.pad = Pad(padding_size, padding_mode='edge')
            self.center_crop = CenterCrop(self.img_size)
            self.random_rotation = RandomApply(
                [self.pad, RandomRotation(30, resample=Image.BILINEAR, fill=fill), self.center_crop], p=0.6)
            ## resized crop
            self.random_resized_crop = RandomApply(
                [RandomResizedCrop(self.img_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)],
                p=0.6)
            ## horizontal flip
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            ## color
            self.random_color_jitter = RandomApply([ColorJitter(brightness=.5, contrast=.2, saturation=.2, hue=.1)], p=0.6)
            ## gaussian blur
            #self.random_gaussian_blur = RandomGaussianBlur(p=0.5, kernel_radius=1, max_sigma=2.0, n_channels=self.n_channels)
            ## composition
            self.augment = Compose(
                [to_PIL_image, self.random_rotation, self.random_resized_crop, self.random_horizontal_flip, self.random_color_jitter, to_tensor])

        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def get_image(self, image_idx, augment=False, transform=True):
        image = self.images[image_idx]
        if augment and self.data_augmentation:
            image = self.augment(image)
        if transform and self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
            img_tensor = self.augment(img_tensor)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'obs': img_tensor, 'label': label, 'index': idx}



class CIFAR100Dataset(torchvision.datasets.CIFAR100):

    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = 'dataset'
        default_config.download = True
        default_config.split = "train"

        # process data
        default_config.preprocess = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        if self.config.split == "train":
            train = True
        elif self.config.split == "valid":
            train = False
            warnings.warn(
                "WARNING: CIFAR100 does not have a valid dataset so the test set is given, should NOT BE USED for model selection")
        elif self.config.split == "test":
            train = False
        else:
            raise ValueError("CIFAR100 dataset does not have a {} split".format((self.config.split)))
        torchvision.datasets.CIFAR100.__init__(
            self, root=os.path.join(self.config['data_root'], "cifar100"),
            train=train, download=self.config.download)

        # self.data transform to N*C*H*W, float type between 0 and 1, preprocess
        self.images = torch.from_numpy(np.transpose(self.data, (0, 3, 1, 2)))
        self.images = self.images.float() / 255.0
        if self.config.preprocess is not None:
            self.images = self.config.preprocess(self.images)
        del self.data

        self.n_channels = self.images.shape[1]
        self.img_size = (self.images.shape[2], self.images.shape[3])
        self.n_images = len(self.images)

        self.labels = torch.LongTensor(self.targets)

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            ## rotation
            if self.n_channels == 1:
                fill = (0,)
            else:
                fill = 0
            radius = max(self.img_size[0], self.img_size[1]) / 2
            padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
            self.pad = Pad(padding_size, padding_mode='edge')
            self.center_crop = CenterCrop(self.img_size)
            self.random_rotation = RandomApply(
                [self.pad, RandomRotation(30, resample=Image.BILINEAR, fill=fill), self.center_crop], p=0.6)
            ## resized crop
            self.random_resized_crop = RandomApply(
                [RandomResizedCrop(self.img_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)],
                p=0.6)
            ## horizontal flip
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            ## color
            self.random_color_jitter = RandomApply([ColorJitter(brightness=.5, contrast=.2, saturation=.2, hue=.1)], p=0.6)
            ## gaussian blur
            #self.random_gaussian_blur = RandomGaussianBlur(p=0.5, kernel_radius=1, max_sigma=2.0, n_channels=self.n_channels)
            ## composition
            self.augment = Compose([to_PIL_image, self.random_rotation, self.random_resized_crop, self.random_horizontal_flip, self.random_color_jitter, to_tensor])

        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def get_image(self, image_idx, augment=False, transform=True):
        image = self.images[image_idx]
        if augment and self.data_augmentation:
            image = self.augment(image)
        if transform and self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
            img_tensor = self.augment(img_tensor)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'obs': img_tensor, 'label': label, 'index': idx}


# ==================================
# 3D Datasets
# ===============================

class Mnist3dDataset(Dataset):
    """ Download from: https://www.kaggle.com/daavoo/3d-mnist, see tuto https://www.kaggle.com/shivamb/3d-convolutions-understanding-use-case """

    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = None
        default_config.split = "train"

        # process data
        default_config.preprocess = None
        default_config.img_size = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.data_root is None:
            self.n_images = 0
            if self.config.img_size is not None:
                self.img_size = self.config.img_size
                self.images = torch.zeros((0, 1, self.config.img_size[0], self.config.img_size[1], self.config.img_size[2]))
                self.n_channels = 1
            else:
                raise ValueError("If data_root not given, the img_size must be specified in the config")
            self.labels = torch.zeros((0, 1), dtype=torch.long)

        else:
            # load HDF5 Mnist3d dataset
            dataset_filepath = os.path.join(self.config.data_root, '3Dmnist', 'full_dataset_vectors.h5')
            with h5py.File(dataset_filepath, 'r') as file:
                if self.config.split == "train":
                    X = file["X_train"][:]
                    Y = file["y_train"][:]
                elif self.config.split in ["valid", "test"]:
                    X = file["X_test"][:]
                    Y = file["y_test"][:]
                self.n_images = int(X.shape[0])
                self.has_labels = True
                self.labels = torch.LongTensor(Y)
                self.images = torch.Tensor(X).float().reshape(-1, 1, 16, 16, 16)
                if self.config.preprocess is not None:
                    self.images = self.config.preprocess(self.images)

                self.n_channels = 1
                if self.images.ndim == 4: # grayscale B*D*H*W
                    self.images = self.images.unsqueeze(1) # B*C*D*H*W
                self.img_size = (self.images.shape[2], self.images.shape[3], self.images.shape[4])

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            # TODO: MNIST 3D AUGMENT
            ## TODO: TensorRandomRotation
            ## resized crop
            self.random_resized_crop = TensorRandomResizedCrop(p=0.6, size=self.img_size, scale=(0.9, 1.0), ratio_x=(0.75, 1.3333333333333333), ratio_y=(0.75, 1.3333333333333333), interpolation='trilinear')
            ## composition
            self.augment = Compose([self.random_resized_crop])


        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform


    def update(self, n_images, images, labels=None):
        """update online the dataset"""
        if labels is None:
            labels = torch.Tensor([-1] * n_images)
        assert n_images == images.shape[0] == labels.shape[0], print(
            'ERROR: the given dataset size ({0}) mismatch with observations size ({1}) and labels size ({2})'.format(
                n_images, images.shape[0], labels.shape[0]))

        self.n_images = int(n_images)
        self.images = images
        self.labels = labels

    def get_image(self, image_idx, augment=False, transform=True):
        image = self.images[image_idx]
        if augment and self.data_augmentation:
            image = self.augment(image)
        if transform and self.transform is not None:
            image = self.transform(image)
        return image

    def get_augmented_batch(self, image_ids, augment=True, transform=True):
        images_aug = []
        for img_idx in image_ids:
            image_aug = self.get_image(img_idx, augment=augment, transform=transform)
            images_aug.append(image_aug)
        images_aug = torch.stack(images_aug, dim=0)
        return images_aug

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]


        if self.data_augmentation:
            img_tensor = self.augment(img_tensor)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1


        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'obs': img_tensor, 'label': label, 'index': idx}

    def save(self, output_npz_filepath):
        np.savez(output_npz_filepath, n_images=self.n_images, images=np.stack(self.images),
                 labels=np.asarray(self.labels))
        return



# ===========================
# Lenia Dataset
# ===========================

class LENIADataset(Dataset):
    """ Lenia dataset"""

    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = None
        default_config.split = "train"

        # process data
        default_config.preprocess = None
        default_config.img_size = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.data_root is None:
            self.n_images = 0
            self.images = torch.zeros((0, 1, self.config.img_size[0], self.config.img_size[1]))
            if self.config.img_size is not None:
                self.img_size = self.config.img_size
                self.n_channels = 1
            self.labels = torch.zeros((0, 1), dtype=torch.long)

        else:
            # load HDF5 lenia dataset
            dataset_filepath = os.path.join(self.config.data_root, 'dataset', 'dataset.h5')
            with h5py.File(dataset_filepath, 'r') as file:
                if 'n_data' in file[self.config.split]:
                    self.n_images = int(file[self.config.split]['n_data'])
                else:
                    self.n_images = int(file[self.config.split]['observations'].shape[0])

                self.has_labels = bool('labels' in file[self.config.split])
                if self.has_labels:
                    self.labels = torch.LongTensor(file[self.config.split]['labels'])
                else:
                    self.labels = torch.LongTensor([-1] * self.n_images)

                self.images = torch.Tensor(file[self.config.split]['observations']).float()
                if self.config.preprocess is not None:
                    self.images = self.config.preprocess(self.images)

                self.n_channels = 1
                if self.images.ndim == 3:
                    self.images = self.images.unsqueeze(1)
                self.img_size = (self.images.shape[2], self.images.shape[3])

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            # LENIA Augment
            self.random_center_crop = TensorRandomCenterCrop(p=0.6, size=self.img_size, scale=(1.0, 2.0), ratio_x=(1., 1.), interpolation='bilinear')
            self.random_roll = TensorRandomRoll(p=(0.6, 0.6), max_delta=(0.5,0.5))
            self.random_spherical_rotation = TensorRandomSphericalRotation(p=0.6, max_degrees=20, n_channels=self.n_channels, img_size=self.img_size)
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            self.random_vertical_flip = RandomVerticalFlip(0.2)
            self.augment = Compose([self.random_center_crop, self.random_roll, self.random_spherical_rotation, to_PIL_image, self.random_horizontal_flip, self.random_vertical_flip, to_tensor])


        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def update(self, n_images, images, labels=None):
        """update online the dataset"""
        if labels is None:
            labels = torch.Tensor([-1] * n_images)
        assert n_images == images.shape[0] == labels.shape[0], print(
            'ERROR: the given dataset size ({0}) mismatch with observations size ({1}) and labels size ({2})'.format(
                n_images, images.shape[0], labels.shape[0]))

        self.n_images = int(n_images)
        self.images = images
        self.labels = labels

    def get_image(self, image_idx, augment=False, transform=True):
        image = self.images[image_idx]
        if augment and self.data_augmentation:
            image = self.augment(image)
        if transform and self.transform is not None:
            image = self.transform(image)
        return image

    def get_augmented_batch(self, image_ids, augment=True, transform=True):
        images_aug = []
        for img_idx in image_ids:
            image_aug = self.get_image(img_idx, augment=augment, transform=transform)
            images_aug.append(image_aug)
        images_aug = torch.stack(images_aug, dim=0)
        return images_aug

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]


        if self.data_augmentation:
            img_tensor = self.augment(img_tensor)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1


        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'obs': img_tensor, 'label': label, 'index': idx}

    def save(self, output_npz_filepath):
        np.savez(output_npz_filepath, n_images=self.n_images, images=np.stack(self.images),
                 labels=np.asarray(self.labels))
        return





# ===========================
# Quadruplet Dataset
# ===========================

def load_quadruplets_tl_from_annotations(datapoints, loss_type="triplet"):
    if loss_type == "triplet":
        # construct triplets from data
        refs = []
        positives = []
        negatives = []
        fourth = []
        for d in datapoints:
            # if not pass, add two triplets
            if d[-1] == 0:
                refs.append(d[0])
                positives.append(d[1])
                negatives.append(d[2])
                fourth.append(d[3])
                refs.append(d[0])
                positives.append(d[1])
                negatives.append(d[3])
                fourth.append(d[2])
        refs = np.array(refs).reshape(-1, 1)
        positives = np.array(positives).reshape(-1, 1)
        negatives = np.array(negatives).reshape(-1, 1)
        fourth = np.array(fourth).reshape(-1, 1)
        quadruplets = np.concatenate([refs, positives, negatives, fourth], axis=1)


    elif loss_type == "quadruplet":
        # construct triplets from data
        positive_pairs = []
        negative_pairs = []
        for d in datapoints:
            # if not pass, add two triplets
            if d[-1] == 0:
                for i in range(4):
                    for j in range(i + 1, 4):
                        if (i, j) != (0, 1):
                            positive_pairs.append([d[0], d[1]])
                            negative_pairs.append([d[i], d[j]])

        quadruplets = np.concatenate([positive_pairs, negative_pairs], axis=1)

    return quadruplets



class QuadrupletDataset(Dataset):
    """Triplet dataset but still returns quadruplets """

    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = None
        default_config.split = "train"
        default_config.n_quadruplets_per_epoch = None

        # quadruplet annotation if given
        default_config.annotations_filepath = None
        default_config.use_annotated_quadruplets = False
        #default_config.use_annotations_with_loss_type = "triplet" # either "triplet" or "quadruplet"
        #default_config.n_annotated_quadruplets_per_epoch = 100

        # process data
        default_config.img_size = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, dataset_name='lenia', config=None, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        # call super class to load data, augmentation and transform
        dataset_class = get_dataset(dataset_name)
        base_dataset = dataset_class(config, **kwargs)
        self.n_images = base_dataset.n_images
        self.images = base_dataset.images
        self.labels = base_dataset.labels
        self.data_augmentation = base_dataset.data_augmentation
        if self.config.data_augmentation:
            self.augment = base_dataset.augment
        self.transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        self.get_image = base_dataset.get_image

        # define number of quadruplets per epoch
        if self.config.n_quadruplets_per_epoch is None:
            self.config.n_quadruplets_per_epoch = int(self.n_images / 4)

        # load annotated quadruplets
        if self.config.annotations_filepath is not None:
            datapoints = np.load(self.config.annotations_filepath)[self.config.split]
            # annotated combinaisons
            self.annotated_quadruplets = []
            for d in datapoints:
                if d[-1] == 0:
                    self.annotated_quadruplets.append([d[0], d[1], d[2], d[3]])

            if self.config.use_annotated_quadruplets:
                # all combinaisons

                quadruplets = load_quadruplets_tl_from_annotations(datapoints, loss_type=self.config.use_annotations_with_loss_type)
                inds = np.arange(quadruplets.shape[0])
                np.random.shuffle(inds)
                self.quadruplets = quadruplets[inds].copy()

                if self.config.n_annotated_quadruplets_per_epoch is None:
                    self.config.n_annotated_quadruplets_per_epoch = quadruplets.shape[0]
                if self.config.n_quadruplets_per_epoch < self.config.n_annotated_quadruplets_per_epoch:
                    self.config.n_quadruplets_per_epoch = self.config.n_annotated_quadruplets_per_epoch
                    warnings.warn("WARNING: n_quadruplets_per_epoch < n_annotated_quadruplets_per_epoch, augmenting it!")


    def __len__(self):
        return self.config.n_quadruplets_per_epoch

    def __getitem__(self, index):
        if self.config.use_annotated_quadruplets and index < self.config.n_annotated_quadruplets_per_epoch:
            quadruplet_ids = self.quadruplets[index]
        else:
            # choose random images
            triplet_ids = np.random.choice(self.n_images, 3)
            quadruplet_ids = np.concatenate([[triplet_ids[0]], triplet_ids])

        cur_images = [self.images[i] for i in quadruplet_ids]
        cur_labels = [self.labels[i] for i in quadruplet_ids]

        if self.data_augmentation:
            for i in range(4):
                cur_images[i] = self.augment(cur_images[i])

        if self.transform is not None:
            for i in range(4):
                cur_images[i] = self.transform(cur_images[i])

        if self.target_transform is not None:
            for i in range(4):
                cur_labels[i] = self.target_transform(cur_labels[i])

        data_ref = {"obs": cur_images[0], "label": cur_labels[0], "index": quadruplet_ids[0]}
        data_a = {"obs": cur_images[1], "label": cur_labels[1], "index": quadruplet_ids[1]}
        data_b = {"obs": cur_images[2], "label": cur_labels[2], "index": quadruplet_ids[2]}
        data_c = {"obs": cur_images[3], "label": cur_labels[3], "index": quadruplet_ids[3]}

        return (data_ref, data_a, data_b, data_c)





'''

class DatasetLeniaHDF5(Dataset):
    """ Dataset to train auto-encoders representations during exploration"""

    def __init__(self, filepath, split='train', preprocess=None, data_augmentation=False):

        # 1) Load the HDF5 file

        self.filepath = filepath
        self.split = split
        with h5py.File(self.filepath, 'r') as file:
            if 'n_data' in file[self.split]:
                self.n_images = int(file[self.split]['n_data'])
            else:
                self.n_images = int(file[self.split]['observations'].shape[0])

            self.has_labels = bool('labels' in file[self.split])
            if self.has_labels:
                label_dtype = file[self.split]['labels'].dtype
                if (label_dtype == int) or (label_dtype == "<f4"):
                    self.n_classes = np.max(file[self.split]['labels']) - np.min(file[self.split]['labels']) + 1
                elif (label_dtype == np.bool):
                    self.n_classes = len(file[self.split]['labels'][0])
                else:
                    raise ValueError(
                        'The dataset label dtype {!r} has to be int (vategorical label) or np.bool (one-hot label)'.format(
                            label_dtype))
                self.labels = torch.Tensor(file[self.split]['labels'])
            else:
                self.labels = torch.Tensor([-1] * self.n_images)

            self.images = torch.Tensor(file[self.split]['observations']).float()
            self.img_size = (self.images.shape[2], self.images.shape[3])

        self.preprocess = preprocess

        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            radius = max(self.img_size[0], self.img_size[1]) / 2
            padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
            self.spheric_pad = SphericPad(
                padding_size=padding_size)  # max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            self.random_vertical_flip = RandomVerticalFlip(0.2)
            self.random_rotation = RandomRotation(40)
            self.center_crop = CenterCrop(self.img_size)
            self.roll_y = Roll(shift=0, dim=1)
            self.roll_x = Roll(shift=0, dim=2)

    def update(self, n_images, images, labels=None):
        if labels is None:
            labels = torch.Tensor([-1] * n_images)
        assert n_images == images.shape[0] == labels.shape[0], print(
            'ERROR: the given dataset size ({0}) mismatch with observations size ({1}) and labels size ({2})'.format(
                n_images, images.shape[0], labels.shape[0]))

        self.n_images = int(n_images)
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
            # random rolled translation (ie pixels shifted outside appear on the other side of image))
            p_y = p_x = 0.3

            if np.random.random() < p_y:
                ## the maximum translation is of half the image size
                max_dy = 0.5 * self.img_size[0]
                shift_y = int(np.round(np.random.uniform(-max_dy, max_dy)))
                self.roll_y.shift = shift_y
                img_tensor = self.roll_y(img_tensor)

            if np.random.random() < p_x:
                max_dx = 0.5 * self.img_size[1]
                shift_x = int(np.round(np.random.uniform(-max_dx, max_dx)))
                self.roll_x.shift = shift_x
                img_tensor = self.roll_x(img_tensor)

            # random spherical padding + rotation (avoid "black holes" when rotating)
            p_r = 0.3

            if np.random.random() < p_r:
                img_tensor = self.spheric_pad(
                    img_tensor.view(1, img_tensor.size(0), img_tensor.size(1), img_tensor.size(2))).squeeze(0)
                img_PIL = to_PIL_image(img_tensor)
                img_PIL = self.random_rotation(img_PIL)
                img_PIL = self.center_crop(img_PIL)
                img_tensor = to_tensor(img_PIL)

            img_PIL = to_PIL_image(img_tensor)
            # random horizontal flip
            img_PIL = self.random_horizontal_flip(img_PIL)
            # random vertical flip
            img_PIL = self.random_vertical_flip(img_PIL)
            # convert back to tensor
            img_tensor = to_tensor(img_PIL)

        if self.preprocess:
            img_tensor = self.preprocess(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        return {'obs': img_tensor, 'label': label}

    def save(self, output_npz_filepath):
        np.savez(output_npz_filepath, n_images=self.n_images, images=np.stack(self.images),
                 labels=np.asarray(self.labels))
        return


class DatasetHDF5(Dataset):
    """
    Dataset to train auto-encoders representations during exploration from datatsets in hdf5 files.

    TODO: add a cache for loaded objects to be faster (see https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5)
    TODO: solve the bug with hdf5 and multiple workers process 
    TODO: add data_augmentation and preprocess
    """

    def __init__(self, filepath, split='train', preprocess=None, data_augmentation=False):

        self.filepath = filepath
        self.split = split

        # HDF5 file isn’t pickleable and to send Dataset to workers’ processes it needs to be serialised with pickle, self.file will be opened in __getitem__ 
        self.data_group = None

        with h5py.File(self.filepath, 'r') as file:
            if 'n_data' in file[self.split]:
                self.n_data = int(file[self.split]['n_data'])
            else:
                self.n_data = int(file[self.split]['observations'].shape[0])

            self.has_labels = bool('labels' in file[self.split])
            if self.has_labels:
                label_dtype = file[self.split]['labels'].dtype
                if (label_dtype == int) or (label_dtype == "<f4"):
                    self.n_classes = np.max(file[self.split]['labels']) - np.min(file[self.split]['labels']) + 1
                elif (label_dtype == np.bool):
                    self.n_classes = len(file[self.split]['labels'][0])
                else:
                    raise ValueError(
                        'The dataset label dtype {!r} has to be int (vategorical label) or np.bool (one-hot label)'.format(
                            label_dtype))

        self.preprocess = preprocess
        self.data_augmentation = data_augmentation

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        # open the HDF5 file here and store as the singleton. Do not open it each time as it introduces huge overhead.
        if self.data_group is None:
            self.data_group = h5py.File(self.filepath, "r")[self.split]

        # image
        img_tensor = torch.from_numpy(self.data_group['observations'][idx, :, :]).float()

        # label
        label = -1
        if self.has_labels:
            label = self.data_group['labels'][idx]
            if label.dtype == float:
                label = int(label)

        return {'obs': img_tensor, 'label': label}


# ===========================
# Triplet/Quadruplet Dataset
# ===========================
class ImageTripletDatasetFromHuman(Dataset):
    def __init__(self, dataset_filepath, anno_filepath, split='train', data_size=None, use_pass=False,
                 preprocess=None, data_augmentation=False):

        self.dataset_filepath = dataset_filepath
        assert split == 'train' or split == 'test'
        self.split = split
        self.use_pass = use_pass
        self.preprocess = preprocess
        self.data_augmentation = data_augmentation

        datapoints = np.load(anno_filepath)[self.split]

        # construct triplets from data
        refs = []
        positives = []
        negatives = []
        fourth = []
        for d in datapoints:
            if self.use_pass:
                # if pass, add 6 triplets (positive pair being AA, negative pairs AB, AC, AD)
                if d[-1] == 1:
                    for i in range(4):
                        for j in range(i + 1, 4):
                            refs.append(d[i])
                            positives.append(d[i])
                            negatives.append(d[j])
                            fourth.append(d[j])
            # if not pass, add two triplets
            if d[-1] == 0:
                refs.append(d[0])
                positives.append(d[1])
                negatives.append(d[2])
                fourth.append(d[3])
                refs.append(d[0])
                positives.append(d[1])
                negatives.append(d[3])
                fourth.append(d[2])
        refs = np.array(refs).reshape(-1, 1)
        positives = np.array(positives).reshape(-1, 1)
        negatives = np.array(negatives).reshape(-1, 1)
        fourth = np.array(fourth).reshape(-1, 1)
        quadruplets = np.concatenate([refs, positives, negatives, fourth], axis=1)

        inds = np.arange(quadruplets.shape[0])

        np.random.shuffle(inds)
        with h5py.File(dataset_filepath, 'r') as file:
            self.n_images = len(file[self.split]['observations'])
            self.img_size = (file[self.split]['observations'].shape[2], file[self.split]['observations'].shape[3])

        self.quadruplets = quadruplets[inds].copy()

        if data_size is None:
            self.data_size = quadruplets.shape[0]
        else:
            self.data_size = data_size

        # save annotated quadruplets for evaluation
        self.annotated_quadruplets = []
        for d in datapoints:
            # if not pass, add two triplets
            if d[-1] == 0:
                self.annotated_quadruplets.append([d[0], d[1], d[2], d[3]])

        # data augmentation
        if self.data_augmentation:
            self.random_center_crop = RandomCenterCrop(0.8, crop_ratio=(1, 2), keep_img_size=True)
            radius = max(self.img_size[0], self.img_size[1]) / 2
            padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
            self.spheric_pad = SphericPad(padding_size=padding_size)  # max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            self.random_vertical_flip = RandomVerticalFlip(0.2)
            self.random_rotation = RandomRotation(20, fill=(0,))
            self.center_crop = CenterCrop(self.img_size)
            self.roll_y = Roll(shift=0, dim=1)
            self.roll_x = Roll(shift=0, dim=2)

    def get_image(self, image_idx):
        with h5py.File(self.dataset_filepath, 'r') as file:
            image = torch.Tensor(file[self.split]['observations'][image_idx]).float()
        return image

    def __getitem__(self, index):
        if index < self.quadruplets.shape[0]:
            quadruplet_ids = self.quadruplets[index]
        else:
            # choose random images
            triplet_ids = np.random.choice(self.n_images, 3)
            quadruplet_ids = np.concatenate([[triplet_ids[0]], triplet_ids])

        with h5py.File(self.dataset_filepath, 'r') as file:
            images = [torch.Tensor(file[self.split]['observations'][quadruplet_ids[i]]).float() for i in range(4)]
            labels = [int(file[self.split]['labels'][quadruplet_ids[i]]) for i in range(4)]


        if self.data_augmentation:
            for i in range(4):
                img_tensor = images[i]

                # random center crop
                self.random_center_crop(img_tensor)

                # random rolled translation (ie pixels shifted outside appear on the other side of image))
                p_y = p_x = 0.8

                if np.random.random() < p_y:
                    ## the maximum translation is of half the image size
                    max_dy = 0.5 * self.img_size[0]
                    shift_y = int(np.round(np.random.uniform(-max_dy, max_dy)))
                    self.roll_y.shift = shift_y
                    img_tensor = self.roll_y(img_tensor)

                if np.random.random() < p_x:
                    max_dx = 0.5 * self.img_size[1]
                    shift_x = int(np.round(np.random.uniform(-max_dx, max_dx)))
                    self.roll_x.shift = shift_x
                    img_tensor = self.roll_x(img_tensor)

                # random spherical padding + rotation (avoid "black holes" when rotating)
                p_r = 0.8

                if np.random.random() < p_r:
                    img_tensor = self.spheric_pad(
                        img_tensor.view(1, img_tensor.size(0), img_tensor.size(1), img_tensor.size(2))).squeeze(0)
                    img_PIL = to_PIL_image(img_tensor)
                    img_PIL = self.random_rotation(img_PIL)
                    img_PIL = self.center_crop(img_PIL)
                    img_tensor = to_tensor(img_PIL)

                img_PIL = to_PIL_image(img_tensor)
                # random horizontal flip
                img_PIL = self.random_horizontal_flip(img_PIL)
                # random vertical flip
                img_PIL = self.random_vertical_flip(img_PIL)
                # convert back to tensor
                images[i] = to_tensor(img_PIL)


        if self.preprocess is not None:
            for i in range(4):
                images[i] = self.preprocess(images[i]).float()


        data_ref = {"obs": images[0], "label": labels[0]}
        data_a = {"obs": images[1], "label": labels[1]}
        data_b = {"obs": images[2], "label": labels[2]}
        data_c = {"obs": images[3], "label": labels[3]}

        return (data_ref, data_a, data_b, data_c)

    def __len__(self):
        return self.data_size


class ImageQuadrupletDatasetFromHuman(Dataset):
    def __init__(self, dataset_filepath, anno_filepath, split='train', data_size=None, use_pass=False,
                 preprocess=None, data_augmentation=False):

        self.dataset_filepath = dataset_filepath
        assert split == 'train' or split == 'test'
        self.split = split
        self.use_pass = use_pass
        self.preprocess = preprocess
        self.data_augmentation = data_augmentation

        datapoints = np.load(anno_filepath)[self.split]

        # construct triplets from data
        positive_pairs = []
        negative_pairs = []
        for d in datapoints:
            if self.use_pass:
                # if pass, add 6 triplets (positive pair being AA, negative pairs AB, AC, AD)
                if d[-1] == 1:
                    for i in range(4):
                        for j in range(i + 1, 4):
                            positive_pairs.append([d[i], d[i]])
                            negative_pairs.append([d[i], d[j]])
            # if not pass, add two triplets
            if d[-1] == 0:
                for i in range(4):
                    for j in range(i + 1, 4):
                        if (i, j) != (0, 1):
                            positive_pairs.append([d[0], d[1]])
                            negative_pairs.append([d[i], d[j]])

        quadruplets = np.concatenate([positive_pairs, negative_pairs], axis=1)

        inds = np.arange(quadruplets.shape[0])

        np.random.shuffle(inds)
        with h5py.File(self.dataset_filepath, 'r') as file:
            self.n_images = len(file[self.split]['observations'])
            self.img_size = (file[self.split]['observations'].shape[2], file[self.split]['observations'].shape[3])

        self.quadruplets = quadruplets[inds].copy()

        if data_size is None:
            self.data_size = quadruplets.shape[0]
        else:
            self.data_size = data_size

        # save annotated quadruplets for evaluation
        self.annotated_quadruplets = []
        for d in datapoints:
            # if not pass, add two triplets
            if d[-1] == 0:
                self.annotated_quadruplets.append([d[0], d[1], d[2], d[3]])

        # data augmentation
        if self.data_augmentation:
            self.random_center_crop = RandomCenterCrop(0.8, crop_ratio=(1, 2), keep_img_size=True)
            radius = max(self.img_size[0], self.img_size[1]) / 2
            padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
            self.spheric_pad = SphericPad(
                padding_size=padding_size)  # max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            self.random_vertical_flip = RandomVerticalFlip(0.2)
            self.random_rotation = RandomRotation(20, fill=(0,))
            self.center_crop = CenterCrop(self.img_size)
            self.roll_y = Roll(shift=0, dim=1)
            self.roll_x = Roll(shift=0, dim=2)


    def get_image(self, image_idx):
        with h5py.File(self.dataset_filepath, 'r') as file:
            image = torch.Tensor(file[self.split]['observations'][image_idx]).float()
        return image

    def __getitem__(self, index):
        if index < self.quadruplets.shape[0]:
            quadruplet_ids = self.quadruplets[index]
        else:
            # choose random images
            triplet_ids = np.random.choice(self.n_images, 3)
            quadruplet_ids = np.concatenate([[triplet_ids[0]], triplet_ids])

        with h5py.File(self.dataset_filepath, 'r') as file:
            images = [torch.Tensor(file[self.split]['observations'][quadruplet_ids[i]]).float() for i in range(4)]
            labels = [int(file[self.split]['labels'][quadruplet_ids[i]]) for i in range(4)]

        if self.data_augmentation:
            for i in range(4):
                img_tensor = images[i]

                # random center crop
                self.random_center_crop(img_tensor)

                # random rolled translation (ie pixels shifted outside appear on the other side of image))
                p_y = p_x = 0.3

                if np.random.random() < p_y:
                    ## the maximum translation is of half the image size
                    max_dy = 0.5 * self.img_size[0]
                    shift_y = int(np.round(np.random.uniform(-max_dy, max_dy)))
                    self.roll_y.shift = shift_y
                    img_tensor = self.roll_y(img_tensor)

                if np.random.random() < p_x:
                    max_dx = 0.5 * self.img_size[1]
                    shift_x = int(np.round(np.random.uniform(-max_dx, max_dx)))
                    self.roll_x.shift = shift_x
                    img_tensor = self.roll_x(img_tensor)

                # random spherical padding + rotation (avoid "black holes" when rotating)
                p_r = 0.3

                if np.random.random() < p_r:
                    img_tensor = self.spheric_pad(
                        img_tensor.view(1, img_tensor.size(0), img_tensor.size(1), img_tensor.size(2))).squeeze(0)
                    img_PIL = to_PIL_image(img_tensor)
                    img_PIL = self.random_rotation(img_PIL)
                    img_PIL = self.center_crop(img_PIL)
                    img_tensor = to_tensor(img_PIL)

                img_PIL = to_PIL_image(img_tensor)
                # random horizontal flip
                img_PIL = self.random_horizontal_flip(img_PIL)
                # random vertical flip
                img_PIL = self.random_vertical_flip(img_PIL)
                # convert back to tensor
                images[i] = to_tensor(img_PIL)

        if self.preprocess is not None:
            for i in range(4):
                images[i] = self.preprocess(images[i]).float()

        data_pos_a = {"obs": images[0], "label": labels[0]}
        data_pos_b = {"obs": images[1], "label": labels[1]}
        data_neg_a = {"obs": images[2], "label": labels[2]}
        data_neg_b = {"obs": images[3], "label": labels[3]}

        return (data_pos_a, data_pos_b, data_neg_a, data_neg_b)

    def __len__(self):
        return self.data_size

'''