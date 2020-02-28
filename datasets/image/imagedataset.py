import goalrepresent as gr
from goalrepresent.helper.nnmodulehelper import Roll, SphericPad
import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import torchvision
from torchvision.transforms import CenterCrop, ToTensor, ToPILImage, RandomHorizontalFlip, RandomResizedCrop, \
    RandomVerticalFlip, RandomRotation
import warnings

to_tensor = ToTensor()
to_PIL_image = ToPILImage()


# ================
# Generic Datasets
# ================

class HOLMESDataset(Dataset):

    @staticmethod
    def default_config():
        default_config = gr.Config()
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None
        default_config.download = True

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

        # initially dataset lists are empty
        self.n_images = 0
        self.images = [] # list or torch tensor of size N*C*H*W
        self.labels = [] # list or torch tensor

        self.datasets = {}
        for dataset in self.config.datasets:
            dataset_class = get_torchvisiondataset(dataset["name"])
            self.datasets[dataset["name"]] = dataset_class(config=dataset.config)

    def update(self, n_images, images, labels=None):
        """ Update the current dataset lists """
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

        if self.data_augmentation is not None:
            pass

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'obs': img_tensor, 'label': label}

    def save(self, output_npz_filepath):
        np.savez(output_npz_filepath, n_images=self.n_images, images=np.stack(self.images),
                 labels=np.asarray(self.labels))




# =================
# Torchvision Datasets
# =================
def get_torchvisiondataset(dataset_name):
    """
    dataset_name: string such that the model called is <dataset_name>Dataset
    """
    return eval("gr.datasets.{}Dataset".format(dataset_name.upper()))

class MNISTDataset(torchvision.datasets.MNIST):
    name = 'mnist'
    num_classes = 10

    def __init__(self, config=None, **kwargs):
        if config.split == "train":
            train = True
        elif config.split == "valid":
            train = False
            warnings.warn("WARNING: MNIST does not have a valid dataset so the test set is given, should NOT BE USED for model selection")
        elif config.split == "test":
            train = False
        else:
            raise ValueError("MNIST dataset does not have a {} split".format((config.split)))
        torchvision.datasets.MNIST.__init__(
            self, root=os.path.join(config['data_root'], self.name),
            train=train, download=config.download)

        # self.data transform to N*C*H*W, float type between 0 and 1, preprocess
        self.images = self.data.unsqueeze(1)
        self.images = self.images.float() / 255.0
        if config.preprocess is not None:
            self.images = config.preprocess(self.images)
        del self.data

        self.labels = self.targets
        del self.targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
       return {'obs': self.images[index], 'label': self.labels[index]}


class SVHNDataset(torchvision.datasets.SVHN):
    name = 'svhn'
    num_classes = 10

    def __init__(self, config=None, **kwargs):
        if config.split == "valid":
            config.split = "test"
            warnings.warn(
                "WARNING: SVHN does not have a valid dataset so the test set is given, should NOT BE USED for model selection")
        torchvision.datasets.SVHN.__init__(
            self, root=os.path.join(config['data_root'], self.name),
            split=config.split, download=config.download)

        # self.data transform to N*C*H*W, float type between 0 and 1, preprocess
        self.images = torch.from_numpy(self.data)
        self.images = self.images.float() / 255.0
        if config.preprocess is not None:
            self.images = config.preprocess(self.images)
        del self.data

        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
       return {'obs': self.images[index], 'label': self.labels[index]}

class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    name = 'cifar10'
    num_classes = 10

    def __init__(self, config=None, **kwargs):
        if config.split == "train":
            train = True
        elif config.split == "valid":
            train = False
            warnings.warn(
                "WARNING: CIFAR10 does not have a valid dataset so the test set is given, should NOT BE USED for model selection")
        elif config.split == "test":
            train = False
        else:
            raise ValueError("CIFAR10 dataset does not have a {} split".format((config.split)))
        torchvision.datasets.CIFAR10.__init__(
            self, root=os.path.join(config['data_root'], self.name),
            train=train, download=config.download)

        # self.data transform to N*C*H*W, float type between 0 and 1, preprocess
        self.images = torch.from_numpy(self.data)
        self.images = self.images.float() / 255.0
        if config.preprocess is not None:
            self.images = config.preprocess(self.images)
        del self.data

        self.labels = self.targets
        del self.targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
       return {'obs': self.images[index], 'label': self.labels[index]}



class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    name = 'cifar100'
    num_classes = 100

    def __init__(self, config=None, **kwargs):
        if config.split == "train":
            train = True
        elif config.split == "valid":
            train = False
            warnings.warn(
                "WARNING: CIFAR100 does not have a valid dataset so the test set is given, should NOT BE USED for model selection")
        elif config.split == "test":
            train = False
        else:
            raise ValueError("CIFAR100 dataset does not have a {} split".format((config.split)))
        torchvision.datasets.CIFAR100.__init__(
            self, root=os.path.join(config['data_root'], self.name),
            train=train, transform=config.transform, download=config.download)

        # self.data transform to N*C*H*W, float type between 0 and 1, preprocess
        self.images = torch.from_numpy(self.data)
        self.images = self.images.float() / 255.0
        if config.preprocess is not None:
            self.images = config.preprocess(self.images)
        del self.data

        self.labels = self.targets
        del self.targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
       return {'obs': self.images[index], 'label': self.labels[index]}



class DatasetLenia(Dataset):
    """ Dataset to train auto-encoders representations during exploration"""

    def __init__(self, img_size, preprocess=None, data_augmentation=False):

        self.n_images = 0
        self.images = []
        self.labels = []

        self.img_size = img_size

        self.preprocess = preprocess

        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            radius = max(self.img_size[0], self.img_size[1]) / 2
            padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
            self.spheric_pad = SphericPad(
                padding_size=padding_size)  # max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            self.random_vertical_flip = RandomVerticalFlip(0.2)
            self.random_resized_crop = RandomResizedCrop(size=self.img_size)
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
                self.roll_y.shift = shift_x
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

