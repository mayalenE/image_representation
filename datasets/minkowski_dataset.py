from addict import Dict
import numpy as np
import MinkowskiEngine as ME
import torch
from torch.utils.data import Dataset

def sparse_collation(data_list):
    
    # Create batched coordinates for the SparseTensor input
    coords = [data["coords"] for data in data_list]
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats = [data["feats"] for data in data_list]
    bfeats = torch.cat(feats, 0).float()

    blabels = [data["label"] for data in data_list]
    binds = [data["index"] for data in data_list]

    return {'coords': bcoords, 'feats': bfeats, 'label': blabels, 'index': binds}

class RandomDataset(Dataset):

    @staticmethod
    def default_config():
        default_config = Dict()

        # process data
        default_config.preprocess = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, n_images=100, n_channels=10, img_size=(64, 64), dtype=torch.float32, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        self.n_images = n_images
        self.n_channels = n_channels
        self.img_size = img_size
        self.dtype = dtype


        self.coords = [torch.stack(torch.where(torch.randint(0,2,size=self.img_size).bool())).t().int().contiguous() for _ in range(self.n_images)]
        self.feats = [torch.arange(len(self.coords[i]) * self.n_channels).view(len(self.coords[i]), self.n_channels).to(self.dtype) for i in range(self.n_images)]


        if self.config.preprocess is not None:
            self.coords, self.feats = self.config.preprocess(self.coords, self.feats)

        max_label = 10
        self.labels = (torch.rand(self.n_images) * max_label).int()

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            ## composition
            self.augment = []

        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def get_image(self, image_idx, augment=False, transform=True):
        coords = self.coords[image_idx]
        feats = self.feats[image_idx]
        if augment and self.data_augmentation:
            coords, feats = self.augment(coords, feats)
        if transform and self.transform is not None:
            coords, feats = self.transform(coords, feats)
        return coords, feats

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        coords_tensor = self.coords[idx]
        feats_tensor = self.feats[idx]

        if self.data_augmentation:
            coords_tensor, feats_tensor = self.augment(coords_tensor, feats_tensor)

        if self.transform is not None:
            coords_tensor, feats_tensor = self.transform(coords_tensor, feats_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            label = self.target_transform(label)

        inds = ME.utils.sparse_quantize(
            coords_tensor, return_index=True, return_maps_only=True
        )

        return {'coords': coords_tensor[inds], 'feats': feats_tensor[inds], 'label': label, 'index': idx}
