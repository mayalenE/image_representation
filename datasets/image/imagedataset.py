import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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
                if (label_dtype == int):
                    self.n_classes = np.max(file[self.split]['labels']) - np.min(file[self.split]['labels']) + 1
                elif (label_dtype == np.bool):
                    self.n_classes = len(file[self.split]['labels'][0])
                else:
                    raise ValueError('The dataset label dtype {!r} has to be int (vategorical label) or np.bool (one-hot label)'.format(label_dtype))

        self.preprocess = preprocess 
        self.data_augmentation = data_augmentation

    def __len__(self):
        return self.n_data


    def __getitem__(self, idx):
        # open the HDF5 file here and store as the singleton. Do not open it each time as it introduces huge overhead.
        if self.data_group is None:
            self.data_group = h5py.File(self.filepath , "r")[self.split]
        
        # image
        img_tensor = torch.from_numpy(self.data_group['observations'][idx,:,:]).float()    

        # label
        label = -1
        if self.has_labels:
            label = self.data_group['labels'][idx]

        return {'obs': img_tensor, 'label': label}