from addict import Dict
import numpy as np
import MinkowskiEngine as ME
import torch
from torch.utils.data import Dataset
import os
import glob
import open3d as o3d
import h5py
from collections import OrderedDict
import warnings

class HDF5Dataset(Dataset):
    """ Represents an abstract HDF5 dataset.

    Input params:
        filepath: Path to the dataset (if None: creates an empty dataset).
        preprocess: PyTorch transform to apply when creating a new data tensor instance (default=None).
        transform: PyTorch transform to apply on-the-fly to every data tensor instance (default=None).
        label_transform: PyTorch transform to apply on-the-fly to every data label instance (default=None).
        Configuration:
            obs_size: size of a data tensor (D,H,W) for 3D or (H,W) for 2D
            n_channels: number of channels of a data tensor
            load_data: If True, loads all the data immediately into RAM. Use this if
                the dataset is fits into memory. Otherwise, leave this at false and
                the data will load lazily.
            data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
    """

    @staticmethod
    def default_config():
        default_config = Dict()

        # data info
        default_config.coords_size = 0 # 4 for (B,D,H,W) tensors or 3 for (B,H,W) tensors
        default_config.coords_dtype = "int"
        default_config.feats_size = 0  #n_channels
        default_config.feats_dtype = "float32"
        default_config.label_size = (1, )
        default_config.label_dtype = "long"

        # load data
        default_config.load_data = False
        default_config.data_cache_size = 100

        return default_config

    def __init__(self, filepath=None, coords_preprocess=None, feats_preprocess=None, coords_transform=None, feats_transform=None, label_transform=None, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        self.filepath = filepath
        self.coords_preprocess = coords_preprocess
        self.feats_preprocess = feats_preprocess
        self.coords_transform = coords_transform
        self.feats_transform = feats_transform
        self.label_transform = label_transform

        if not os.path.exists(self.filepath):
            warnings.warn(f"Filepath {filepath} does not exists, creating an empty Dataset")

            self.data_ids = set()
            self.data_cache = OrderedDict()
            self.data_ids_in_cache = []

            # create empty dataset to be filled incrementally by the add_data function
            # different tensor sizes can be handled by h5py special dtype (each tensor must be flatten to 1D though)
            coords_dtype = h5py.special_dtype(vlen=np.dtype(self.config.coords_dtype))
            feats_dtype = h5py.special_dtype(vlen=np.dtype(self.config.feats_dtype))
            with h5py.File(self.filepath, "w") as file:
                file.attrs["n_obs"] = 0
                # unlimited number of data can be appended with maxshape=None
                file.create_dataset("coordinates", (0,) , coords_dtype, maxshape=(None,))
                file.create_dataset("features", (0,), feats_dtype, maxshape=(None,))
                file.create_dataset("labels", (0,) + self.config.label_size, self.config.label_dtype, maxshape=(None,) + self.config.label_size)

        else:
            # load HDF5  dataset
            with h5py.File(self.filepath, "r") as file:
                if "n_obs" in file:
                    n_obs = int(file["n_obs"])
                else:
                    n_obs = int(file["coordinates"].shape[0])

                self.data_ids = set(range(n_obs))
                self.data_cache = OrderedDict()
                self.data_ids_in_cache = []

                if self.config.load_data:
                    for data_idx in self.data_ids:
                        self.data_cache[data_idx] = {
                            "coords": torch.Tensor(file["coordinates"][data_idx]).type(eval(f"torch.{self.config.coords_dtype}")).view(-1, self.config.coords_size),
                            "feats": torch.Tensor(file["features"][data_idx]).type(eval(f"torch.{self.config.feats_dtype}")).view(-1, self.config.feats_size),
                            "label": torch.Tensor(file["labels"][data_idx]).type(eval(f"torch.{self.config.label_dtype}"))}
                        self.data_ids_in_cache.append(data_idx)
                else:
                    max_idx = min(len(self.data_ids), self.config.data_cache_size)
                    for data_idx in range(max_idx):
                        self.data_cache[data_idx] = {
                            "coords": torch.Tensor(file["coordinates"][data_idx]).type(eval(f"torch.{self.config.coords_dtype}")).view(-1, self.config.coords_size),
                            "feats": torch.Tensor(file["features"][data_idx]).type(eval(f"torch.{self.config.feats_dtype}")).view(-1, self.config.feats_size),
                            "label": torch.Tensor(file["labels"][data_idx]).type(eval(f"torch.{self.config.label_dtype}"))}
                        self.data_ids_in_cache.append(data_idx)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data = self.get_data(idx)
        coords = data["coords"]
        feats = data["feats"]
        label = data["label"]

        if self.coords_transform is not None:
            coords = self.coords_transform(coords)

        if self.feats_transform is not None:
            feats = self.feats_transform(feats)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return {"coords": coords, "feats": feats, "label": label, "index": idx}

    def get_data(self, data_idx):
        if not data_idx in self.data_ids:
            raise ValueError(f"data with key ID {data_idx} is not in the database")

        elif data_idx in self.data_ids_in_cache:
            return self.data_cache[data_idx]

        else:
            with h5py.File(self.filepath, "r") as file:
                return {"coords": torch.Tensor(file["coordinates"][data_idx]).type(eval(f"torch.{self.config.coords_dtype}")).view(-1, self.config.coords_size),
                        "feats": torch.Tensor(file["features"][data_idx]).type(eval(f"torch.{self.config.feats_dtype}")).view(-1, self.config.feats_size),
                        "label": torch.Tensor(file["labels"][data_idx]).type(eval(f"torch.{self.config.label_dtype}"))}

    def add_data(self, coords, feats, label, add_to_cache=True):

        new_data_idx = len(self.data_ids)
        assert new_data_idx not in self.data_ids

        assert isinstance(coords, torch.Tensor)
        assert isinstance(feats, torch.Tensor)
        assert isinstance(label, torch.Tensor)

        if coords.dtype != self.config.coords_dtype:
            coords = coords.type(eval(f"torch.{self.config.coords_dtype}"))
        if self.coords_preprocess is not None:
            coords = self.coords_preprocess(coords)

        if feats.dtype != self.config.feats_dtype:
            feats = feats.type(eval(f"torch.{self.config.feats_dtype}"))
        if self.feats_preprocess is not None:
            feats = self.feats_preprocess(feats)

        if label.dtype != self.config.label_dtype:
            label = label.type(eval(f"torch.{self.config.label_dtype}"))

        assert (coords.shape[-1] == self.config.coords_size)
        assert (feats.shape[-1] == self.config.feats_size)
        assert (label.shape == self.config.label_size)

        with h5py.File(self.filepath, "a") as file:
            cur_size = file.attrs["n_obs"]
            file["coordinates"].resize(cur_size + 1, axis=0)
            file["coordinates"][new_data_idx] = coords.detach().cpu().numpy().flatten()
            file["features"].resize(cur_size + 1, axis=0)
            file["features"][new_data_idx] = feats.detach().cpu().numpy().flatten()
            file["labels"].resize(cur_size + 1, axis=0)
            file["labels"][new_data_idx, :] = label.detach().cpu().numpy()
            file.attrs["n_obs"] = cur_size + 1

        if self.config.load_data:
            self.data_cache[new_data_idx] = {"coords": coords.detach().cpu(), "feats": feats.detach().cpu(), "label": label.detach().cpu()}
            self.data_ids_in_cache.append(new_data_idx)

        elif add_to_cache:
            # remove last item from cache when not enough size
            if len(self.data_ids_in_cache) > self.config.data_cache_size:
                del (self.data_cache[self.data_ids_in_cache[0]])
                del (self.data_ids_in_cache[0])
            self.data_cache[new_data_idx] = {"coords": coords.detach().cpu(), "feats": feats.detach().cpu(), "label": label.detach().cpu()}
            self.data_ids_in_cache.append(new_data_idx)

        self.data_ids.add(new_data_idx)
        return

    def empty_cache(self):
        self.data_cache = OrderedDict()
        self.data_ids_in_cache = []
        return


# ===========================================================================================================================================================================================

def sparse_collation(data_list):
    
    # Create batched coordinates for the SparseTensor input
    coords = [data["coords"] for data in data_list]
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats = [data["feats"] for data in data_list]
    bfeats = torch.cat(feats, 0).float()

    labels = [torch.tensor([data["label"]]).long() for data in data_list]
    blabels = torch.cat(labels, 0)
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
        self.feats = [torch.ones((len(self.coords[i]), self.n_channels)).to(self.dtype) for i in range(self.n_images)]


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


# ==================================
# Torch3D Datasets
# ===============================

def resample_mesh(mesh_cad, density=1):
    """
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.
    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud
    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (
        (1 - np.sqrt(r[:, 0:1])) * A
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    return P

class ModelNet40Dataset(Dataset):

    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = 'dataset'
        default_config.download = True
        default_config.split = "train"

        # Resample mesh specifis
        default_config.resolution = 128
        default_config.density = 30000

        # process data
        default_config.preprocess = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.split == "train":
            split = "train"
        elif self.config.split in ["valid", "test"]:
            split = "test"
        fnames = glob.glob(os.path.join(self.config.data_root, "ModelNet40/", f"chair/{split}/*.off"))
        fnames = sorted([os.path.relpath(fname, os.path.join(self.config.data_root, "ModelNet40/")) for fname in fnames])
        assert len(fnames) > 0, "No file loaded"

        self.fnames = []
        self.n_images = 0
        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        for file in fnames:

            mesh_file = os.path.join(self.config.data_root, "ModelNet40/", file)
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
            vertices = np.asarray(pcd.vertices)
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector(
                (vertices - vmin) / (vmax - vmin).max()
            )

            # Oversample points and copy
            xyz = resample_mesh(pcd, density=self.config.density)

            if len(xyz) < 1000:
                print(f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}.")

            else:
                self.fnames.append(file)
                self.n_images += 1

        self.n_channels = 1
        self.img_size = (self.config.resolution, self.config.resolution, self.config.resolution) #TODO: check here!
        self.dtype = torch.float32


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
        file = self.fnames[idx]
        mesh_file = os.path.join(self.config.data_root, "ModelNet40/", file)
        pcd = o3d.io.read_triangle_mesh(mesh_file)
        # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
        vertices = np.asarray(pcd.vertices)
        vmax = vertices.max(0, keepdims=True)
        vmin = vertices.min(0, keepdims=True)
        pcd.vertices = o3d.utility.Vector3dVector(
            (vertices - vmin) / (vmax - vmin).max()
        )
        xyz = resample_mesh(pcd, density=self.config.density)
        xyz = xyz * self.config.resolution
        coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)

        # image
        coords_tensor = coords
        feats_tensor = torch.from_numpy(xyz[inds]).float()

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

        return {'coords': coords_tensor, 'feats': feats_tensor, 'label': label, 'index': idx}


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

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.data_root is None:
            self.n_images = 0
            if self.config.img_size is not None:
                self.img_size = self.config.img_size
                self.coords = []
                self.feats = []
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
                self.img_size = (16, 16, 16)
                self.n_channels = 1
                self.coords = []
                self.feats = []
                images = torch.Tensor(X).float().reshape((-1, self.n_channels, ) + self.img_size)
                images_sparse = ME.to_sparse(images)
                for idx in range(len(images)):
                    self.coords.append(images_sparse.coordinates_at(idx))
                    self.feats.append(images_sparse.features_at(idx))

        if self.config.preprocess is not None:
            self.coords, self.feats = self.config.preprocess(self.coords, self.feats)


        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            self.augment = []

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


# ===========================
# SimCells Dataset
# ===========================

class SimCellsDataset(Dataset):
    """ SimCells dataset"""

    @staticmethod
    def default_config():
        default_config = Dict()

        # load data
        default_config.data_root = None
        default_config.split = "train"

        # process data
        default_config.preprocess = None
        default_config.img_size = None
        default_config.n_channels = 4
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.data_root is None:
            self.n_images = 0
            if self.config.img_size is not None:
                self.coords = []
                self.feats = []
                self.img_size = self.config.img_size
                self.n_channels = self.config.n_channels
            self.labels = torch.zeros((0, 1), dtype=torch.long)

        else:
            # TODO: load HDF5 SimCells dataset
            pass

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            self.augment = [] #TODO: augmentation

        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def add(self, image, label):
        coords = image.C
        feats = image.F
        if self.config.preprocess is not None:
            coords, feats = self.config.preprocess(coords, feats)
        self.coords.append(coords)
        self.feats.append(feats)
        self.labels = torch.cat([self.labels, label])
        self.n_images += 1

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


        return {'coords': coords_tensor, 'feats': feats_tensor, 'label': label, 'index': idx}

