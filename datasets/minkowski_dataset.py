from addict import Dict
import numpy as np
import MinkowskiEngine as ME
import torch
from torch.utils.data import Dataset
import os
import glob
import open3d as o3d
import h5py

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

