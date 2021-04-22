import numbers
import math
from image_representation.utils.torch_nn_module import Roll, SphericPad
from image_representation.utils.torch_functional import PI
import numpy as np
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

to_PIL = transforms.ToPILImage()
to_Tensor = transforms.ToTensor()

''' ---------------------------------------------
               PREPROCESS DATA HELPER
-------------------------------------------------'''

class TensorRandomFlip(object):
    def __init__(self, p=0.5, dim_flip=-1):
        self.p = p
        self.dim_flip = dim_flip

    def __call__(self, x):
        if torch.rand(()) < self.p:
            return x.flip(self.dim_flip)


class TensorRandomGaussianBlur(object):
    def __init__(self, p=0.5, kernel_radius=5, max_sigma=5, n_channels=1, spatial_dims=2):
        self.p = p
        self.kernel_size = 2 * kernel_radius + 1
        self.padding_size = int((self.kernel_size - 1) / 2)
        self.max_sigma = max_sigma
        self.n_channels = n_channels
        self.spatial_dims = spatial_dims

    def gaussian_kernel(self, kernel_size, sigma):
        """
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        mesh_grids = torch.stack([x_grid, y_grid], dim=-1)
        """

        # implementation of meshgrid in torch of shape (kernel_size, kernel_size, kernel_size if Z, 2)
        mesh_coords = [torch.arange(kernel_size)] * kernel_size
        mesh_grids = [None] * self.spatial_dims
        for dim in range(self.spatial_dims):
            view_size = [1, 1, 1]
            view_size[dim] = -1
            repeat_size = [kernel_size, kernel_size, kernel_size]
            repeat_size[dim] = 1
            mesh_grids[dim] = mesh_coords[dim].view(tuple(view_size)).repeat(repeat_size)
        mesh_grids = torch.stack(mesh_grids, dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * PI * variance)) * torch.exp(-torch.sum((mesh_grids - mean) ** 2., dim=-1) / (2 * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, *gaussian_kernel.size())
        gaussian_kernel = gaussian_kernel.repeat(self.n_channels, 1, *([1]*self.spatial_dims))
        return gaussian_kernel

    def __call__(self, x):
        if torch.rand(()) < self.p:
            sigma = int((torch.rand(()) * (1.0 - self.max_sigma) + self.max_sigma).round())
            x = x.view((1, ) + tuple(x.size()))
            x = F.pad(x, pad=self.padding_size, mode='reflect')
            kernel = self.gaussian_kernel(self.kernel_size, sigma)
            if self.spatial_dims == 2:
                x = F.conv2d(x, kernel, groups=self.n_channels).squeeze(0)
            elif self.spatial_dims == 3:
                x = F.conv3d(x, kernel, groups=self.n_channels).squeeze(0)
        return x


class TensorRandomSphericalRotation(object):
    def __init__(self, p=0.5, max_degrees=20, n_channels=1, img_size=(64, 64)):
        self.p = p
        self.spatial_dims = len(img_size)
        radius = max(img_size) / 2
        padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
        # max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
        self.spheric_pad = SphericPad(padding_size=padding_size)
        if n_channels == 1:
            fill = (0,)
        else:
            fill = 0
        if self.spatial_dims == 2:
            self.random_rotation = transforms.RandomRotation(max_degrees, resample=Image.BILINEAR, fill=fill)
            self.center_crop = transforms.CenterCrop(img_size)
        if self.spatial_dims == 3:
            if isinstance(max_degrees, numbers.Number):
                self.max_degress = (max_degrees, max_degrees, max_degrees)
            elif isinstance(max_degrees, tuple) or isinstance(max_degrees, list):
                assert len(max_degrees) == 3, "the number of rotation is 3, must provide tuple of length 3"
                self.max_degress = tuple(max_degrees)

    def __call__(self, x):
        if np.random.random() < self.p:
            x = x.view((1,) + tuple(x.size()))
            if self.spatial_dims == 2:
                x = self.spheric_pad(x).squeeze(0)
                img_PIL = to_PIL(x)
                img_PIL = self.random_rotation(img_PIL)
                img_PIL = self.center_crop(img_PIL)
                x = to_Tensor(img_PIL)
            elif self.spatial_dims == 3:
                x = self.spheric_pad(x).squeeze(0)
                theta_x = float(torch.empty(1).uniform_(-float(self.max_degrees[0]), float(self.max_degrees[0])).item()) * math.pi / 180.0
                theta_y = float(torch.empty(1).uniform_(-float(self.max_degrees[1]), float(self.max_degrees[1])).item()) * math.pi / 180.0
                theta_z = float(torch.empty(1).uniform_(-float(self.max_degrees[2]), float(self.max_degrees[2])).item()) * math.pi / 180.0
                R_x = torch.tensor([[1., 0., 0.],
                                    [0., math.cos(theta_x), -math.sin(theta_x)],
                                    [0., math.sin(theta_x), math.cos(theta_x)]])
                R_y = torch.tensor([[math.cos(theta_y), 0., math.sin(theta_y)],
                                    [0., 1.0, 0.0],
                                    [-math.sin(theta_y), 0.0, math.cos(theta_y)]])
                R_z = torch.tensor([[math.cos(theta_z), -math.sin(theta_z), 0.0],
                                    [math.sin(theta_z), math.cos(theta_z), 0.0],
                                    [0., 0., 1.]])
                R = R_z.matmul(R_y.matmul(R_x)).unsqueeze(0) # batch_size = 1
                grid = F.affine_grid(torch.cat([R, torch.zeros(3,1)], dim=-1), size=x.size())
                x = F.functional.grid_sample(x, grid)
        return x


class TensorRandomRoll(object):
    def __init__(self, p=(0.5, 0.5), max_delta=(0.5, 0.5), spatial_dims=2):
        self.spatial_dims = spatial_dims

        if isinstance(p, numbers.Number):
            self.p = tuple([p] * self.spatial_dims)
        else:
            self.p = p

        if isinstance(max_delta, numbers.Number):
            self.max_delta = tuple([max_delta] * self.spatial_dims)
        else:
            self.max_delta = max()

        assert len(self.p) == len(self.max_delta) == self.spatial_dims

        self.roll = [None] * self.spatial_dims
        for dim in range(self.spatial_dims):
            self.roll[dim] = Roll(shift=0, dim=dim)

    def __call__(self, x):

        for dim in range(self.spatial_dims):

            if np.random.random() < self.p[dim]:

                shift_dim = int(np.round(np.random.uniform(-self.max_delta[dim] * x.shape[1+dim], self.max_delta[dim] * x.shape[1+dim]))) #x: C*D*H*W
                self.roll[dim].shift = shift_dim
                x = self.roll[dim](x)

        return x


class TensorRandomResizedCrop(object):
    """
    Reimplementation of torchvision.transforms.RandomResizedCrop to deal with 2D or 3D tensors
    """

    def __init__(self, p, size, scale=(1., 1.), ratio_x=(1., 1.), ratio_y=(1., 1.), interpolation='bilinear'):
        self.p = p
        if (scale[0] > scale[1]) or (ratio_x[0] > ratio_x[1]) or (ratio_y[0] > ratio_y[1]):
            raise ValueError("Scale and ratio should be of kind (min, max)")
        self.scale = scale
        self.ratio_x = ratio_x
        self.ratio_y = ratio_y
        self.out_size = size
        self.spatial_dims = len(self.out_size)
        self.interpolation = interpolation

    def __call__(self, x):
        if np.random.random() < self.p:

            area = torch.prod(torch.tensor(self.out_size)).item()
            log_ratio_x = tuple(torch.log(torch.tensor(self.ratio_x)))
            if self.spatial_dims == 3:
                log_ratio_y = tuple(torch.log(torch.tensor(self.ratio_y)))

            patch_size = [None] * self.spatial_dims
            for _ in range(10):
                target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
                ratio_x = torch.exp(torch.empty(1).uniform_(log_ratio_x[0], log_ratio_x[1])).item()
                patch_xsize = int(round(math.pow(target_area * ratio_x, 1./self.spatial_dims)))
                if self.spatial_dims == 2:
                    ratio_y = 1. / ratio_x
                    patch_ysize = int(round(math.pow(target_area * ratio_y, 1./self.spatial_dims)))
                    if 0 < patch_xsize <= x.shape[-1] and 0 < patch_ysize <= x.shape[-2]:
                        patch_size = [patch_ysize, patch_xsize]
                elif self.spatial_dims == 3:
                    ratio_y = torch.exp(torch.empty(1).uniform_(log_ratio_y[0], log_ratio_y[1])).item()
                    patch_ysize = int(round(math.pow(target_area * ratio_y, 1./self.spatial_dims)))
                    ratio_z = 1. / (ratio_x * ratio_y)
                    patch_zsize = int(round(math.pow(target_area * ratio_z, 1./self.spatial_dims)))
                    if 0 < patch_xsize <= x.shape[-1] and 0 < patch_ysize <= x.shape[-2] and 0 < patch_zsize <= x.shape[-3]:
                        patch_size = [patch_zsize, patch_ysize, patch_xsize]

            if None in patch_size:
                for dim in range(len(patch_size)):
                    patch_size[dim] = x.shape[1+dim]

            return random_crop_preprocess(x, patch_size, out_size=self.out_size, interpolation=self.interpolation)

        else:
            return x

class TensorRandomCenterCrop(object):
    def __init__(self, p, size, scale=(1., 1.), ratio_x=(1., 1.), ratio_y=(1.,1.), interpolation='bilinear'):
        self.p = p
        if (scale[0] > scale[1]) or (ratio_x[0] > ratio_x[1]) or (ratio_y[0] > ratio_y[1]):
            raise ValueError("Scale and ratio should be of kind (min, max)")
        self.scale = scale
        self.ratio_x = ratio_x
        self.ratio_y = ratio_y
        self.out_size = size
        self.spatial_dims = len(self.out_size)
        self.interpolation = interpolation


    def __call__(self, x):
        if np.random.random() < self.p:

            area = torch.prod(torch.tensor(self.out_size)).item()
            log_ratio_x = tuple(torch.log(torch.tensor(self.ratio_x)))
            if self.spatial_dims == 3:
                log_ratio_y = tuple(torch.log(torch.tensor(self.ratio_y)))

            patch_size = [None] * self.spatial_dims
            for _ in range(10):
                target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
                ratio_x = torch.exp(torch.empty(1).uniform_(log_ratio_x[0], log_ratio_x[1])).item()
                patch_xsize = int(round(math.pow(target_area * ratio_x, 1./self.spatial_dims)))
                if self.spatial_dims == 2:
                    ratio_y = 1. / ratio_x
                    patch_ysize = int(round(math.pow(target_area * ratio_y, 1./self.spatial_dims)))
                    if 0 < patch_xsize <= x.shape[-1] and 0 < patch_ysize <= x.shape[-2]:
                        patch_size = [patch_ysize, patch_xsize]
                elif self.spatial_dims == 3:
                    ratio_y = torch.exp(torch.empty(1).uniform_(log_ratio_y[0], log_ratio_y[1])).item()
                    patch_ysize = int(round(math.pow(target_area * ratio_y, 1./self.spatial_dims)))
                    ratio_z = 1. / (ratio_x * ratio_y)
                    patch_zsize = int(round(math.pow(target_area * ratio_z, 1./self.spatial_dims)))
                    if 0 < patch_xsize <= x.shape[-1] and 0 < patch_ysize <= x.shape[-2] and 0 < patch_zsize <= x.shape[-3]:
                        patch_size = [patch_zsize, patch_ysize, patch_xsize]

            if None in patch_size:
                for dim in range(len(patch_size)):
                    patch_size[dim] = x.shape[1+dim]

            return centroid_crop_preprocess(x, patch_size, out_size=self.out_size, interpolation=self.interpolation)

        else:
            return x


def resized_crop(x, bbox, out_size, mode='bilinear'):
    """
    arg: x, tensor Cx(D)xHxW
    Reimplementation of torchvision.transforms.functional.resized_crop to deal with 2D or 3D images
    """
    x = crop(x, bbox)
    x = torch.nn.functional.interpolate(x.unsqueeze(0), out_size, mode=mode).squeeze(0)
    return x


def crop(x, bbox):
    """
    arg: x, tensor Cx(D)xHxW
    """
    spatial_dims = len(x.size()[1:])
    if spatial_dims == 2:
        return x[:, bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]
    elif spatial_dims == 3:
        return x[:, bbox[0]:bbox[0]+bbox[3], bbox[1]:bbox[1]+bbox[4], bbox[2]:bbox[2]+bbox[5]]


def centroid_crop_preprocess(x, patch_size, out_size=None, interpolation='bilinear'):
    """
    arg: x, tensor Cx(D)xHxW
    """

    img_size = tuple(x.size()[1:])
    spatial_dims = len(img_size)

    padding_size = round(max(*[patch_size[dim] / 2 for dim in range(spatial_dims)]))
    spheric_pad = SphericPad(padding_size=padding_size)
    x = spheric_pad(x.view((1, ) + tuple(x.size()))).squeeze(0)

    # crop around center of mass (mY and mX describe the position of the centroid of the image)
    image = x.numpy()
    meshgrids = np.meshgrid(*[range(img_size[dim]) for dim in range(spatial_dims)])

    m00 = np.sum(image)
    bbox = [None]*(2*spatial_dims) #y0,x0,h,w for 2D or z0,y0,x0,d,h,w for 3D
    for dim in range(spatial_dims):
        dim_power1_image = meshgrids[dim] * image
        if m00 == 0:
            m_dim = (img_size[dim] - 1) / 2.0
        else:
            m_dim = np.sum(dim_power1_image) / m00
        m_dim += padding_size
        bbox[dim] = (m_dim - patch_size[dim]) / 2
        bbox[spatial_dims+dim] = patch_size[dim]

    if out_size is not None:
        patch = resized_crop(x, tuple(bbox), out_size, mode=interpolation)
    else:
        patch = crop(x, tuple(bbox))

    return patch


def random_crop_preprocess(x, patch_size, out_size=None, interpolation='bilinear'):
    '''
    arg: x, tensor Cx(D)xHxW
    '''

    img_size = tuple(x.size()[1:])
    spatial_dims = len(img_size)

    # set the seed as mX*mY(*mZ) for reproducibility ((mZ,mY,mX) describe the position of the centroid of the image)
    local_seed = 1.0
    image = x.numpy()
    meshgrids = np.meshgrid(*[range(img_size[dim]) for dim in range(spatial_dims)])
    m00 = np.sum(image)
    for dim in range(spatial_dims):
        dim_power1_image = meshgrids[dim] * image
        if m00 == 0:
            m_dim = (img_size[dim] - 1) / 2.0
        else:
            m_dim = np.sum(dim_power1_image) / m00
        local_seed *= m_dim

    ## raw set seed
    global_rng_state = random.getstate()
    random.seed(local_seed)

    n_trials = 0
    best_patch_activation = 0
    selected_patch = False

    activation = m00 / torch.prod(torch.tensor(img_size)).item()
    while 1:
        bbox = [None] * spatial_dims + patch_size #y0,x0,h,w for 2D or z0,y0,x0,d,h,w for 3D
        # random sampling of origin crop
        for dim in range(spatial_dims):
            bbox[dim] = random.randint(0, img_size[dim] - patch_size[dim])
        if out_size is not None:
            patch = resized_crop(x, tuple(bbox), out_size, mode=interpolation)
        else:
            patch = crop(x, tuple(bbox))
        patch_activation = patch
        for dim in range(spatial_dims):
            patch_activation = patch_activation.sum(-1) / patch_size[-(dim+1)]

        if patch_activation > (activation * 0.5):
            selected_patch = patch
            break

        if patch_activation >= best_patch_activation:
            best_patch_activation = patch_activation
            selected_patch = patch

        n_trials += 1
        if n_trials == 20:
            break

    ## reput global random state
    random.setstate(global_rng_state)

    return selected_patch

