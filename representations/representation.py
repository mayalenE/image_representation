from abc import ABCMeta, abstractmethod
from addict import Dict
import torch

class Representation(metaclass=ABCMeta):
    """
    Base Representation Class
    """

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

    def save(self, filepath='representation.pickle'):
        torch.save(self, filepath)
        return

    @staticmethod
    def load(filepath='representation.pickle', map_location='cpu'):
        representation = torch.load(filepath, map_location=map_location)
        return representation

    @abstractmethod
    def calc_embedding(self, x):
        """
        x:  array-like (n_samples, n_features)
        Return an array with the representations (n_samples, n_latents)
    	"""
        pass


    def calc_distance(self, z1, z2, low=None, high=None):
        """
        Standard Euclidean distance between representation vectors.
        """
        # normalize representations
        if low is not None and high is not None:
            z1 = z1 - low
            z1 = z1 / (high - low)
            z2 = z2 - low
            z2 = z2 / (high - low)

        if len(z1) == 0 or len(z2) == 0:
            return torch.tensor([])

        dist = (z1 - z2).pow(2).sum(-1).sqrt()

        return dist
