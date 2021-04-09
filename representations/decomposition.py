from image_representation import Representation
from addict import Dict
import numpy as np
from sklearn.decomposition import PCA


class PCARepresentation(Representation):

    @staticmethod
    def default_config():
        default_config = Representation.default_config()

        # parameters
        default_config.parameters = Dict()
        default_config.parameters.random_state = None

        return default_config

    def __init__(self, n_features=28 * 28, n_latents=10, config=None, **kwargs):
        Representation.__init__(self, config=config, **kwargs)

        # input size (flatten)
        self.n_features = n_features
        # latent size
        self.n_latents = n_latents
        # feature range
        self.feature_range = (0.0, 1.0)

        self.algorithm = PCA()
        self.update_algorithm_parameters()

    def fit(self, X_train, update_range=True):
        ''' 
        X_train: array-like (n_samples, n_features)
        '''
        X_train = np.nan_to_num(X_train)
        if update_range:
            self.feature_range = (X_train.min(axis=0), X_train.max(axis=0))  # save (min, max) for normalization
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(
            scale == 0)] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
        X_train = (X_train - self.feature_range[0]) / scale
        X_transformed = self.algorithm.fit_transform(X_train)
        return X_transformed

    def calc_embedding(self, x):
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(
            scale == 0)] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
        x = (x - self.feature_range[0]) / scale
        x = self.algorithm.transform(x)
        return x

    def update_algorithm_parameters(self):
        self.algorithm.set_params(n_components=self.n_latents, **self.config.parameters)
