from image_representation import Representation
from addict import Dict
from sklearn.manifold import TSNE
from umap import UMAP
import numpy as np

class TSNERepresentation(Representation):
    @staticmethod
    def default_config():
        default_config = Representation.default_config()

        # parameters
        default_config.parameters = Dict()
        default_config.parameters.perplexity = 30.0
        default_config.parameters.init = "random"
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

        self.algorithm = TSNE(n_components=self.n_latents)
        self.update_algorithm_parameters()

    def fit(self, X_train, update_range=True):
        ''' 
        X_train: array-like (n_samples, n_features)
        '''
        X_train = np.nan_to_num(X_train)
        if update_range:
            self.feature_range = (X_train.min(axis=0), X_train.max(axis=0))  # save (min, max) for normalization
        X_train = (X_train - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        self.algorithm.fit(X_train)

    def calc_embedding(self, x):
        x = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        x = self.algorithm.transform(x)
        return x

    def update_algorithm_parameters(self):
        self.algorithm.set_params(**self.config.parameters, verbose=False)


class UMAPRepresentation(Representation):

    @staticmethod
    def default_config():
        default_config = Dict()

        # parameters
        default_config.parameters = Dict()
        default_config.parameters.n_neighbors = 15
        default_config.parameters.metric = 'euclidean'
        default_config.parameters.init = 'spectral'
        default_config.parameters.random_state = None
        default_config.parameters.min_dist = 0.1

        return default_config

    def __init__(self, n_features=28 * 28, n_latents=10, config=None, **kwargs):
        Representation.__init__(self, config=config, **kwargs)

        # input size (flatten)
        self.n_features = n_features
        # latent size
        self.n_latents = n_latents
        # feature range
        self.feature_range = (0.0, 1.0)

        self.algorithm = UMAP()
        self.update_algorithm_parameters()

    def fit(self, X_train, update_range=True):
        ''' 
        X_train: array-like (n_samples, n_features)
        '''
        X_train = np.nan_to_num(X_train)
        if update_range:
            self.feature_range = (X_train.min(axis=0), X_train.max(axis=0))  # save (min, max) for normalization
        X_train = (X_train - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        self.algorithm.fit(X_train)

    def calc_embedding(self, x):
        x = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        x = self.algorithm.transform(x)
        return x

    def update_algorithm_parameters(self):
        self.algorithm.set_params(n_components=self.n_latents, **self.config.parameters, verbose=False)
