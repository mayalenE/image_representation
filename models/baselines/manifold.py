import goalrepresent as gr
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP


class TSNEModel(gr.BaseModel):
    '''
    TSNE Model Class
    '''

    @staticmethod
    def default_config():
        default_config = gr.BaseModel.default_config()

        # hyperparameters
        default_config.hyperparameters = gr.Config()
        default_config.hyperparameters.perplexity = 30.0
        default_config.hyperparameters.init = "random"
        default_config.hyperparameters.random_state = None

        return default_config

    def __init__(self, n_features=28 * 28, n_latents=10, config=None, **kwargs):
        gr.BaseModel.__init__(self, config=config, **kwargs)

        # store the initial parameters used to create the model
        self.init_params = locals()
        del self.init_params['self']

        # input size (flatten)
        self.n_features = n_features
        # latent size
        self.n_latents = n_latents
        # feature range
        self.feature_range = (0.0, 1.0)

        self.algorithm = TSNE(n_components=self.n_latents)
        self.update_hyperparameters(self.config.hyperparameters)

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

    def update_hyperparameters(self, hyperparameters):
        self.algorithm.set_params(**self.config.hyperparameters, verbose=False)
        # update config
        self.config.hyperparameters = gr.config.update_config(hyperparameters, self.config.hyperparameters)


class UMAPModel(gr.BaseModel):
    '''
    UMAP Model Class
    '''

    @staticmethod
    def default_config():
        default_config = gr.BaseModel.default_config()

        # hyperparameters
        default_config.hyperparameters = gr.Config()
        default_config.hyperparameters.n_neighbors = 15
        default_config.hyperparameters.metric = 'euclidean'
        default_config.hyperparameters.init = 'spectral'
        default_config.hyperparameters.random_state = None
        default_config.hyperparameters.min_dist = 0.1

        return default_config

    def __init__(self, n_features=28 * 28, n_latents=10, config=None, **kwargs):
        gr.BaseModel.__init__(self, config=config, **kwargs)

        # store the initial parameters used to create the model
        self.init_params = locals()
        del self.init_params['self']

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

    def update_hyperparameters(self, hyperparameters):
        gr.BaseModel.update_hyperparameters(self, hyperparameters)
        self.set_algorithm_parameters()

    def update_algorithm_parameters(self):
        self.algorithm.set_params(n_components=self.n_latents, **self.config.hyperparameters, verbose=False)
