import goalrepresent as gr
import numpy as np
from sklearn.decomposition import PCA

class PCAModel(gr.BaseModel):
    '''
    PCA Model Class
    '''
    @staticmethod
    def default_config():
        default_config = super().default_config()
        
        # hyperparameters
        default_config.hyperparameters = gr.Config()
        default_config.hyperparameters.random_state=None
        
        return default_config
    
    def __init__(self, n_features=28*28, n_latents = 10, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        
        # store the initial parameters used to create the model
        self.init_params = locals()
        del self.init_params['self']
        
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
            self.feature_range = (X_train.min(axis=0), X_train.max(axis=0)) # save (min, max) for normalization
        X_train = (X_train - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        self.algorithm.fit(X_train)

        
    def calc_embedding(self, x):
        x = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        x = self.algorithm.transform(x)
        
        return x
    
    def update_hyperparameters(self, hyperparameters):
        super().update_hyperparameters(hyperparameters)
        self.update_algorithm_parameters()

    def update_algorithm_parameters(self):
        self.algorithm.set_params(n_components=self.n_latents, **self.config.hyperparameters, verbose=False)
