"""
script containing all the core Abstract Class with the skeletton to follow
"""
from abc import ABCMeta, abstractmethod
import goalrepresent as gr
from goalrepresent.helper import datahelper, randomhelper

class BaseEncoder (metaclass=ABCMeta):
    """
    Base Encoder class
    Each Model has an encoder mapping input x to goal latent representation z
    """
    @staticmethod
    def default_config():
        default_config = gr.Config()
        return default_config
    

    def __init__(self, config=None, **kwargs):
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())
    
    @abstractmethod  
    def calc_embedding(self, x):
        pass


class BaseModel(metaclass=ABCMeta):
    """
    Base Model Class
    """
    @staticmethod
    def default_config():
        default_config = gr.Config()
        return default_config
    

    def __init__(self, config=None, **kwargs):
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())
        
        
    @abstractmethod
    def calc_embedding(self, x):
        """
        x:  array-like (n_samples, n_features)
        Return an array with the representations (n_samples, n_latents)
    	"""
        pass
    
    @abstractmethod
    def get_encoder(self):
        """
        Returns a  BaseEncoder with a calc_embedding() function
        """
        pass
     
    @abstractmethod
    def get_decoder(self):
        """
        Returns an nn.Module decoder with a forward() function, or None
        """
        pass                    
    
    @abstractmethod
    def save_checkpoint(self, checkpoint_filepath):
        pass

    @staticmethod
    def get_model(model_name):
        """
        model_name: string such that the model called is <model_name>Model
        """
        return eval("gr.models.{}Model".format(model_name))


class BaseEvaluationModel(metaclass=ABCMeta):
    """
    Base Evaluation Model Class
    """
    @staticmethod
    def default_config():
        default_config = gr.Config()
        
        default_config.output_name = None
        
        return default_config
    

    def __init__(self, representation_model, config=None, **kwargs):
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())
        
        ## pluggin the representation pretrained encoder 
        self.representation_encoder = representation_model.get_encoder()
        
    @abstractmethod
    def run_training (self, train_loader=None, valid_loader=None, keep_best_model=True, logger=None):
        pass
    
    @abstractmethod
    def run_representation_testing(self, dataloader, testing_config = None):
        pass
    
    @abstractmethod
    def do_evaluation_pass(self, dataloader, logger=None):
        pass
    
    @abstractmethod
    def visualize_results(self, visualization_config = None):
        pass
    
    @abstractmethod
    def save_checkpoint(self, checkpoint_filepath):
        pass
    
    
    @staticmethod
    def get_evaluationmodel(model_name):
        """
        model_name: string such that the model called is <model_name>Model
        """
        return eval("gr.evaluationmodels.{}Model".format(model_name))
    
    
class BaseRepresentation(metaclass=ABCMeta):
    """
    Base Representation Class
    """
    @staticmethod
    def default_config():
        default_config = gr.Config()
        # seed 
        default_config.seed = 0
        
        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())
        
        # set seed
        randomhelper.set_seed(self.config.seed)
        
    def save(self, filepath = 'representation.pickle'):
        datahelper.save(self, filepath)
        return
    
    @staticmethod
    def load(filepath = 'representation.pickle', map_location='cpu', config=None):
        representation = datahelper.load(filepath, map_location = map_location, config = config)
        randomhelper.set_seed(representation.config.seed)
        if hasattr(representation, 'model'):
            if hasattr(representation.model.config, 'device'):
                if map_location == 'cpu':
                    representation.model.set_device(use_gpu=False)
                elif map_location == "cuda:0":
                    representation.model.set_device(use_gpu=True)
        
        return representation

    @abstractmethod
    def preprocess(self, observations):
        pass
    
    @abstractmethod    
    def calc(self, x):
        pass
