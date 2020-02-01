import goalrepresent as gr
from goalrepresent import evaluationmodels
import numpy as np
import os
from sklearn import svm
import torch

def get_algorithm(algorithm_name):
    """
    algorithm_name: string such that the algorithm called is sklearn.svm.<algorithm_name>
    """
    return eval("svm.{}".format(algorithm_name))


class SVMClassifierModel(evaluationmodels.BaseClassifierModel):
    """
    SVM Classifier Evaluation Model Class
    """
    @staticmethod
    def default_config():
        default_config = evaluationmodels.BaseClassifierModel.default_config()
        
        # hyperparameters
        default_config.algorithm = gr.Config()
        default_config.algorithm.name = "SVC"
        default_config.algorithm.hyperparameters = gr.Config()
        default_config.algorithm.hyperparameters.kernel = "linear"
        default_config.algorithm.hyperparameters.probability = True
        default_config.algorithm.hyperparameters.max_iter = -1

        return default_config
    

    def __init__(self, representation_model, config=None, **kwargs):
        super().__init__(representation_model, config=config, **kwargs)
        
        self.set_algorithm(self.config.algorithm.name, self.config.algorithm.hyperparameters)
        
        
    def set_algorithm(self, algorithm_name, algorithm_hyperparameters):
        algorithm_class = get_algorithm(algorithm_name)
        self.algorithm = algorithm_class(**algorithm_hyperparameters)
        
        # update config
        self.config.algorithm.name = algorithm_name
        self.config.algorithm.parameters = gr.config.update_config(algorithm_hyperparameters, self.config.algorithm.hyperparameters)
        
        
    def forward(self, x):
        encoder_outputs = self.representation_encoder(x)
        z_normalized = (encoder_outputs["z"] - self.feature_range[0]) 
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(scale==0)] = 1.0 # trick when some some latents are the same for every point (no scale and divide by 1)
        z_normalized = z_normalized / scale
        predicted_y = torch.from_numpy(self.algorithm.predict_log_proba(z_normalized.detach().numpy()))
        return predicted_y
    
    
    def fit(self, z_train, y_train, update_range=True):
        ''' 
        z_train: array-like (n_samples, n_features)
        y_train: array-like target values (n_samples, )
        '''
        z_train = np.nan_to_num(z_train)
        if update_range: 
            self.feature_range = (z_train.min(axis=0), z_train.max(axis=0)) # save (min, max) for normalization
        z_train = (z_train - self.feature_range[0]) 
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(scale==0)] = 1.0 # trick when some some latents are the same for every point (no scale and divide by 1)
        z_train = z_train / scale
        self.algorithm.fit(z_train, y_train)
        
        self.n_epochs += 1
        
        
    def run_training(self, train_loader=None, valid_loader=None,  keep_best_model=True, logger=None):
        """
        logger: tensorboard X summary writer
        """        
        self.n_classes = train_loader.dataset.dataset.n_classes
        
        if valid_loader is not None:
            do_validation = True
            
        # construction of X_train (n_samples, n_features)
        n_train_samples = len(train_loader.dataset.indices)
        z_train = np.empty((n_train_samples, self.representation_encoder.n_latents))
        y_train = np.empty(n_train_samples)
        for batch_idx, batch_data in enumerate(train_loader):
            x =  batch_data['obs']
            y = batch_data['label'].squeeze()
            # forward
            encoder_outputs = self.representation_encoder(x)
            start = batch_idx*train_loader.batch_size
            end = min((batch_idx+1)*train_loader.batch_size, n_train_samples)
            z_train[start:end, :] = encoder_outputs["z"].detach().numpy()
            y_train[start:end] = y
        
        # fit the svm algorithm
        self.fit(z_train, y_train)
        
        #  log train and valid error
        ## train
        _, train_losses = self.do_evaluation_pass(train_loader)
        if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
            for k, v in train_losses.items():
                logger.add_scalars("loss/{}".format(k), {"train": np.mean(v)}, self.n_epochs)
        ## valid
        if do_validation:
           _,_ = self.do_evaluation_pass(valid_loader, logger=logger)
        
        # save the trained SVM
        self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))
        
    
    def run_representation_testing(self, dataloader, testing_config = None, train_loader=None, valid_loader=None, logger=None):
        test_data = dict()
        
        # run testing on the test data
        test_predictions, test_losses = self.do_evaluation_pass(dataloader)
        
        #  transform log probabilities -> probabilities predictions
        test_predictions["predicted_y"] =  np.exp(test_predictions["predicted_y"])
        
        test_data["predictions"] = test_predictions
        test_data["error"] = test_losses
        
        
        return test_data
    

    def save_checkpoint(self, checkpoint_filepath):
        # save classifier if we want to relaunch testing from that point
        checkpoint = {
        "type": self.__class__.__name__,
        "config": self.config,
        "algorithm": self.algorithm,
        }
        
        torch.save(checkpoint, checkpoint_filepath)
    
    
