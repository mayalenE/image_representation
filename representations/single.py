import goalrepresent as gr
from goalrepresent import models
import os
from tensorboardX import SummaryWriter

class SingleModelRepresentation(gr.BaseRepresentation):
    """
    Representation with single model Class
    """ 
    @staticmethod
    def default_config():
        default_config = gr.BaseRepresentation.default_config()
        
        # model parameters
        default_config.model = gr.Config()
        default_config.model.name = "VAE"
        default_config.model.config = models.VAEModel.default_config()
        
        # training parameters
        default_config.training = gr.Config()
        default_config.training.n_epochs = 0
        default_config.training.output_folder = None
        
        default_config.testing = gr.Config()
        default_config.testing.output_folder = None
        default_config.testing.evaluationmodels = []
        return default_config
    
    
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        
         # model
        self.set_model(self.config.model.name, self.config.model.config)


    def set_model(self, model_name, model_config):
        model_class = gr.BaseModel.get_model(model_name)
        self.model = model_class(config=model_config)
        
        # update config
        self.config.model.name = model_name
        self.config.model.config = gr.config.update_config(model_config, self.config.model.config)
        
    def run_training(self, train_loader, valid_loader, training_config, keep_best_model=True, logging=True):
        # prepare output folders
        output_folder = training_config.output_folder
        if (output_folder is not None) and (not os.path.exists (output_folder)):
            os.makedirs(output_folder)
            
        checkpoint_folder = os.path.join(output_folder, "checkpoints")
        if (checkpoint_folder is not None) and (not os.path.exists (checkpoint_folder)):
            os.makedirs(checkpoint_folder)
        self.model.config.checkpoint.folder = checkpoint_folder
        
        # prepare logger
        if logging: 
            logging_folder = os.path.join(output_folder, "logging")
            if (logging_folder is not None) and (not os.path.exists (logging_folder)):
                os.makedirs(logging_folder)
    
            logger = SummaryWriter(logging_folder, 'w')
        else:
            logger = None
        
        # run_training
        self.model.run_training(train_loader, training_config.n_epochs, valid_loader, logger=logger)
        
        # export scalar data to JSON for external processing
        if logger is not None:
            logger.export_scalars_to_json(os.path.join(output_folder, "output_scalars.json"))
            logger.close()
            
        # if we want the representation to keep the model that performed best on the test dataset
        if keep_best_model:
            best_model_path = os.path.join(checkpoint_folder, "best_weight_model.pth")
            if os.path.exists(best_model_path):
                best_model = gr.dnn.BaseDNN.load_checkpoint(best_model_path, use_gpu = self.model.config.device.use_gpu)
                self.model = best_model
        
        # update config
        self.config.training = gr.config.update_config(training_config, self.config.training)        
        
    def run_testing(self, test_loader, testing_config, train_loader=None, valid_loader=None, logging=True):
        # prepare output folders
        output_folder = testing_config.output_folder
        if (output_folder is not None) and (not os.path.exists (output_folder)):
            os.makedirs(output_folder)
            
        # loop over different tests
        #TODO: if test already done pass
        test_statistics = {}
        for evalmodel_config in testing_config.evaluationmodels:
            evalmodel_name = evalmodel_config.name
            evalmodel_class = gr.BaseEvaluationModel.get_evaluationmodel(evalmodel_name)
            evalmodel = evalmodel_class(self.model, config=evalmodel_config.config)
            
            # prepare output folders
            curr_output_folder = evalmodel_config.output_folder
            if (curr_output_folder is not None) and (not os.path.exists (curr_output_folder)):
                os.makedirs(curr_output_folder)
            
            checkpoint_folder = os.path.join(curr_output_folder, "checkpoints")
            if (checkpoint_folder is not None) and (not os.path.exists (checkpoint_folder)):
                os.makedirs(checkpoint_folder)
            evalmodel.config.checkpoint.folder = checkpoint_folder
            
            # prepare logger
            if logging: 
                logging_folder = os.path.join(curr_output_folder, "logging")
                if (logging_folder is not None) and (not os.path.exists (logging_folder)):
                    os.makedirs(logging_folder)
        
                logger = SummaryWriter(logging_folder, 'w')
            else:
                logger = None
                
            # train the evaluationmodel if needed
            evalmodel.run_training(train_loader=train_loader, valid_loader=valid_loader, logger=logger)
            # test the representation by this evaluationmodel
            evalmodel_test_statistics = evalmodel.run_representation_testing(test_loader, testing_config=evalmodel_config)
            output_name = curr_output_folder.split("/")[-1]
            test_statistics[output_name] = evalmodel_test_statistics
            
            # export scalar data to JSON for external processing
            if logger is not None:
                logger.export_scalars_to_json(os.path.join(curr_output_folder, "output_scalars.json"))
                logger.close()
            
        # update config
        self.config.testing = gr.config.update_config(testing_config, self.config.testing)
        
        return test_statistics
        
    def preprocess(self, observations):
        x = observations #N*C*H*W 
        return x
        
    def calc(self, x):
        z = self.model.calc_embedding(x)
        return z
        


    
