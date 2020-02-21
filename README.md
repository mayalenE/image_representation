# CORE
/!\ skeletton to follow, all the classes inheriting from a BaseXXX class should have this
## BaseRepresentation
#### config
-  *config.seed*: seed for the experiment (for reproducibility and statistical viability)
#### property attributes
-  *model*: attribute or nn.Module() that has several sub models
#### methods
##### static methods
- *default_config()*: returns the default configuration
##### instance methods
- *preprocess (observations)*: given raw observations, convert them to the correct input format x for the representation
- *calc (x)*: given correct input x, calc the embedding z
- *calc_distance (z1, z2,goal_space_extent)*: calc distance between two embeddings z1,z2 given the goal space boundaries (to normalize)




## BaseModel
#### config
#### property attributes
#### methods
##### static methods
- *default_config()*: returns the default configuration
- *get_model(model_name)*: calls the model class defined with model_name
##### instance methods
- *calc_embedding (x)*: given an input x calc the embedding z
- *get_encoder()*: returns the encoder of the model (mandatory)
- *get_decoder()*: returns the decoder of the model (if no decoder return None)
- *save_checkpoint (checkpoint_filepath)*: saves the model

## BaseEncoder
#### config
#### property attributes
- *n_latents*: number of dimensions of the encoding 
#### methods
##### static methods
- *default_config()*: returns the default configuration
##### instance methods
- *calc_embedding (x)*: given an input x calc the embedding z

## BaseEvaluationModel
#### config
-  *config.output_name*:
#### property attributes
- *representation_encoder*: the evalmodel are all build up on top of the representation's encoder (this variable is untouched even if retraining on top)
#### methods
##### static methods
- *default_config()*: returns the default configuration
- *get_evaluationmodel(model_name)*: calls the evaluationmodel class defined with model_name
##### instance methods
- *run_training (train_loader, valid_loader=None, logger=None)*: trains the evaluation model given a training/validation dataset (optional)
- *run_representation_testing (test_loader, testing_config = None)*: tests the representation plugged with the evaluation model given a test dataset. For each evaluation model specifies how and what to output, and the testing_config specifies if and where to save the results. Always returns the loss.
- *do_evaluation_pass(dataloader, logger=None)*: given a dataset (train/valid/test), loop over it and returns the loss per data-element, optional logging
- *visualize_results( visualization_config = None)*:
- *save_checkpoint (checkpoint_filepath)*: saves the model


## dnn.BaseDNN
#### config
- *config.network*:
	- *config.network.name*:
	- *config.network.parameters*:
	- *config.network.initialization*:
	- *config.network.initialization.name*:
	- *config.network.initialization.parameters*:
- *config.device*:
	- *config.device.use_gpu*
- *config.loss*:
	- *config.loss.name*
	- *config.loss.parameters*
- *config.optimizer*:
	- *config.optimizer.name*
	- *config.optimizer.parameters*
- *config.logging*:
- *config.checkpoint*:
	- *config.checkpoint.folder*:
#### property attributes
- *network*: attribute or AttrDict() that has several sub networks (encoder, decoder, etc)
- *loss_f*: attribute or AttrDict() that has several sub losses(discriminator, generator)
- *optimizer*: attribute or AttrDict() that has several sub optimizers (discriminator, generator)
- *n_epochs*:
#### methods
##### static methods
- *default_config()*: returns the default configuration
- *load_checkpoint(checkpoint_filepath, use_gpu = False)*:
##### instance methods
- *__init\__(config=None, **kwargs)*:
- *set_network(network_name, network_parameters)*:
- *init_network(initialization_name, initialization_parameters)*:
- *set_device(use_gpu)*:
- *push_variable_to_device(x)*:
- *set_loss(loss_name, loss_parameters)*:
- *set_optimizer (optimizer_name, optimizer_parameters)*:
- *run_training (train_loader, n_epochs, valid_loader = None, training_logger=None)*:
- *train_epoch (train_loader, logger = None)*:
- *valid_epoch (valid_loader, logger = None)*:
- *save_checkpoint (checkpoint_filepath)*:

## evaluationmodels.classification.BaseClassifier
#### config
- *config.loss*:
	- *config.loss.name*
	- *config.loss.parameters*
- *config.optimizer*:
	- *config.optimizer.name*
	- *config.optimizer.parameters*
- *config.logging*:
- *config.checkpoint*:
	- *config.checkpoint.folder*:
#### property attributes
- *encoder*:
- *n_epochs*:
#### methods
##### static methods
##### instance methods


# dnn
# models
# evaluationmodels
# representations
# datasets
# helper
- *save (object, filepath)*: save the object as it is
- *load (filepath, map_location, config)*: loads a saved object on the desired device, with possibility to update the config (warning though as the config might not correspond anymore to saved params)