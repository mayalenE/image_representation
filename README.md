# ImageRepresentation: Code Skeleton

## addict.Dict
Class which implements a dictionary that provides attribute-style access.  
This class is used to implement configurations of all morphosearch classes, typically initialized in the
class `__init__` function with:

```
self.config = self.__class__.default_config()
self.config.update(config)
self.config.update(kwargs)
```



## image_representation.Representation
The main API methods that this class needs to implement are:
- **calc_embedding(x)**: given an input image x calculate the embedding
- **save(filepath)**: saves the model

## image_representation.TorchNNRepresentation
Base class of representations that are also torch neural modules. Inherits from image_representation.Representation and torch.nn.Module.  
- **config**: 
	- **config.network**:
		- *config.network.name*:
		- *config.network.parameters*:
		- *config.network.weights_init*:
		- *config.network.weights_init.name*:
		- *config.network.weights_init.parameters*:
	- **config.device**: 'cpu', 'cuda'
	- **config.loss**:
		- *config.loss.name*
		- *config.loss.parameters*
	- **config.optimizer**:
		- *config.optimizer.name*
		- *config.optimizer.parameters*
	- **config.logging**:
	- **config.checkpoint**:
		- *config.checkpoint.folder*:
- **network**: torch.nn.Module or Dict of torch.nn.Modules with several sub networks (encoder, decoder, etc)
- **loss_f**: torch.nn.functional or Dict of sub losses(discriminator, generator)
- **optimizer**: torch.optim or Dict of sub optimizers (discriminator, generator)
- **n_epochs**: number of training epochs
- **n_latents**: number of dimensions of the encoding 

Aditionnally to Representation's main API methods, the following main API methods must be implemented:
- **set_network(network_name, network_parameters)**:
- **init_network(weights_init_name, weights_init_parameters)**:
- **set_loss(loss_name, loss_parameters)**:
- **set_optimizer (optimizer_name, optimizer_parameters)**:
- **run_training (train_loader, n_epochs, valid_loader = None, training_logger=None)**:
- **train_epoch (train_loader, logger = None)**:
- **valid_epoch (valid_loader, logger = None)**:
- **save(filepath)**:
- **load(filepath, map_loaction='cpu')**:
- **calc_embedding (x)**: given an input image x calc the embedding z

