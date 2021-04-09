# Table of Contents

* [Install and Run](#imagerepresentation-install-and-run)
  
* [Code Skeleton](#imagerepresentation-code-skeleton)
  
* [RoadMap](#imagerepresentation-roadmap)

---

# ImageRepresentation: Install and Run

1. If you do not already have it, please install [Conda](https://www.anaconda.com/)
2. Create *morphosearch* conda environment: `conda create --name morphosearch python=3.6`
3. Activate *morphosearch* conda environment: `conda activate morphosearch`
4. If you do not already have it, please create a package folder that you will link to your conda env: `mkdir <path_to_packages_folder>`
5. Into your package folder, clone the following packages:  
    b. `git clone git@github.com:mayalenE/imagerepresentation.git`
5. Include thos packages in the conda environment:  
   `echo <path_to_packages_folder> "$HOME/miniconda3/envs/morphosearch/lib/python3.6/site-packages/my_packages.pth"`
6. Install the required conda packages in the environment (*requirements.txt* file can be found in imagerepresentation directory):  
   `while read requirement; do conda install --yes $requirement --channel default --channel anaconda --channel conda-forge --channel pytorch; done < requirements.txt`

# ImageRepresentation: Code Skeleton
The main classes (***System***, ***OutputRepresentation***, ***OutputFitness*** and ***Explorer***) are implemented in `core.py`.


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
- **load(checkpoint_filepath, map_loaction='cpu')**:
- **save(checkpoint_filepath, map_loaction='cpu')**:
- **calc_embedding (x)**: given an input image x calc the embedding z

# ImageRepresentation: RoadMap