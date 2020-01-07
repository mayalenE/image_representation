import bisect
import goalrepresent as gr
from goalrepresent import dnn
from goalrepresent.dnn.solvers import initialization 
from goalrepresent.dnn.networks import decoders
import numpy as np
import os
import sys
import time
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
import warnings

transform_to_PIL_image = ToPILImage()
transform_to_tensor = ToTensor()

class ReconstructorModel(dnn.BaseDNN, gr.BaseEvaluationModel):
    """
    Reconstructor Evaluation Model Class
    """
    @staticmethod
    def default_config():
        default_config = gr.BaseEvaluationModel.default_config()
        
        # freeze or not the decoder
        default_config.freeze_encoder = True
        default_config.freeze_decoder = True
        
        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Burgess"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64,64)
        default_config.network.parameters.n_conv_layers = 4
        
        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()
        
        # device parameters
        default_config.device = gr.Config()
        default_config.device.use_gpu = True
        
        # loss parameters
        default_config.loss = gr.Config()
        default_config.loss.name = "VAE"
        default_config.loss.parameters = gr.Config()
        
        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        
        # logging config (will save every X epochs)
        default_config.logging = gr.Config()
        default_config.logging.record_loss_every = 1
        default_config.logging.record_top10_images_every = 1

        # checkpoint config
        default_config.checkpoint = gr.Config()
        default_config.checkpoint.folder = None
        default_config.checkpoint.save_model_every = 10
        return default_config
    

    def __init__(self, representation_model, config=None, **kwargs):
        super(dnn.BaseDNN, self).__init__() # calls nn.Module constructor
        super(nn.Module, self).__init__(representation_model, config=config, **kwargs) # calls gr.BaseEvaluationModel constructor
        
        # store the initial parameters used to create the model
        self.init_params = locals()
        del self.init_params['self']
        
        # define the device to use (gpu or cpu)
        if self.config.device.use_gpu and not torch.cuda.is_available():
            self.config.device.use_gpu = False
            warnings.warn("Cannot set model device as GPU because not available, setting it to CPU")
        
        # network
        self.set_network(representation_model, self.config.network.name, self.config.network.parameters)
        if self.config.freeze_encoder:
            for param in self.network.encoder.parameters():
                param.requires_grad = False
        if self.config.freeze_decoder:
            for param in self.network.decoder.parameters():
                param.requires_grad = False
        else:
            self.set_optimizer(self.config.optimizer.name, self.config.optimizer.parameters)
        self.set_device(self.config.device.use_gpu)
        
        # loss function
        self.set_loss(self.config.loss.name, self.config.loss.parameters)
        
        self.n_epochs = 0
                
    
    def set_network(self, representation_model, network_name, network_parameters):
        self.network = nn.Module()
        self.network.encoder = self.representation_encoder
        self.network.decoder = representation_model.get_decoder()
        if self.network.decoder is None:
            decoder_class = decoders.get_decoder(network_name)
            network_parameters.n_latents = self.network.encoder.n_latents
            self.network.decoder = decoder_class(**network_parameters)
            self.init_network(self.config.network.initialization.name, self.config.network.initialization.parameters)
        # update config
        self.config.network.name = network_name
        self.config.network.parameters = gr.config.update_config(network_parameters, self.config.network.parameters)
        
    def init_network(self, initialization_name, initialization_parameters):
        initialization_class = initialization.get_initialization(initialization_name)
        # only initialize the decoder (keep the pretrained encoder)
        self.network.decoder.apply(initialization_class)
        
        # update config
        self.config.network.initialization.name = initialization_name
        self.config.network.initialization.parameters = gr.config.update_config(initialization_parameters, self.config.network.initialization.parameters)
        
    def set_optimizer(self, optimizer_name, optimizer_parameters):
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        # if encoder frozen only optimize the decoder, else optimize all together
        if self.config.freeze_encoder:
            self.optimizer = optimizer_class(self.network.decoder.parameters(), **optimizer_parameters)
        else:
            self.optimizer = optimizer_class(self.network.parameters(), **optimizer_parameters)
        
        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters, self.config.optimizer.parameters)
    
    def reparameterize(self, mu, logvar):
        mu = self.push_variable_to_device(mu)
        logvar = self.push_variable_to_device(logvar)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x):
        x = self.push_variable_to_device(x)
        encoder_outputs = self.network.encoder(x)
        mu = encoder_outputs[0]
        logvar = encoder_outputs[1]
        z = self.reparameterize(mu, logvar)
        if self.network.decoder.__class__ is gr.models.hierarchicaltree.HierarchicalTreeDecoder:
            path_taken = encoder_outputs[-2]
            recon_x = self.network.decoder(z, path_taken)
        else:
            recon_x = self.network.decoder(z)
        return recon_x, mu, logvar, z
    
    
    def train_epoch (self, train_loader, logger=None):
        self.train()
        losses = {}
        for data in train_loader:
            x =  Variable(data['obs'])
            x = self.push_variable_to_device(x)
            # forward
            outputs = self.forward(x)
            recon_x, mu, logvar, z = outputs
            batch_losses = self.loss_f(x, recon_x, mu, logvar, logger=logger)
            # backward
            loss = batch_losses['total']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())
          
        for k, v in losses.items():
            losses [k] = np.mean (v)
        
        self.n_epochs += 1
        
        return losses
    
    def do_evaluation_pass(self, dataloader, logger = None):
        self.eval()
        predictions = {}
        losses = {}
        log_top10_images = (logger is not None) and (self.n_epochs % self.config.logging.record_top10_images_every == 0)
        log_loss = (logger is not None) and (self.n_epochs % self.config.logging.record_loss_every == 0)
        
        if log_top10_images:
                top10_scores = [sys.float_info.max]*10
                top10_images = [None]*10
                top10_reconstructions = [None]*10
                bottom10_scores = [0]*10
                bottom10_images = [None]*10
                bottom10_reconstructions = [None]*10
        
        # loop over the different batches
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                x =  Variable(data['obs'])
                # forward
                recon_x, mu, logvar, z = self.forward(x)
                loss_inputs = {'recon_x': recon_x, 'mu': mu, 'logvar': logvar}
                loss_targets = {'x': x}
                batch_losses = self.loss_f(loss_inputs, loss_targets, reduction=False, logger=logger)
                
                # save results
                if self.config.loss.parameters.reconstruction_dist == "bernouilli":
                    recon_x = torch.sigmoid(recon_x)
                if 'recon_x' in predictions:
                    predictions['recon_x'] = np.vstack([predictions['recon_x'], recon_x.detach().cpu().numpy()])
                else:
                    predictions['recon_x'] = recon_x.detach().cpu().numpy()
                    
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis = -1)
                    else:
                        losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis = -1)])
                        
                # update top-10 and bottom-10 classified images
                if log_top10_images:
                    for img_idx_wrt_batch, img in enumerate(x.cpu().data):
                        recon_loss = batch_losses['recon'][img_idx_wrt_batch].data.item()
                        if recon_loss < top10_scores[-1]:
                            i = bisect.bisect(top10_scores, recon_loss)
                            top10_scores.insert(i, recon_loss)
                            top10_scores = top10_scores[:-1]
                            top10_images.insert(i, img.cpu().detach())
                            top10_images = top10_images[:-1]
                            top10_reconstructions.insert(i, recon_x[img_idx_wrt_batch].cpu().detach())
                            top10_reconstructions = top10_reconstructions[:-1]
                        elif recon_loss > bottom10_scores[0]:
                            i = bisect.bisect(bottom10_scores, recon_loss)
                            bottom10_scores.insert(i, recon_loss)
                            bottom10_scores = bottom10_scores[1:]
                            bottom10_images.insert(i, img.cpu().detach())
                            bottom10_images = bottom10_images[1:]
                            bottom10_reconstructions.insert(i, recon_x[img_idx_wrt_batch].cpu().detach())
                            bottom10_reconstructions = bottom10_reconstructions[1:]                 

        # log results   
        if log_top10_images:
            vizu_tensor_list = [None] * (2*10)
            vizu_tensor_list[0::2] = [top10_images[n] for n in range(10)]
            vizu_tensor_list[1::2] = [top10_reconstructions[n] for n in range(10)]
            img = make_grid(vizu_tensor_list, nrow = 2, padding=0)
            logger.add_image('top10_reconstructions', img, self.n_epochs)
            vizu_tensor_list = [None] * (2*10)
            vizu_tensor_list[0::2] = [bottom10_images[n] for n in range(10)]
            vizu_tensor_list[1::2] = [bottom10_reconstructions[n] for n in range(10)]
            img = make_grid(vizu_tensor_list, nrow = 2, padding=0)
            logger.add_image('bottom10_reconstructions', img, self.n_epochs)

        if log_loss:
            for k, v in losses.items():
                logger.add_scalars('loss/{}'.format(k), {'valid': np.mean (v)}, self.n_epochs)
                
        return predictions, losses
    
    
    def save_predictions(self, predictions, npz_filepath):
        np.savez(npz_filepath, predictions = predictions)
    
    
    def visualize_results(self):
        pass
    

    def run_training(self, train_loader=None, valid_loader=None, keep_best_model=True, logger=None):
        """
        logger: tensorboard X summary writer
        """   
        if not self.config.freeze_decoder:
            # all the evaluation models are plugged to the frozen representation and trained for 100 epochs
            n_epochs = 50
             
            # Save the graph in the logger
            if logger is not None:
                dummy_input = torch.FloatTensor(1, self.network.encoder.n_channels, self.network.encoder.input_size[0], self.network.encoder.input_size[1]).uniform_(0,1)
                dummy_input = self.push_variable_to_device(dummy_input)
                #logger.add_graph(nn.Sequential(representation.model.encoder, self), dummy_input)
                
            if valid_loader is not None:
                best_valid_loss = sys.float_info.max
                do_validation = True
            
            for epoch in range(n_epochs):
                t0 = time.time()
                train_losses = self.train_epoch (train_loader, logger=logger)
                t1 = time.time()
                
                if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in train_losses.items():
                        logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                    logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1-t0), self.n_epochs)
                
                if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                    self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))
                
                if do_validation:
                    _, valid_losses = self.do_evaluation_pass (valid_loader, logger=logger)
                    valid_loss = np.mean(valid_losses['total'])
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))
            
            # if we want the representation to keep the model that performed best on the valid dataset
            if keep_best_model:
                best_model_path = os.path.join(self.config.checkpoint.folder, "best_weight_model.pth")
                if os.path.exists(best_model_path):
                    best_model = gr.dnn.BaseDNN.load_checkpoint(best_model_path, use_gpu = self.model.config.device.use_gpu)
                    self.network.load_state_dict(best_model.network.state_dict())
                    self.optimizer.load_state_dict(best_model.optimize.state_dict())
        
                    
                    
    def run_representation_testing(self, dataloader, testing_config = None):
        test_data = dict()

        # run testing on the test data
        test_predictions, test_losses = self.do_evaluation_pass(dataloader)
        
        test_data["predictions"] = test_predictions
        test_data["error"] = test_losses
        
        
        return test_data
