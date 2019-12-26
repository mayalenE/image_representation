import copy
from goalrepresent import models
from goalrepresent import dnn
from goalrepresent.dnn import losses
from goalrepresent.dnn.networks import encoders, decoders, discriminators
from itertools import chain
import numpy as np
import os
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image

EPS = 1e-12

""" ========================================================================================================================
GAN Architectures:
========================================================================================================================="""
class DCGANModel(dnn.BaseDNN):
    '''
    DCGAN Class
    '''
    def __init__(self, n_channels = 1, input_size = (64,64), n_latents = 10, model_architecture = "Radford", n_conv_layers = 4, reconstruction_dist = 'bernouilli', use_gpu = True, **kwargs):
        super(DCGANModel, self).__init__()
        
        # store the initial parameters used to create the model
        self.init_params = locals()
        del self.init_params['self']
        
        # define the device to use (gpu or cpu)
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents
        
        # network
        decoder = decoders.get_decoder(model_architecture)
        discriminator = discriminators.get_discriminator(model_architecture)
        self.decoder = decoder(self.n_channels, self.input_size, self.n_conv_layers, self.n_latents)
        self.discriminator = discriminator(self.n_channels, self.input_size, self.n_conv_layers, output_size = 1)
        self.reconstruction_dist = reconstruction_dist
        
        self.n_epochs = 0
    
    def decode(self, z):
        if self.use_gpu and not z.is_cuda:
           z = z.cuda()
        return self.decoder(z)
    
    def discriminate(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        return self.discriminator(x)

    def forward(self, x_real):
        if self.use_gpu and not x_real.is_cuda:
            x_real = x_real.cuda()
        return {'recon_x': x_real, 'mu': torch.zeros((x_real.size(0), self.n_latents)), 'logvar':  torch.zeros((x_real.size(0), self.n_latents))}
    
    def set_optimizer(self, optimizer_name, optimizer_hyperparameters):
        optimizer = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer_generator = optimizer(self.decoder.parameters(), **optimizer_hyperparameters)
        self.optimizer_discriminator = optimizer(self.discriminator.parameters(), **optimizer_hyperparameters)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch (self, train_loader):
        self.train()
        losses = {}
        for data in train_loader:
            x_real = Variable(data['image'], requires_grad = True)    
            
            z_fake = Variable(torch.randn(x_real.size(0), self.n_latents))
            x_fake = self.decode(z_fake)
            
            real_label = Variable(torch.ones(x_real.size(0)))
            fake_label = Variable(torch.zeros(x_real.size(0)))
            
            # (1) Train the discriminator
            output_prob_real = self.discriminate(x_real)
            output_prob_fake = self.discriminate(x_fake.detach())

            if self.use_gpu: 
                output_prob_real = output_prob_real.cuda()
                output_prob_fake = output_prob_fake.cuda()
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()
            
            loss_discriminator = (self.criterion(output_prob_real, real_label) + self.criterion(output_prob_fake, fake_label)) / 2.0
            self.optimizer_discriminator.zero_grad()
            loss_discriminator.backward()
            self.optimizer_discriminator.step()
            
            
            
            # (2) Train the generator
            output_prob_fake = self.discriminate(x_fake)
            if self.use_gpu:
                output_prob_fake = output_prob_fake.cuda()
            loss_generator = self.criterion(output_prob_fake, real_label)
            self.optimizer_generator.zero_grad()
            loss_generator.backward()
            self.optimizer_generator.step()
#            for name, param in self.decoder.named_parameters():
#                if param.requires_grad:
#                    print(name, '{:0.2f}'.format(param.data.sum().item()))
        

            # save losses
            final_losses = {'discriminator': loss_discriminator, 'generator': loss_generator, 'total': loss_discriminator+loss_generator}
            for k, v in final_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())
            # debug only on first batch:        
            #break
                    
        for k, v in losses.items():
            losses [k] = np.mean (v)
        
        self.n_epochs +=1
        return losses
    
    
    def valid_epoch (self, valid_loader, save_image_in_folder=None):
        self.eval()
        losses = {}
        with torch.no_grad():
            for data in valid_loader:
                x_real = Variable(data['image'])    
                
                z_fake = Variable(torch.randn(x_real.size(0), self.n_latents))
                x_fake = self.decode(z_fake)
                
                real_label = Variable(torch.ones(x_real.size(0)))
                fake_label = Variable(torch.zeros(x_real.size(0)))

                output_prob_real = self.discriminate(x_real)
                output_prob_fake = self.discriminate(x_fake)
                
                if self.use_gpu: 
                    output_prob_real = output_prob_real.cuda()
                    output_prob_fake = output_prob_fake.cuda()
                    real_label = real_label.cuda()
                    fake_label = fake_label.cuda()
                
                loss_discriminator = self.criterion(output_prob_real, real_label) + self.criterion(output_prob_fake, fake_label)
                loss_generator = self.criterion(output_prob_fake, real_label)
                
    
                # save losses
                final_losses = {'discriminator': loss_discriminator, 'generator': loss_generator, 'total': loss_discriminator+loss_generator}
                for k, v in final_losses.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())
                # debug only on first batch:        
                #break
           
        for k, v in losses.items():
            losses [k] = np.mean (v)
            
        # save images
        if save_image_in_folder is not None and self.n_epochs % 5 == 0:
            generated_images = x_fake.cpu().data
            n_images = x_fake.size(0)
            vizu_tensor_list = [generated_images[n] for n in range(n_images)]
            filename = os.path.join (save_image_in_folder, 'Epoch{0}.png'.format(self.n_epochs))
            n_cols = (n_images // 4) + 1
            save_image(vizu_tensor_list, filename, nrow=n_cols, padding=0, normalize = True, range = (-1.,1.))

        return losses


class BiGANModel(dnn.BaseDNN):
    '''
    BiGAN Class
    '''
    def __init__(self, n_channels = 1, input_size = (64,64), n_latents = 10, model_architecture = "Burgess", n_conv_layers = 4, reconstruction_dist = 'bernouilli', use_gpu = True, **kwargs):
        super(BiGANModel, self).__init__()
        
        # store the initial parameters used to create the model
        self.init_params = locals()
        del self.init_params['self']
        
        # define the device to use (gpu or cpu)
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents
        
        # network
        encoder = encoders.get_encoder(model_architecture)
        decoder = decoders.get_decoder(model_architecture)
        discriminator = discriminators.get_discriminator(model_architecture)
        self.encoder = encoder(self.n_channels, self.input_size, self.n_conv_layers, self.n_latents)
        self.decoder = decoder(self.n_channels, self.input_size, self.n_conv_layers, self.n_latents)
        self.discriminator = discriminator(self.n_channels, self.input_size, self.n_conv_layers, output_size = 1)
        self.reconstruction_dist = reconstruction_dist
        
        self.n_epochs = 0
        self.n_iters = 0
        
    def encode(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar):
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        if self.use_gpu and not z.is_cuda:
           z = z.cuda()
        return self.decoder(z)
    
    def discriminate(self, x, z):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        if self.use_gpu and not z.is_cuda:
            z = z.cuda()
        return self.discriminator(x, z)

    def forward(self, x_real):
        if self.use_gpu and not x_real.is_cuda:
            x_real = x_real.cuda()
#        mu_real, logvar_real = self.encode(x_real)
#        z_real = self.reparameterize(mu_real, logvar_real)
        z_real = self.encode(x_real)
        mu_real = z_real
        logvar_real = torch.zeros_like(mu_real)
        
        z_fake = Variable(torch.randn(x_real.size()[0], self.n_latents))
        x_fake = self.decode(z_fake)
        
        if self.training:
            noise1 = Variable(torch.Tensor(x_real.size()).normal_(0, 0.1 * (1000 - self.n_iters) / 1000))
            noise2 = Variable(torch.Tensor(x_fake.size()).normal_(0, 0.1 * (1000 - self.n_iters) / 1000))
            if self.use_gpu:
                 noise1 = noise1.cuda()
                 noise2 = noise2.cuda()
            x_real = x_real + noise1
            x_fake = x_fake + noise2
        output_prob_real = self.discriminate(x_real, z_real)
        output_prob_fake = self.discriminate(x_fake, z_fake)
        
        return {'recon_x': self.decode(z_real), 'mu': mu_real, 'logvar': logvar_real, 'sampled_z': z_real, 'z_fake': z_fake, 'x_fake': x_fake, 'output_prob_real': output_prob_real, 'output_prob_fake': output_prob_fake}
    
        
    def calc_embedding(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        mu, logvar = self.encode(x)
        
        return mu
    
    def recon_loss (self, recon_x, x):
        if self.reconstruction_dist == "bernouilli":
            return losses.BCE_with_digits_loss(recon_x, x)  
        elif self.reconstruction_dist == "gaussian":
            return losses.MSE_loss(recon_x,x)
        else:
            raise ValueError ("Unkown decoder distribution: {}".format(self.reconstruction_dist))
            
    def train_loss(self, outputs, inputs):
        output_prob_real = outputs['output_prob_real']
        output_prob_fake = outputs['output_prob_fake']
        real_label = Variable(torch.ones(output_prob_real.size()[0]))
        fake_label = Variable(torch.zeros(output_prob_real.size()[0]))
       
        if self.use_gpu and not output_prob_real.is_cuda:
            output_prob_real = output_prob_real.cuda()
            
        if self.use_gpu and not output_prob_fake.is_cuda:
            output_prob_fake = output_prob_fake.cuda()
            
        if self.use_gpu and not real_label.is_cuda:
            real_label = real_label.cuda()
            
        if self.use_gpu and not fake_label.is_cuda:
            fake_label = fake_label.cuda()

        discriminator_loss = self.criterion(output_prob_real, real_label) + self.criterion(output_prob_fake, fake_label)
        generator_loss = self.criterion(output_prob_real, fake_label) + self.criterion(output_prob_fake, real_label)

        return {'discriminator': discriminator_loss, 'generator': generator_loss, 'total': discriminator_loss-generator_loss}
    
    def valid_losses(self, outputs, inputs):
        valid_losses = self.train_loss(outputs, inputs)
        x = inputs['image']
        recon_x = outputs['recon_x']        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
        recon_loss = self.recon_loss(recon_x, x)
        valid_losses['total'] = recon_loss
        return valid_losses
    
    
    def set_optimizer(self, optimizer_name, optimizer_hyperparameters):
        optimizer = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer_generator = optimizer(chain(self.encoder.parameters(), self.decoder.parameters()), **optimizer_hyperparameters)
        self.optimizer_discriminator = optimizer(self.discriminator.parameters(), **optimizer_hyperparameters)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch (self, train_loader):
        self.train()
        losses = {}
        for data in train_loader:
            input_img = Variable(data['image'], requires_grad = True)    
            
            # forward
            # train discriminator first and save it
            outputs = self.forward(input_img)
            batch_losses = self.train_loss(outputs, data)
            loss_discriminator = batch_losses['discriminator']
            final_losses =  batch_losses
            loss_generator = batch_losses['generator']
            if loss_generator.data.item() < 3.5:
                self.optimizer_discriminator.zero_grad()
                loss_discriminator.backward()
                self.optimizer_discriminator.step()
            backup = copy.deepcopy(self.discriminator)
            # finetune it on this data 10 times
            for K in range(1):
                outputs = self.forward(input_img)
                batch_losses = self.train_loss(outputs, data)
                loss_discriminator = batch_losses['discriminator']
                self.optimizer_discriminator.zero_grad()
                loss_discriminator.backward(retain_graph=True)
                self.optimizer_discriminator.step()
            # train generator with the finetuned discriminator
            outputs = self.forward(input_img)
            batch_losses = self.train_loss(outputs, data)
            loss_generator = batch_losses['generator']
            final_losses ['generator'] = loss_generator
            self.optimizer_generator.zero_grad()
            loss_generator.backward()
            self.optimizer_generator.step()
            
            # reput discriminator not finetuned
            self.discriminator.load_state_dict(backup.state_dict())  
            del backup
#            outputs = self.forward(input_img)
#            batch_losses = self.train_loss(outputs, data)
#            
#            # backward
#            loss_discriminator = batch_losses['discriminator']
#            loss_generator = batch_losses['generator']
#            
#            if loss_generator.data.item() < 3.5:
#                self.optimizer_discriminator.zero_grad()
#                loss_discriminator.backward(retain_graph=True)
#                '''
#                for name, param in self.discriminator.named_parameters():
#                    if param.requires_grad:
#                        print(name, '{:0.2f}'.format(param.grad.data.sum().item()))
#                '''
#                self.optimizer_discriminator.step()
#                            
#
#            self.optimizer_generator.zero_grad()
#            loss_generator.backward()
#            '''
#            for name, param in self.encoder.named_parameters():
#                    if param.requires_grad:
#                        print(name, '{:0.2f}'.format(param.grad.data.sum().item()))
#            for name, param in self.decoder.named_parameters():
#                    if param.requires_grad:
#                        print(name, '{:0.2f}'.format(param.grad.data.sum().item()))
#            #'''
#            self.optimizer_generator.step()

            # save losses
            for k, v in final_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())
                    
            #break
                    
        for k, v in losses.items():
            losses [k] = np.mean (v)
        
        self.n_iters += 1
        self.n_epochs +=1
        return losses
    
    
    def valid_epoch (self, valid_loader, save_image_in_folder=None):
        self.eval()
        losses = {}
        with torch.no_grad():
            for data in valid_loader:
                input_img = Variable(data['image'])
                # forward
                outputs = self.forward(input_img)
                batch_losses = self.valid_losses(outputs, data)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())
                #break
                    
        for k, v in losses.items():
            losses [k] = np.mean (v)
            
        # save images
        if save_image_in_folder is not None and self.n_epochs % 10 == 0:
            input_images = input_img.cpu().data
            output_images = torch.sigmoid(outputs['recon_x']).cpu().data
            n_images = data['image'].size()[0]
            vizu_tensor_list = [None] * (2*n_images)
            vizu_tensor_list[:n_images] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[n_images:] = [output_images[n] for n in range(n_images)]
            filename = os.path.join (save_image_in_folder, 'Epoch{0}.png'.format(self.n_epochs))
            save_image(vizu_tensor_list, filename, nrow=n_images, padding=0)

        return losses

