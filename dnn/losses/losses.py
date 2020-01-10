from abc import ABC, abstractmethod
from goalrepresent.helper import mathhelper
import math
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, loss_inputs, loss_targets, reduction = True, **kwargs):
        pass
    
def get_loss(loss_name):
    """
    loss_name: string such that the loss called is <loss_name>Loss
    """
    return eval("{}Loss".format(loss_name))


class DIMLoss(BaseLoss):
    def __init__(self, alpha=1.0, beta=0.0, gamma = 1.0, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def __call__(self, loss_inputs, loss_targets, reduction = True, **kwargs):
        try:
            global_neg = loss_inputs['global_neg']
            local_neg = loss_inputs['local_neg']
            prior_neg = loss_inputs['prior_neg']
            global_pos = loss_targets['global_pos']
            local_pos = loss_targets['local_pos']
            prior_pos = loss_targets['prior_pos']
        except:
            raise ValueError("VAELoss needs global_neg, local_neg, prior_neg inputs and global_pos, local_pos, prior_pos targets")
        global_loss = _mi_bce_loss(global_pos, global_neg, reduction=reduction)
        local_loss = _mi_bce_loss(local_pos, local_neg, reduction=reduction)
        prior_loss = _mi_bce_loss(prior_pos, prior_neg, reduction=reduction)
        
        total_loss = self.alpha * global_loss + self.beta * local_loss + self.gamma * prior_loss
        
        return {'total': total_loss, 'global': global_loss, 'local': local_loss, 'prior': prior_loss}
    

class VAELoss(BaseLoss):
    def __init__(self, reconstruction_dist="bernouilli", **kwargs):
        self.reconstruction_dist = reconstruction_dist
        
    def __call__(self, loss_inputs, loss_targets, reduction = True, **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_targets['x']
        except:
            raise ValueError("VAELoss needs recon_x, mu, logvar inputs and x targets")
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist, reduction=reduction)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)
        total_loss = recon_loss + KLD_loss
        
        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}
    
    
class BetaVAELoss(BaseLoss):
    def __init__(self, beta = 5.0, reconstruction_dist="bernouilli", **kwargs):
        super().__init(**kwargs)
        self.reconstruction_dist = reconstruction_dist
        self.beta = beta
        
    def __call__(self, loss_inputs, loss_targets, reduction = True, **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_targets['x']
        except:
            raise ValueError("BetaVAELoss needs recon_x, mu, logvar inputs and x targets")
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist, reduction=reduction)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)
        total_loss = recon_loss + self.beta * KLD_loss

        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}
    
    
class AnnealedVAELoss(BaseLoss):
    def __init__(self, gamma=1000.0, capacity=0.0, reconstruction_dist="bernouilli", **kwargs):
        super().__init(**kwargs)
        self.reconstruction_dist = reconstruction_dist
        self.gamma = gamma
        self.capacity = capacity
        
    def __call__(self, loss_inputs, loss_targets, reduction = True, **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_targets['x']
        except:
            raise ValueError("AnnealedVAELoss needs recon_x, mu, logvar inputs and x targets")
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar)
        total_loss = recon_loss +  self.gamma * (KLD_loss - self.capacity).abs()

        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}
    
    
class BetaTCVAELoss(BaseLoss):
    def __init__(self, alpha=1.0, beta=10.0, gamma=1.0, tc_approximate='mss', dataset_size = 0, reconstruction_dist="bernouilli", **kwargs):
        super().__init(**kwargs)
        self.reconstruction_dist = reconstruction_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tc_approximate == tc_approximate
        self.dataset_size = dataset_size
        
    def __call__(self, loss_inputs, loss_targets, reduction = True, **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            sampled_z = loss_inputs['sampled_z']
            x = loss_targets['x']
        except:
            raise ValueError("VAELoss needs recon_x, mu, logvar, sampled_z inputs and x targets")
        # reconstruction loss
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist)
                
        # KL LOSS MODIFIED
        ## calculate log q(z|x) (log density of gaussian(mu,sigma2))
        log_q_zCx = (-0.5 * (math.log(2.0*np.pi) + logvar) - (sampled_z-mu).pow(2) / (2 * logvar.exp())).sum(1) # sum on the latent dimensions (factorized distribution so log of prod is sum of logs)
        
        ## calculate log p(z) (log density of gaussian(0,1))
        log_pz = (-0.5 * math.log(2.0*np.pi) - sampled_z.pow(2) / 2).sum(1)
        
        ## calculate log_qz ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m)) and log_prod_qzi
        batch_size = sampled_z.size(0)
        n_latents = sampled_z.size(1)
        _logqz = -0.5 * (math.log(2.0*np.pi) + logvar.view(1, batch_size, n_latents)) - (sampled_z.view(batch_size, 1, n_latents) - mu.view(1, batch_size, n_latents)).pow(2) / (2 * logvar.view(1, batch_size, n_latents).exp())
        if self.tc_approximate == 'mws':
            # minibatch weighted sampling
            log_prod_qzi = (mathhelper.logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * self.dataset_size)).sum(1)
            log_qz = (mathhelper.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * self.dataset_size))
        elif self.tc_approximate == 'mss':
            # minibatch stratified sampling
            N = self.dataset_size
            M = batch_size - 1
            strat_weight = (N - M) / (N * M)
            W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
            W.view(-1)[::M+1] = 1 / N
            W.view(-1)[1::M+1] = strat_weight
            W[M-1, 0] = strat_weight
            logiw_matrix = Variable(W.log().type_as(_logqz.data))
            log_qz = mathhelper.logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            log_prod_qzi = mathhelper.logsumexp(logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)
        else:
            raise ValueError ('The minibatch approximation of the total correlation "{}" is not defined'.format(self.tc_approximate))

        ## I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        ## TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        ## dw_kl_loss is KL[q(z)||p(z)] (dimension-wise KL term)
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        # TOTAL LOSS
        total_loss = recon_loss + self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss
        
        return {'total': total_loss, 'recon': recon_loss, 'mi': mi_loss, 'tc': tc_loss, 'dw_kl': dw_kl_loss}
    
    
class FCClassifierLoss(BaseLoss):
    def __init__(self, **kwargs):
        pass
        
    def __call__(self, loss_inputs, loss_targets, reduction = True, **kwargs):
        try:
            predicted_y = loss_inputs['predicted_y']
            y = loss_targets['y']
        except:
            raise ValueError("FCClassifierLoss needs predicted_y inputs and y targets")
        CE_loss = _ce_loss(predicted_y, y, reduction=reduction)
        
        return {'total': CE_loss}
    
class SVMClassifierLoss(BaseLoss):
    def __init__(self, **kwargs):
        pass
        
    def __call__(self, loss_inputs, loss_targets, reduction = True, **kwargs):
        try:
            predicted_y = loss_inputs['predicted_y']
            y = loss_targets['y']
        except:
            raise ValueError("SVMlassifierLoss needs predicted_y inputs and y targets")
        # predicted_y is already a log probabilities here
        CE_loss = _nll_loss(predicted_y, y, reduction=reduction)
        
        return {'total': CE_loss}

    
    
"""=======================================
LOSS HELPERS
=========================================="""
  
def _reconstruction_loss(recon_x, x, reconstruction_dist="bernouilli", reduction=True):
    if reconstruction_dist == "bernouilli":
        loss = _bce_with_digits_loss(recon_x, x, reduction=reduction)  
    elif reconstruction_dist == "gaussian":
        loss = _mse_loss(recon_x, x, reduction=reduction)
    else:
        raise ValueError ("Unkown decoder distribution: {}".format(reconstruction_dist))

    return loss


def _kld_loss(mu, logvar, reduction=True):
    if reduction:
        """ Returns the KLD loss D(q,p) where q is N(mu,var) and p is N(0,I) """
        # 0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss_per_latent_dim = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 0 ) / mu.size(0) #we  average on the batch
        #KL-divergence between a diagonal multivariate normal and the standard normal distribution is the sum on each latent dimension
        KLD_loss = torch.sum(KLD_loss_per_latent_dim)
        # we add a regularisation term so that the KLD loss doesnt "trick" the loss by sacrificing one dimension
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim)
    else:
        KLD_loss_per_latent_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        KLD_loss = torch.sum(KLD_loss_per_latent_dim, dim = 1)
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim, dim = 1)

    
    return KLD_loss, KLD_loss_per_latent_dim, KLD_loss_var


def _mse_loss(recon_x, x, reduction=True): 
    if reduction:
        """ Returns the reconstruction loss (mean squared error) summed on the image dims and averaged on the batch size """
        return F.mse_loss(recon_x, x, reduction = "sum" ) / x.size(0) 
    else:
        return F.mse_loss(recon_x, x, reduction = "none").sum(3).sum(2).sum(1) # sum on the image dims but not on the batch dimension

def _bce_loss(recon_x, x, reduction=True):
    if reduction:
        """ Returns the reconstruction loss (binary cross entropy) summed on the image dims and averaged on the batch size """
        return F.binary_cross_entropy(recon_x, x, reduction = "sum" ) / x.size(0)
    else:
        return F.binary_cross_entropy(recon_x, x, reduction = "none").sum(3).sum(2).sum(1)
def _bce_with_digits_loss(recon_x, x, reduction=True): 
    if reduction:
        """ Returns the reconstruction loss (sigmoid + binary cross entropy) summed on the image dims and averaged on the batch size """
        return F.binary_cross_entropy_with_logits(recon_x, x, reduction = "sum" ) / x.size(0) 
    else:
        return F.binary_cross_entropy_with_logits(recon_x, x, reduction = "none").sum(3).sum(2).sum(1)

def _ce_loss(recon_y, y, reduction=True):
    """ Returns the cross entropy loss (softmax + NLLLoss) averaged on the batch size """
    if reduction:
        return F.cross_entropy(recon_y, y, reduction = "sum" ) / y.size(0) 
    else:
        return F.cross_entropy(recon_y, y, reduction = "none")
    
def _nll_loss(recon_y, y, reduction=True):
    """ Returns the cross entropy loss (softmax + NLLLoss) averaged on the batch size """
    if reduction:
        return F.nll_loss(recon_y, y, reduction = "sum" ) / y.size(0)
    else:
        return F.nll_loss(recon_y, y, reduction = "none")
    
def _mi_bce_loss(positive, negative,  reduction=True):
    """ Eq. 4 from the DIM paper is equivalent to binary cross-entropy 
    i.e. minimizing the output of this function is equivalent to maximizing
    the output of jsd_mi(positive, negative)
    """
    if reduction:
        real = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction = "sum") / positive.size(0)
        fake = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction = "sum")
        if len(negative.size()) > 0:
            fake /= negative.size(0)
        return real + fake
    else:
        real = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction = "none").sum(3).sum(2).sum(1)
        fake = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction = "none").sum(3).sum(2).sum(1)
        return real + fake