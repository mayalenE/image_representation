import torch
from image_representation.utils.torch_functional import logsumexp, PI
from image_representation.representations.torch_nn.losses import BaseLoss, _reconstruction_loss, _kld_loss
import math


def get_loss(loss_name):
    """
    loss_name: string such that the loss called is <loss_name>Loss
    """
    return eval("ME{}Loss".format(loss_name))


class MEVAELoss(BaseLoss):
    def __init__(self,  reconstruct_coordinates=True, reconstruct_features=True, reconstruction_dist="bernoulli", **kwargs):

        self.reconstruction_dist = reconstruction_dist
        self.reconstruct_coordinates = reconstruct_coordinates
        self.reconstruct_features = reconstruct_features

        self.input_keys_list = ['x', 'recon_x',
                                'mu', 'logvar',
                                'out_cls', 'out_targets']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            x = loss_inputs['x']
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            out_cls = loss_inputs['out_cls']
            out_targets = loss_inputs['out_targets']

        except:
            raise ValueError("VAELoss needs {} inputs".format(self.input_keys_list))


        # coordinates reconstruction loss
        recon_c_loss = torch.tensor([0.0], device=mu.device)
        if self.reconstruct_coordinates:
            for out_cl, target in zip(out_cls, out_targets):
                curr_loss = _reconstruction_loss(out_cl.F.squeeze(-1), target.type(out_cl.F.dtype), "bernoulli", reduction=reduction)
                recon_c_loss += curr_loss / len(out_cls)

        # feature loss on closest coordinates from target
        recon_f_loss = torch.tensor([0.0], device=mu.device)
        if self.reconstruct_features:
            batch_size = x.C[:, 0].max().item()
            for batch_idx in range(batch_size):
                recon_mask = (recon_x.C[:, 0] == batch_idx)
                x_mask = (x.C[:, 0] == batch_idx)
                if (recon_mask.sum() == 0) or (x_mask.sum() == 0) :
                    continue
                coords_recon = recon_x.C[recon_mask, 1:] #.cpu()
                feats_recon = recon_x.F[recon_mask]
                coords_x = x.C[x_mask, 1:] #.cpu()
                feats_x = x.F[x_mask]
                # divide nearest neighbor computation in chuck of 1000
                sub_batch_size = 1000
                n_sub_batches = coords_recon.shape[0] // sub_batch_size + 1
                for sub_batch_idx in range(n_sub_batches):
                    cur_inds = range(sub_batch_idx*sub_batch_size, min((sub_batch_idx+1)*sub_batch_size, coords_recon.shape[0]))
                    if len(cur_inds) > 0:
                        cur_coords_recon = coords_recon[cur_inds]
                        cur_coords_recon = cur_coords_recon.reshape(-1, 1, cur_coords_recon.shape[-1])
                        closest_x_coodinates = (coords_x - cur_coords_recon).pow(2).sum(-1).min(-1).indices
                        recon_f_loss += _reconstruction_loss(feats_recon[cur_inds],
                                                             feats_x[closest_x_coodinates],
                                                             reconstruction_dist=self.reconstruction_dist, reduction=reduction)

        # KLD loss
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)

        total_loss = (recon_c_loss + recon_f_loss) / 2.0 + KLD_loss

        return {'total': total_loss, 'recon_c': recon_c_loss, 'recon_f': recon_f_loss, 'KLD': KLD_loss}

class MEFVAELoss(BaseLoss):
    def __init__(self, reconstruction_dist="bernoulli", **kwargs):
        self.reconstruction_dist = reconstruction_dist

        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            x = loss_inputs['x'].F
            recon_x = loss_inputs['recon_x'].F
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']

        except:
            raise ValueError("VAELoss needs {} inputs".format(self.input_keys_list))

        recon_f_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist, reduction=reduction)

        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)
        total_loss = recon_f_loss + KLD_loss

        return {'total': total_loss, 'recon': recon_f_loss, 'KLD': KLD_loss}


class MEFBetaVAELoss(MEFVAELoss):
    def __init__(self, beta=5.0, reconstruction_dist="bernoulli", **kwargs):
        MEFVAELoss.__init__(self, reconstruction_dist=reconstruction_dist)
        self.beta = beta

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        vae_loss_dict = MEFVAELoss.__call__(self, loss_inputs=loss_inputs, reduction=reduction)
        bvae_loss_dict = vae_loss_dict
        bvae_loss_dict['total'] = vae_loss_dict['recon'] + self.beta * vae_loss_dict['KLD']

        return bvae_loss_dict


class MEFAnnealedVAELoss(MEFVAELoss):
    def __init__(self, gamma=1000.0, c_min=0.0, c_max=5.0, c_change_duration=100000, reconstruction_dist="bernoulli",
                 **kwargs):
        MEFVAELoss.__init__(self, reconstruction_dist=reconstruction_dist)
        self.gamma = gamma
        self.c_min = c_min
        self.c_max = c_max
        self.c_change_duration = c_change_duration

        # update counters
        self.capacity = self.c_min
        self.n_iters = 0


    def update_encoding_capacity(self):
        if self.n_iters > self.c_change_duration:
            self.capacity = self.c_max
        else:
            self.capacity = min(self.c_min + (self.c_max - self.c_min) * self.n_iters / self.c_change_duration,
                                self.c_max)

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        vae_loss_dict = MEFVAELoss.__call__(self, loss_inputs=loss_inputs, reduction=reduction)
        recon_loss = vae_loss_dict['recon']
        KLD_loss = vae_loss_dict['KLD']
        total_loss = recon_loss + self.gamma * (KLD_loss - self.capacity).abs()

        if total_loss.requires_grad:  # if we are in "train mode", update counters
            self.n_iters += 1
            self.update_encoding_capacity()

        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}


class MEFBetaTCVAELoss(BaseLoss):
    def __init__(self, alpha=1.0, beta=10.0, gamma=1.0, tc_approximate='mss', dataset_size=0,
                 reconstruction_dist="bernoulli", **kwargs):
        self.reconstruction_dist = reconstruction_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tc_approximate = tc_approximate
        self.dataset_size = dataset_size

        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar', 'z']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            x = loss_inputs['x'].F
            recon_x = loss_inputs['recon_x'].F
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            sampled_z = loss_inputs['z'].F
        except:
            raise ValueError("BetaTCVAELoss needs {} inputs".format(self.input_keys_list))
        # reconstruction loss
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist)

        # KL LOSS MODIFIED
        ## calculate log q(z|x) (log density of gaussian(mu,sigma2))
        log_q_zCx = (-0.5 * (math.log(2.0 * PI) + logvar) - (sampled_z - mu).pow(2) / (2 * logvar.exp())).sum(
            1)  # sum on the latent dimensions (factorized distribution so log of prod is sum of logs)

        ## calculate log p(z) (log density of gaussian(0,1))
        log_pz = (-0.5 * math.log(2.0 * PI) - sampled_z.pow(2) / 2).sum(1)

        ## calculate log_qz ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m)) and log_prod_qzi
        batch_size = sampled_z.size(0)
        n_latents = sampled_z.size(1)
        _logqz = -0.5 * (math.log(2.0 * PI) + logvar.view(1, batch_size, n_latents)) - (
                sampled_z.view(batch_size, 1, n_latents) - mu.view(1, batch_size, n_latents)).pow(2) / (
                         2 * logvar.view(1, batch_size, n_latents).exp())
        if self.tc_approximate == 'mws':
            # minibatch weighted sampling
            log_prod_qzi = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(
                batch_size * self.dataset_size)).sum(1)
            log_qz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(
                batch_size * self.dataset_size))
        elif self.tc_approximate == 'mss':
            # minibatch stratified sampling
            N = self.dataset_size
            M = max(batch_size - 1, 1)
            strat_weight = (N - M) / (N * M)
            W = torch.empty((batch_size, batch_size)).fill_(1 / M)
            W.view(-1)[::M + 1] = 1 / N
            W.view(-1)[1::M + 1] = strat_weight
            W[M - 1, 0] = strat_weight
            logiw_matrix = W.log().type_as(_logqz.data)
            log_qz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            log_prod_qzi = logsumexp(logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1,
                                                keepdim=False).sum(1)
        else:
            raise ValueError(
                'The minibatch approximation of the total correlation "{}" is not defined'.format(self.tc_approximate))

        ## I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        ## TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        ## dw_kl_loss is KL[q(z)||p(z)] (dimension-wise KL term)
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        # TOTAL LOSS
        total_loss = recon_loss + self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss

        return {'total': total_loss, 'recon': recon_loss, 'mi': mi_loss, 'tc': tc_loss, 'dw_kl': dw_kl_loss}