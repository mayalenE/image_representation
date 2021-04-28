from abc import ABC, abstractmethod
import torch
from torch.nn import functional as F
from image_representation.utils.torch_functional import logsumexp, PI
from image_representation.utils.minkowski_utils import ME_sparse_to_dense
import math

class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        pass


def get_loss(loss_name):
    """
    loss_name: string such that the loss called is <loss_name>Loss
    """
    return eval("{}Loss".format(loss_name))

class TripletCLRLoss(BaseLoss):
    def __init__(self, margin=1.0, distance='cosine', **kwargs):
        self.distance = distance
        self.margin = margin

        self.input_keys_list = ['z', 'z_aug']


    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            z_ref = loss_inputs['z']
            z_a = loss_inputs['z_aug']
        except:
            raise ValueError("TripletCLRLoss needs {} inputs".format(self.input_keys_list))

        # normalize projection feature vectors
        z_ref = F.normalize(z_ref, dim=1)
        z_a = F.normalize(z_a, dim=1)
        # we artificially create negatives by shuffling the indices
        id_shuffle = torch.randperm(z_a.size()[0], requires_grad=False)
        id_shuffle[id_shuffle == torch.arange(z_a.size()[0])] -= 1
        id_shuffle = id_shuffle.detach()
        z_b = z_a[id_shuffle]

        if self.distance == 'euclidean':
            distance_a = (z_ref - z_a).pow(2).sum(1).sqrt()
            distance_b = (z_ref - z_b).pow(2).sum(1).sqrt()
        elif self.distance == 'cosine':
            distance_a = (1.0 - F.cosine_similarity(z_ref, z_a))
            distance_b = (1.0 - F.cosine_similarity(z_ref, z_b))
        elif self.distance == 'chebyshev':
            distance_a = (z_ref - z_a).abs().max(dim=1)[0]
            distance_b = (z_ref - z_b).abs().max(dim=1)[0]

        triplet_loss = F.relu(distance_a - distance_b + self.margin)

        output_losses = dict()
        output_losses['total'] = triplet_loss

        if reduction == "sum":
            for k, v in output_losses.items():
                output_losses[k] = v.sum()
        elif reduction == "mean":
            for k, v in output_losses.items():
                output_losses[k] = v.mean()

        return output_losses

class SimCLRLoss(BaseLoss):
    def __init__(self, temperature=0.5, distance='cosine', **kwargs):
        self.distance = distance
        self.temperature = temperature

        self.input_keys_list = ['proj_z', 'proj_z_aug']

    def get_correlated_mask(self, batch_size):
        diag = torch.eye(2 * batch_size)
        l1 = torch.diag(torch.ones(batch_size), diagonal=-batch_size)
        l2 = torch.diag(torch.ones(batch_size), diagonal=batch_size)
        mask = (diag + l1 + l2)
        mask = (1 - mask).type(torch.bool)
        return mask


    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            proj_z = loss_inputs['proj_z']
            proj_z_aug = loss_inputs['proj_z_aug']

        except:
            raise ValueError("SimCLRLoss needs {} inputs".format(self.input_keys_list))

        batch_size = proj_z.shape[0]

        # normalize projection feature vectors
        proj_z = F.normalize(proj_z, dim=1)
        proj_z_aug = F.normalize(proj_z_aug, dim=1)

        representations = torch.cat([proj_z, proj_z_aug], dim=0)
        if self.distance == 'cosine':
            similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self.get_correlated_mask(batch_size).to(similarity_matrix.device)

        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(similarity_matrix.device).long()
        loss = _ce_loss(logits, labels, reduction=reduction)

        clr_loss = loss / (2 * batch_size)

        output_losses = dict()
        output_losses['total'] = clr_loss

        return output_losses


class TripletLoss(BaseLoss):
    def __init__(self, margin=0.0, distance='squared_euclidean', combined_loss_name=None, combined_loss_parameters=None,
                 combined_loss_scale_factor=1.0, use_attention=False, **kwargs):
        self.margin = margin
        self.distance = distance

        if combined_loss_name is None:
            self.combined_loss = None
        else:
            combined_loss_class = get_loss(combined_loss_name)
            self.combined_loss = combined_loss_class(**combined_loss_parameters)
            self.combined_loss_scale_factor = combined_loss_scale_factor

        self.use_attention = use_attention

        if self.distance == 'cosine' or self.distance == 'squared_cosine':
            assert self.margin < 1.0, print("cosine distance is between 0 and 1 so margin has to be < 1")
            assert not self.use_attention, print("cosine distance does not support attention for the moment")


        self.input_keys_list = ['x_ref_outputs', 'x_a_outputs', 'x_b_outputs', 'x_c_outputs']
        if self.use_attention:
            self.input_keys_list.append('attention')

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            x_ref_outputs = loss_inputs['x_ref_outputs']
            x_a_outputs = loss_inputs['x_a_outputs']
            x_b_outputs = loss_inputs['x_b_outputs']
            x_c_outputs = loss_inputs['x_c_outputs']
            z_ref = x_ref_outputs["z"]
            z_a = x_a_outputs["z"]
            z_b = x_b_outputs["z"]
            if self.use_attention:
                attention = loss_inputs["attention"]
            else:
                attention = torch.ones_like(z_ref)
        except:
            raise ValueError("TripletLoss needs {} inputs".format(self.input_keys_list))

        if self.distance == 'euclidean':
            distance_a = ((z_ref - z_a).pow(2) * attention).sum(1).sqrt()
            distance_b = ((z_ref - z_b).pow(2) * attention).sum(1).sqrt()
        elif self.distance == 'cosine':
            distance_a = (1.0-F.cosine_similarity(z_ref, z_a))
            distance_b = (1.0-F.cosine_similarity(z_ref, z_b))
        elif self.distance == 'chebyshev':
            distance_a = ((z_ref - z_a).abs() * attention).max(dim=1)[0]
            distance_b = ((z_ref - z_b).abs() * attention).max(dim=1)[0]
        elif self.distance == 'squared_euclidean':
            distance_a = ((z_ref - z_a).pow(2) * attention).sum(1)
            distance_b = ((z_ref - z_b).pow(2) * attention).sum(1)
        elif self.distance == 'squared_cosine':
            distance_a = (1.0-F.cosine_similarity(z_ref, z_a)).pow(2)
            distance_b = (1.0-F.cosine_similarity(z_ref, z_b)).pow(2)
        elif self.distance == 'squared_chebyshev':
            distance_a = ((z_ref - z_a).abs() * attention).max(dim=1)[0].pow(2)
            distance_b = ((z_ref - z_b).abs() * attention).max(dim=1)[0].pow(2)

        triplet_loss = F.relu(distance_a - distance_b + self.margin)

        output_losses = dict()
        if self.combined_loss is None:
            output_losses['total'] = triplet_loss

        else:
            combined_loss = dict()
            combined_loss_ref = self.combined_loss(x_ref_outputs, reduction=reduction)
            combined_loss_a = self.combined_loss(x_a_outputs, reduction=reduction)
            combined_loss_b = self.combined_loss(x_b_outputs, reduction=reduction)
            combined_loss_c = self.combined_loss(x_c_outputs, reduction=reduction)
            for k in combined_loss_ref.keys():
                combined_loss[k] = (combined_loss_ref[k] + combined_loss_a[k] + combined_loss_b[k] + combined_loss_c[k]) / 4.0
            combined_loss_scaled = combined_loss['total'] / self.combined_loss_scale_factor

            total_loss = triplet_loss + combined_loss_scaled

            output_losses['total'] = total_loss
            output_losses['triplet'] = triplet_loss
            output_losses['combined'] = combined_loss_scaled + combined_loss_b[k]
            for k, v in combined_loss.items():
                if k != 'total':
                    output_losses[k] = v

        if reduction == "sum":
            for k, v in output_losses.items():
                output_losses[k] = v.sum()
        elif reduction == "mean":
            for k, v in output_losses.items():
                output_losses[k] = v.mean()

        return output_losses


class QuadrupletLoss(BaseLoss):
    def __init__(self, margin=0.0, distance='euclidean', combined_loss_name=None, combined_loss_parameters=None, combined_loss_scale_factor=1.0, use_attention = False, **kwargs):
        self.margin = margin
        self.distance = distance

        if combined_loss_name is None:
            self.combined_loss = None
        else:
            combined_loss_class = get_loss(combined_loss_name)
            self.combined_loss = combined_loss_class(**combined_loss_parameters)
            self.combined_loss_scale_factor = combined_loss_scale_factor

        self.use_attention = use_attention

        if self.distance == 'cosine' or self.distance == 'squared_cosine':
            assert self.margin < 1.0, print("cosine distance is between 0 and 1 so margin has to be < 1")
            assert not self.use_attention, print("cosine distance does not support attention for the moment")

        self.input_keys_list = ['x_pos_a_outputs', 'x_pos_b_outputs', 'x_neg_a_outputs', 'x_neg_b_outputs']
        if self.use_attention:
            self.input_keys_list.append('attention')


    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            x_pos_a_outputs = loss_inputs['x_pos_a_outputs']
            x_pos_b_outputs = loss_inputs['x_pos_b_outputs']
            x_neg_a_outputs = loss_inputs['x_neg_a_outputs']
            x_neg_b_outputs = loss_inputs['x_neg_b_outputs']
            z_pos_a = x_pos_a_outputs["z"]
            z_pos_b = x_pos_b_outputs["z"]
            z_neg_a = x_neg_a_outputs["z"]
            z_neg_b = x_neg_b_outputs["z"]
            if self.use_attention:
                attention = loss_inputs["attention"]
            else:
                attention = torch.ones_like(z_pos_a)
        except:
            raise ValueError("QuadrupletLoss needs {} inputs".format(self.input_keys_list))

        if self.distance == 'euclidean':
            distance_pos = ((z_pos_a - z_pos_b).pow(2) * attention).sum(1).sqrt()
            distance_neg = ((z_neg_a - z_neg_b).pow(2) * attention).sum(1).sqrt()
        elif self.distance == 'cosine':
            distance_pos = (1.0 - F.cosine_similarity(z_pos_a, z_pos_b))
            distance_neg = (1.0 - F.cosine_similarity(z_neg_a, z_neg_b))
        elif self.distance == 'chebyshev':
            distance_pos = ((z_pos_a - z_pos_b).abs() * attention).max(dim=1)[0]
            distance_neg = ((z_neg_a - z_neg_b).abs() * attention).max(dim=1)[0]
        elif self.distance == 'squared_euclidean':
            distance_pos = ((z_pos_a - z_pos_b).pow(2) * attention).sum(1)
            distance_neg = ((z_neg_a - z_neg_b).pow(2) * attention).sum(1)
        elif self.distance == 'squared_cosine':
            distance_pos = (1.0 - F.cosine_similarity(z_pos_a, z_pos_b)).pow(2)
            distance_neg = (1.0 - F.cosine_similarity(z_neg_a, z_neg_b)).pow(2)
        elif self.distance == 'squared_chebyshev':
            distance_pos = ((z_pos_a - z_pos_b).abs() * attention).max(dim=1)[0].pow(2)
            distance_neg = ((z_neg_a - z_neg_b).abs() * attention).max(dim=1)[0].pow(2)

        quadruplet_loss = F.relu(distance_pos - distance_neg + self.margin)

        output_losses = dict()
        if self.combined_loss is None:
            output_losses['total'] = quadruplet_loss

        else:
            combined_loss = dict()
            combined_loss_pos_a = self.combined_loss(x_pos_a_outputs, reduction=reduction)
            combined_loss_pos_b = self.combined_loss(x_pos_b_outputs, reduction=reduction)
            combined_loss_neg_a = self.combined_loss(x_neg_a_outputs, reduction=reduction)
            combined_loss_neg_b = self.combined_loss(x_neg_b_outputs, reduction=reduction)
            for k in combined_loss_pos_a.keys():
                combined_loss[k] = (combined_loss_pos_a[k] + combined_loss_pos_b[k] + combined_loss_neg_a[k] + combined_loss_neg_b[k]) / 4.0
            combined_loss_scaled = combined_loss['total'] / self.combined_loss_scale_factor

            total_loss = quadruplet_loss + combined_loss_scaled

            output_losses['total'] = total_loss
            output_losses['quadruplet'] = quadruplet_loss
            output_losses['combined'] = combined_loss_scaled
            for k, v in combined_loss.items():
                if k != 'total':
                    output_losses[k] = v

        if reduction == "sum":
            for k, v in output_losses.items():
                output_losses[k] = v.sum()
        elif reduction == "mean":
            for k, v in output_losses.items():
                output_losses[k] = v.mean()

        return output_losses


class DIMLoss(BaseLoss):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.input_keys_list = ['global_pos', 'global_neg', 'local_pos', 'local_neg', 'prior_pos', 'prior_neg']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            global_pos = loss_inputs['global_pos']
            global_neg = loss_inputs['global_neg']
            local_pos = loss_inputs['local_pos']
            local_neg = loss_inputs['local_neg']
            prior_pos = loss_inputs['prior_pos']
            prior_neg = loss_inputs['prior_neg']

        except:
            raise ValueError("DIMLoss needs {} inputs".format(self.input_keys_list))
        global_loss = _gan_loss(global_pos, global_neg, reduction=reduction)
        local_loss = _gan_loss(local_pos, local_neg, reduction=reduction)
        prior_loss = _gan_loss(prior_pos, prior_neg, reduction=reduction)

        total_loss = self.alpha * global_loss + self.beta * local_loss + self.gamma * prior_loss

        return {'total': total_loss, 'global': global_loss, 'local': local_loss, 'prior': prior_loss}


class BiGANLoss(BaseLoss):
    def __init__(self, **kwargs):
        self.input_keys_list = ['prob_pos', 'prob_neg']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            prob_pos = loss_inputs['prob_pos']
            prob_neg = loss_inputs['prob_neg']
        except:
            raise ValueError("BiGANLoss needs {} inputs".format(self.input_keys_list))

        discriminator_loss = _gan_loss(prob_pos, prob_neg)
        generator_loss = _gan_loss(prob_neg, prob_pos)
        total_loss = discriminator_loss.detach() + generator_loss.detach()

        return {'discriminator': discriminator_loss, 'generator': generator_loss, 'total': total_loss}


class VAEGANLoss(BaseLoss):
    def __init__(self, reconstruction_dist="bernoulli", **kwargs):
        self.reconstruction_dist = reconstruction_dist
        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar', 'prob_pos', 'prob_neg']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_inputs['x']
            prob_pos = loss_inputs['prob_pos']
            prob_neg = loss_inputs['prob_neg']
        except:
            raise ValueError("VAELoss needs {} inputs".format(self.input_keys_list))
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist, reduction=reduction)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)
        vae_loss = recon_loss + KLD_loss

        discriminator_loss = _gan_loss(prob_pos, prob_neg)
        generator_loss = _gan_loss(prob_neg, prob_pos)

        total_loss = vae_loss + discriminator_loss + generator_loss

        return {'total': total_loss, 'vae': vae_loss, 'recon': recon_loss, 'KLD': KLD_loss,
                'discriminator': discriminator_loss, 'generator': generator_loss}


class VAELoss(BaseLoss):
    def __init__(self, reconstruction_dist="bernoulli", **kwargs):
        self.reconstruction_dist = reconstruction_dist

        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar',
                                'gfi_cls', 'gfi_target',
                                'lfi_cls', 'lfi_target',
                                'recon_x_cls', 'recon_x_target']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            x = loss_inputs['x']
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            gfi_cls = loss_inputs['gfi_cls']
            gfi_target = loss_inputs['gfi_target']
            lfi_cls = loss_inputs['lfi_cls']
            lfi_target = loss_inputs['lfi_target']
            recon_x_cls = loss_inputs['recon_x_cls']
            recon_x_target = loss_inputs['recon_x_target']

        except:
            raise ValueError("VAELoss needs {} inputs".format(self.input_keys_list))

        recon_c_loss_gfi = _reconstruction_loss(gfi_cls.F.squeeze(), gfi_target.type(gfi_cls.dtype), "bernoulli", reduction=reduction)
        recon_c_loss_lfi = _reconstruction_loss(lfi_cls.F.squeeze(), lfi_target.type(lfi_cls.dtype), "bernoulli", reduction=reduction)
        recon_c_loss_x = _reconstruction_loss(recon_x_cls.F.squeeze(), recon_x_target.type(recon_x_cls.dtype), "bernoulli", reduction=reduction)
        recon_c_loss = (recon_c_loss_gfi + recon_c_loss_lfi + recon_c_loss_x) / 3.0

        # x = ME_sparse_to_dense(x)[0]
        # recon_x = ME_sparse_to_dense(recon_x, shape=x.shape)[0]
        # recon_f_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist, reduction=reduction)
        recon_f_loss = torch.tensor([0.0], device=recon_c_loss.device)

        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)
        total_loss = recon_c_loss + recon_f_loss + KLD_loss

        return {'total': total_loss, 'recon_c': recon_c_loss, 'recon_f': recon_f_loss, 'KLD': KLD_loss}


class BetaVAELoss(BaseLoss):
    def __init__(self, beta=5.0, reconstruction_dist="bernoulli", **kwargs):
        self.reconstruction_dist = reconstruction_dist
        self.beta = beta

        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_inputs['x']
        except:
            raise ValueError("BetaVAELoss needs {} inputs".format(self.input_keys_list))
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist, reduction=reduction)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar, reduction=reduction)
        total_loss = recon_loss + self.beta * KLD_loss

        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}


class AnnealedVAELoss(BaseLoss):
    def __init__(self, gamma=1000.0, c_min=0.0, c_max=5.0, c_change_duration=100000, reconstruction_dist="bernoulli",
                 **kwargs):
        self.reconstruction_dist = reconstruction_dist
        self.gamma = gamma
        self.c_min = c_min
        self.c_max = c_max
        self.c_change_duration = c_change_duration

        # update counters
        self.capacity = self.c_min
        self.n_iters = 0

        self.input_keys_list = ['x', 'recon_x', 'mu', 'logvar']

    def update_encoding_capacity(self):
        if self.n_iters > self.c_change_duration:
            self.capacity = self.c_max
        else:
            self.capacity = min(self.c_min + (self.c_max - self.c_min) * self.n_iters / self.c_change_duration,
                                self.c_max)

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            x = loss_inputs['x']
        except:
            raise ValueError("AnnealedVAELoss needs {} inputs".format(self.input_keys_list))
        recon_loss = _reconstruction_loss(recon_x, x, self.reconstruction_dist)
        KLD_loss, KLD_per_latent_dim, KLD_var = _kld_loss(mu, logvar)
        total_loss = recon_loss + self.gamma * (KLD_loss - self.capacity).abs()

        if total_loss.requires_grad:  # if we are in "train mode", update counters
            self.n_iters += 1
            self.update_encoding_capacity()

        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}


class BetaTCVAELoss(BaseLoss):
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
            recon_x = loss_inputs['recon_x']
            mu = loss_inputs['mu']
            logvar = loss_inputs['logvar']
            sampled_z = loss_inputs['z']
            x = loss_inputs['x']
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
            W = torch.Tensor(batch_size, batch_size, requires_grad=True).fill_(1 / M)
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


class FCClassifierLoss(BaseLoss):
    def __init__(self, **kwargs):
        self.input_keys_list = ['y', 'predicted_y']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            predicted_y = loss_inputs['predicted_y']
            y = loss_inputs['y']
        except:
            raise ValueError("FCClassifierLoss needs {} inputs".format(self.input_keys_list))
        CE_loss = _ce_loss(predicted_y, y, reduction=reduction)

        return {'total': CE_loss}


class SVMClassifierLoss(BaseLoss):
    def __init__(self, **kwargs):
        self.input_keys_list = ['y', 'predicted_y']

    def __call__(self, loss_inputs, reduction="mean", **kwargs):
        try:
            predicted_y = loss_inputs['predicted_y']
            y = loss_inputs['y']
        except:
            raise ValueError("SVMlassifierLoss needs {} inputs".format(self.input_keys_list))
        # predicted_y is already a log probabilities here
        CE_loss = _nll_loss(predicted_y, y, reduction=reduction)

        return {'total': CE_loss}


"""=======================================
LOSS HELPERS
=========================================="""


def _reconstruction_loss(recon_x, x, reconstruction_dist="bernoulli", reduction="mean"):
    if reconstruction_dist == "bernoulli":
        loss = _bce_with_logits_loss(recon_x, x, reduction=reduction)
    elif reconstruction_dist == "gaussian":
        loss = _mse_loss(recon_x, x, reduction=reduction)
    else:
        raise ValueError("Unkown decoder distribution: {}".format(reconstruction_dist))
    return loss


def _kld_loss(mu, logvar, reduction="mean"):
    """ Returns the KLD loss D(q,p) where q is N(mu,var) and p is N(0,I) """
    if reduction == "mean":
        # 0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss_per_latent_dim = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0) / mu.size(
            0)  # we  average on the batch
        # KL-divergence between a diagonal multivariate normal and the standard normal distribution is the sum on each latent dimension
        KLD_loss = torch.sum(KLD_loss_per_latent_dim)
        # we add a regularisation term so that the KLD loss doesnt "trick" the loss by sacrificing one dimension
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim)
    elif reduction == "sum":
        # 0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss_per_latent_dim = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        # KL-divergence between a diagonal multivariate normal and the standard normal distribution is the sum on each latent dimension
        KLD_loss = torch.sum(KLD_loss_per_latent_dim)
        # we add a regularisation term so that the KLD loss doesnt "trick" the loss by sacrificing one dimension
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim)
    elif reduction == "none":
        KLD_loss_per_latent_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        KLD_loss = torch.sum(KLD_loss_per_latent_dim, dim=1)
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim, dim=1)
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')

    return KLD_loss, KLD_loss_per_latent_dim, KLD_loss_var


def _mse_loss(recon_x, x, reduction="mean"):
    """ Returns the reconstruction loss (mean squared error) summed on the image dims and averaged on the batch size """
    if reduction == "mean":
        mse_loss =  F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    elif reduction == "sum":
        mse_loss = F.mse_loss(recon_x, x, reduction="sum")
    elif reduction == "none":
        mse_loss = F.mse_loss(recon_x, x, reduction="none")
        mse_loss = mse_loss.view(mse_loss.size(0), -1).sum(1)
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')
    return mse_loss


def _bce_loss(recon_x, x, reduction="mean"):
    """ Returns the reconstruction loss (binary cross entropy) summed on the image dims and averaged on the batch size """
    if reduction == "mean":
        bce_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    elif reduction == "sum":
        bce_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    elif reduction == "none":
        bce_loss = F.binary_cross_entropy(recon_x, x, reduction="none")
        bce_loss = bce_loss.view(bce_loss.size(0), -1).sum(1)
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')
    return bce_loss


def _bce_with_logits_loss(recon_x, x, reduction="mean"):
    """ Returns the reconstruction loss (sigmoid + binary cross entropy) summed on the image dims and averaged on the batch size """
    if reduction == "mean":
        bce_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum") / x.size(0)
    elif reduction == "sum":
        bce_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    elif reduction == "none":
        bce_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="none")
        bce_loss = bce_loss.view(bce_loss.size(0), -1).sum(1)
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')
    return bce_loss


def _ce_loss(recon_y, y, reduction="mean"):
    """ Returns the cross entropy loss (softmax + NLLLoss) averaged on the batch size """
    if reduction == "mean":
        ce_loss =  F.cross_entropy(recon_y, y, reduction="sum") / y.size(0)
    elif reduction == "sum":
        ce_loss = F.cross_entropy(recon_y, y, reduction="sum")
    elif reduction == "none":
        ce_loss = F.cross_entropy(recon_y, y, reduction="none")
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')
    return ce_loss


def _nll_loss(recon_y, y, reduction="mean"):
    """ Returns the cross entropy loss (softmax + NLLLoss) averaged on the batch size """
    if reduction == "mean":
        nll_loss = F.nll_loss(recon_y, y, reduction="sum") / y.size(0)
    elif reduction == "sum":
        nll_loss = F.nll_loss(recon_y, y, reduction="sum")
    elif reduction == "none":
        nll_loss = F.nll_loss(recon_y, y, reduction="none")
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')
    return nll_loss


def _gan_loss(positive, negative, reduction="mean"):
    """ Eq. 4 from the DIM paper is equivalent to binary cross-entropy 
    i.e. minimizing the output of this function is equivalent to maximizing
    the output of jsd_mi(positive, negative)
    """
    if reduction == "mean":
        real = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction="sum") / positive.size(
            0)
        fake = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction="sum")
        if len(negative.size()) > 0:
            fake /= negative.size(0)
    elif reduction == "sum":
        real = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction="sum")
        fake = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction="sum")
    elif reduction == "none":
        real = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction="none")
        real = real.view(real.size(0), -1).sum(1)
        fake = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction="none")
        fake = fake.view(fake.size(0), -1).sum(1)
    else:
        raise ValueError('reduction must be either "mean" | "sum" | "none" ')

    return real + fake
