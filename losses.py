import torch
import torch.nn as nn
import torch.distributions as dist
from models.LSA.model import BaseModule

def kl_loss_fn(z_post, sum_samples=True, correct=False, sumdim=(1,2,3)):
    z_prior = dist.Normal(0, 1.0)
    kl_div = dist.kl_divergence(z_post, z_prior)
    if correct:
        kl_div = torch.sum(kl_div, dim=sumdim)
    else:
        kl_div = torch.mean(kl_div, dim=sumdim)
    if sum_samples:
        return torch.mean(kl_div)
    else:
        return kl_div

def rec_loss_fn(recon_x, x, sum_samples=True, correct=False, sumdim=(1,2,3)):
    if correct:
        x_dist = dist.Laplace(recon_x, 1.0)
        log_p_x_z = x_dist.log_prob(x)
        log_p_x_z = torch.sum(log_p_x_z, dim=sumdim)
    else:
        log_p_x_z = -torch.abs(recon_x - x)
        log_p_x_z = torch.mean(log_p_x_z, dim=sumdim)
    if sum_samples:
        return -torch.mean(log_p_x_z)
    else:
        return -log_p_x_z

def geco_beta_update(beta, error_ema, goal, step_size, min_clamp=1e-10, max_clamp=1e4, speedup=None):
    constraint = (error_ema - goal).detach()
    if speedup is not None and constraint > 0.0:
        beta = beta * torch.exp(speedup * step_size * constraint)
    else:
        beta = beta * torch.exp(step_size * constraint)
    if min_clamp is not None:
        beta = np.max((beta.item(), min_clamp))
    if max_clamp is not None:
        beta = np.min((beta.item(), max_clamp))
    return beta

def get_ema(new, old, alpha):
    if old is None:
        return new
    return (1.0 - alpha) * new + alpha * old