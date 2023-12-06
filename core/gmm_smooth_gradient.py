from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn.modules.loss import MSELoss
from tqdm import trange, tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.utils.plot_utils import saveallforms
from core.gaussian_mixture_lib import GaussianMixture, GaussianMixture_torch
from core.gmm_special_diffusion_lib import GMM_density, GMM_logprob, GMM_scores
from core.gmm_special_diffusion_lib import GMM_density_torch, GMM_logprob_torch, GMM_scores_torch
def diffuse_gmm(gmm, t, sigma):
  lambda_t = marginal_prob_std_np(t, sigma)**2 # variance
  noise_cov = np.eye(gmm.dim) * lambda_t
  covs_dif = [cov + noise_cov for cov in gmm.covs]
  return GaussianMixture(gmm.mus, covs_dif, gmm.weights)


def marginal_prob_std_np(t, sigma):
  return np.sqrt( (sigma**(2*t) - 1) / 2 / np.log(sigma) )


def generate_spiral_samples_torch(n_points, a=1, b=0.2):
    """Generate points along a spiral using PyTorch.
    Parameters:
    - n_points (int): Number of points to generate.
    - a, b (float): Parameters that define the spiral shape.
    Returns:
    - torch.Tensor: Batch of vectors representing points on the spiral.
    """
    theta = torch.linspace(0, 4 * np.pi, n_points)  # angle theta
    r = a + b * theta  # radius
    x = r * torch.cos(theta)  # x = r * cos(theta)
    y = r * torch.sin(theta)  # y = r * sin(theta)
    spiral_batch = torch.stack((x, y), dim=1)
    return spiral_batch


def generate_ring_samples_torch(n_points, R=1, ):
    """
    Generate points along a Ring using PyTorch.
    Parameters:
    - n_points (int): Number of points to generate.
    - R: Radius of the ring.
    Returns:
    - torch.Tensor: Batch of vectors representing points on the spiral.
    """
    theta = torch.linspace(0, 2 * np.pi, n_points + 1, )  # angle theta
    theta = theta[:-1]
    x = R * torch.cos(theta)  # x = r * cos(theta)
    y = R * torch.sin(theta)  # y = r * sin(theta)
    spiral_batch = torch.stack((x, y), dim=1)
    return spiral_batch

#%%
ring_pnts = generate_ring_samples_torch(6, R=1)
# gmm_ring_pnts = GaussianMixture_torch(ring_pnts,
#                   [torch.eye(2) * 0.001 for _ in range(6)],
#                   torch.ones(6) / 6)
#%%
from scipy.ndimage import gaussian_filter
xlim = (-2, 2)
ylim = (-2, 2)
ngrid = 100

sigma_t = 1
XX, YY = torch.meshgrid(torch.linspace(*xlim, ngrid),
                        torch.linspace(*ylim, ngrid), )
pnt_vecs = torch.stack((XX, YY), dim=-1).reshape(-1, 2)
score_vecs = GMM_scores_torch(ring_pnts, sigma_t, pnt_vecs)
prob_vec = GMM_logprob_torch(ring_pnts, sigma_t, pnt_vecs)
pnt_vecs_grid = pnt_vecs.reshape(ngrid, ngrid, 2)
score_vecs_grid = score_vecs.reshape(ngrid, ngrid, 2)
#%%
subsample = 4
plt.figure(figsize=(8, 8))
plt.quiver(pnt_vecs_grid[::subsample, ::subsample, 0].numpy(),
              pnt_vecs_grid[::subsample, ::subsample, 1].numpy(),
              score_vecs_grid[::subsample, ::subsample, 0].numpy(),
              score_vecs_grid[::subsample, ::subsample, 1].numpy())
plt.scatter(ring_pnts[:, 0], ring_pnts[:, 1], s=180, color="r")
plt.axis("equal")
plt.xlim(xlim)
plt.ylim(ylim)
plt.title(f"Original Score field, sigma_t={sigma_t}", fontsize=16)
plt.tight_layout()
plt.show()
#%%
sigma_t = 0.2
fd_delta = 1.9
logprob_vec_pos0 = GMM_logprob_torch(ring_pnts, sigma_t, pnt_vecs + fd_delta * torch.tensor([1, 0]))
logprob_vec_neg0 = GMM_logprob_torch(ring_pnts, sigma_t, pnt_vecs - fd_delta * torch.tensor([1, 0]))
logprob_vec_pos1 = GMM_logprob_torch(ring_pnts, sigma_t, pnt_vecs + fd_delta * torch.tensor([0, 1]))
logprob_vec_neg1 = GMM_logprob_torch(ring_pnts, sigma_t, pnt_vecs - fd_delta * torch.tensor([0, 1]))
score_vecs_fd = torch.stack([(logprob_vec_pos0 - logprob_vec_neg0) / (2 * fd_delta),
                                (logprob_vec_pos1 - logprob_vec_neg1) / (2 * fd_delta)], dim=-1)
pnt_vecs_grid = pnt_vecs.reshape(ngrid, ngrid, 2)
score_vecs_fd_grid = score_vecs_fd.reshape(ngrid, ngrid, 2)

subsample = 4
plt.figure(figsize=(8, 8))
plt.quiver(pnt_vecs_grid[::subsample, ::subsample, 0].numpy(),
              pnt_vecs_grid[::subsample, ::subsample, 1].numpy(),
              score_vecs_fd_grid[::subsample, ::subsample, 0].numpy(),
              score_vecs_fd_grid[::subsample, ::subsample, 1].numpy())
plt.scatter(ring_pnts[:, 0], ring_pnts[:, 1], s=180, color="r")
plt.axis("equal")
plt.xlim(xlim)
plt.ylim(ylim)
plt.title(f"Smoothed Score field, sigma_t={sigma_t}, with finite diff delta={fd_delta}", fontsize=16)
plt.tight_layout()
plt.show()


#%%

sigma_t = 0.01
XX, YY = torch.meshgrid(torch.linspace(*xlim, ngrid),
                        torch.linspace(*ylim, ngrid), )
pnt_vecs = torch.stack((XX, YY), dim=-1).reshape(-1, 2)
pnt_vecs_grid = pnt_vecs.reshape(ngrid, ngrid, 2)
score_vecs = GMM_scores_torch(ring_pnts, sigma_t, pnt_vecs)
sigma_smooth = 0.5
delta = (xlim[1] - xlim[0]) / ngrid
u_smoothed = gaussian_filter(score_vecs_grid.numpy()[:,:,0], sigma=sigma_smooth / delta)
v_smoothed = gaussian_filter(score_vecs_grid.numpy()[:,:,1], sigma=sigma_smooth / delta)

subsample = 4
plt.figure(figsize=(8, 8))
plt.quiver(pnt_vecs_grid[::subsample, ::subsample, 0].numpy(),
              pnt_vecs_grid[::subsample, ::subsample, 1].numpy(),
              u_smoothed[::subsample, ::subsample],
              v_smoothed[::subsample, ::subsample])
plt.scatter(ring_pnts[:, 0], ring_pnts[:, 1], s=180, color="r")
plt.axis("equal")
plt.xlim(xlim)
plt.ylim(ylim)
plt.title(f"Smoothed Score field, sigma_t={sigma_t}, with smooth kernel={sigma_smooth}", fontsize=16)
plt.tight_layout()
plt.show()
#%%
from scipy.integrate import solve_ivp
def f_EDM_vec(t, x, mus, ):
    """Right hand side of the VP SDE probability flow ODE, vectorized version"""
    return - GMM_scores(mus, t, x.T).T


def f_fdsmooth_EDM_vec(t, x, mus, fd_delta=0.5):
    """Right hand side of the VP SDE probability flow ODE, vectorized version"""
    perturb_mat = np.eye(2) * fd_delta
    score_fd_vecs = np.zeros_like(x)
    for i in range(2):
        lp_pos = GMM_logprob(mus, t, x.T + perturb_mat[i:i+1, :])
        lp_neg = GMM_logprob(mus, t, x.T - perturb_mat[i:i+1, :])
        score_fd = (lp_pos - lp_neg) / (2 * fd_delta)
        score_fd_vecs[i, :] = score_fd
    return - score_fd_vecs.T


def exact_delta_gmm_reverse_diff_EDM(mus, xT, t_eval=None,
                                 sigma_max=80, sigma_min=0.002):
    sol = solve_ivp(lambda t, x: f_EDM_vec(t, x, mus, ),
                    (sigma_max, sigma_min), xT, method="RK45",
                    vectorized=True, t_eval=t_eval)
    return sol.y[:, -1], sol


def fdsmooth_delta_gmm_reverse_diff_EDM(mus, xT, t_eval=None, fd_delta=0.5,
                                 sigma_max=80, sigma_min=0.002):
    sol = solve_ivp(lambda t, x: f_fdsmooth_EDM_vec(t, x, mus, fd_delta=fd_delta),
                    (sigma_max, sigma_min), xT, method="RK45",
                    vectorized=True, t_eval=t_eval)
    return sol.y[:, -1], sol
#%%
from tqdm import trange, tqdm
sigma_max = 80
sigma_min=0.002
t_eval = np.logspace(np.log10(sigma_max-0.0001), np.log10(sigma_min+1E-3), 20)
sample_col = []
x_traj_col = []
for i in trange(1000):
    xT = np.random.randn(2) * sigma_max
    x_sample, sol = exact_delta_gmm_reverse_diff_EDM(ring_pnts.numpy(), xT, t_eval=t_eval,
                                                     sigma_max=sigma_max, sigma_min=sigma_min)
    sample_col.append(x_sample)
    x_traj_col.append(sol.y)
sample_col = np.stack(sample_col, axis=0)
x_traj_col = np.stack(x_traj_col, axis=0)
#%%
plt.figure(figsize=(8, 8))
plt.scatter(ring_pnts[:, 0], ring_pnts[:, 1], s=180, color="r")
plt.scatter(sample_col[:, 0], sample_col[:, 1], s=20, color="k", alpha=0.1)
plt.axis("equal")
plt.xlim(xlim)
plt.ylim(ylim)
plt.title(f"Samples from EDM, sigma_max={sigma_max}, sigma_min={sigma_min}", fontsize=16)
plt.tight_layout()
plt.show()

#%%
sigma_max = 80
sigma_min = 0.002
t_eval = np.logspace(np.log10(sigma_max-0.0001), np.log10(sigma_min+1E-3), 50)
sample_fd_col = []
x_traj_fd_col = []
for i in trange(100):
    xT = np.random.randn(2) * sigma_max
    x_sample, sol = fdsmooth_delta_gmm_reverse_diff_EDM(ring_pnts.numpy(), xT, t_eval=t_eval,
                         fd_delta=0.9, sigma_max=sigma_max, sigma_min=sigma_min)
    sample_fd_col.append(x_sample)
    x_traj_fd_col.append(sol.y)
sample_fd_col = np.stack(sample_fd_col, axis=0)
x_traj_fd_col = np.stack(x_traj_fd_col, axis=0)
#%%
# plot all traj
plt.figure(figsize=(8, 8))
plt.scatter(ring_pnts[:, 0], ring_pnts[:, 1], s=180, color="r")
plt.scatter(sample_fd_col[:, 0], sample_fd_col[:, 1], s=20, color="k", alpha=0.1)
plt.plot(x_traj_fd_col[:, 0, :].T,
         x_traj_fd_col[:, 1, :].T, color="k", alpha=0.7)
plt.axis("equal")
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.scatter(ring_pnts[:, 0], ring_pnts[:, 1], s=180, color="r")
plt.scatter(sample_fd_col[:, 0], sample_fd_col[:, 1], s=20, color="k", alpha=0.1)
plt.axis("equal")
plt.xlim(xlim)
plt.ylim(ylim)
plt.title(f"Samples from EDM with finite diff score, "
          f"sigma_max={sigma_max}, sigma_min={sigma_min}, fd delta={fd_delta}", fontsize=16)
plt.tight_layout()
plt.show()