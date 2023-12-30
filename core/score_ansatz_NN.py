from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn.modules.loss import MSELoss
from tqdm import trange, tqdm #.notebook
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.utils.plot_utils import saveallforms
from core.gaussian_mixture_lib import GaussianMixture, GaussianMixture_torch
#%%
def marginal_prob_std(t, sigma):
  """Note that this std -> 0, when t->0
  So it's not numerically stable to sample t=0 in the dataset
  Note an earlier version missed the sqrt...
  """
  return torch.sqrt( (sigma**(2*t) - 1) / 2 / torch.log(torch.tensor(sigma)) ) # sqrt fixed Jun.19


def denoise_loss_fn(model, x, marginal_prob_std_f, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability, sample t uniformly from [eps, 1.0]
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std_f(random_t,)
  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1)))
  return loss


def train_score_td(X_train_tsr, score_model_td=None,
                   sigma=25,
                   lr=0.005,
                   nepochs=750,
                   eps=1E-3,
                   batch_size=None):
    ndim = X_train_tsr.shape[1]
    if score_model_td is None:
        score_model_td = ScoreModel_Time(sigma=sigma, ndim=ndim)
    marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)
    optim = Adam(score_model_td.parameters(), lr=lr)
    pbar = trange(nepochs)
    for ep in pbar:
        if batch_size is None:
            loss = denoise_loss_fn(score_model_td, X_train_tsr, marginal_prob_std_f, eps=eps)
        else:
            idx = torch.randint(0, X_train_tsr.shape[0], (batch_size,))
            loss = denoise_loss_fn(score_model_td, X_train_tsr[idx], marginal_prob_std_f, eps=eps)

        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"step {ep} loss {loss.item():.3f}")
        if ep == 0:
            print(f"step {ep} loss {loss.item():.3f}")
    return score_model_td



def reverse_diffusion_time_dep(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False):
  """
  score_model_td: if `exact` is True, use a gmm of class GaussianMixture
                  if `exact` is False. use a torch neural network that takes vectorized x and t as input.
  """
  lambdaT = (sigma**2 - 1) / (2 * np.log(sigma))
  xT = np.sqrt(lambdaT) * np.random.randn(sampN, ndim)
  x_traj_rev = np.zeros((*xT.shape, nsteps, ))
  x_traj_rev[:,:,0] = xT
  dt = 1 / nsteps
  for i in range(1, nsteps):
    t = 1 - i * dt
    tvec = torch.ones((sampN)) * t
    eps_z = np.random.randn(*xT.shape)
    if exact:
      gmm_t = diffuse_gmm(score_model_td, t, sigma)
      score_xt = gmm_t.score(x_traj_rev[:,:,i-1])
    else:
      with torch.no_grad():
        score_xt = score_model_td(torch.tensor(x_traj_rev[:,:,i-1]).float(), tvec).numpy()
    # simple Euler-Maryama integration of SGD
    x_traj_rev[:,:,i] = x_traj_rev[:,:,i-1] + eps_z * (sigma ** t) * np.sqrt(dt) + score_xt * dt * sigma**(2*t)
  return x_traj_rev


def visualize_diffusion_distr(x_traj_rev, leftT=0, rightT=-1, explabel=""):
  if rightT == -1:
    rightT = x_traj_rev.shape[2]-1
  figh, axs = plt.subplots(1,2,figsize=[12,6])
  sns.kdeplot(x=x_traj_rev[:,0,leftT], y=x_traj_rev[:,1,leftT], ax=axs[0])
  axs[0].set_title("Density of Gaussian Prior of $x_T$\n before reverse diffusion")
  plt.axis("equal")
  sns.kdeplot(x=x_traj_rev[:,0,rightT], y=x_traj_rev[:,1,rightT], ax=axs[1])
  axs[1].set_title(f"Density of $x_0$ samples after {rightT} step reverse diffusion")
  plt.axis("equal")
  plt.suptitle(explabel)
  return figh
#%% Sample dataset generation
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
import math
import torch
import torch.nn.functional as F
def gaussian_mixture_logprob_score_torch(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model in PyTorch
    :param x: [N batch, N dim]
    :param mus: [N comp, N dim]
    :param Us: [N comp, N dim, N dim]
    :param Lambdas: [N comp, N dim]
    :param weights: [N comp,] or None
    :return:
    """
    ndim = x.shape[-1]
    logdetSigmas = torch.sum(torch.log(Lambdas), dim=-1)  # [N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    rot_residuals = torch.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists = torch.sum(rot_residuals ** 2 / Lambdas[None, :, :], dim=-1)  # [N batch, N comp]
    if weights is not None:
        logprobs = (-0.5 * (logdetSigmas[None, :] + MHdists) +
                    torch.log(weights))  # - 0.5 * ndim * torch.log(2 * torch.pi)  # [N batch, N comp]
    else:
        logprobs = -0.5 * (logdetSigmas[None, :] + MHdists)
    participance = F.softmax(logprobs, dim=-1)  # [N batch, N comp]
    compo_score_vecs = torch.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas[None, :, :]),
                                    Us)  # [N batch, N comp, N dim]
    score_vecs = torch.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
    # logsumexp trick
    logprob = torch.logsumexp(logprobs, dim=-1)  # [N batch,]
    logprob -= 0.5 * ndim * math.log(2 * torch.pi)
    return logprob, score_vecs


def gaussian_mixture_score_batch_sigma_torch(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model in PyTorch
    :param x: [N batch,N dim]
    :param mus: [N comp, N dim]
    :param Us: [N comp, N dim, N dim]
    :param Lambdas: [N batch, N comp, N dim]
    :param weights: [N comp,] or None
    :return:
    """
    if Lambdas.ndim == 2:
        Lambdas = Lambdas[None, :, :]
    ndim = x.shape[-1]
    logdetSigmas = torch.sum(torch.log(Lambdas), dim=-1)  # [N batch, N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    rot_residuals = torch.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists = torch.sum(rot_residuals ** 2 / Lambdas, dim=-1)  # [N batch, N comp]
    if weights is not None:
        logprobs = (-0.5 * (logdetSigmas + MHdists) +
                    torch.log(weights))  # - 0.5 * ndim * torch.log(2 * torch.pi)  # [N batch, N comp]
    else:
        logprobs = -0.5 * (logdetSigmas + MHdists)
    participance = F.softmax(logprobs, dim=-1)  # [N batch, N comp]
    compo_score_vecs = torch.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas),
                                    Us)  # [N batch, N comp, N dim]
    score_vecs = torch.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
    return score_vecs
#%%
import torch.nn as nn
class GMM_ansatz_net(nn.Module):
    def __init__(self, ndim, n_components, sigma=5.0):
        super().__init__()
        self.ndim = ndim
        self.n_components = n_components
        self.mus = nn.Parameter(torch.randn(n_components, ndim))
        self.Us = nn.Parameter(torch.randn(n_components, ndim, ndim))
        self.logLambdas = nn.Parameter(torch.randn(n_components, ndim))
        self.logweights = nn.Parameter(torch.log(torch.ones(n_components) / n_components))
        self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

    def forward(self, x, t):
        """
        x: (batch, ndim)
        sigma: (batch, )
        """
        sigma = self.marginal_prob_std_f(t)
        return gaussian_mixture_score_batch_sigma_torch(x, self.mus, self.Us,
               self.logLambdas.exp()[None, :, :] + sigma[:, None, None]**2, self.logweights.exp())


#%%
import os
from sklearn.model_selection import train_test_split
figdir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/HaimDiffusionRNNProj/Ring_GMM_ansatz_train"
os.makedirs(figdir, exist_ok=True)
train_pnts = 20
# ring_X = generate_ring_samples_torch(train_pnts)
ring_X = generate_spiral_samples_torch(train_pnts, a=0.4, b=0.15)
# train test split
Xtrain, Xtest = ring_X, torch.empty(0, 2)  # train_test_split(ring_X, test_size=0.0001, random_state=42)
# Xtrain, Xtest = train_test_split(ring_X, test_size=0.0001, random_state=42)
sigma_max = 10
gmm_components = 50
# mlp_width = 8
# mlp_depth = 3 # note, 2 layer usually doesn't work. 3 layer works.
# act_fun = nn.Tanh
# cfg_str = f"mlp {mlp_depth} layer width{mlp_width} {act_fun.__name__} sigma{sigma_max}"
cfg_str = f"sigma{sigma_max} dense ring {gmm_components} components"
# cfg_str = f"sigma{sigma_max} dense spiral, {gmm_components} components"
# dataset_str = f"ring_{train_pnts}"
dataset_str = f"spiral_{train_pnts}"
for batch_size in [1024]: # 128, 256, 512,
    for epochs in [250, 500, 750, 1000, 1500, 2000, 3000]:# :
        torch.manual_seed(42)
        score_model_td = GMM_ansatz_net(ndim=2, n_components=gmm_components, sigma=sigma_max)
        score_model_td = train_score_td(Xtrain, score_model_td=score_model_td,
                        sigma=sigma_max, lr=0.05, nepochs=epochs, batch_size=batch_size)
        #%%
        x_traj_denoise = reverse_diffusion_time_dep(score_model_td, sampN=3000, sigma=sigma_max, nsteps=1000, ndim=2, exact=False)
        figh = visualize_diffusion_distr(x_traj_denoise,
                             explabel=f"Time Dependent NN trained from weighted denoising\nepoch{epochs} batch {batch_size}\n{cfg_str}")
        figh.axes[1].set_xlim([-2.5, 2.5])
        figh.axes[1].set_ylim([-2.5, 2.5])
        saveallforms(figdir, f"{dataset_str}_NN_contour_train_Ncomp{gmm_components}_batch{batch_size}_ep{epochs:04d}_sde")
        figh.show()
        #%%
        plt.figure(figsize=(7, 7))
        plt.scatter(x_traj_denoise[:, 0, -1], x_traj_denoise[:, 1, -1], alpha=0.1, lw=0.1, color="k", label="score net gen samples")
        plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=200, alpha=0.9, label="train", marker="o")
        plt.scatter(Xtest[:, 0], Xtest[:, 1], s=200, alpha=0.9, label="test", marker="o")
        plt.axis("image")
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.title(f"NN Generated Samples\nepoch{epochs} batch {batch_size}\n{cfg_str}")
        plt.legend()
        plt.tight_layout()
        saveallforms(figdir, f"{dataset_str}_NN_samples_train_Ncomp{gmm_components}_batch{batch_size}_ep{epochs:04d}_sde")
        plt.show()
#%%
# len(list(score_model_td.parameters()))
# count the total number of parameters
sum(p.numel() for p in score_model_td.parameters() if p.requires_grad)




#%%
# GMM ansatz
ndim = 2
n_components = 10
mus = torch.randn(n_components, ndim)
covs = torch.randn(n_components, ndim, ndim)
covs = covs @ covs.transpose(-1, -2)
weights = torch.ones(n_components) / n_components
#%%
x = torch.randn(1000, ndim)

torch.softmax()








