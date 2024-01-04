import sys
sys.path.append("/n/home12/binxuwang/Github/DiffusionMemorization")
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
# from core.gaussian_mixture_lib import GaussianMixture, GaussianMixture_torch

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
                   batch_size=None,
                   device="cpu",
                   callback_func=lambda score_model_td, epochs, loss: None,
                   callback_epochs=[]):
    ndim = X_train_tsr.shape[1]
    if score_model_td is None:
        score_model_td = ScoreModel_Time(sigma=sigma, ndim=ndim)
    score_model_td.to(device)
    X_train_tsr = X_train_tsr.to(device)
    marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)
    optim = Adam(score_model_td.parameters(), lr=lr)
    pbar = trange(nepochs)
    score_model_td.train()
    loss_traj = []
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
        if ep in callback_epochs:
            callback_func(score_model_td, ep, loss)
        loss_traj.append(loss.item())
    return score_model_td, loss_traj



def reverse_diffusion_time_dep(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False, device="cpu"):
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


def reverse_diffusion_time_dep_torch(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False, device="cpu"):
  """More efficient version that run solely on device
  score_model_td: if `exact` is True, use a gmm of class GaussianMixture
                  if `exact` is False. use a torch neural network that takes vectorized x and t as input.
  """
  lambdaT = (sigma**2 - 1) / (2 * math.log(sigma))
  xT = math.sqrt(lambdaT) * torch.randn(sampN, ndim, device=device)
  x_traj_rev = torch.zeros((sampN, ndim, nsteps), device="cpu")
  x_traj_rev[:, :, 0] = xT.cpu()
  dt = 1 / nsteps
  x_next = xT
  for i in range(1, nsteps):
      t = 1 - i * dt
      tvec = torch.ones((sampN,), device=device) * t
      eps_z = torch.randn_like(xT)
      with torch.no_grad():
        score_xt = score_model_td(x_next, tvec)
      # if exact:
      #     gmm_t = diffuse_gmm(score_model_td, t, sigma, device)
      #     score_xt = gmm_t.score(x_traj_rev[:, :, i-1])
      # else:
      x_next = x_next + eps_z * (sigma ** t) * math.sqrt(dt) + score_xt * dt * sigma**(2*t)
      x_traj_rev[:, :, i] = x_next.cpu()

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

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps.
  Basically it multiplexes a scalar `t` into a vector of `sin(2 pi k t)` and `cos(2 pi k t)` features.
  """
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, t):
    t_proj = t.view(-1, 1) * self.W[None, :] * 2 * math.pi
    return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class ScoreModel_Time(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, sigma, ndim=2, nlayers=5, nhidden=50, time_embed_dim=10,
               act_fun=nn.Tanh):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    layers = []
    layers.extend([nn.Linear(time_embed_dim + ndim, nhidden),
                   act_fun()])
    for _ in range(nlayers-2):
        layers.extend([nn.Linear(nhidden, nhidden),
                         act_fun()])
    layers.extend([nn.Linear(nhidden, ndim)])
    self.net = nn.Sequential(*layers)
    self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

  def forward(self, x, t):
    std_vec = self.marginal_prob_std_f(t)[:, None,]
    t_embed = self.embed(t)
    pred = self.net(torch.cat((x / (1 + std_vec ** 2).sqrt(),
                               t_embed), dim=1))
    # this additional steps provides an inductive bias.
    # the neural network output on the same scale,
    pred = pred / std_vec
    return pred


class ScoreModel_Time_edm(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, sigma, ndim=2, nlayers=5, nhidden=50, time_embed_dim=10,
               act_fun=nn.Tanh):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    layers = []
    layers.extend([nn.Linear(time_embed_dim + ndim, nhidden), act_fun()])
    for _ in range(nlayers - 2):
        layers.extend([nn.Linear(nhidden, nhidden), act_fun()])
    layers.extend([nn.Linear(nhidden, ndim)])
    self.net = nn.Sequential(*layers)
    self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

  def forward(self, x, t):
    std_vec = self.marginal_prob_std_f(t)[:, None,]
    ln_std_vec = torch.log(std_vec) / 4
    t_embed = self.embed(ln_std_vec)
    pred = self.net(torch.cat((x / (1 + std_vec ** 2).sqrt(),
                               t_embed), dim=1))
    # this additional steps provides an inductive bias.
    # the neural network output on the same scale,
    pred = pred / std_vec - x / (1 + std_vec ** 2)
    return pred

import os
import argparse
import pickle as pkl
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiments with different configurations.")

    parser.add_argument("--shape", type=str, default="ring", help="Dataset to use")
    parser.add_argument("--train_pnts", type=int, default=20, help="Number of training points")
    # parser.add_argument("--spiral_a", type=float, default=0.4, help="Spiral parameter a")
    # parser.add_argument("--spiral_b", type=float, default=0.15, help="Spiral parameter b")
    parser.add_argument("--time_embed_dim", type=int, nargs="+", default=[16, 32, 64, 128], help="Time embedding dimensions")
    parser.add_argument("--mlp_depth", type=int, nargs="+", default=[2, 3, 4, 6, 8], help="MLP depths")
    parser.add_argument("--mlp_width", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], help="MLP widths")
    parser.add_argument("--epochs", type=int, nargs="+", default=[250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000], help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--sigma_max", type=float, default=10, help="Maximum sigma value")

    return parser.parse_args()

rootdir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/HaimDiffusionRNNProj"

args = parse_arguments()
train_pnts = args.train_pnts
batch_size = args.batch_size
if args.shape == "ring":
    ring_X = generate_ring_samples_torch(train_pnts)
    dataset_str = f"ring_{train_pnts}"
elif args.shape == "spiral":
    ring_X = generate_spiral_samples_torch(train_pnts, a=0.4, b=0.15)
    dataset_str = f"spiral_{train_pnts}"

figdir = join(rootdir, "Shape2d_MLP_train_kempner", dataset_str)
os.makedirs(figdir, exist_ok=True)
Xtrain, Xtest = ring_X, torch.empty(0, 2)
sigma_max = args.sigma_max
time_embed_dim_list = args.time_embed_dim
mlp_depth_list = args.mlp_depth
mlp_width_list = args.mlp_width
epochs_list = args.epochs
lr = args.lr
# train_pnts = 20
# batch_size = 2048
# ring_X = generate_ring_samples_torch(train_pnts)
# ring_X = generate_spiral_samples_torch(train_pnts, a=0.4, b=0.15)
# train test split
# Xtrain, Xtest = ring_X, torch.empty(0, 2)
# Xtrain, Xtest = train_test_split(ring_X, test_size=0.0001, random_state=42)
# sigma_max = 10
# gmm_components = 50
# cfg_str = f"sigma{sigma_max} dense ring {gmm_components} components"
# dataset_str = f"ring_{train_pnts}"
# dataset_str = f"spiral_{train_pnts}"
plt.switch_backend("Agg")
max_epoch = 5000
stats_col = []
for time_embed_dim in time_embed_dim_list:
    for mlp_depth in mlp_depth_list:
        for mlp_width in mlp_width_list:
            model_cfg_str = f"sigma{sigma_max} | mlp depth {mlp_depth} layer | width {mlp_width} units | time embed dim {time_embed_dim}"
            cfg_fn_str = f"temb{time_embed_dim}_depth{mlp_depth}_width{mlp_width}"
            torch.manual_seed(42)
            score_model_td = ScoreModel_Time_edm(sigma=sigma_max, ndim=2, 
                            nlayers=mlp_depth, nhidden=mlp_width, time_embed_dim=time_embed_dim,)
            print(model_cfg_str)
            print(f"Batch size {batch_size}")
            print(f"Dataset {dataset_str}")
            print("Parameter count", sum(p.numel() for p in score_model_td.parameters() if p.requires_grad))
            
            def training_callback(score_model_td, epochs, loss):
                x_traj_denoise = reverse_diffusion_time_dep_torch(score_model_td, 
                        sampN=3000, sigma=sigma_max, nsteps=1000, ndim=2, exact=False, device="cuda")
                try:
                    figh = visualize_diffusion_distr(x_traj_denoise,
                                        explabel=f"Time Dependent NN trained from weighted denoising\nepoch{epochs} batch {batch_size}\n{model_cfg_str}")
                    figh.axes[1].set_xlim([-2.8, 2.8])
                    figh.axes[1].set_ylim([-2.8, 2.8])
                    saveallforms(figdir, f"{dataset_str}_NN_contour_train_{cfg_fn_str}_batch{batch_size}_ep{epochs:04d}_sde")
                    figh.show()
                    
                    plt.figure(figsize=(7, 7))
                    plt.scatter(x_traj_denoise[:, 0, -1], x_traj_denoise[:, 1, -1], alpha=0.1, lw=0.1, color="k", label="score net gen samples")
                    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=200, alpha=0.9, label="train", marker="o")
                    plt.scatter(Xtest[:, 0], Xtest[:, 1], s=200, alpha=0.9, label="test", marker="o")
                    plt.axis("image")
                    plt.xlim(-2.8, 2.8)
                    plt.ylim(-2.8, 2.8)
                    plt.title(f"NN Generated Samples\nepoch{epochs} batch {batch_size}\n{model_cfg_str}")
                    plt.legend()
                    plt.tight_layout()
                    saveallforms(figdir, f"{dataset_str}_NN_samples_train_{cfg_fn_str}_batch{batch_size}_ep{epochs:04d}_sde")
                    plt.show()
                except Exception as e:
                    print(e)
                    print("Error in plotting")
            
            score_model_td, loss_traj = train_score_td(Xtrain, score_model_td=score_model_td,
                            sigma=sigma_max, lr=lr, nepochs=max_epoch, batch_size=batch_size, device="cuda",
                            callback_func=training_callback, callback_epochs=epochs_list)
            stats = {"loss_init": loss_traj[0], "loss_end" : loss_traj[-1], 
                    "loss_end_10": np.mean(loss_traj[-10:]), "loss_end_100": np.mean(loss_traj[-100:]),
                    "mlp_detail": model_cfg_str, 
                    "depth": mlp_depth, "width": mlp_width, "time_embed_dim": time_embed_dim,
                    "batch_size": batch_size, "epochs": max_epoch, "lr": lr, 
                    "dataset": dataset_str, "sigma": sigma_max, }
            stats_col.append(stats)
            pkl.dump((stats, loss_traj), 
                        open(join(figdir, f"{dataset_str}_NN_train_{cfg_fn_str}_batch{batch_size}_ep{max_epoch:04d}.pkl"), "wb"))
            print(stats)
            plt.close("all")

import pandas as pd
stats_df = pd.DataFrame(stats_col)
stats_df.to_csv(join(figdir, f"{dataset_str}_NN_train_stats.csv"), index=False)

