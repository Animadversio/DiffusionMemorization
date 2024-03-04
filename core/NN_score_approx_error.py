#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn.modules.loss import MSELoss
from tqdm.autonotebook import trange, tqdm #.notebook
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.utils.plot_utils import saveallforms
#%%
def marginal_prob_std(t, sigma):
  """Note that this std -> 0, when t->0
  So it's not numerically stable to sample t=0 in the dataset
  Note an earlier version missed the sqrt...
  """
  return torch.sqrt( (sigma**(2*t) - 1) / 2 / torch.log(torch.tensor(sigma)) ) # sqrt fixed Jun.19


def marginal_prob_std_np(t, sigma):
  return np.sqrt( (sigma**(2*t) - 1) / 2 / np.log(sigma) )



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
#%%
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
    t_proj = t[:, None] * self.W[None, :] * 2 * math.pi
    return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class ScoreModel_Time(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, sigma, ):
    super().__init__()
    self.embed = GaussianFourierProjection(10, scale=1)
    self.net = nn.Sequential(nn.Linear(12, 50),
               nn.Tanh(),
               nn.Linear(50,50),
               nn.Tanh(),
               nn.Linear(50,2))
    self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

  def forward(self, x, t):
    t_embed = self.embed(t)
    pred = self.net(torch.cat((x, t_embed), dim=1))
    # this additional steps provides an inductive bias.
    # the neural network output on the same scale,
    pred = pred / self.marginal_prob_std_f(t)[:, None,]
    return pred


from core.gaussian_mixture_lib import GaussianMixture, GaussianMixture_torch
def diffuse_gmm(gmm, t, sigma):
  lambda_t = marginal_prob_std_np(t, sigma)**2 # variance
  noise_cov = np.eye(gmm.dim) * lambda_t
  covs_dif = [cov + noise_cov for cov in gmm.covs]
  return GaussianMixture(gmm.mus, covs_dif, gmm.weights)


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

mu1 = np.array([0,1.0])
Cov1 = np.array([[1.0,0.2],
          [0.2,2.0]])
mu2 = np.array([2.0,-3.5])
Cov2 = np.array([[2.0,-1.0],
          [-1.0,2.0]])
mu3 = np.array([-6.5,-5.0])
Cov3 = np.array([[4.0,0.5],
          [0.5,2.0]])
mu4 = np.array([-5.5,5.0])
Cov4 = np.array([[1.0,1.0],
          [1.0,3.0]])
gmm3 = GaussianMixture([mu1,mu2,mu3,mu4],
                      [Cov1,Cov2,Cov3,Cov4],
                      [1.0,1.0,1.0,1.0])
#%%
# X_train_samp, _, _ = gmm3.sample(N=5000)
# X_train_samp = torch.tensor(X_train_samp).float()

gmm_samps3, _, _ = gmm3.sample(20000)
scorevecs3 = gmm3.score(gmm_samps3)
X_train_tsr3 = torch.tensor(gmm_samps3).float()
y_train_tsr3 = torch.tensor(scorevecs3).float()
# X_train_tsr3, y_train_tsr3, _, _ = sample_X_and_score(gmm3, trainN=20000)
#%% Trainging the score model
sigma = 25
score_model_td3 = ScoreModel_Time(sigma=sigma, )
marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)
optim = Adam(score_model_td3.parameters(), lr=0.005)
pbar = trange(750)
for ep in pbar:
  loss = denoise_loss_fn(score_model_td3, X_train_tsr3, marginal_prob_std_f, 0.05)
  optim.zero_grad()
  loss.backward()
  optim.step()
  pbar.set_description(f"step {ep} loss {loss.item():.3f}")
  if ep == 0:
    print(f"step {ep} loss {loss.item():.3f}")


#%%
x_traj_denoise3 = reverse_diffusion_time_dep(score_model_td3, sampN=2000, sigma=25, nsteps=200, ndim=2, exact=False)
figh = visualize_diffusion_distr(x_traj_denoise3, explabel="Time Dependent NN trained from weighted denoising")
#%%
plt.figure(figsize=(7, 7))
sns.kdeplot(x=x_traj_denoise3[:,0,-1], y=x_traj_denoise3[:,1,-1], shade_lowest=False, cmap="Blues", label="Denoised")
sns.kdeplot(x=X_train_tsr3[:,0], y=X_train_tsr3[:,1], shade_lowest=False, cmap="Reds", label="Original")
plt.axis("image")
plt.title("Time Dependent NN trained from weighted denoising")
plt.legend()
plt.show()

#%% Characterize error across time
import pandas as pd
figdir = r"E:\OneDrive - Harvard University\DiffusionMemorization\Figures\GMM2d_figures"
#%%
score_model_td3.eval()
score_model_td3.cpu()
df_col = []
gmm3_t1 = diffuse_gmm(gmm3, 1, sigma)
for t in np.arange(0.025, 1.01, 0.025):
    time_th = torch.tensor(t).float()
    gmm3_t = diffuse_gmm(gmm3, time_th.item(), sigma)
    gmm_samps3_t, _, _ = gmm3_t.sample(10000)
    scorevecs3_t = gmm3_t.score(gmm_samps3_t)
    X_test_tsr3 = torch.tensor(gmm_samps3_t).float()
    scorevec_nn = score_model_td3(X_test_tsr3, time_th.repeat(X_test_tsr3.shape[0]))
    mse_vec = ((scorevec_nn.detach().numpy() - scorevecs3_t)**2).sum(axis=1)

    gmm_samps3_t0, _, _ = gmm3.sample(10000)
    scorevecs3_t0 = gmm3.score(gmm_samps3_t0)
    X_test_tsr3_t0 = torch.tensor(gmm_samps3_t0).float()
    scorevec_nn_t0 = score_model_td3(X_test_tsr3_t0, time_th.repeat(X_test_tsr3_t0.shape[0]))
    mse_vec_t0 = ((scorevec_nn_t0.detach().numpy() - scorevecs3_t0)**2).sum(axis=1)

    gmm_samps3_t1, _, _ = gmm3_t1.sample(10000)
    scorevecs3_t1 = gmm3_t.score(gmm_samps3_t1)
    X_test_tsr3_t1 = torch.tensor(gmm_samps3_t1).float()
    scorevec_nn_t1 = score_model_td3(X_test_tsr3_t1, time_th.repeat(X_test_tsr3_t1.shape[0]))
    mse_vec_t1 = ((scorevec_nn_t1.detach().numpy() - scorevecs3_t1)**2).sum(axis=1)
    df_col.append({"t": t, "mse": mse_vec.mean(), "mse_std": mse_vec.std(),
                     "mse_t0": mse_vec_t0.mean(), "mse_std_t0": mse_vec_t0.std(),
                   "mse_t1": mse_vec_t1.mean(), "mse_std_t1": mse_vec_t1.std()})
    print(f"t={t:.3f}, mse={mse_vec.mean():.5f}, mse_std={mse_vec.std():.5f} , mse_t0={mse_vec_t0.mean():.5f}, mse_std_t0={mse_vec_t0.std():.5f}, mse_t1={mse_vec_t1.mean():.5f}, mse_std_t1={mse_vec_t1.std():.5f}")

df_td3 = pd.DataFrame(df_col)
#%%
df_td3.to_csv(join(figdir, "mse_score_field_time_dep.csv"))
#%%
plt.figure(figsize=(6, 5))
plt.plot(df_td3["t"], df_td3["mse"], label="Time Dependent NN trained from weighted denoising")
plt.xlabel("Diffusion time")
plt.ylabel("MSE")
plt.title("MSE of score field Estimated with p_t(x) across time")
plt.legend()
saveallforms(figdir, "mse_score_field_expectation_p_xt", )
plt.show()
#%%
plt.figure(figsize=(6, 5))
plt.plot(df_td3["t"], df_td3["mse_t0"], label="Time Dependent NN trained from weighted denoising")
plt.xlabel("Diffusion time")
plt.ylabel("MSE")
plt.title("MSE of score field Estimated with p_t(x) across time")
plt.legend()
saveallforms(figdir, "mse_score_field_expectation_p_x0", )
plt.show()
#%%
plt.figure(figsize=(6, 5))
plt.plot(df_td3["t"], df_td3["mse_t1"], label="Time Dependent NN trained from weighted denoising")
plt.xlabel("Diffusion time")
plt.ylabel("MSE")
plt.title("MSE of score field Estimated with p_t(x) across time")
plt.legend()
saveallforms(figdir, "mse_score_field_expectation_p_x1", )
plt.show()




#%%
# visualize the trained score model's score field vs the original score field
score_model_td3.eval()
score_model_td3.cpu()
for t in np.arange(0, 1.01, 0.025):
  time_th = torch.tensor(t).float()
  xx, yy = np.meshgrid(np.linspace(-15, 15, 100), np.linspace(-15, 15, 100))
  xvec = xx.reshape(-1)
  yvec = yy.reshape(-1)
  xy = np.stack((xvec, yvec), axis=1)
  xy = torch.tensor(xy).float()
  scorevec_nn = score_model_td3(xy, time_th.repeat(xy.shape[0]))\
    .detach().numpy()
  gmm3_t = diffuse_gmm(gmm3, time_th.item(), sigma)
  scorevec_analy = gmm3_t.score(xy.numpy())
  density_analy = gmm3_t.pdf(xy.numpy()).reshape(100,100)
  scoremap_nn = scorevec_nn[:,:].reshape(100,100,-1)
  scoremap_analy = scorevec_analy[:,:].reshape(100,100,-1)
  #%%
  # subslc = slice(None, None, 6)
  subslc = slice(None, None, 4)
  plt.figure(figsize=(7, 7))
  plt.quiver(xx[subslc,subslc], yy[subslc,subslc],
             scoremap_nn[subslc,subslc,0],
             scoremap_nn[subslc,subslc,1],
             color="blue", alpha=0.5, label="NN approx")
  plt.quiver(xx[subslc,subslc], yy[subslc,subslc],
             scoremap_analy[subslc,subslc,0],
             scoremap_analy[subslc,subslc,1],
             color="red", alpha=0.5, label="Analytical")
  sns.kdeplot(x=X_train_tsr3[:,0], y=X_train_tsr3[:,1],
              cmap="Reds", label="data dist.")
  plt.contour(xx, yy, density_analy, levels=10, colors="black", alpha=0.5)
  plt.title(f"Score Field Comparison at t={time_th.item():.3f}")
  plt.axis("image")
  plt.tight_layout()
  plt.legend()
  saveallforms(figdir, f"scorefield_comparison_t{time_th.item():.3f}_dense")
  plt.show()
  # break

#%%
# make the gif out of the figurs
import imageio
import os
import glob
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
#%%
# make the gif out of the figurs
import imageio
from os.path import join
gifpath = join(figdir, "scorefield_comparison.gif")
gifpath = join(figdir, "scorefield_comparison_dense.gif")
images = []
for t in np.arange(0, 1.01, 0.025):
  # images.append(imageio.imread(join(figdir, f"scorefield_comparison_t{t:.3f}.png")))
  images.append(imageio.imread(join(figdir, f"scorefield_comparison_t{t:.3f}_dense.png")))
imageio.mimsave(gifpath, images)
imageio.mimsave(gifpath.replace(".gif",".mp4"), images)
