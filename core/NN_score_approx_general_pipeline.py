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
def diffuse_gmm(gmm, t, sigma):
  lambda_t = marginal_prob_std_np(t, sigma)**2 # variance
  noise_cov = np.eye(gmm.dim) * lambda_t
  covs_dif = [cov + noise_cov for cov in gmm.covs]
  return GaussianMixture(gmm.mus, covs_dif, gmm.weights)


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
#%% Score Network architectures
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

  def __init__(self, sigma, ndim=2, nhidden=50, time_embed_dim=10,
               act_fun=nn.Tanh):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    self.net = nn.Sequential(nn.Linear(time_embed_dim + ndim, nhidden),
               act_fun(),
               nn.Linear(nhidden, nhidden),
               act_fun(),
               nn.Linear(nhidden, nhidden),
               act_fun(),
               nn.Linear(nhidden, nhidden),
               act_fun(),
               nn.Linear(nhidden, ndim))
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

  def __init__(self, sigma, ndim=2, nhidden=50, time_embed_dim=10,
               act_fun=nn.Tanh):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    self.net = nn.Sequential(nn.Linear(time_embed_dim + ndim, nhidden),
               act_fun(),
               nn.Linear(nhidden, nhidden),
               act_fun(),
               nn.Linear(nhidden, nhidden),
               act_fun(),
               nn.Linear(nhidden, nhidden),
               act_fun(),
               nn.Linear(nhidden, ndim))
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
#%%


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
from sklearn.model_selection import train_test_split
spiral_X = generate_spiral_samples_torch(1000)
Xtrain, Xtest = train_test_split(spiral_X, test_size=0.2, random_state=42)
#%%
plt.figure(figsize=(7, 7))
plt.scatter(Xtrain[:,0], Xtrain[:,1], s=1, alpha=0.5, label="train")
plt.scatter(Xtest[:,0], Xtest[:,1], s=1, alpha=0.5, label="test")
plt.axis("image")
plt.title("Spiral data train/test split")
plt.show()
#%%
spiral_X = generate_spiral_samples_torch(1000)
# train test split
Xtrain, Xtest = train_test_split(spiral_X, test_size=0.2, random_state=42)
score_model_td = ScoreModel_Time(sigma=10, ndim=2, act_fun=nn.Tanh, nhidden=128, time_embed_dim=32)
score_model_td = train_score_td(Xtrain, score_model_td=score_model_td,
                                sigma=10, lr=0.005, nepochs=2000, batch_size=2048)
# x_traj_denoise = reverse_diffusion_time_dep(score_model_td, sampN=1000, sigma=10, nsteps=1000, ndim=2, exact=False)
#%%
x_traj_denoise = reverse_diffusion_time_dep(score_model_td, sampN=2000, sigma=10, nsteps=1000, ndim=2, exact=False)
figh = visualize_diffusion_distr(x_traj_denoise, explabel="Time Dependent NN trained from weighted denoising")
figh.show()

plt.figure(figsize=(7, 7))
plt.scatter(x_traj_denoise[:, 0, -1], x_traj_denoise[:, 1, -1], alpha=0.2, lw=0.1, color="k")
plt.scatter(Xtrain[:,0], Xtrain[:,1], s=1, alpha=0.5, label="train")
plt.scatter(Xtest[:,0], Xtest[:,1], s=1, alpha=0.5, label="test")
plt.axis("image")
plt.show()


#%%
ring_X = generate_ring_samples_torch(6)
# train test split
# Xtrain, Xtest = train_test_split(ring_X, test_size=0.0001, random_state=42)
for epochs in [250, 500, 750, 1000, 2000,]:
    Xtrain, Xtest = ring_X, torch.empty(0, 2)#train_test_split(ring_X, test_size=0.0001, random_state=42)
    torch.manual_seed(42)
    score_model_td = ScoreModel_Time(sigma=10, ndim=2, act_fun=nn.Tanh, nhidden=128, time_embed_dim=32)
    score_model_td = train_score_td(Xtrain, score_model_td=score_model_td,
                    sigma=10, lr=0.005, nepochs=epochs, batch_size=1024)
    # x_traj_denoise = reverse_diffusion_time_dep(score_model_td, sampN=1000, sigma=10, nsteps=1000, ndim=2, exact=False)
    #%%
    x_traj_denoise = reverse_diffusion_time_dep(score_model_td, sampN=3000, sigma=10, nsteps=1000, ndim=2, exact=False)
    figh = visualize_diffusion_distr(x_traj_denoise, explabel="Time Dependent NN trained from weighted denoising")
    figh.show()
    #%%
    plt.figure(figsize=(7, 7))
    plt.scatter(x_traj_denoise[:, 0, -1], x_traj_denoise[:, 1, -1], alpha=0.2, lw=0.1, color="k")
    plt.scatter(Xtrain[:,0], Xtrain[:,1], s=160, alpha=0.9, label="train", marker="o")
    plt.scatter(Xtest[:,0], Xtest[:,1], s=160, alpha=0.9, label="test", marker="o")
    plt.axis("image")
    plt.title("epoch{}".format(epochs))
    plt.show()


#%%
plt.plot(x_traj_denoise[:, 0, :].T, x_traj_denoise[:, 1, :].T, alpha=0.2, lw=0.1, color="k")
plt.axis("image")
plt.show()
#%%
npnts = 500
mus = np.random.randn(npnts, 2)
delta_eps = 1E-5
covs = [np.eye(2) * delta_eps for _ in range(npnts)]
weights = [1/npnts for _ in range(npnts)]
delta_gmm = GaussianMixture([*mus], covs, weights)
#%%

figdir = r"E:\OneDrive - Harvard University\DiffusionMemorization\Figures\GMM2dDelta_figure"
#%%
# X_train_samp, _, _ = gmm3.sample(N=5000)
# X_train_samp = torch.tensor(X_train_samp).float()

gmm_samps, _, _ = delta_gmm.sample(5000)
# scorevecs = delta_gmm.score(gmm_samps)
X_train_tsr = torch.tensor(gmm_samps).float()
# y_train_tsr3 = torch.tensor(scorevecs3).float()
# X_train_tsr3, y_train_tsr3, _, _ = sample_X_and_score(gmm3, trainN=20000)
#%% Trainging the score model
sigma = 25
score_model_td3 = ScoreModel_Time(sigma=sigma, )
marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)
optim = Adam(score_model_td3.parameters(), lr=0.005)
pbar = trange(750)
for ep in pbar:
  loss = denoise_loss_fn(score_model_td3, X_train_tsr, marginal_prob_std_f, 0.05)
  optim.zero_grad()
  loss.backward()
  optim.step()
  pbar.set_description(f"step {ep} loss {loss.item():.3f}")
  if ep == 0:
    print(f"step {ep} loss {loss.item():.3f}")

#%%
plt.figure(figsize=(7, 7))
plt.scatter(X_train_tsr[:,0], X_train_tsr[:,1], s=1, alpha=0.5)
plt.axis("image")
plt.title("Original")
plt.show()
#%%
x_traj_denoise = reverse_diffusion_time_dep(score_model_td3, sampN=1000, sigma=25, nsteps=200, ndim=2, exact=False)
figh = visualize_diffusion_distr(x_traj_denoise, explabel="Time Dependent NN trained from weighted denoising")
figh.show()
#%%
x_traj_denoise_NN = reverse_diffusion_time_dep(score_model_td3, sampN=1000, sigma=25, nsteps=500, ndim=2, exact=False)
x_traj_denoise_analy = reverse_diffusion_time_dep(delta_gmm, sampN=1000, sigma=25, nsteps=500, ndim=2, exact=True)
#%%
plt.figure(figsize=(7, 7))
plt.plot(x_traj_denoise_analy[:, 0, :].T, x_traj_denoise_analy[:, 1, :].T, alpha=0.2, lw=0.1, color="k")
plt.axis("image")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
#%%
plt.figure(figsize=(7, 7))
plt.plot(X_train_tsr[:, 0], X_train_tsr[:, 1], "x", alpha=0.2, lw=0.1, color="k", label="training data")
plt.plot(x_traj_denoise_analy[:, 0, -1], x_traj_denoise_analy[:, 1, -1], "o", alpha=0.2, lw=0.1, color="r", label="DDPM sample with analytical score")
plt.plot(x_traj_denoise_NN[:, 0, -1], x_traj_denoise[:, 1, -1], "o", alpha=0.2, lw=0.1, color="b", label="DDPM sample with NN score")
plt.legend()
plt.axis("image")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title("Diffusion sample with analytical score, NN score\n vs training data", fontsize=16)
saveallforms(figdir,"diffusion_sample_NN_vs_analy_vs_dataset")
plt.show()
#%%
# split the plot above into 2 panels
plt.figure(figsize=(12, 6.5))
plt.subplot(121)
plt.plot(X_train_tsr[:, 0], X_train_tsr[:, 1], "x", alpha=0.2, lw=0.1, color="k", label="training data")
plt.plot(x_traj_denoise_analy[:, 0, -1], x_traj_denoise_analy[:, 1, -1], "o", alpha=0.2, lw=0.1, color="r", label="DDPM sample with analytical score")
plt.legend(fontsize=14)
plt.axis("image")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title("Sample with analytical score vs training data", fontsize=14)
plt.subplot(122)
plt.plot(X_train_tsr[:, 0], X_train_tsr[:, 1], "x", alpha=0.2, lw=0.1, color="k", label="training data")
plt.plot(x_traj_denoise_NN[:, 0, -1], x_traj_denoise[:, 1, -1], "o", alpha=0.2, lw=0.1, color="b", label="DDPM sample with NN score")
plt.legend(fontsize=14)
plt.axis("image")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title("Sample with NN approx score vs training data", fontsize=14)
plt.tight_layout()
saveallforms(figdir,"diffusion_sample_NN_vs_analy_vs_dataset_split")
plt.show()
#%%
# plot the nearest neighbor distance between the training data and the diffusion sample
from scipy.spatial.distance import cdist
dist_NN_analy = cdist(X_train_tsr, x_traj_denoise_analy[:, :, -1]).min(axis=0)
dist_NN_NN = cdist(X_train_tsr, x_traj_denoise_NN[:, :, -1]).min(axis=0)
#%%
plt.figure(figsize=(5, 4))
plt.hist(dist_NN_analy, bins=np.linspace(0, 1, 101),
         alpha=0.35, label="analytical score", color="r")
plt.hist(dist_NN_NN, bins=np.linspace(0, 1, 101),
         alpha=0.35, label="NN score", color="b")
plt.legend()
plt.xlabel("L2 Distance to nearest training data")
plt.ylabel("Count")
plt.title("Distance of diffusion sample to nearest training data")
saveallforms(figdir, "nearest_data_dist_NN_vs_analy_hist")
plt.show()

#%%
plt.figure(figsize=(7, 7))
sns.kdeplot(x=x_traj_denoise[:,0,-1], y=x_traj_denoise[:,1,-1], shade_lowest=False, cmap="Blues", label="Denoised")
sns.kdeplot(x=X_train_tsr[:,0], y=X_train_tsr[:,1], shade_lowest=False, cmap="Reds", label="Original")
plt.axis("image")
plt.title("Time Dependent NN trained from weighted denoising")
plt.legend()
plt.show()

#%% Characterize error across time
"""Compute the MSE loss of score over time"""
import pandas as pd
score_model_td3.eval()
score_model_td3.cpu()
df_col = []
delta_gmm_t1 = diffuse_gmm(delta_gmm, 1, sigma)
for t in np.arange(0.025, 1.01, 0.025):
    time_th = torch.tensor(t).float()
    delta_gmm_t = diffuse_gmm(delta_gmm, time_th.item(), sigma)
    gmm_samps3_t, _, _ = delta_gmm_t.sample(10000)
    scorevecs3_t = delta_gmm_t.score(gmm_samps3_t)
    X_test_tsr3 = torch.tensor(gmm_samps3_t).float()
    scorevec_nn = score_model_td3(X_test_tsr3, time_th.repeat(X_test_tsr3.shape[0]))
    mse_vec = ((scorevec_nn.detach().numpy() - scorevecs3_t)**2).sum(axis=1)

    gmm_samps3_t0, _, _ = delta_gmm.sample(10000)
    scorevecs3_t0 = delta_gmm.score(gmm_samps3_t0)
    X_test_tsr3_t0 = torch.tensor(gmm_samps3_t0).float()
    scorevec_nn_t0 = score_model_td3(X_test_tsr3_t0, time_th.repeat(X_test_tsr3_t0.shape[0]))
    mse_vec_t0 = ((scorevec_nn_t0.detach().numpy() - scorevecs3_t0)**2).sum(axis=1)

    gmm_samps3_t1, _, _ = delta_gmm_t1.sample(10000)
    scorevecs3_t1 = delta_gmm_t1.score(gmm_samps3_t1)
    X_test_tsr3_t1 = torch.tensor(gmm_samps3_t1).float()
    scorevec_nn_t1 = score_model_td3(X_test_tsr3_t1, time_th.repeat(X_test_tsr3_t1.shape[0]))
    mse_vec_t1 = ((scorevec_nn_t1.detach().numpy() - scorevecs3_t1)**2).sum(axis=1)
    df_col.append({"t": t, "mse": mse_vec.mean(), "mse_std": mse_vec.std(),
                     "mse_t0": mse_vec_t0.mean(), "mse_std_t0": mse_vec_t0.std(),
                   "mse_t1": mse_vec_t1.mean(), "mse_std_t1": mse_vec_t1.std()})
    print(f"t={t:.3f}, mse={mse_vec.mean():.5f}, mse_std={mse_vec.std():.5f} , mse_t0={mse_vec_t0.mean():.5f}, mse_std_t0={mse_vec_t0.std():.5f}, mse_t1={mse_vec_t1.mean():.5f}, mse_std_t1={mse_vec_t1.std():.5f}")

df_td = pd.DataFrame(df_col)
#%%
df_td.to_csv(join(figdir, "mse_score_field_time_dep.csv"))
#%%
plt.figure(figsize=(6, 5))
plt.plot(df_td["t"], df_td["mse"], )
plt.xlabel("Diffusion time")
plt.ylabel("MSE")
plt.title("MSE of NN estimated score field and p_t(x) score field")
saveallforms(figdir, "mse_score_field_expectation_p_xt", )
plt.show()

#%%
plt.figure(figsize=(6, 5))
plt.plot(df_td["t"], df_td["mse_t0"], )#label="Time Dependent NN trained from weighted denoising")
plt.xlabel("Diffusion time")
plt.ylabel("MSE")
plt.title("MSE of NN estimated score field and p_0(x) score field")
saveallforms(figdir, "mse_score_field_expectation_p_x0", )
plt.show()

#%%
plt.figure(figsize=(6, 5))
plt.plot(df_td["t"], df_td["mse_t1"], )#label="Time Dependent NN trained from weighted denoising")
plt.xlabel("Diffusion time")
plt.ylabel("MSE")
plt.title("MSE of NN estimated score field and p_1(x) score field")
saveallforms(figdir, "mse_score_field_expectation_p_x1", )
plt.show()




#%%
"""
visualize the trained score model's score field vs the original score field and make animation out of it. 
"""
score_model_td3.eval()
score_model_td3.cpu()
for t in np.arange(0, 1.01, 0.025):
  time_th = torch.tensor(t).float()
  xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
  xvec = xx.reshape(-1)
  yvec = yy.reshape(-1)
  xy = np.stack((xvec, yvec), axis=1)
  xy = torch.tensor(xy).float()
  scorevec_nn = score_model_td3(xy, time_th.repeat(xy.shape[0]))\
    .detach().numpy()
  delta_gmm_t = diffuse_gmm(delta_gmm, time_th.item(), sigma)
  scorevec_analy = delta_gmm_t.score(xy.numpy())
  density_analy = delta_gmm_t.pdf(xy.numpy()).reshape(100, 100)
  scoremap_nn = scorevec_nn[:,:].reshape(100,100,-1)
  scoremap_analy = scorevec_analy[:,:].reshape(100,100,-1)
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
  sns.kdeplot(x=X_train_tsr[:,0], y=X_train_tsr[:,1],
              cmap="Reds", label="data dist.")
  plt.contour(xx, yy, density_analy, levels=10, colors="black", alpha=0.5)
  plt.title(f"Score Field Comparison at t={time_th.item():.3f}")
  plt.axis("image")
  plt.tight_layout()
  plt.legend()
  saveallforms(figdir, f"scorefield_comparison_t{time_th.item():.3f}_dense_focus")
  plt.show()


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
gifpath = join(figdir, "scorefield_comparison_dense.gif")
images = []
for t in np.arange(0, 1.01, 0.025):
  images.append(imageio.imread(join(figdir, f"scorefield_comparison_t{t:.3f}_dense.png")))
imageio.mimsave(gifpath, images)
imageio.mimsave(gifpath.replace(".gif",".mp4"), images)
#%%
gifpath = join(figdir, "scorefield_comparison_dense_focus.gif")
images = []
for t in np.arange(0, 1.01, 0.025):
  images.append(imageio.imread(join(figdir, f"scorefield_comparison_t{t:.3f}_dense_focus.png")))
imageio.mimsave(gifpath, images)
imageio.mimsave(gifpath.replace(".gif",".mp4"), images)