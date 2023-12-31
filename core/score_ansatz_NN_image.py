
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
from os.path import join
from core.utils.plot_utils import saveallforms
from core.gaussian_mixture_lib import GaussianMixture, GaussianMixture_torch


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
                   clipnorm=None,):
    ndim = X_train_tsr.shape[1]
    if score_model_td is None:
        score_model_td = ScoreModel_Time_edm(sigma=sigma, ndim=ndim)
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
        if clipnorm is not None:
            torch.nn.utils.clip_grad_norm_(score_model_td.parameters(),
                                           max_norm=clipnorm)
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

#%%
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
    logprobs = -0.5 * (logdetSigmas + MHdists)  # [N batch, N comp]
    if weights is not None:
        logprobs += torch.log(weights)  # - 0.5 * ndim * torch.log(2 * torch.pi)  # [N batch, N comp]
    participance = F.softmax(logprobs, dim=-1)  # [N batch, N comp]
    compo_score_vecs = torch.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas),
                                    Us)  # [N batch, N comp, N dim]
    score_vecs = torch.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
    return score_vecs


def gaussian_mixture_lowrank_score_batch_sigma_torch(x,
                 mus, Us, Lambdas, sigma, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model in PyTorch
    :param x: [N batch,N dim]
    :param mus: [N comp, N dim]
    :param Us: [N comp, N dim, N rank]
    :param Lambdas: [N comp, N rank]
    :param sigma: [N batch,] or []
    :param weights: [N comp,] or None
    :return:
    """
    if Lambdas.ndim == 2:
        Lambdas = Lambdas[None, :, :]
    ndim = x.shape[-1]
    nrank = Us.shape[-1]
    logdetSigmas = torch.sum(torch.log(Lambdas + sigma[:, None, None] ** 2), dim=-1)  # [N batch, N comp,]
    logdetSigmas += (ndim - nrank) * 2 * torch.log(sigma)[:, None]  # [N batch, N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    residual_sqnorm = torch.sum(residuals ** 2, dim=-1)  # [N batch, N comp]
    Lambda_tilde = Lambdas / (Lambdas + sigma[:, None, None] ** 2)  # [N batch, N comp, N rank]
    rot_residuals = torch.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists_lowrk = torch.sum(rot_residuals ** 2 * Lambda_tilde, dim=-1)  # [N batch, N comp]
    logprobs = -0.5 * (logdetSigmas +
                       (residual_sqnorm - MHdists_lowrk) / sigma[:, None] ** 2)  # [N batch, N comp]
    if weights is not None:
        logprobs += torch.log(weights)
    participance = F.softmax(logprobs, dim=-1)  # [N batch, N comp]
    compo_score_vecs = - residuals + torch.einsum("BCD,CED->BCE",
                                    (rot_residuals * Lambda_tilde),
                                    Us)  # [N batch, N comp, N dim]
    score_vecs = torch.einsum("BC,BCE->BE", participance, compo_score_vecs) / (sigma[:, None] ** 2)  # [N batch, N dim]
    return score_vecs


class GMM_ansatz_net(nn.Module):

    def __init__(self, ndim, n_components, sigma=5.0):
        super().__init__()
        self.ndim = ndim
        self.n_components = n_components
        # normalize the weights
        mus = torch.randn(n_components, ndim)
        Us = torch.randn(n_components, ndim, ndim)
        mus = mus / torch.norm(mus, dim=-1, keepdim=True)
        Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
        # TODO: orthonormalize Us
        self.mus = nn.Parameter(mus)
        self.Us = nn.Parameter(Us)
        self.logLambdas = nn.Parameter(torch.randn(n_components, ndim))
        self.logweights = nn.Parameter(torch.log(torch.ones(n_components) / n_components))
        self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

    def forward(self, x, t):
        """
        x: (batch, ndim)
        sigma: (batch, )
        """
        sigma = self.marginal_prob_std_f(t, )
        return gaussian_mixture_score_batch_sigma_torch(x, self.mus, self.Us,
               self.logLambdas.exp()[None, :, :] + sigma[:, None, None] ** 2, self.logweights.exp())


class GMM_ansatz_net_lowrank(nn.Module):
    def __init__(self, ndim, n_components, n_rank, sigma=5.0):
        super().__init__()
        self.ndim = ndim
        self.n_components = n_components
        # normalize the weights
        mus = torch.randn(n_components, ndim)
        Us = torch.randn(n_components, ndim, n_rank)
        mus = mus / torch.norm(mus, dim=-1, keepdim=True)
        Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
        # TODO: orthonormalize Us
        self.mus = nn.Parameter(mus)
        self.Us = nn.Parameter(Us)
        self.logLambdas = nn.Parameter(torch.randn(n_components, n_rank))
        self.logweights = nn.Parameter(torch.log(torch.ones(n_components) / n_components))
        self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

    def forward(self, x, t):
        """
        x: (batch, ndim)
        sigma: (batch, )
        """
        sigma = self.marginal_prob_std_f(t, )
        return gaussian_mixture_lowrank_score_batch_sigma_torch(x, self.mus, self.Us,
               self.logLambdas.exp(), sigma[:], self.logweights.exp())


class Gauss_ansatz_net(nn.Module):
    def __init__(self, ndim, n_rank=None, sigma=5.0):
        super().__init__()
        self.ndim = ndim
        # normalize the weights
        mus = torch.randn(ndim)
        if n_rank is None:
            n_rank = ndim
        Us = torch.randn(ndim, n_rank)
        mus = mus / torch.norm(mus, dim=-1, keepdim=True)
        Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
        # TODO: orthonormalize Us
        self.mus = nn.Parameter(mus)
        self.Us = nn.Parameter(Us)
        self.logLambdas = nn.Parameter(torch.randn(n_rank))
        self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

    def forward(self, x, t):
        """
        x: (batch, ndim)
        sigma: (batch, )
        """
        sigma = self.marginal_prob_std_f(t, )
        # ndim = x.shape[-1]
        # nrank = Us.shape[-1]
        residuals = (x[:, :] - self.mus[None, :])  # [N batch, N dim]
        # residual_sqnorm = torch.sum(residuals ** 2, dim=-1)  # [N batch, ]
        Lambdas = self.logLambdas.exp()[None, :]
        Lambda_tilde = Lambdas / (Lambdas + sigma[:, None] ** 2)  # [N batch, N rank]
        rot_residuals = torch.einsum("BD,DE->BE", residuals, self.Us)  # [N batch, N comp, N dim]
        # MHdists_lowrk = torch.sum(rot_residuals ** 2 * Lambda_tilde, dim=-1)  # [N batch, N comp]
        compo_score_vecs = - residuals + torch.einsum("BE,DE->BD",
                              (rot_residuals * Lambda_tilde), self.Us)  # [N batch, N comp, N dim]
        score_vecs = compo_score_vecs / (sigma[:, None] ** 2)  # [N batch, N dim]
        return score_vecs

#%%
def test_lowrank_score_correct(n_components = 5, npnts = 40):
    # test low rank version
    ndim = 2
    n_rank = 1
    xs = torch.randn(npnts, ndim)
    mus = torch.randn(n_components, ndim)
    Us = torch.randn(n_components, ndim, n_rank)
    mus = mus / torch.norm(mus, dim=-1, keepdim=True)
    Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
    Us_ortho = Us[:, [-1, -2], :] * torch.tensor([1, -1])[None, :, None]
    # test ortho
    assert torch.allclose(torch.einsum("CDr,CDr->Cr", Us, Us_ortho), torch.zeros(n_components, n_rank))
    Lambdas_lowrank = torch.randn(n_components, n_rank).exp()
    # sigma = torch.tensor([1.0])
    sigma = torch.rand(npnts)
    score_lowrank = gaussian_mixture_lowrank_score_batch_sigma_torch(xs, mus, Us, Lambdas_lowrank, sigma)
    # built full rank basis
    Us_full = torch.cat((Us, Us_ortho), dim=-1)
    # build full rank noise covariance
    Lambdas_full = torch.cat((Lambdas_lowrank[None, :, :] + sigma[:, None, None] ** 2,
                              (sigma[:, None, None] ** 2).repeat(1, n_components, ndim - n_rank)), dim=-1)
    score_fullrank = gaussian_mixture_score_batch_sigma_torch(xs, mus, Us_full, Lambdas_full,)
    assert torch.allclose(score_lowrank, score_fullrank, atol=1e-4, rtol=1e-4)

test_lowrank_score_correct()
#%%
def test_lowrank_gauss_score_correct(n_components = 1, npnts = 40, ndim = 3,
    n_rank = 2):
    # test low rank version
    xs = torch.randn(npnts, ndim)
    mus = torch.randn(n_components, ndim)
    Us = torch.randn(n_components, ndim, n_rank)
    ts = torch.rand(npnts)
    sigmas = marginal_prob_std(ts, 5.0)
    mus = mus / torch.norm(mus, dim=-1, keepdim=True)
    Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
    # test ortho
    Lambdas_lowrank = torch.randn(n_components, n_rank).exp()
    score_lowrank = gaussian_mixture_lowrank_score_batch_sigma_torch(xs, mus, Us, Lambdas_lowrank, sigmas)

    net = Gauss_ansatz_net(ndim, n_rank, sigma=5.0)
    net.mus.data = mus[0]
    net.Us.data = Us[0]
    net.logLambdas.data = torch.log(Lambdas_lowrank)[0]
    # built full rank basis
    score_gauss = net(xs, ts)

    assert torch.allclose(score_lowrank, score_gauss, atol=1e-4, rtol=1e-4)

test_lowrank_gauss_score_correct()
#%%
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

dataset = MNIST(root="~/Datasets", train=True, download=False, transform=transforms.ToTensor())
Xtsr = dataset.data.float() / 255
Xtrain = Xtsr.reshape(Xtsr.shape[0], -1)
ytrain = dataset.targets
#%%
Xtrain_norm = (Xtrain - Xtrain.mean()) / Xtrain.std()
Xmean = Xtrain_norm.mean(dim=0)
covmat = torch.cov((Xtrain_norm - Xmean).T)
eigval, eigvec = torch.linalg.eigh(covmat.to(torch.float64))
eigval = eigval.flip(dims=(0,))  # sort from largest to smallest
eigvec = eigvec.flip(dims=(1,))  # sort from largest to smallest
assert torch.allclose(eigvec.T @ eigvec, torch.eye(eigvec.shape[0]).to(torch.float64))
assert torch.allclose(eigvec @ torch.diag(eigval) @ eigvec.T, covmat.to(torch.float64))
#%%
mtg = make_grid(eigvec.reshape(1, 28, 28, -1).permute(3,0,1,2)[:100], nrow=10)
plt.figure(figsize=[10,10])
plt.imshow(mtg.permute(1, 2, 0) / mtg.max())
plt.axis("off")
plt.tight_layout()
plt.show()

#%%
logLamda_init = torch.log(eigval + 1E-5).float()
U_init = eigvec.float()

#%%
ndim = Xtrain.shape[1]
sigma_max = 10
epochs = 2000
batch_size = 2048
torch.manual_seed(42)
score_model_gauss = Gauss_ansatz_net(ndim=ndim, sigma=sigma_max)
# Data PC initialization
score_model_gauss.logLambdas.data = logLamda_init[:]
score_model_gauss.Us.data = U_init[:, :]
# perturb = torch.randn(gmm_components, ndim)
# perturb = perturb / torch.norm(perturb, dim=-1, keepdim=True)
score_model_gauss.mus.data = Xmean[:]  # + perturb * 1
#%%
score_model_gauss = train_score_td(Xtrain_norm, score_model_td=score_model_gauss,
        sigma=sigma_max, lr=0.001, nepochs=epochs, batch_size=batch_size, clipnorm=1)
# loss around 400 over 800 epochs, still very bad.
# parameter is over too much 30 Million for 50 components
# Training takes 20 mins for 2000 epochs

#%%
ndim = Xtrain.shape[1]
gmm_components = 10
sigma_max = 10
epochs = 2000
batch_size = 2048
torch.manual_seed(42)
score_model_gmm = GMM_ansatz_net(ndim=ndim,
             n_components=gmm_components, sigma=sigma_max)
# Data PC initialization
score_model_gmm.logLambdas.data = logLamda_init[None, :].repeat(gmm_components, 1)
score_model_gmm.Us.data = U_init[None, :, :].repeat(gmm_components, 1, 1)
perturb = torch.randn(gmm_components, ndim)
perturb = perturb / torch.norm(perturb, dim=-1, keepdim=True)
score_model_gmm.mus.data = Xmean[None, :].repeat(gmm_components, 1) + perturb * 10
#%%
score_model_gmm = train_score_td(Xtrain_norm, score_model_td=score_model_gmm,
        sigma=sigma_max, lr=0.0005, nepochs=epochs, batch_size=batch_size, clipnorm=1)
# 10 full Gaussian GMM 0.0005 lr, 2000 epochså
# step 0 loss 283.743
# step 1999 loss 108.833: 100%|██████████| 2000/2000 [07:53<00:00,  4.23it/s]
#%%
print("number of parameters",
sum(p.numel() for p in score_model_gmm.parameters() if p.requires_grad))
# 6 million parameters
#%%
samples = reverse_diffusion_time_dep(score_model_gmm, sampN=100, sigma=sigma_max, nsteps=1000, ndim=ndim, exact=False)
#%%
mtg = make_grid(torch.from_numpy(samples[:,:,-1].reshape(-1, 1, 28, 28)), nrow=10)
plt.figure(figsize=[10,10])
plt.imshow(mtg.permute(1, 2, 0))
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
centroids = score_model_gmm.mus.detach().reshape(-1, 1, 28, 28)
mtg = make_grid(centroids, nrow=5)
plt.imshow(mtg.permute(1, 2, 0) )
plt.show()

#%%
ndim = Xtrain.shape[1]
gmm_components = 50
sigma_max = 10
epochs = 2000
batch_size = 2048
torch.manual_seed(42)
score_model_gmm = GMM_ansatz_net(ndim=ndim,
             n_components=gmm_components, sigma=sigma_max)
# Data PC initialization
score_model_gmm.logLambdas.data = logLamda_init[None, :].repeat(gmm_components, 1)
score_model_gmm.Us.data = U_init[None, :, :].repeat(gmm_components, 1, 1)
perturb = torch.randn(gmm_components, ndim)
perturb = perturb / torch.norm(perturb, dim=-1, keepdim=True)
score_model_gmm.mus.data = Xmean[None, :].repeat(gmm_components, 1) + perturb * 10
#%%
score_model_gmm = train_score_td(Xtrain_norm, score_model_td=score_model_gmm,
        sigma=sigma_max, lr=0.0005, nepochs=epochs, batch_size=batch_size, clipnorm=1)
# 50 full Gaussian GMM 0.0005 lr, 2000 epochså
# step 0 loss 268.291
# step 1999 loss 114.897: 100%|██████████| 2000/2000 [37:20<00:00,  1.12s/it]
#%%
print("number of parameters",
sum(p.numel() for p in score_model_gmm.parameters() if p.requires_grad))
# 30 million parameters
#%%
samples = reverse_diffusion_time_dep(score_model_gmm, sampN=100, sigma=sigma_max, nsteps=1000, ndim=ndim, exact=False)
#%%
mtg = make_grid(torch.from_numpy(samples[:,:,-1].reshape(-1, 1, 28, 28)), nrow=10)
plt.figure(figsize=[10,10])
plt.imshow(mtg.permute(1, 2, 0))
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
centroids = score_model_gmm.mus.detach().reshape(-1, 1, 28, 28)
mtg = make_grid(centroids, nrow=8)
plt.imshow(mtg.permute(1, 2, 0) )
plt.show()




#%%
ndim = Xtrain.shape[1]
nrank = 350
gmm_components = 10
sigma_max = 10
epochs = 2000
batch_size = 2048
torch.manual_seed(42)
score_model_lowrk = GMM_ansatz_net_lowrank(ndim=ndim,
           n_components=gmm_components, n_rank=nrank, sigma=sigma_max)
# Data PC initialization
score_model_lowrk.logLambdas.data = logLamda_init[None, :nrank].repeat(gmm_components, 1)
score_model_lowrk.Us.data = U_init[None, :, :nrank].repeat(gmm_components, 1, 1)
perturb = torch.randn(gmm_components, ndim)
perturb = perturb / torch.norm(perturb, dim=-1, keepdim=True)
score_model_lowrk.mus.data = Xmean[None, :] + perturb * 25
#%%
score_model_lowrk = train_score_td(Xtrain_norm, score_model_td=score_model_lowrk,
        sigma=sigma_max, lr=0.0002, nepochs=epochs, batch_size=batch_size, clipnorm=1)
# step 0 loss 375.200
# step 1999 loss 118.352: 100%|██████████| 2000/2000 [10:10<00:00,  3.27it/s]
# 10 mins, loss around 120,
# 5510700 parameters 5M
#%%
print("number of parameters",
sum(p.numel() for p in score_model_lowrk.parameters() if p.requires_grad))
#%%
samples = reverse_diffusion_time_dep(score_model_lowrk, sampN=100, sigma=sigma_max, nsteps=1000, ndim=ndim, exact=False)
#%%
mtg = make_grid(torch.from_numpy(samples[:,:,-1].reshape(-1, 1, 28, 28)), nrow=10)
plt.figure(figsize=[10,10])
plt.imshow(mtg.permute(1, 2, 0))
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
centroids = score_model_lowrk.mus.detach().reshape(-1, 1, 28, 28)
mtg = make_grid(centroids, nrow=5)
plt.imshow(mtg.permute(1, 2, 0) )
plt.show()



#%% Baseline, MLP score model with time embedding
epochs = 2000
batch_size = 2048
score_model_edm = ScoreModel_Time_edm(sigma=sigma_max, ndim=ndim,
                nlayers=8, nhidden=512, time_embed_dim=64,
                act_fun=nn.Tanh)
score_model_edm = train_score_td(Xtrain_norm, score_model_td=score_model_edm,
        sigma=sigma_max, lr=0.001, nepochs=epochs, batch_size=batch_size)
# loss around 200, could be slower
# 1 mins for 2000 epochs
#%%
# count the total number of parameters
sum(p.numel() for p in score_model_gmm.parameters() if p.requires_grad)
#%%
# count the total number of parameters
sum(p.numel() for p in score_model_edm.parameters() if p.requires_grad)
#%%
score_model_gauss(Xtrain_norm[:1], torch.tensor([.2])).norm()
#%%
score_model_edm(Xtrain_norm[:1], torch.tensor([.2])).norm()


#%%
from torchvision.utils import make_grid
centroids = score_model_lowrk.mus.detach().reshape(-1, 1, 28, 28)
mtg = make_grid(centroids)
plt.imshow(mtg.permute(1, 2, 0) / mtg.max())
plt.show()

#%%
Us = score_model_td.Us.detach()
Lambdas = score_model_td.logLambdas.exp().detach()
compon = 0
PCsort = Lambdas[compon, :].argsort(descending=True)
Us_sorted = Us[compon, :, PCsort,]

#%%
mtg = make_grid(Us_sorted[:, :100].reshape(1, 28, 28, -1).permute(3, 0, 1, 2), nrow=10)
plt.imshow(mtg.permute(1, 2, 0) / mtg.max())
plt.show()

#%%
plt.imshow(Us_sorted[:, 5].reshape(28, 28))
plt.show()

#%%







#%%
from sklearn.mixture import GaussianMixture
gmm_sk = GaussianMixture(n_components=gmm_components, covariance_type="full",
                         verbose=1, tol=1e-3, max_iter=1000)
gmm_sk.fit(Xtrain)
# will take forever to converge




#%%
from sklearn.model_selection import train_test_split
figdir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/HaimDiffusionRNNProj/Ring_GMM_ansatz_train"
import os
os.makedirs(figdir, exist_ok=True)
# ring_X = generate_ring_samples_torch(50)
ring_X = generate_spiral_samples_torch(20, a=0.4, b=0.15)
# train test split
Xtrain, Xtest = ring_X, torch.empty(0, 2)  # train_test_split(ring_X, test_size=0.0001, random_state=42)
# Xtrain, Xtest = train_test_split(ring_X, test_size=0.0001, random_state=42)
sigma_max = 10
gmm_components = 150
# mlp_width = 8
# mlp_depth = 3 # note, 2 layer usually doesn't work. 3 layer works.
# act_fun = nn.Tanh
# cfg_str = f"mlp {mlp_depth} layer width{mlp_width} {act_fun.__name__} sigma{sigma_max}"
cfg_str = f"sigma{sigma_max} dense"
cfg_str = f"sigma{sigma_max} dense spiral, {gmm_components} components"
for batch_size in [1024]: # 128, 256, 512,
    for epochs in [250, 500, 750, 1000, 1500, 2000,]:# :
        torch.manual_seed(42)
        score_model_td = GMM_ansatz_net(ndim=2, n_components=gmm_components)
        score_model_td = train_score_td(Xtrain, score_model_td=score_model_td,
                        sigma=sigma_max, lr=0.05, nepochs=epochs, batch_size=batch_size)
        #%%
        x_traj_denoise = reverse_diffusion_time_dep(score_model_td, sampN=3000, sigma=sigma_max, nsteps=1000, ndim=2, exact=False)
        figh = visualize_diffusion_distr(x_traj_denoise,
                             explabel=f"Time Dependent NN trained from weighted denoising\nepoch{epochs} batch {batch_size}\n{cfg_str}")
        figh.axes[1].set_xlim([-2.5, 2.5])
        figh.axes[1].set_ylim([-2.5, 2.5])
        saveallforms(figdir, f"ring_NN_contour_train_Ncomp{gmm_components}_batch{batch_size}_ep{epochs:04d}_sde")
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
        saveallforms(figdir, f"ring_NN_samples_train_Ncomp{gmm_components}_batch{batch_size}_ep{epochs:04d}_sde")
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








