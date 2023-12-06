import torch

def generate_ring_samples_torch(n_points, R=1, ):
    """
    Generate points along a Ring using PyTorch.
    Parameters:
    - n_points (int): Number of points to generate.
    - R: Radius of the ring.
    Returns:
    - torch.Tensor: Batch of vectors representing points on the spiral.
    """
    theta = torch.linspace(0, 2 * torch.pi, n_points + 1, )  # angle theta
    theta = theta[:-1]
    x = R * torch.cos(theta)  # x = r * cos(theta)
    y = R * torch.sin(theta)  # y = r * sin(theta)
    spiral_batch = torch.stack((x, y), dim=1)
    return spiral_batch


def generate_spiral_samples_torch(n_points, a=1, b=0.2):
    """Generate points along a spiral using PyTorch.
    Parameters:
    - n_points (int): Number of points to generate.
    - a, b (float): Parameters that define the spiral shape.
    Returns:
    - torch.Tensor: Batch of vectors representing points on the spiral.
    """
    theta = torch.linspace(0, 4 * torch.pi, n_points)  # angle theta
    r = a + b * theta  # radius
    x = r * torch.cos(theta)  # x = r * cos(theta)
    y = r * torch.sin(theta)  # y = r * sin(theta)
    spiral_batch = torch.stack((x, y), dim=1)
    return spiral_batch

# pnts = generate_spiral_samples_torch(40, a = 0.1, b = 0.1)
pnts = generate_ring_samples_torch(6, R=5)

#%%
import matplotlib.pyplot as plt
from core.gmm_special_diffusion_lib import GMM_logprob
ngrid = 200
conv_sigma = 0.1
xx, yy = torch.meshgrid(torch.linspace(-8, 8, ngrid),
                        torch.linspace(-8, 8, ngrid))
query_pnts = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
logprob_gmm = GMM_logprob(pnts.numpy(), sigma=conv_sigma, x=query_pnts.numpy())
#%%
plt.figure(figsize=(8, 8))
plt.contour(xx.numpy(), yy.numpy(),
             logprob_gmm.reshape(ngrid, ngrid),
             50, cmap="jet")
plt.axis("image")
plt.show()
#%%
from torch import einsum
ndim = 2
sigma2 = conv_sigma ** 2
Eye = torch.eye(ndim)
# M2 = torch.mean(pnts[:, None, :] * pnts[:, :, None], dim=0)
Xi2 = einsum("Ba,Bb->ab", pnts, pnts) / pnts.shape[0]
Xi3 = einsum("Ba,Bb,Bc->abc", pnts, pnts, pnts) / pnts.shape[0]
Xi4 = einsum("Ba,Bb,Bc,Bd->abcd", pnts, pnts, pnts, pnts) / pnts.shape[0]
# Xi3 = torch.mean(pnts[:, None, None, :] * pnts[:, None, :, None] * pnts[:, :, None, None, None], dim=0)
moment1 = torch.mean(pnts, dim=0)
moment2 = Xi2 + torch.eye(ndim) * sigma2
moment3 = Xi3 + sigma2 * (einsum("ab,c->abc", Eye, moment1) \
              + einsum("ac,b->abc", Eye, moment1) \
              + einsum("bc,a->abc", Eye, moment1))
moment4 = Xi4 + sigma2 * (einsum("ab,cd->abcd", Eye, Xi2) + \
                          einsum("cd,ab->abcd", Eye, Xi2) + \
                          einsum("ac,bd->abcd", Eye, Xi2) + \
                          einsum("bd,ac->abcd", Eye, Xi2) + \
                          einsum("ad,bc->abcd", Eye, Xi2) + \
                          einsum("bc,ad->abcd", Eye, Xi2) ) \
          + sigma2 **2 * (einsum("ab,cd->abcd", Eye, Eye) +
                          einsum("ac,bd->abcd", Eye, Eye) +
                          einsum("ad,bc->abcd", Eye, Eye))

cumulant1 = moment1
cumulant2 = moment2 - einsum("a,b->ab", moment1, moment1)
#%%
cumulant3 = moment3 - einsum("a,bc->abc", moment1, moment2) \
                    - einsum("b,ac->abc", moment1, moment2) \
                    - einsum("c,ab->abc", moment1, moment2) \
        + 2 * einsum("a,b,c->abc", moment1, moment1, moment1)
#%%
cumulant4 = moment4 - einsum("a,bcd->abcd", moment1, moment3) \
                    - einsum("b,acd->abcd", moment1, moment3) \
                    - einsum("c,abd->abcd", moment1, moment3) \
                    - einsum("d,abc->abcd", moment1, moment3) \
                    - einsum("ab,cd->abcd", moment2, moment2) \
                    - einsum("ac,bd->abcd", moment2, moment2) \
                    - einsum("ad,bc->abcd", moment2, moment2) \
        + 2 * einsum("a,b,cd->abcd", moment1, moment1, moment2) \
        + 2 * einsum("a,c,bd->abcd", moment1, moment1, moment2) \
        + 2 * einsum("a,d,bc->abcd", moment1, moment1, moment2) \
        + 2 * einsum("c,d,ab->abcd", moment1, moment1, moment2) \
        + 2 * einsum("b,d,ac->abcd", moment1, moment1, moment2) \
        + 2 * einsum("b,c,ad->abcd", moment1, moment1, moment2) \
        - 6 * einsum("a,b,c,d->abcd", moment1, moment1, moment1, moment1)
#%%
cov_mat = cumulant2
mu_vec = cumulant1
prec_mat = torch.inverse(cov_mat)
#%%
import math
gauss_logprob = -0.5 * einsum("Ba,ab,Bb->B", query_pnts, prec_mat, query_pnts) \
                - 0.5 * torch.logdet(cov_mat) \
                - 0.5 * ndim * math.log(2 * math.pi)
affine_pnts = einsum("ba,Ba->Bb", prec_mat, query_pnts - mu_vec[None, :])
cumulant3_perturb = (3 * einsum("abc,ab,Bc->B", cumulant3, prec_mat, affine_pnts) \
 - einsum("abc,Ba,Bb,Bc->B", cumulant3, affine_pnts, affine_pnts, affine_pnts) )
cumulant4_perturb = (einsum("abcd,Ba,Bb,Bc,Bd->B", cumulant4, affine_pnts, affine_pnts, affine_pnts, affine_pnts) \
                     - 6 * einsum("abcd,ab,Bc,Bd->B", cumulant4, prec_mat, affine_pnts, affine_pnts) \
                     + 3 * einsum("abcd,ab,cd->", cumulant4, prec_mat, prec_mat))
#%%
edgeworth_logprob = gauss_logprob + torch.log(1 - cumulant3_perturb / 6 + cumulant4_perturb / 24)
edgeworth_logprob = gauss_logprob + torch.log(1 - cumulant3_perturb / 6 + cumulant4_perturb / 24)
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.contour(xx.numpy(), yy.numpy(),
             gauss_logprob.reshape(ngrid, ngrid).numpy(),
             50, cmap="jet")
plt.axis("image")
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.contour(xx.numpy(), yy.numpy(),
             gauss_logprob.reshape(ngrid, ngrid).numpy(),
             50, cmap="jet")
plt.scatter(pnts[:, 0], pnts[:, 1], s=180, color="r")
plt.axis("image")
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.contourf(xx.numpy(), yy.numpy(),
             edgeworth_logprob.reshape(ngrid, ngrid).numpy(),
             50, cmap="jet")
plt.scatter(pnts[:, 0], pnts[:, 1], s=180, color="r")
plt.axis("image")
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.contourf(xx.numpy(), yy.numpy(),
             edgeworth_logprob.reshape(ngrid, ngrid).exp().numpy(),
             50, cmap="jet")
plt.scatter(pnts[:, 0], pnts[:, 1], s=180, color="r")
plt.axis("image")
plt.show()
#%% 3d surf plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx.numpy(), yy.numpy(),
                edgeworth_logprob.reshape(ngrid, ngrid).exp().numpy(), cmap=cm.jet)
# change perspective
ax.view_init(azim=30, elev=70)
plt.show()

#%%
plt.figure(figsize=(8, 8))
plt.contour(xx.numpy(), yy.numpy(),
             cumulant3_perturb.reshape(ngrid, ngrid).numpy(),
             50, cmap="jet")
plt.axis("image")
plt.show()

