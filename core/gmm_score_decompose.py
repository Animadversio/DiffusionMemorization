"""
A demo figure for NeuroMatch 2023 tutorial
Showing that the optimal direction for denoising is the gradient of the log probability
"""
# %load_ext autoreload
# %autoreload 2
#%%
from core.gaussian_mixture_lib import GaussianMixture
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from scipy.stats import multivariate_normal
def kdeplot(pnts, label="", ax=None, titlestr=None, **kwargs):
  if ax is None:
    ax = plt.gca()#figh, axs = plt.subplots(1,1,figsize=[6.5, 6])
  sns.kdeplot(x=pnts[:,0], y=pnts[:,1], ax=ax, label=label, **kwargs)
  if titlestr is not None:
    ax.set_title(titlestr)

def quiver_plot(pnts, vecs, *args, **kwargs):
  plt.quiver(pnts[:, 0], pnts[:,1], vecs[:, 0], vecs[:, 1], *args, **kwargs)

figdir = r"E:\OneDrive - Harvard University\Neuromatch2023_Tutorial\Score_decompose"

#%%
# from numpy.random.mtrand import sample
# mean and covariance of the 1,2,3 Gaussian branch.
mu1 = np.array([0,1.0])
Cov1 = np.array([[1.0,0.0],
          [0.0,1.0]])

mu2 = np.array([2.0,-1.0])
Cov2 = np.array([[2.0,0.5],
          [0.5,1.0]])

gmm = GaussianMixture([mu1,mu2],[Cov1,Cov2],[1.0,1.0])
#%%
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pnts = np.stack([xx, yy], axis=-1).reshape(-1, 2)
prob, prob_branch = gmm.pdf_decompose(pnts)
#%%
#%%
gmm_samps, _, _ = gmm.sample(5000)
#%%
figh, ax = plt.subplots(1,1,figsize=[6,6])
kdeplot(gmm_samps, )
plt.contour(xx, yy, prob.reshape(100,100), cmap="Reds")
plt.title("Empirical density of Gaussian mixture density")
plt.axis("image");
plt.show()
#%% Demo of Gaussian mixtrue
mu1 = np.array([0,1.0])
Cov1 = np.array([[1.0,0.0],
          [0.0,1.0]])

mu2 = np.array([2.0,-1.0])
Cov2 = np.array([[2.0,0.5],
          [0.5,1.0]])

gmm = GaussianMixture([mu1,mu2],[Cov1,Cov2],[1.0,1.0])
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pnts = np.stack([xx, yy], axis=-1).reshape(-1, 2)
prob, prob_branch = gmm.pdf_decompose(pnts)
#%%
gmm_samps_few = np.array([[2.25, 1.0],
                          [-0.5, -1.6],])
# gmm_samps_few, _, _ = gmm.sample(10)
scorevecs_few = gmm.score(gmm_samps_few)
gradvec_list, participance = gmm.score_decompose(gmm_samps_few)

plt.figure(figsize=[8,8])
quiver_plot(gmm_samps_few, scorevecs_few, color="black", alpha=0.4, scale=20, width=0.007, label="score of the mixture")
quiver_plot(gmm_samps_few, gradvec_list[0], color="blue", alpha=0.4, scale=20, width=0.007, label="score of gauss mode1")
quiver_plot(gmm_samps_few, gradvec_list[1], color="orange", alpha=0.4, scale=20, width=0.007, label="score of gauss mode2")
plt.contour(xx, yy, prob.reshape(100,100), cmap="Greys")
plt.contour(xx, yy, prob_branch[:,0].reshape(100,100), cmap="Blues")
plt.contour(xx, yy, prob_branch[:,1].reshape(100,100), cmap="Oranges")
plt.scatter(gmm.mus[:,0], gmm.mus[:,1], marker="x", color="black", s=100, label="mode mean")
# quiver_plot(gmm_samps_few, scorevecs_few, scale=15, alpha=0.7, width=0.003)
plt.title("Score vector field $\log p(x)$")
plt.axis("image")
plt.legend()
saveallforms(figdir, "gmm_score_decompose_demo", )
plt.show()

#%%
mu1 = np.array([0,1.0])
Cov1 = np.array([[1.0,0.0],
          [0.0,1.0]])

mu2 = np.array([2.0,-1.0])
Cov2 = np.array([[1.0,0.0],
          [0.0,1.0]])

gmm = GaussianMixture([mu1,mu2],[Cov1,Cov2],[1.0,1.0])
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pnts = np.stack([xx, yy], axis=-1).reshape(-1, 2)
prob, prob_branch = gmm.pdf_decompose(pnts)
#%%
gmm_samps_few = np.array([[ 2.25,  1.00],
                          [-0.25, -1.25],])
# gmm_samps_few, _, _ = gmm.sample(10)
scorevecs_few = gmm.score(gmm_samps_few)
gradvec_list, participance = gmm.score_decompose(gmm_samps_few)

mus = np.array(gmm.mus)
plt.figure(figsize=[8,8])
quiver_plot(gmm_samps_few, scorevecs_few, color="black", alpha=0.4, scale=20, width=0.007, label="score of the mixture")
quiver_plot(gmm_samps_few, gradvec_list[0], color="blue", alpha=0.4, scale=20, width=0.007, label="score of gauss mode1")
quiver_plot(gmm_samps_few, gradvec_list[1], color="orange", alpha=0.4, scale=20, width=0.007, label="score of gauss mode2")
plt.contour(xx, yy, prob.reshape(100,100), cmap="Greys",)
plt.contour(xx, yy, prob_branch[:,0].reshape(100,100), cmap="Blues",)
plt.contour(xx, yy, prob_branch[:,1].reshape(100,100), cmap="Oranges",)
plt.scatter(mus[:,0], mus[:,1], marker="x", color="black", s=100, label="Original data point x_0")
plt.scatter(gmm_samps_few[:,0], gmm_samps_few[:,1], marker="o", color="black", s=100, label="Sampled data point x_t")
# quiver_plot(gmm_samps_few, scorevecs_few, scale=15, alpha=0.7, width=0.003)
plt.title("Score vector field $\log p(x)$")
plt.axis("image")
plt.legend()
plt.xlim([-3,5])
plt.ylim([-4,4])
saveallforms(figdir, "points_score_decompose_demo", )
plt.show()

#%%
mu1 = np.array([0,1.0])
Cov1 = np.array([[.3,-0.1],
          [-0.1,.5]])

mu2 = np.array([2.0,-1.0])
Cov2 = np.array([[.5,-0.2],
          [-0.2,.5]])

mu3 = np.array([-1.0,-1.0])
Cov3 = np.array([[.5,0.3],
          [0.3,.5]])
gmm3 = GaussianMixture([mu1,mu2,mu3],[Cov1,Cov2,Cov3],[1.0,1.0,1.0])
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
pnts = np.stack([xx, yy], axis=-1).reshape(-1, 2)
prob, prob_branch = gmm3.pdf_decompose(pnts)
scorevecs_all = gmm3.score(pnts)
gradvec_all_list, participance_all = gmm3.score_decompose(pnts)
#%%
gmm_samps_few = np.array([[ 2.25,  1.00],
                          [-0.25, -1.25],])
# gmm_samps_few, _, _ = gmm.sample(10)
# scorevecs_few = gmm.score(gmm_samps_few)
# gradvec_list, participance = gmm.score_decompose(gmm_samps_few)
#%%
sparse_msk = np.zeros((100,100), dtype=bool)
sparse_msk[2::8,2::8] = True
sparse_msk = sparse_msk.reshape(-1)
#%%
mus = np.array(gmm3.mus)
plt.figure(figsize=[8,8])
quiver_plot(pnts[sparse_msk], gradvec_all_list[0][sparse_msk,:], color="blue", alpha=0.5, scale=100, width=0.005, label="score of gauss mode0")
quiver_plot(pnts[sparse_msk], gradvec_all_list[1][sparse_msk,:], color="orange", alpha=0.5, scale=100, width=0.005, label="score of gauss mode1")
quiver_plot(pnts[sparse_msk], gradvec_all_list[2][sparse_msk,:], color="green", alpha=0.5, scale=100, width=0.005, label="score of gauss mode2")
# quiver_plot(gmm_samps_few, scorevecs_few, color="black", alpha=0.4, scale=20, width=0.007, label="score of the mixture")
# quiver_plot(gmm_samps_few, gradvec_list[0], color="blue", alpha=0.4, scale=20, width=0.007, label="score of gauss mode1")
# quiver_plot(gmm_samps_few, gradvec_list[1], color="orange", alpha=0.4, scale=20, width=0.007, label="score of gauss mode2")
# plt.contour(xx, yy, prob.reshape(100,100), cmap="Greys",)
plt.contour(xx, yy, prob_branch[:,0].reshape(100,100), cmap="Blues",)
plt.contour(xx, yy, prob_branch[:,1].reshape(100,100), cmap="Oranges",)
plt.contour(xx, yy, prob_branch[:,2].reshape(100,100), cmap="Greens",)
plt.scatter(mus[:,0], mus[:,1], marker="x", color="black", s=100, label="Original data point x_0")
# plt.scatter(gmm_samps_few[:,0], gmm_samps_few[:,1], marker="o", color="black", s=100, label="Sampled data point x_t")
# quiver_plot(gmm_samps_few, scorevecs_few, scale=15, alpha=0.7, width=0.003)
plt.title("Score vector field $\log p(x)$")
plt.axis("image")
plt.legend()
plt.xlim([-3,3])
plt.ylim([-3,3])
saveallforms(figdir, "branches_conditional", )
plt.show()
#%%
mus = np.array(gmm3.mus)
clrs = ["blue", "orange", "green"]
cmaps = ["Blues", "Oranges", "Greens"]
for branch in range(3):
    plt.figure(figsize=[8, 8])
    quiver_plot(pnts[sparse_msk], gradvec_all_list[branch][sparse_msk, :], color=clrs[branch], alpha=0.5, scale=100, width=0.005,
                label=f"score of the mode {branch}")
    plt.contour(xx, yy, prob_branch[:, branch].reshape(100, 100), cmap=cmaps[branch], )
    plt.scatter(mus[:, 0], mus[:, 1], marker="x", color="black", s=100, label="Mode center x_0")
    plt.axis("image")
    plt.legend()
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    saveallforms(figdir, f"branches_conditional_mode{branch}", )
    plt.show()
#%%
plt.figure(figsize=[8, 8])
quiver_plot(pnts[sparse_msk], scorevecs_all[sparse_msk, :], color="gray", alpha=0.5, scale=100, width=0.005,
            label=f"score of the all modes")
plt.contour(xx, yy, prob.reshape(100, 100), cmap="Greys", )
plt.scatter(mus[:, 0], mus[:, 1], marker="x", color="black", s=100, label="Mode center x_0")
plt.axis("image")
plt.legend()
plt.xlim([-3, 3])
plt.ylim([-3, 3])
saveallforms(figdir, f"branches_conditional_mode_mixture", )
plt.show()