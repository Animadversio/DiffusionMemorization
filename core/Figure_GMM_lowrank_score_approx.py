#%%
import os
import sys
from os.path import join
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid, save_image
from core.gmm_special_dynamics import alpha
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionMemorization")
# from train_edm import edm_sampler, EDM, create_model
# from core.edm_utils import get_default_config, create_edm
from core.utils.plot_utils import saveallforms
# set pandas display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#%%
figroot = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/DiffusionHiddenLinear"
figsumdir = join(figroot, "GMM_lowrk_approx_summary")
os.makedirs(figsumdir, exist_ok=True)

# %% [markdown]
# ### MNIST score with varying rank and components

# %%
ckptdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/base_mnist_20240129-1342/checkpoints/"
df_gmm_rk = pd.read_csv(join(ckptdir, "..", "MNIST_edm_1000k_epoch_gmm_exp_var_gmm_rk.csv"))

# %%
# preprocess the dataframe to extact the rank and components
df_gmm_rk["St_residual"] = 1 - df_gmm_rk["St_EV"]
df_gmm_rk["Dt_residual"] = 1 - df_gmm_rk["Dt_EV"]
df_gmm_rk[['n_cluster', 'n_rank']] = df_gmm_rk['name'].str.extract(r'gmm_(\d+)_mode_(\d+)_rank')
df_gmm_rk['n_cluster'] = df_gmm_rk['n_cluster'].astype(float)
df_gmm_rk['n_rank'] = df_gmm_rk['n_rank'].astype(float)

# %%
df_gmm_rk.columns

# %%
for sigma in df_gmm_rk.sigma.unique():
    # Create a heatmap
    res_mat = []
    for n_clusters in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for n_rank in [8, 16, 32, 64, 96, 128, 256, 512, 768, 1024]:
            res = 1 - df_gmm_rk[(df_gmm_rk["name"] == f"gmm_{n_clusters}_mode_{n_rank}_rank") & 
                                (df_gmm_rk.sigma == sigma)]["St_EV"].values
            res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, "residual": res[0]})
    res_mat = pd.DataFrame(res_mat)
    res_mat_pivot = res_mat.pivot_table(index="n_clusters", columns="n_rank",
                                        values="residual", aggfunc="mean")
    plt.figure(figsize=(10, 8))
    sns.heatmap(res_mat_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Residual Matrix of GMM MNIST Dataset | sigma=%.2f" % sigma)
    plt.ylabel("Number of Modes")
    plt.xlabel("Rank")
    saveallforms(figsumdir, f"MNIST_GMM_rank_residual_heatmap_sigma{sigma}")
    plt.show()

# %%
# alterantive heatmap with scientific notation
for sigma in df_gmm_rk.sigma.unique():
    # Create a heatmap
    res_mat = []
    for n_clusters in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for n_rank in [8, 16, 32, 64, 96, 128, 256, 512, 768, 1024]:
            res = 1 - df_gmm_rk[(df_gmm_rk["name"] == f"gmm_{n_clusters}_mode_{n_rank}_rank") & 
                                (df_gmm_rk.sigma == sigma)]["St_EV"].values
            res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, "residual": res[0]})
    res_mat = pd.DataFrame(res_mat)
    res_mat_pivot = res_mat.pivot_table(index="n_clusters", columns="n_rank",
                                        values="residual", aggfunc="mean")
    plt.figure(figsize=(10, 8))
    sns.heatmap(res_mat_pivot, annot=True, fmt=".1e", cmap="YlGnBu")
    plt.title("Residual Matrix of GMM MNIST Dataset | sigma=%.2f" % sigma)
    plt.ylabel("Number of Modes")
    plt.xlabel("Rank")
    plt.show()
# %%
df_gmm_rk.n_rank.unique()

# %%
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_gmm_rk[(df_gmm_rk.n_rank == 1024)],
            x="n_cluster", y="St_residual", hue="sigma", 
            palette="RdYlBu", lw=1.5, marker="o", markersize=5, alpha=0.4)
plt.yscale("log")
plt.xscale("log")
plt.title("Score residual with Varying Number of Modes | Rank=1024")
plt.show()

# %%
