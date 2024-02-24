# %% [markdown]
# ## GMM vs EDM 
# %%
import os
import sys
from os.path import join
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid, save_image
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionMemorization")
from train_edm import edm_sampler, EDM, create_model
from core.edm_utils import get_default_config, create_edm
from core.utils.plot_utils import saveallforms
# set pandas display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# %%
figroot = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/DiffusionHiddenLinear"
figsumdir = join(figroot, "GMM_EDM_summary")
os.makedirs(figsumdir, exist_ok=True)
#%%
def visualize_train_run_score_cmp(df_syn, plot_var="St_residual", hue_order=["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                        "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"], 
                        train_run_name="CIFAR10 EDM 5k epochs initial", 
                        savename="cifar10_mini_edm_gmm_score_residual_ev_initial_epochs"):
    
    if plot_var == "St_residual":
        cmp_variable = "score"
    elif plot_var == "Dt_residual":
        cmp_variable = "denoiser"
    else:
        cmp_variable = plot_var
    figh, axs = plt.subplots(2, 5, figsize=(18, 8))
    axs = axs.flatten()
    for i, sigma in enumerate([1.0e-02, 5.0e-02, 1.0e-01, 5.0e-01, 1.0e+00, 1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, ]):
        df_syn_sigma = df_syn[(df_syn.sigma == sigma)]
        sns.lineplot(data=df_syn_sigma, x="epoch", y=plot_var, hue="name", 
                    hue_order=hue_order, palette="RdYlBu", lw=2.0, ax=axs[i])
        axs[i].set_yscale("log")
        axs[i].set_ylim(None, 1)
        axs[i].set_title(f"sigma={sigma}")
        if i < 9:
            axs[i].legend().remove()
    plt.suptitle(f"Residual explained variance of EDM {cmp_variable} by Gaussian and GMM {cmp_variable}s ({train_run_name})", fontsize=18)
    plt.tight_layout()
    saveallforms(figsumdir, savename, figh, ["pdf", "png"])
    plt.show()
    return figh, axs


# %% [markdown]
# ### MNIST EDM training process vs GMM scores
# %% [markdown]
# #### Early Phase MNIST
ckptdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/base_mnist_20240130-2207/checkpoints/"
df_syn_mnst = pd.read_csv(join(ckptdir, "..", "MNIST_edm_25k_epoch_gmm_exp_var.csv"))
# %%
df_syn_mnst.name.unique()
#%%
mnist_score_names = ["mean isotropic", "gaussian", 'gmm_2_mode', 
                'gmm_5_mode', 'gmm_10_mode', 'gmm_20_mode', 'gmm_50_mode', 
                'gmm_100_mode', 'gmm_200_mode','gmm_500_mode', "gmm delta"]

visualize_train_run_score_cmp(df_syn_mnst, plot_var="St_residual", 
                              hue_order=mnist_score_names, 
                              train_run_name="MNIST EDM 25k epochs initial phase",
                              savename="mnist_mini_edm_gmm_score_residual_ev_initial_epochs");

visualize_train_run_score_cmp(df_syn_mnst, plot_var="Dt_residual", 
                              hue_order=mnist_score_names, 
                              train_run_name="MNIST EDM 25k epochs initial phase",
                              savename="mnist_mini_edm_gmm_denoiser_residual_ev_initial_epochs");

# %% [markdown]
# #### Late Phase MNIST

# %%
ckptdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/base_mnist_20240129-1342/checkpoints/"
df_syn_mnst_lt = pd.read_csv(join(ckptdir, "..", "MNIST_edm_1000k_epoch_gmm_exp_var.csv"))

mnist_score_names = ["mean isotropic", "gaussian", 'gmm_2_mode', 
                'gmm_5_mode', 'gmm_10_mode', 'gmm_20_mode', 'gmm_50_mode', 
                'gmm_100_mode', 'gmm_200_mode','gmm_500_mode', "gmm delta"]

visualize_train_run_score_cmp(df_syn_mnst_lt, plot_var="St_residual", 
                              hue_order=mnist_score_names, 
                              train_run_name="MNIST EDM 1000k epochs initial phase",
                              savename="mnist_mini_edm_gmm_score_residual_ev_late_epochs");

visualize_train_run_score_cmp(df_syn_mnst_lt, plot_var="Dt_residual", 
                              hue_order=mnist_score_names, 
                              train_run_name="MNIST EDM 1000k epochs initial phase",
                              savename="mnist_mini_edm_gmm_denoiser_residual_ev_late_epochs");


# %% [markdown]
# ### CIFAR10 EDM training process vs GMM scores

# %% [markdown]
# #### Early phase ckpt stats

# %%
ckptdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/base_cifar10_20240130-2317/checkpoints"
df_syn_cf = pd.read_csv(join(ckptdir, "..", "edm_50k_epoch_gmm_exp_var.csv"))
# df_syn_cf.sigma.unique()
cifar_score_names = ["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                       "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"]
visualize_train_run_score_cmp(df_syn_cf, plot_var="St_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 EDM 50k epochs initial phase",
                              savename="cifar10_mini_edm_gmm_score_residual_ev_initial_epochs");

visualize_train_run_score_cmp(df_syn_cf, plot_var="Dt_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 EDM 50k epochs initial phase",
                              savename="cifar10_mini_edm_gmm_denoiser_residual_ev_initial_epochs");

# %%
sigma = 0.01
plt.figure(figsize=(7, 7))
sns.lineplot(data=df_syn_cf[(df_syn_cf.sigma == sigma) & (df_syn_cf.epoch > 0) & (df_syn.epoch < 2E4)], 
            x="epoch", y="St_residual", palette="RdYlBu", hue="name", lw=2.5, alpha=0.8,
            hue_order=["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                       "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"])
# TODO: add the shaded errorbar from `St_EV_std`
plt.yscale("log")
plt.ylim(None, 1.5)
plt.title(f"Residual explained variance of score | sigma={sigma}", fontsize=16)
saveallforms(figsumdir, "cifar10_mini_edm_gmm_score_residual_ev_initial_epochs_sigma001")
plt.show()

# %% [markdown]
# #### Late phase ckpt CIFAR10

# %%
ckptdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/base_cifar10_20240130-2318/checkpoints"
df_syn_cf_lt = pd.read_csv(join(ckptdir, "..", "edm_365k_epoch_gmm_exp_var.csv"))
# df_syn_cf.sigma.unique()
cifar_score_names = ["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                       "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"]
visualize_train_run_score_cmp(df_syn_cf_lt, plot_var="St_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 EDM 365k epochs late phase",
                              savename="cifar10_mini_edm_gmm_score_residual_ev_late_epochs");
visualize_train_run_score_cmp(df_syn_cf_lt, plot_var="Dt_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 EDM 365k epochs late phase",
                              savename="cifar10_mini_edm_gmm_denoiser_residual_ev_late_epochs");


# %%
sigma = 0.02
plt.figure(figsize=(7, 7))
sns.lineplot(data=df_syn_cf_lt[(df_syn_cf_lt.sigma == sigma) & (df_syn_cf_lt.epoch > 0)], 
            x="epoch", y="St_residual", palette="RdYlBu", hue="name", lw=2.5, alpha=0.8,
            hue_order=["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                       "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"])
# TODO: add the shaded errorbar from `St_EV_std`
plt.yscale("log")
plt.ylim(None, 1.0)
plt.title(f"Residual explained variance of score \n sigma={sigma} (late training)", fontsize=16)
plt.show()

# %%
sigma = 0.05
plt.figure(figsize=(7, 7))
sns.lineplot(data=df_syn_cf_lt[(df_syn_cf_lt.sigma == sigma) & (df_syn_cf_lt.epoch > 0)], 
            x="epoch", y="St_residual", palette="RdYlBu", hue="name", lw=2.5, alpha=0.8,
            hue_order=["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                       "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"])
# TODO: add the shaded errorbar from `St_EV_std`
plt.yscale("log")
plt.ylim(None, 1.0)
plt.title(f"Residual explained variance of score \n sigma={sigma} (late training)", fontsize=16)
plt.show()

# %% [markdown]
# ### CIFAR EDM trained 

# %%
tabdir = r"/n/home12/binxuwang/Github/DiffusionMemorization/Tables"
df_syn_cifar2 = pd.read_csv(join(tabdir, "cifar10_uncond_edm_5k_epoch_gmm_exp_var.csv"))
cifar_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 'gmm_10_mode', 
                     'gmm_20_mode', 'gmm_50_mode', 'gmm_100_mode', 'gmm_200_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_cifar2, plot_var="St_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 origin EDM 5k epochs early phase",
                              savename="cifar10_edm_gmm_score_residual_ev_early_epochs");
visualize_train_run_score_cmp(df_syn_cifar2, plot_var="Dt_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 origin EDM 5k epochs early phase",
                              savename="cifar10_edm_gmm_denoiser_residual_ev_early_epochs");

# %% [markdown]
# ### AFHQ 64 Dataseet

# %%
df_syn_afhq = pd.read_csv(join(tabdir, "AFHQ_edm_5k_epoch_gmm_exp_var_fixed.csv"))
df_syn_afhq.name.unique()
afhq_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 
                    'gmm_10_mode', 'gmm_20_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_afhq, plot_var="St_residual", 
                              hue_order=afhq_score_names, 
                              train_run_name="AFHQ origin EDM 5k epochs initial phase",
                              savename="afhq_edm_gmm_score_residual_ev_initial_epochs");
visualize_train_run_score_cmp(df_syn_afhq, plot_var="Dt_residual",
                                hue_order=afhq_score_names, 
                                train_run_name="AFHQ origin EDM 5k epochs initial phase",
                                savename="afhq_edm_gmm_denoiser_residual_ev_initial_epochs");

# %% [markdown]
# #### Late training
# %%
df_syn_afhq_lt = pd.read_csv(join(tabdir, "AFHQ_edm_35k_epoch_gmm_exp_var.csv"))
afhq_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 
                    'gmm_10_mode', 'gmm_20_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_afhq_lt, plot_var="St_residual", 
                              hue_order=afhq_score_names, 
                              train_run_name="AFHQ origin EDM 35k epochs late phase",
                              savename="afhq_edm_gmm_score_residual_ev_late_epochs");
visualize_train_run_score_cmp(df_syn_afhq_lt, plot_var="Dt_residual",
                                hue_order=afhq_score_names, 
                                train_run_name="AFHQ origin EDM 35k epochs late phase",
                                savename="afhq_edm_gmm_denoiser_residual_ev_late_epochs");

# %% [markdown]
# ### FFHQ dataset
# %%
df_syn_ffhq = pd.read_csv(join(tabdir,"FFHQ_edm_5k_epoch_gmm_exp_var.csv"))
ffhq_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 
                    'gmm_10_mode', 'gmm_20_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_ffhq, plot_var="St_residual",
                                hue_order=ffhq_score_names, 
                                train_run_name="FFHQ origin EDM 5k epochs initial phase",
                                savename="ffhq_edm_gmm_score_residual_ev_initial_epochs");
visualize_train_run_score_cmp(df_syn_ffhq, plot_var="Dt_residual",
                                hue_order=ffhq_score_names, 
                                train_run_name="FFHQ origin EDM 5k epochs initial phase",
                                savename="ffhq_edm_gmm_denoiser_residual_ev_initial_epochs");

# %% [markdown]
# #### FFHQ Late stage training
df_syn_ffhq_lt = pd.read_csv(join(tabdir,"FFHQ_edm_35k_epoch_gmm_exp_var.csv"))
ffhq_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 
                    'gmm_10_mode', 'gmm_20_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_ffhq_lt, plot_var="St_residual",
                                hue_order=ffhq_score_names, 
                                train_run_name="FFHQ origin EDM 35k epochs late phase",
                                savename="ffhq_edm_gmm_score_residual_ev_late_epochs");
visualize_train_run_score_cmp(df_syn_ffhq_lt, plot_var="Dt_residual",
                                hue_order=ffhq_score_names, 
                                train_run_name="FFHQ origin EDM 35k epochs late phase",
                                savename="ffhq_edm_gmm_denoiser_residual_ev_late_epochs");


# %%






# %% [markdown]
# ### MNIST score with varying rank and components

# %%
ckptdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/base_mnist_20240129-1342/checkpoints/"
df_gmm_rk = pd.read_csv(join(ckptdir, "..", "MNIST_edm_1000k_epoch_gmm_exp_var_gmm_rk.csv"))

# %%
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(res_mat.pivot_table(index="n_clusters", columns="n_rank", 
                        values="residual", aggfunc="mean"), 
                annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Residual Matrix of GMM MNIST Dataset | sigma=%.2f" % sigma)
    plt.ylabel("Number of Modes")
    plt.xlabel("Rank")
    plt.show()

# %%
df_gmm_rk.n_rank.unique()

# %%
df_gmm_rk[(df_gmm_rk.n_rank == 1024) & 
        (df_gmm_rk.sigma == 1.0)]

# %%
sns.lineplot(data=df_gmm_rk[(df_gmm_rk.n_rank == 1024)],
            x="n_cluster", y="St_residual", hue="sigma", 
            palette="RdYlBu", lw=1.5)
plt.yscale("log")
plt.xscale("log")
plt.show()
