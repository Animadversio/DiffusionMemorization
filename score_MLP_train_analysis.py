#%%
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import glob
#%%
rootdir = r"/Users/binxuwang/DL_Projects/HaimDiffusionRNNProj/Shape2d_MLP_train_kempner"
rootdir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/HaimDiffusionRNNProj/Shape2d_MLP_train_kempner"
dataset_str = "ring_20"
expdir = join(rootdir, f"{dataset_str}_lr_scaling")
# list pkl files
pkl_list = glob.glob(join(expdir, "*.pkl"))

# %%
pkl_list
# %%
"""Fixed width, vary depth"""
# mlp_depth = 3
mlp_width = 256
temb_dim = 32
plt.figure()
for mlp_depth in [2, 3, 4, 6, 8]:
    filename = f"{dataset_str}_NN_train_temb{temb_dim}_depth{mlp_depth}_width{mlp_width}_batch2048_ep5001.pkl"
    try:
        data = pkl.load(open(join(expdir, filename), "rb"))
        meta, loss_traj = data
    except FileNotFoundError:
        continue
    # smooth loss
    loss_traj_smooth = np.convolve(loss_traj, np.ones(25)/25, mode="valid")
    plt.plot(loss_traj_smooth, label=f"depth={mlp_depth}")
plt.legend()
plt.title(f"Smooth loss traj\nMLP width={mlp_width}, temb_dim={temb_dim}")
plt.show()
# %%
"""Fixed depth, vary width"""
mlp_depth = 3
# mlp_width = 64
temb_dim = 32
plt.figure()
for mlp_width in [8, 16, 32, 64, 128, 256, 512, 1024, ]:
    filename = f"{dataset_str}_NN_train_temb{temb_dim}_depth{mlp_depth}_width{mlp_width}_batch2048_ep5001.pkl"
    try:
        data = pkl.load(open(join(expdir, filename), "rb"))
        meta, loss_traj = data
    except FileNotFoundError:
        continue
    # smooth loss
    loss_traj_smooth = np.convolve(loss_traj, np.ones(25)/25, mode="valid")
    plt.plot(loss_traj_smooth, label=f"width={mlp_width}")
plt.title(f"Smooth loss traj\nMLP depth={mlp_depth}, temb_dim={temb_dim}")
plt.legend()
# plt.ylim([0.7, 1.1])
plt.show()

# %%
"""Fixed depth, vary width"""
mlp_depth = 3
# mlp_width = 64
temb_dim = 32
plt.figure()
for mlp_width in [8, 16, 32, 64, 128, 256, 512, 1024, ]:
    filename = f"{dataset_str}_NN_train_temb{temb_dim}_depth{mlp_depth}_width{mlp_width}_batch2048_ep5001.pkl"
    try:
        data = pkl.load(open(join(expdir, filename), "rb"))
        meta, loss_traj = data
    except FileNotFoundError:
        continue
    # smooth loss
    loss_traj_smooth = np.convolve(loss_traj, np.ones(25)/25, mode="valid")
    plt.loglog(loss_traj_smooth, label=f"width={mlp_width}")
plt.title(f"Smooth loss traj\nMLP depth={mlp_depth}, temb_dim={temb_dim}")
plt.legend()
# plt.ylim([0.7, 1.1])
plt.show()

#%%
"""Fixed depth, width, vary temb_dim"""
mlp_depth = 4
mlp_width = 32
temb_dim = 32
plt.figure()
for temb_dim in [16, 32, 64, 128,]:
    filename = f"{dataset_str}_NN_train_temb{temb_dim}_depth{mlp_depth}_width{mlp_width}_batch2048_ep5001.pkl"
    try:
        data = pkl.load(open(join(expdir, filename), "rb"))
        meta, loss_traj = data
    except FileNotFoundError:
        continue
    # smooth loss
    loss_traj_smooth = np.convolve(loss_traj, np.ones(25)/25, mode="valid")
    plt.plot(loss_traj_smooth, label=f"t emb dim={temb_dim}")
plt.title(f"Smooth loss traj\nMLP depth={mlp_depth}, width={mlp_width}")
plt.legend()
# plt.ylim([0.7, 1.1])
plt.show()

# %%
import torch
import torch.nn as nn
ndim = 32
batch = 1000
for ndim in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    Wtmp = nn.Linear(ndim, ndim)
    in_vec = torch.randn(batch, ndim)
    out_vec = Wtmp(in_vec)
    plt.semilogx(np.ones(batch)*ndim, 
             out_vec.std(dim=1).detach().numpy(), 
             label=f"ndim={ndim}",
             marker=".", alpha=0.5)
    # out_vec.std(dim=1)
plt.legend()
plt.show()
# %%
for ndim in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    Wtmp = nn.Linear(ndim, ndim)
    print(ndim, (Wtmp.weight.mean()).item(), "+-",(Wtmp.weight.std()** 2 * ndim).item())
# %%
