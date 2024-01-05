#%%
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import glob
import platform
#%%
if platform.system() == "Darwin":
    rootdir = r"/Users/binxuwang/DL_Projects/HaimDiffusionRNNProj/Shape2d_MLP_train_kempner"
elif platform.system() == "Linux":
    rootdir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/HaimDiffusionRNNProj/Shape2d_MLP_train_kempner"
#%%
dataset_str = "spiral_50" # "ring_20"
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
plt.ylim([0.7, 1.1])
plt.show()

#%%
loss_traj_smooth[-1]

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
plt.ylim([0.7, 1.1])
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
#%%

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
#%%
rootdir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/HaimDiffusionRNNProj/Shape2d_GMM_ansatz_train_kempner"
dataset_str = "spiral_50" # "ring_20"
dataset_str = "ring_20" # "ring_20"
expdir = join(rootdir, f"{dataset_str}")
# list pkl files
pkl_list = glob.glob(join(expdir, "*.pkl"))
#%%
def get_loss_trajs(pattern, Ncomp_list, expdir):
    loss_traj_list = [  ]
    for Ncomp in Ncomp_list:
        try:
            filename = pattern % Ncomp
            data = pkl.load(open(join(expdir, filename), "rb"))
            meta, loss_traj = data
        except FileNotFoundError:
            continue
        loss_traj_list.append(loss_traj)
    return loss_traj_list
    
    
def plot_sweep_pattern(pattern, Ncomp_list, expdir, var_name="Ncomp",
                       title_desc="", ylim=None, xlim=None, logscale="plot"):
    plot_func = {"plot": plt.plot, 
                 "semilogx": plt.semilogx, 
                 "semilogy": plt.semilogy, 
                 "loglog": plt.loglog}[logscale]
    plt.figure()
    for Ncomp in Ncomp_list:
        filename = pattern % Ncomp
        try:
            data = pkl.load(open(join(expdir, filename), "rb"))
            meta, loss_traj = data
        except FileNotFoundError:
            continue
        # smooth loss
        loss_traj_smooth = np.convolve(loss_traj, np.ones(25)/25, mode="valid")
        plot_func(loss_traj_smooth, label=f"{var_name}={Ncomp}")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(f"Smooth loss traj\n{title_desc}")
    plt.show()


pattern = f"{dataset_str}_ansatz_NN_train_Ncomp%d_batch2048_ep5001.pkl"
plot_sweep_pattern(pattern, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], expdir, 
                   var_name="Ncomp", title_desc=f"GMM ansatz", 
                   logscale="semilogx")
#%%
Ncomps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
Ncomps = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50, 64, 96, 128, 192, 256, 384, 512, 1024]
loss_col = get_loss_trajs(pattern, Ncomps, expdir,)
plt.figure()
end_loss = []
for i, (Ncomp, loss_traj) in enumerate(zip(Ncomps,loss_col)):
    end_loss.append(np.mean(loss_traj[-50:]))
    # loss_traj_smooth = np.convolve(loss_traj, np.ones(25)/25, mode="valid")
    # plt.semilogx(loss_traj_smooth, label=f"Ncomp={Ncomps[i]}")
plt.semilogx(Ncomps, end_loss, marker="o")
plt.xlabel("Ncomp")
plt.ylabel("Loss")
plt.title("Loss vs. Ncomp")
plt.show()
# %%
for ndim in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    Wtmp = nn.Linear(ndim, ndim)
    print(ndim, (Wtmp.weight.mean()).item(), "+-",(Wtmp.weight.std()** 2 * ndim).item())
# %%
