# %%


# %%
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Define the transformation to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# Download and load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='~/Datasets', train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.MNIST(root='~/Datasets', train=False, transform=transform)

# Create data loaders
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# %% [markdown]
# ## EDM vs Analytical Score

# %%
def load_create_edm(config, model_path):
    # model_path = f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_75000.pth"
    unet = create_model(config)
    edm = EDM(model=unet, cfg=config)
    checkpoint = torch.load(model_path, map_location=device)
    # logger.info(f"loaded model: {model_name}")
    edm.model.load_state_dict(checkpoint)
    for param in edm.model.parameters():
        param.requires_grad = False
    edm.model.eval()
    return edm

model_path = f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_75000.pth"
unet = create_model(config)
edm = EDM(model=unet, cfg=config)
checkpoint = torch.load(model_path, map_location=device)
# logger.info(f"loaded model: {model_name}")
edm.model.load_state_dict(checkpoint)
for param in edm.model.parameters():
    param.requires_grad = False
edm.model.eval();

# %% [markdown]
# ### Load and Examine the EDM trained model w.r.t. scores

# %%
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
# %% [markdown]
# ### Util functions for 2d slicing projection

class CoordSystem:
    
    def __init__(self, basis1, basis2, origin=None):
        # Ensure the basis vectors are normalized and orthogonal
        # if origin is None:
        #     origin = torch.zeros_like(basis1)
        self.reference = origin
        self.basis1 = basis1 / torch.norm(basis1)
        self.basis2 = basis2 / torch.norm(basis2)
        self.basis_matrix = torch.stack([self.basis1, self.basis2], dim=0)

    def project_vector(self, vectors):
        # Project a vector onto the basis using matrix algebra
        return (vectors @ self.basis_matrix.T)

    def ortho_project_vector(self, vectors):
        # Project a vector onto the perpendicular space of the basis using matrix algebra
        return vectors - (vectors @ self.basis_matrix.to(vectors.dtype).T) @ self.basis_matrix.to(vectors.dtype)
    
    def project_points(self, points):
        # Project a set of points onto the basis
        return (points - self.reference) @ self.basis_matrix.T


def orthogonal_grid(x1, x2, x3, grid_size):
    # Step 1: Find the Basis Vectors
    v1 = x2 - x1
    v = x3 - x1
    # Orthogonalize v with respect to v1 using the Gram-Schmidt process
    proj_v1_v = np.dot(v, v1) / np.dot(v1, v1) * v1
    v2 = v - proj_v1_v

    # Step 2: Normalize the Basis Vectors
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)

    # Step 3: Create the Grid Points
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Scaling factors for v1 and v2
            scale_v1 = i / (grid_size - 1)
            scale_v2 = j / (grid_size - 1)
            # Generate the grid point
            grid_point = x1 + scale_v1 * v1_normalized + scale_v2 * v2_normalized
            grid_points.append(grid_point)

    return np.array(grid_points)

def orthogonal_grid_torch(x1, x2, x3, grid_nums=(10, 10), 
                          x_range=(0, 1), y_range=(0, 1)):
    # Step 1: Find the Basis Vectors
    v1 = x2 - x1
    v = x3 - x1
    # Orthogonalize v with respect to v1 using the Gram-Schmidt process
    proj_v1_v = torch.dot(v, v1) / torch.dot(v1, v1) * v1
    v2 = v - proj_v1_v
    v1_norm = torch.norm(v1)
    v2_norm = torch.norm(v2)
    # Step 2: Normalize the Basis Vectors
    v1_normalized = v1 / v1_norm
    v2_normalized = v2 / v2_norm
    coordsys = CoordSystem(v1_normalized, v2_normalized, origin=x1)
    # Step 3: Create the Grid Points
    grid_vecs = []
    norm_coords = []
    plane_coords = []
    for ti in torch.linspace(x_range[0], x_range[1], grid_nums[0]):
        for tj in torch.linspace(y_range[0], y_range[1], grid_nums[1]):
            # Scaling factors for v1 and v2
            scale_v1 = ti * v1_norm
            scale_v2 = tj * v2_norm
            # Generate the grid point
            grid_vec = x1 + scale_v1 * v1_normalized + scale_v2 * v2_normalized
            grid_vecs.append(grid_vec)
            norm_coords.append([ti, tj])
            plane_coords.append([scale_v1, scale_v2])
    return torch.stack(grid_vecs), \
            torch.tensor(norm_coords), \
            torch.tensor(plane_coords), coordsys

# %% [markdown]
# ### Visualization pipeline function

# %%
from core.analytical_score_lib import mean_isotropic_score, Gaussian_score, delta_GMM_score
from core.analytical_score_lib import explained_var_vec
from core.analytical_score_lib import sample_Xt_batch, sample_Xt_batch
from core.gaussian_mixture_lib import gaussian_mixture_score_batch_sigma_torch, \
    gaussian_mixture_lowrank_score_batch_sigma_torch, compute_cluster

def score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape,
                           sigmas=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0], titlestr="",
                           grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25)):
    anchors_tsr = torch.stack(anchors)
    grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
            *anchors, grid_nums=grid_nums, x_range=x_range, y_range=y_range)

    mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=grid_nums[1])
    plt.figure(figsize=(10, 10))
    plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()
    for sigma in sigmas:
        edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
        score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
        score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
        score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
        score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
        # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
        # Calculate the vector field
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                            ("gmm delta", score_gmm_Xt),
                            ("gaussian", score_gaussian_Xt), 
                            # ("gaussian regularize", score_gaussian_reg_Xt),
                            ("mean isotropic", score_mean_Xt), 
                            # ("mean + std isotropic", score_mean_std_Xt), 
                            # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                            # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                            ]):
            # Create a grid for the quiver plot
            vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
            anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
            axs[i].invert_yaxis()
            axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
            axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
            axs[i].set_aspect('equal')
            axs[i].set_title(name)
        plt.suptitle(f"score vector field {titlestr}\nsigma={sigma:f}")
        plt.show()

# %%
def score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="l2",
                           sigmas=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0], titlestr="",
                           grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25),
                           show_image=True):
    anchors_tsr = torch.stack(anchors)
    grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
            *anchors, grid_nums=grid_nums, x_range=x_range, y_range=y_range)
    if show_image:
        mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=grid_nums[1])
        plt.figure(figsize=(10, 10))
        plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.show()
    for sigma in sigmas:
        edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
        score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
        score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
        score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
        score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
        # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
        # Calculate the vector field
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                            ("gmm delta", score_gmm_Xt),
                            ("gaussian", score_gaussian_Xt), 
                            # ("gaussian regularize", score_gaussian_reg_Xt),
                            ("mean isotropic", score_mean_Xt), 
                            # ("mean + std isotropic", score_mean_std_Xt), 
                            # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                            # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                            ]):
            # Create a grid for the quiver plot
            
            anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
            axs[i].invert_yaxis()
            if magnitude == "l2":
                vec_norm = torch.norm(vector_field, dim=-1).cpu().numpy()
            elif magnitude == "proj_l2":
                vec_proj = coordsys.project_vector(vector_field)
                vec_norm = torch.norm(vec_proj, dim=-1).cpu().numpy()
            elif magnitude == "ortho_l2":
                vec_ortho = coordsys.ortho_project_vector(vector_field.double())
                vec_norm = torch.norm(vec_ortho, dim=-1).cpu().numpy()
            # axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
            im = axs[i].imshow(vec_norm.reshape(grid_nums), origin="lower", 
                          extent=[plane_coords[:, 1].min(), plane_coords[:, 1].max(), 
                                  plane_coords[:, 0].min(), plane_coords[:, 0].max(), ],
                          vmin=0, vmax=vec_norm.max())
            axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
            axs[i].invert_yaxis()
            axs[i].set_aspect('equal')
            axs[i].set_title(name)
            # add colorbar
            # divider = make_axes_locatable(axs[i])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, )
            # cbar.set_clim(0, vec_norm.max())
        plt.suptitle(f"score vector field {titlestr}\nsigma={sigma:f} magnitude={magnitude}")
        plt.show()

# %%
fid_batch_size = 100
with torch.no_grad():
    noise = torch.randn([fid_batch_size, config.channels, config.img_size, config.img_size],
                        generator=torch.cuda.manual_seed(config.seed), device=config.device)
    samples = edm_sampler(edm, noise, num_steps=config.total_steps, use_ema=False).detach().cpu()
    samples.mul_(0.5).add_(0.5)
samples = torch.clamp(samples, 0., 1.).cpu()
# samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
# samples = samples.reshape((-1, config.img_size, config.img_size, config.channels))
# all_samples.append(samples)

plt.figure(figsize=(10, 10))
plt.imshow((make_grid(samples*255.0, nrow=10).permute(1, 2, 0)).numpy().astype(np.uint8))
plt.axis('off')
plt.show()

# %% [markdown]
# ### Load Dataset in EDM convention

# %%
transform = transforms.Compose([
    torchvision.transforms.Resize(32), # config.img_size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Download and load the MNIST dataset
train_edm_dataset = torchvision.datasets.MNIST(root='~/Datasets', train=True, transform=transform, download=False)
test_edm_dataset = torchvision.datasets.MNIST(root='~/Datasets', train=False, transform=transform)

edm_Xtsr = torch.stack([train_edm_dataset[i][0] for i in range(len(train_edm_dataset))])
edm_Xmat = edm_Xtsr.view(edm_Xtsr.shape[0], -1).cuda()
edm_Xtsr_test = torch.stack([test_edm_dataset[i][0] for i in range(len(test_edm_dataset))])
edm_Xmat_test = edm_Xtsr_test.view(edm_Xtsr_test.shape[0], -1)
ytsr_test = torch.tensor(test_edm_dataset.targets)
edm_imgshape = tuple(edm_Xtsr.shape[1:])
edm_Xmean = edm_Xmat.mean(dim=0)
edm_Xcov = torch.cov(edm_Xmat.T, )

eigvals, eigvecs = torch.linalg.eigh(edm_Xcov)
eigvals = torch.flip(eigvals, dims=(0,))
eigvecs = torch.flip(eigvecs, dims=(1,))
print(eigvals.shape, eigvecs.shape)
print(eigvals[0:10].sqrt())

# %% [markdown]
# ## Visualize learned scores by projection


# %%
# TODO Define test function for the score projection
torch.allclose(coordsys.ortho_project_vector(grid_vecs) + 
               coordsys.project_vector(grid_vecs) @ coordsys.basis_matrix, 
               grid_vecs, atol=1E-5)

# %%
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        edm_Xmat[0, :], edm_Xmat[1, :], edm_Xmat[2, :], 
        grid_nums=(16, 16), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))
assert torch.allclose(plane_coords.cuda(), coordsys.project_points(grid_vecs))
plt.scatter(plane_coords[:, 0], plane_coords[:, 1])

# %%
# Compute score on the grid points
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        edm_Xmat[0, :], edm_Xmat[1, :], edm_Xmat[2, :], 
        grid_nums=(16, 16), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))
sigma = 0.2
edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)

# %% [markdown]
# ### Demo visualization

# %%
idxs = np.random.choice(edm_Xmat.shape[0], 3, replace=False)
anchors = [edm_Xmat[idxs[0], :], edm_Xmat[idxs[1], :], edm_Xmat[idxs[2], :]]
anchors_tsr = torch.stack(anchors)
print(f"Score vector field among 3 random samples, {idxs} labels {ytsr[idxs].numpy()}")
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        *anchors, grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))

mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=25)
plt.figure(figsize=(10, 10))
plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
for sigma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
    score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
    score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
    score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
    # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
    # Calculate the vector field
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                        ("gmm delta", score_gmm_Xt),
                        ("gaussian", score_gaussian_Xt), 
                        # ("gaussian regularize", score_gaussian_reg_Xt),
                        ("mean isotropic", score_mean_Xt), 
                        # ("mean + std isotropic", score_mean_std_Xt), 
                        # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                        # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                        ]):
        # Create a grid for the quiver plot
        vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
        anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
        axs[i].invert_yaxis()
        axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
        axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
        axs[i].set_aspect('equal')
        axs[i].set_title(name+f" score vector field\nsigma={sigma:f}")
    plt.show()


# %%
idxs = np.random.choice(edm_Xmat.shape[0], 3, replace=False)
anchors = [edm_Xmat[idxs[0], :], edm_Xmat[idxs[1], :], edm_Xmat[idxs[2], :]]
anchors_tsr = torch.stack(anchors)
print(f"Score vector field among 3 random samples, {idxs} labels {ytsr[idxs].numpy()}")
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        *anchors, grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))

mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=25)
plt.figure(figsize=(10, 10))
plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
for sigma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 8.0]:
    edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
    score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
    score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
    score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
    # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
    # Calculate the vector field
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                        ("gmm delta", score_gmm_Xt),
                        ("gaussian", score_gaussian_Xt), 
                        # ("gaussian regularize", score_gaussian_reg_Xt),
                        ("mean isotropic", score_mean_Xt), 
                        # ("mean + std isotropic", score_mean_std_Xt), 
                        # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                        # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                        ]):
        # Create a grid for the quiver plot
        vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
        anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
        axs[i].invert_yaxis()
        axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
        axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
        axs[i].set_aspect('equal')
        axs[i].set_title(name+f" score vector field\nsigma={sigma:f}")
    plt.show()


# %%
import matplotlib.pyplot as plt
dist2anchor = torch.cdist(edm_Xmat[0:1, :], edm_Xmat)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
# anchors = [edm_Xmat[idxs[0], :], edm_Xmat[idxs[1], :], edm_Xmat[idxs[2], :]]
anchors_tsr = torch.stack(anchors)
print(f"Score vector field among 3 random samples, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}")
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        *anchors, grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))

mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=25)
plt.figure(figsize=(10, 10))
plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
for sigma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
    score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
    score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
    score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
    # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
    # Calculate the vector field
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                        ("gmm delta", score_gmm_Xt),
                        ("gaussian", score_gaussian_Xt), 
                        # ("gaussian regularize", score_gaussian_reg_Xt),
                        ("mean isotropic", score_mean_Xt), 
                        # ("mean + std isotropic", score_mean_std_Xt), 
                        # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                        # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                        ]):
        # Create a grid for the quiver plot
        vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
        anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
        axs[i].invert_yaxis()
        axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
        axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
        axs[i].set_aspect('equal')
        axs[i].set_title(name+f" score vector field\nsigma={sigma:f}")
    plt.show()


# %%
epoch = 200000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")


# %% [markdown]
# #### Off plane component of score vector

# %%
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
# knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )

# %%
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
# knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )

# %%
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
# knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )

# %%
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
dist2anchor = torch.cdist(edm_Xmat_test[12:13, :], edm_Xmat_test)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 nearest samples in TEST set, {knnidx[0].cpu()} labels {ytsr_test[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )

# %%
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
dist2anchor = torch.cdist(edm_Xmat[10:11, :], edm_Xmat)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 nearest samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )


# %% [markdown]
# #### Training set, 3 Random samples 

# %%
epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, 
                       titlestr=titlestr,)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), show_image=False)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), show_image=False)

# %% [markdown]
# #### Training set, nearest neighbors to one point

# %%

epoch = 300000
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
dist2anchor = torch.cdist(edm_Xmat[10:11, :], edm_Xmat)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 nearest samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, 
                       titlestr=titlestr,)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), show_image=False)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), show_image=False)

# %% [markdown]
# #### Test set, nearest neighbors to one point

# %%
epoch = 300000
edm_Xmat_test = edm_Xmat_test.to(device)
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
dist2anchor = torch.cdist(edm_Xmat_test[12:13, :], edm_Xmat_test)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 nearest samples in TEST set, {knnidx[0].cpu()} labels {ytsr_test[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, 
                       titlestr=titlestr,)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), show_image=False)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), show_image=False)

# %% [markdown]
# #### Test set: random point samples

# %%
epoch = 300000
edm_Xmat_test = edm_Xmat_test.to(device)
edm = load_create_edm(config, f"/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth")
# dist2anchor = torch.cdist(edm_Xmat_test[10:11, :], edm_Xmat_test)
# knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
# knnidx = torch.randint(edm_Xmat_test.shape[0], (1, 3))
knnidx = torch.tensor([[8847, 8929, 9063]])
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TEST set, {knnidx[0].cpu()} labels {ytsr_test[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, 
                       titlestr=titlestr,)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), show_image=False)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), show_image=False)

# %% [markdown]
# ### Combined Vector field comparison

# %%
import matplotlib.pyplot as plt
anchors = [edm_Xmat[0, :], edm_Xmat[1, :], edm_Xmat[4, :]]
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        *anchors, grid_nums=(16, 16), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))
anchors_tsr = torch.stack(anchors)

# sigma = 1.0
for sigma in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 3.5, 5.0]:
    edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
    score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
    score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
    score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
    # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
    # Calculate the vector field

    fig, ax = plt.subplots()
    ax.scatter(coordsys.project_points(anchors_tsr).cpu()[:, 0],
            coordsys.project_points(anchors_tsr).cpu()[:, 1], color='r', marker='x')
    for name, vector_field, clr in [("edm", score_edm, "black"),
                        ("gmm delta", score_gmm_Xt, "red"),
                        ("gaussian", score_gaussian_Xt, "blue"), 
                        # ("gaussian regularize", score_gaussian_reg_Xt),
                        # ("mean isotropic", score_mean_Xt), 
                        # ("mean + std isotropic", score_mean_std_Xt), 
                        # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                        # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                        ]:
        vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
        ax.quiver(plane_coords[:, 0], plane_coords[:, 1], vec_proj[:, 0], vec_proj[:, 1], 
                label=name, color=clr, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_title(f"Score vector field comparison\nsigma={sigma:.2f}")
    plt.legend()
    plt.show()



