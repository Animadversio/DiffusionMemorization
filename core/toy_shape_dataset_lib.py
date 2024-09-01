import torch
import numpy as np

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


def random_radius_function(angles, amplitudes, phases, R0=1):
    num_modes = len(amplitudes)
    freqs = np.arange(1, num_modes+1)
    radii = R0 + np.sum(amplitudes[None, :] * np.cos(freqs[None, :] * angles[:,None] + phases[None, :]), axis=1)
    return radii


def create_random_star_shape(num_points, num_modes):
    amplitudes = 0.4 * np.random.rand(num_modes) - 0.2
    phases = 2 * np.pi * np.random.rand(num_modes)
    angles = np.linspace(0, 2*np.pi, num_points)
    radii = random_radius_function(angles, amplitudes, phases)
    radius_fun = lambda angles: random_radius_function(angles, amplitudes, phases)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    pnts = np.vstack((x, y)).T
    return pnts, radius_fun, amplitudes, phases


def generate_random_star_shape_torch(num_points, num_modes=10):
    pnts, radius_fun, amplitudes, phases = create_random_star_shape(num_points, num_modes)
    return torch.from_numpy(pnts), radius_fun, amplitudes, phases

