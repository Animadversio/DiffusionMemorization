#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Initialize the first layer weights and biases
        thetas = torch.linspace(0, 2 * np.pi, hidden_dim)
        init_weights = torch.stack((torch.cos(thetas), torch.sin(thetas)), dim=1)  # Corrected shape
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc1.weight = nn.Parameter(init_weights)
        self.fc1.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Initialize the second layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def generate_ellipsoid_dataset(n_points, a, b, noise=0.1):
    """
    Generate a dataset of points on an ellipsoid centered at the origin.
    
    Parameters:
    - n_points: Number of points to generate.
    - a, b: Semi-major and semi-minor axes of the ellipsoid.
    - noise: Standard deviation of Gaussian noise to be added to the ellipsoid values.
    
    Returns:
    - A tuple of tensors (inputs, targets), where inputs are the coordinates of the points,
      and targets are the ellipsoid values (ideally close to 1, with some noise).
    """
    angles = torch.linspace(0, 2 * np.pi, n_points)
    x = a * torch.cos(angles) + torch.randn(n_points) * noise
    y = b * torch.sin(angles) + torch.randn(n_points) * noise
    inputs = torch.stack([x, y], dim=1)
    # targets = torch.ones(n_points,2) + torch.randn(n_points,2) * noise
    targets = inputs  + torch.randn(n_points,2) * noise # torch.zeros(n_points,2)
    return inputs, targets

#%%
# Generate the dataset
a, b = 1, 1  # Semi-axes lengths for the ellipsoid
n_points = 1000
inputs, targets = generate_ellipsoid_dataset(n_points, a, b, noise=0.0)
# Define the network, loss function, and optimizer
criterion = nn.MSELoss()
# Recreate the model with the corrected class
model = TwoLayerNN(input_dim=2, hidden_dim=500, output_dim=2)
# Redefine the optimizer because the model parameters have changed
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 1000
# Retrain the model with the corrected architecture
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs.float())
    loss = criterion(outputs.squeeze(), targets.float())
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

final_loss = loss.item()
final_loss
#%%
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dynamics_system(t, x, model):
    """
    Defines the dynamics of the system: dx/dt = -x + mlp(x).
    
    Parameters:
    - t: The time variable (not used in the equation but required by solve_ivp).
    - x: The state vector of the system.
    - model: The trained PyTorch model.
    
    Returns:
    - The derivative of x with respect to time.
    """
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add batch dimension
    mlp_output = model(x_tensor).squeeze().detach().numpy()  # Get MLP output and convert back to numpy
    dxdt =  -x + mlp_output # 
    return dxdt

# Time span for the simulation
t_span = [0, 10]
t_eval = np.linspace(*t_span, 1000)
# generate random initial conditions
n_points = 500
x0 = torch.randn(n_points, 2)
# solve the system for each initial condition
solutions = []
for i in trange(n_points):
    solution = solve_ivp(dynamics_system, t_span, x0[i], args=(model,), t_eval=t_eval, method='RK45')
    solutions.append(solution.y)
solution_tsr = torch.tensor(solutions)

# Plotting the dynamics of the system
plt.figure(figsize=(6, 6))
for i in range(0,n_points,10):
    plt.plot(solution_tsr[i, 0], solution_tsr[i, 1], alpha=0.5)
plt.title('Phase Space Trajectories')


#%%
# plot the dynmic system as vector field.
xgrid = np.linspace(-2.5, 2.5, 20)
ygrid = np.linspace(-2.5, 2.5, 20)
XX, YY = np.meshgrid(xgrid, ygrid)
U, V = np.zeros_like(XX), np.zeros_like(YY)
pnts = np.stack([XX, YY], axis=-1)
pnts_tsr = pnts.reshape(-1,2)
mn = model(torch.tensor(pnts_tsr, dtype=torch.float32)).detach().numpy()
U = mn[:,0].reshape(20,20)
V = mn[:,1].reshape(20,20)
plt.figure(figsize=(6,6))
plt.quiver(XX, YY, U - XX, V - YY)
# plot the 
plt.plot(inputs[:,0],inputs[:,1])
plt.axis("image")
plt.show()

#%%
# piece-wise linear dynamic system vector field.




# %%
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.scatter(solution_tsr[:,0,0], solution_tsr[:,1,0], cmap='coolwarm')
plt.subplot(1,2,2)
plt.scatter(solution_tsr[:,0,-1], solution_tsr[:,1,-1], cmap='coolwarm')
plt.show()
# %%

# Initial conditions for the system
x0 = [3, 2]

# Solve the system
solution = solve_ivp(dynamics_system, t_span, x0, args=(model,), t_eval=t_eval, method='RK45')

# Plotting the dynamics of the system
plt.figure(figsize=(12, 5))

# Trajectory over time
plt.subplot(1, 2, 1)
plt.plot(t_eval, solution.y[0], label='$x_1$', alpha=0.5)
plt.plot(t_eval, solution.y[1], label='$x_2$', alpha=0.5)
plt.title('Dynamics Over Time')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()

# Phase space trajectory
plt.subplot(1, 2, 2)
plt.plot(solution.y[0], solution.y[1], label='Trajectory')
plt.title('Phase Space Trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

plt.tight_layout()
plt.show()

#%%