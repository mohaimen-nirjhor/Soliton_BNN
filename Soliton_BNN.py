import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Define 1-soliton F(x,t)
# -----------------------------
def define_soliton_F(k, x, t, delta=0.0):
    theta = k * x - k**3 * t + delta
    return 1 + torch.exp(theta)

# -----------------------------
# Neural Network Architecture
# -----------------------------
class FNet(nn.Module):
    def __init__(self, hidden_dim=20):
        super(FNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensure F(x,t) > 0
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# -----------------------------
# Utility: Compute nth derivative w.r.t x
# -----------------------------
def grad_n(f, x, n):
    for _ in range(n):
        f = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    return f

# -----------------------------
# Full Hirota Bilinear Residual
# -----------------------------
def hirota_residual(model, x, t, k, delta):
    x_flat = x.flatten().view(-1, 1).requires_grad_(True)
    t_flat = t.flatten().view(-1, 1).requires_grad_(True)

    # Analytical F and u
    F_target = define_soliton_F(k, x_flat, t_flat, delta)
    log_F_target = torch.log(F_target)
    u_true = 2 * grad_n(log_F_target, x_flat, 2)

    # Predicted F and u
    F_pred = model(x_flat, t_flat)
    log_F_pred = torch.log(torch.clamp(F_pred, min=1e-6))
    u_pred = 2 * grad_n(log_F_pred, x_flat, 2)

    # Penalty on u error
    u_penalty = torch.mean((u_pred - u_true)**2)

    # MSE between F_pred and F_target
    mse_loss = torch.mean((F_pred - F_target)**2)

    # Direct penalty to encourage learning of true form
    theta = k * x_flat - k**3 * t_flat + delta
    direct_penalty = torch.mean((F_pred - (1 + torch.exp(theta)))**2)

    # Full Hirota Bilinear Operator
    F_x   = grad_n(F_pred, x_flat, 1)
    F_xx  = grad_n(F_pred, x_flat, 2)
    F_xxx = grad_n(F_pred, x_flat, 3)
    F_xxxx = grad_n(F_pred, x_flat, 4)
    F_t   = grad_n(F_pred, t_flat, 1)

    term1 = 2 * (F_xxxx * F_pred - 4 * F_xxx * F_x + 3 * F_xx**2)
    term2 = 2 * (F_xx * F_t - F_x * F_t)
    residual = term1 + term2

    reg = torch.mean(torch.exp(-x_flat**2))  # Optional regularization

    # Total loss
    residual_loss = torch.mean(residual**2)
    total_loss = mse_loss + 1e-4 * (u_penalty + direct_penalty + residual_loss + reg)

    # For visualization
    u_output = 2 * grad_n(torch.log(torch.clamp(F_pred, min=1e-6)), x_flat, 2)

    return total_loss, u_output, F_pred

# -----------------------------
# Initialize Weights
# -----------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -----------------------------
# Hyperparameters & Setup
# -----------------------------
hidden_dim = 20
num_epochs = 1000
lr = 5e-4

model = FNet(hidden_dim=hidden_dim).to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

k = torch.FloatTensor(1).uniform_(0.8, 1.2).to(device)
delta = 0.0
x_vals = torch.linspace(-20, 20, 400, device=device).view(-1, 1)
t_vals = torch.linspace(0, 20, 200, device=device).view(-1, 1)

x, t = torch.meshgrid(x_vals.squeeze(), t_vals.squeeze(), indexing='ij')
x_flat = x.flatten().view(-1, 1).requires_grad_(True)
t_flat = t.flatten().view(-1, 1).requires_grad_(True)

assert x_flat.size(0) == t_flat.size(0), "Mismatch in number of elements between x and t"

# -----------------------------
# Training Loop
# -----------------------------
loss_history = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss, _, _ = hirota_residual(model, x_flat, t_flat, k, delta)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    loss_history.append(loss.item())

    if epoch % 200 == 0:
        print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.4e}, LR: {scheduler.optimizer.param_groups[0]['lr']:.1e}")

# -----------------------------
# Visualization
# -----------------------------
model.eval()
with torch.no_grad():
    _, u_pred, _ = hirota_residual(model, x, t, k, delta)
    U = u_pred.detach().cpu().numpy().reshape(len(x_vals), len(t_vals))

# 1D Profile
plt.figure(figsize=(8, 6))
t_index = 50
plt.plot(x_vals.cpu().numpy(), U[:, t_index], label=f't = {t_vals[t_index].item():.2f}, k = {k.item():.2f}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Soliton Profile at Fixed t')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 3D Surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
X = x.detach().cpu().numpy()
T = t.detach().cpu().numpy()
ax.plot_surface(X, T, U, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Soliton Surface')
plt.tight_layout()
plt.show()

# Contour Plot
plt.figure(figsize=(10, 6))
plt.contourf(X, T, U, levels=50, cmap='coolwarm')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Contour of u(x,t)')
plt.tight_layout()
plt.show()
