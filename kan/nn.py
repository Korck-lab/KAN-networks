import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Set global seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



class SplineActivation(nn.Module):
    def __init__(self, num_knots=10):
        super(SplineActivation, self).__init__()
        self.num_knots = num_knots
        self.knots = nn.Parameter(torch.linspace(-1, 1, num_knots))
        self.coefficients = nn.Parameter(torch.zeros(num_knots))

    def forward(self, x):
        x_clipped = torch.clamp(x, -1, 1)
        spline_values = torch.zeros_like(x_clipped)
        for i in range(self.num_knots):
            spline_values += self.coefficients[i] * torch.abs(x_clipped - self.knots[i])
        return spline_values

class KAN(nn.Module):

    # He initialization function
    def _he_init(self):
        for layer in [self.phi1, self.psi1, self.phi2, self.psi2, self.phi3, self.psi3]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    # Xavier initialization function
    def _xavier_init(self):
        for layer in [self.phi1, self.psi1, self.phi2, self.psi2, self.phi3, self.psi3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def __init__(self, input_dim, hidden_dim, kernel_type='spline', num_knots=10):
        super(KAN, self).__init__()
        self.phi1 = nn.Linear(input_dim, hidden_dim)
        self.psi1 = nn.Linear(input_dim, hidden_dim)
        self.phi2 = nn.Linear(hidden_dim, hidden_dim)
        self.psi2 = nn.Linear(hidden_dim, hidden_dim)
        self.phi3 = nn.Linear(hidden_dim, 1)
        self.psi3 = nn.Linear(hidden_dim, 1)
        self._he_init()

        if kernel_type == 'spline':
            self.spline1 = SplineActivation(num_knots)
            self.spline2 = SplineActivation(num_knots)
            self.spline3 = SplineActivation(num_knots)
            self.spline4 = SplineActivation(num_knots)
            self.spline5 = SplineActivation(num_knots)
            self.spline6 = SplineActivation(num_knots)
        elif kernel_type == 'piecewise':
            self.spline1 = PiecewiseLinearActivation(num_knots)
            self.spline2 = PiecewiseLinearActivation(num_knots)
            self.spline3 = PiecewiseLinearActivation(num_knots)
            self.spline4 = PiecewiseLinearActivation(num_knots)
            self.spline5 = PiecewiseLinearActivation(num_knots)
            self.spline6 = PiecewiseLinearActivation(num_knots)
        elif kernel_type == 'relu':
            self.spline1 = nn.ReLU()
            self.spline2 = nn.ReLU()
            self.spline3 = nn.ReLU()
            self.spline4 = nn.ReLU()
            self.spline5 = nn.ReLU()
            self.spline6 = nn.ReLU()
        elif kernel_type == 'sigmoid':
            self.spline1 = nn.Sigmoid()
            self.spline2 = nn.Sigmoid()
            self.spline3 = nn.Sigmoid()
            self.spline4 = nn.Sigmoid()
            self.spline5 = nn.Sigmoid()
            self.spline6 = nn.Sigmoid()
        elif kernel_type == 'tanh':
            self.spline1 = nn.Tanh()
            self.spline2 = nn.Tanh()
            self.spline3 = nn.Tanh()
            self.spline4 = nn.Tanh()
            self.spline5 = nn.Tanh()
            self.spline6 = nn.Tanh()

        # Apply He initialization to all layers
        # self.apply(he_init)

    # Custom normalization function
    @staticmethod
    def custom_layer_norm(outputs):
        mean = torch.mean(outputs, dim=0, keepdim=True)
        std = torch.std(outputs, dim=0, keepdim=True)
        return (outputs - mean) / std

    def forward(self, x):
        def apply_activation(act, x):
            if isinstance(act, SplineActivationWithDerivative):
                return act(x)[0]  # Get the spline values, ignore derivatives
            else:
                return act(x)

        x1 = apply_activation(self.spline1, self.phi1(x))
        x2 = apply_activation(self.spline2, self.psi1(x))
        sum_x = x1 + x2
        x3 = apply_activation(self.spline3, self.phi2(sum_x))
        x4 = apply_activation(self.spline4, self.psi2(sum_x))
        sum_x2 = x3 + x4
        x5 = apply_activation(self.spline5, self.phi3(sum_x2))
        x6 = apply_activation(self.spline6, self.psi3(sum_x2))
        output = x5 + x6
        return torch.sigmoid(output)

def train_model(model, X_train, y_train, epochs=2000, learning_rate=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model, loss_list

# Generate synthetic data
X, y = make_circles(n_samples=2000, noise=0.2, factor=0.2)
y = y.reshape(-1, 1)

X_train, X_test = torch.FloatTensor(X)
y_train, y_test = torch.FloatTensor(y)

# Train the models with different kernels
kernels = ['spline', 'relu', 'sigmoid', 'tanh', 'piecewise']
models = {}
losses = {}
input_dim, hidden_dim = 2, 3
num_knots = 10
epochs, learning_rate = 1000, 0.01

for kernel in kernels:
    print(f"\nTraining KAN model with {kernel} kernel:")
    kan_model = KAN(input_dim, hidden_dim, kernel_type=kernel, num_knots=num_knots)
    trained_model, loss_list = train_model(kan_model, X_train, y_train, epochs=epochs, learning_rate=learning_rate)
    models[kernel] = trained_model
    losses[kernel] = loss_list

# Plot the loss curves
plt.figure(figsize=(10, 8))
for kernel, loss_list in losses.items():
    plt.plot(loss_list, label=f'{kernel} kernel')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves for Different KAN Kernels')
plt.show()
c=len(kernels)//2
r=c+len(kernels)%2
aspect_r=(12/2)*c
aspect_c=(10/2)*r

# Evaluate and plot decision boundaries
fig, axes = plt.subplots(r, c, figsize=(aspect_r, aspect_c))

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
for i, kernel in enumerate(kernels):
    model = models[kernel]
    model.eval()
    with torch.no_grad():
        Z = model(grid).reshape(xx.shape)

    ax = axes[i // 2, i % 2]
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='o')
    ax.set_title(f'{kernel} kernel')

plt.tight_layout()
plt.show()

# Evaluate and print accuracy for each model
for kernel, model in models.items():
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        print(f'Accuracy for {kernel} kernel: {accuracy:.4f}')
