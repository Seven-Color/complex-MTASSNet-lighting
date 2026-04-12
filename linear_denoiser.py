"""
Linear Denoiser Implementation based on:
Ghane, R., Akhtiamov, D., & Hassibi, B. (2026). 
Precise Performance of Linear Denoisers in the Proportional Regime.

Key idea: Train a linear denoiser W by injecting synthetic noise Σ₁ 
(different from actual noise Σ_z) to learn denoising directly.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math


class LinearDenoiser(nn.Module):
    """
    Linear denoiser W that learns to map noisy inputs to clean outputs.
    Trained using synthetic noise injection approach from the paper.
    """
    
    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        
        # Learnable linear transformation
        self.W = nn.Linear(self.input_dim, self.output_dim, bias=False)
        
    def forward(self, x_noisy: torch.Tensor) -> torch.Tensor:
        """Denoise noisy input"""
        return self.W(x_noisy)
    
    def get_weight_matrix(self) -> torch.Tensor:
        return self.W.weight


class OptimalLinearDenoiser:
    """
    Compute the optimal linear denoiser in closed form.
    
    Given: clean data x ~ N(0, Σ), noisy data x + z where z ~ N(0, Σ_z)
    Training with synthetic noise Σ_1: x + u where u ~ N(0, Σ_1)
    
    The optimal W minimizes E[||W(x+u) - x||^2]
    """
    
    def __init__(self, sigma: np.ndarray, sigma_1: np.ndarray):
        """
        Args:
            sigma: Covariance matrix of clean data Σ (d×d)
            sigma_1: Synthetic noise covariance Σ₁ (d×d)
        """
        self.sigma = sigma
        self.sigma_1 = sigma_1
        
    def compute_optimal_W(self) -> np.ndarray:
        """
        Closed-form solution for optimal W:
        W* = Σ (Σ + Σ_1)^{-1}
        
        This is the matrix that minimizes E[||Wx - x||²] for input x+u ~ N(0, Σ+Σ_1)
        """
        # Compute (Σ + Σ_1)^{-1}
        sigma_sum = self.sigma + self.sigma_1
        sigma_sum_inv = np.linalg.inv(sigma_sum)
        
        # W* = Σ (Σ + Σ_1)^{-1}
        W_opt = self.sigma @ sigma_sum_inv
        return W_opt
    
    def compute_empirical_W(self, X: np.ndarray, sigma_1: np.ndarray) -> np.ndarray:
        """
        Empirical Wiener filter: estimate Σ from samples, then compute W
        
        W_emp = Σ̂ (Σ̂ + Σ_1)^{-1}
        """
        n, d = X.shape
        # Sample covariance
        sigma_hat = np.cov(X.T, bias=True)
        sigma_sum = sigma_hat + sigma_1
        sigma_sum_inv = np.linalg.inv(sigma_sum)
        W_emp = sigma_hat @ sigma_sum_inv
        return W_emp


class DenoiserTrainer:
    """
    Trainer for the linear denoiser using the synthetic noise injection method.
    """
    
    def __init__(self, input_dim: int, device: str = 'cuda'):
        self.input_dim = input_dim
        self.device = device
        self.model = LinearDenoiser(input_dim).to(device)
        
    def train_step(self, X_clean: torch.Tensor, sigma_1: torch.Tensor, 
                   optimizer: torch.optim.Optimizer) -> float:
        """
        One training step: given clean samples, inject synthetic noise,
        train W to recover clean data.
        
        Args:
            X_clean: Clean data (n, d)
            sigma_1: Synthetic noise covariance (d, d)
            optimizer: Optimizer
        """
        n, d = X_clean.shape
        
        # Generate synthetic noisy samples: x + u where u ~ N(0, Σ_1)
        # Use Cholesky decomposition for sampling
        L = torch.linalg.cholesky(sigma_1)
        noise = torch.randn(n, d, device=self.device)
        X_noisy = X_clean + torch.matmul(noise, L.T)
        
        # Predict clean from noisy
        X_pred = self.model(X_noisy)
        
        # MSE loss: ||W(x+u) - x||^2
        loss = torch.mean((X_pred - X_clean) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(self, X_clean: torch.Tensor, sigma_1: torch.Tensor,
              epochs: int = 1000, lr: float = 0.01) -> list:
        """Train the linear denoiser"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            loss = self.train_step(X_clean, sigma_1, optimizer)
            losses.append(loss)
            
        return losses


class TransformerDenoiser(nn.Module):
    """
    Transformer-based denoiser extending the linear denoiser concept.
    Uses self-attention to capture more complex denoising patterns.
    """
    
    def __init__(self, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection (can be used as linear denoiser)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Learnable "denoising matrix" similar to linear denoiser
        self.D = nn.Parameter(torch.eye(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model) or (batch, d_model)
        """
        # Handle both single vector and sequence cases
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Output projection with learnable denoising
        x = self.output_proj(x)
        
        # Apply learned denoising transform
        x = torch.matmul(x, self.D)
        
        if x.shape[1] == 1:
            x = x.squeeze(1)
            
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DiffusionDenoiser(nn.Module):
    """
    Diffusion model style denoiser combining linear denoiser concept
    with neural network for more complex noise patterns.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        
        self.dim = dim
        
        # Linear denoiser component (as in paper)
        self.linear_denoiser = nn.Linear(dim, dim, bias=False)
        
        # Neural network component for residual
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Learnable mixing coefficient (like σ_1 optimization)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x_noisy: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Denoise noisy input
        
        The linear component captures the optimal linear denoising (W* from paper)
        The MLP learns residual corrections
        """
        # Linear denoising component
        x_linear = self.linear_denoiser(x_noisy)
        
        # Residual learning
        x_residual = self.mlp(x_noisy)
        
        # Mix based on learned parameter (analogous to optimizing over σ_1)
        alpha = torch.sigmoid(self.alpha)
        x_denoised = alpha * x_linear + (1 - alpha) * x_residual
        
        return x_denoised


def compute_generalization_error(W: np.ndarray, sigma: np.ndarray, 
                                  sigma_1: np.ndarray, sigma_z: np.ndarray) -> float:
    """
    Compute theoretical generalization error from the paper.
    
    Error = Tr(W Σ W^T) + Tr(Σ_z) - 2Tr(W Σ)
    
    For the trained denoiser on actual noise distribution.
    """
    # Expected error on clean data + actual noise
    term1 = np.trace(W @ sigma @ W.T)
    term2 = np.trace(sigma_z)
    term3 = 2 * np.trace(W @ sigma)
    
    error = term1 + term2 - term3
    return error


def optimize_sigma_1(sigma: np.ndarray, sigma_z: np.ndarray, 
                      kappa: float = 2.0) -> np.ndarray:
    """
    Optimize the synthetic noise covariance Σ_1 for best denoising.
    
    The paper shows we can optimize Σ_1 to minimize generalization error.
    For isotropic case, this has closed-form solution.
    """
    d = sigma.shape[0]
    
    # For isotropic case: σ_1 = α*I
    # The paper suggests using a multiple of the noise we're trying to remove
    # For optimal performance, Σ_1 should be proportional to Σ_z
    
    # Heuristic: scale sigma_z by kappa (sample complexity ratio)
    sigma_1_opt = kappa * sigma_z
    
    return sigma_1_opt


# ============== Demo / Testing ==============

def demo_linear_denoiser():
    """Demonstrate the linear denoiser"""
    print("=" * 60)
    print("Linear Denoiser Demo")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup
    d = 100  # dimension
    n = 200  # samples (n/d = 2, κ = 2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create synthetic data
    # Generate covariance matrix with decaying eigenvalues (realistic scenario)
    eigenvalues = np.exp(-np.linspace(0, 3, d))
    V = np.random.randn(d, d)
    V, _ = np.linalg.qr(V)
    sigma = V @ np.diag(eigenvalues) @ V.T
    
    # Actual noise covariance
    sigma_z = 0.1 * np.eye(d)
    
    # Synthetic noise for training (key idea from paper)
    sigma_1 = 0.3 * np.eye(d)
    
    # Generate clean samples
    L = np.linalg.cholesky(sigma)
    noise = np.random.randn(d, n)
    X_clean = L @ noise  # (d, n)
    X_clean = X_clean.T  # (n, d)
    
    # Add actual noise for testing
    L_z = np.linalg.cholesky(sigma_z)
    noise_z = L_z @ np.random.randn(d, n)  # (d, n)
    X_noisy_test = X_clean + noise_z.T  # (n, d)
    
    # Convert to torch
    X_clean_tensor = torch.from_numpy(X_clean).float().to(device)
    
    # Train linear denoiser
    trainer = DenoiserTrainer(d, device)
    sigma_1_tensor = torch.from_numpy(sigma_1).float().to(device)
    
    print(f"Training on {n} samples with dimension {d} (κ = {n/d})")
    
    losses = trainer.train(X_clean_tensor, sigma_1_tensor, epochs=500, lr=0.05)
    
    # Get learned W
    W_learned = trainer.model.get_weight_matrix().detach().cpu().numpy()
    
    # Compare with analytical optimal
    opt_denoiser = OptimalLinearDenoiser(sigma, sigma_1)
    W_optimal = opt_denoiser.compute_optimal_W()
    
    # Compute errors
    error_learned = compute_generalization_error(W_learned, sigma, sigma_1, sigma_z)
    error_optimal = compute_generalization_error(W_optimal, sigma, sigma_1, sigma_z)
    
    # Empirical Wiener filter
    W_emp = opt_denoiser.compute_empirical_W(X_clean, sigma_1)
    error_emp = compute_generalization_error(W_emp, sigma, sigma_1, sigma_z)
    
    print(f"\nGeneralization Error (on actual noise):")
    print(f"  Learned denoiser:   {error_learned:.4f}")
    print(f"  Analytical optimal:  {error_optimal:.4f}")
    print(f"  Empirical Wiener:    {error_emp:.4f}")
    print(f"\nW matrix similarity (learned vs optimal): {np.corrcoef(W_learned.flatten(), W_optimal.flatten())[0,1]:.4f}")
    
    # Test on noisy data
    X_noisy_tensor = torch.from_numpy(X_noisy_test).float().to(device)
    X_denoised = trainer.model(X_noisy_tensor).detach().cpu().numpy()
    
    # Compute denoising quality
    snr_before = np.mean(X_noisy_test ** 2) / np.mean((X_noisy_test - X_clean) ** 2)
    snr_after = np.mean(X_denoised ** 2) / np.mean((X_denoised - X_clean) ** 2)
    
    print(f"\nDenoising results:")
    print(f"  SNR before: {10 * np.log10(snr_before):.2f} dB")
    print(f"  SNR after:  {10 * np.log10(snr_after):.2f} dB")
    
    return trainer.model


def demo_transformer_denoiser():
    """Demonstrate transformer-based denoiser"""
    print("\n" + "=" * 60)
    print("Transformer Denoiser Demo")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    d_model = 64
    batch_size = 16
    seq_len = 8
    
    model = TransformerDenoiser(d_model=d_model, nhead=4, num_layers=2)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Single vector test
    x_single = torch.randn(batch_size, d_model)
    output_single = model(x_single)
    print(f"Single input shape: {x_single.shape}")
    print(f"Single output shape: {output_single.shape}")
    
    return model


def demo_diffusion_denoiser():
    """Demonstrate diffusion-style denoiser"""
    print("\n" + "=" * 60)
    print("Diffusion Denoiser Demo")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    dim = 128
    batch_size = 32
    
    model = DiffusionDenoiser(dim=dim, hidden_dim=256, num_layers=3)
    
    x_noisy = torch.randn(batch_size, dim)
    x_denoised = model(x_noisy)
    
    print(f"Input shape: {x_noisy.shape}")
    print(f"Output shape: {x_denoised.shape}")
    print(f"Alpha (mixing coefficient): {torch.sigmoid(model.alpha).item():.4f}")
    
    # Test with timestep (for diffusion compatibility)
    timestep = torch.tensor([0.5])
    x_denoised_t = model(x_noisy, timestep)
    print(f"With timestep - output shape: {x_denoised_t.shape}")
    
    return model


if __name__ == "__main__":
    print("Linear Denoiser Implementation")
    print("Based on: Ghane et al. (2026) - Precise Performance in Proportional Regime\n")
    
    # Run demos
    linear_model = demo_linear_denoiser()
    transformer_model = demo_transformer_denoiser()
    diffusion_model = demo_diffusion_denoiser()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)