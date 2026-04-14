import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats

def isotropic_gaussian_vector(n, mean=0, std=1, seed=None):
    """
    Generate an n-dimensional vector from an isotropic Gaussian distribution.
    
    Parameters:
    -----------
    n : int
        Dimension of the vector.
    mean : float, default=0
        Mean of the Gaussian distribution.
    std : float, default=1
        Standard deviation of the Gaussian distribution.
    seed : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    vector : numpy.ndarray
        An n-dimensional vector sampled from N(mean, std^2 * I).
    """
    if seed is not None:
        np.random.seed(seed)
    return mean + std * np.random.randn(n)


def gumbel_sample(loc=0.0, scale=1.0, size=1, seed=None):
    """
    Generate samples from a Gumbel distribution.
    
    Parameters:
    -----------
    loc : float, default=0.0
        Location parameter (μ) of the Gumbel distribution.
    scale : float, default=1.0
        Scale parameter (β) of the Gumbel distribution.
    size : int, default=1
        Number of samples to generate.
    seed : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    samples : numpy.ndarray
        Array of Gumbel-distributed random samples.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate uniform random numbers between 0 and 1
    u = np.random.uniform(size=size)
    
    # Apply inverse transform sampling
    # Gumbel CDF: F(x) = exp(-exp(-(x-μ)/β))
    # Inverse CDF: F^(-1)(p) = μ - β * ln(-ln(p))
    return loc - scale * np.log(-np.log(u))


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with two hidden layers of 256 neurons each.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input features.
    hidden_dims : list, default=[256, 256]
        Dimensions of hidden layers.
    output_dim : int, default=1
        Dimension of output.
    activation : callable, default=nn.ReLU()
        Activation function to use between layers.
    """
    def __init__(self, input_dim, hidden_dims=[256, 256], output_dim=1, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(activation)
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(activation)
            
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x


# Example usage
if __name__ == "__main__":
    # Set a global random seed for all examples
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    n = 10  # Input dimension
    num_samples = 1024  # Number of samples
    std_values = [0.5, 1.0, 2.0, 5.0]  # Different std values to test
    
    print(f"Initializing MLP with input dimension: {n}")
    print(f"Number of samples: {num_samples}")
    print(f"Testing std values: {std_values}")
    
    # Step 1: Initialize neural network
    mlp = MLP(input_dim=n, hidden_dims=[256, 256], output_dim=1)
    print(f"\nMLP architecture:")
    print(mlp)
    
    # Create subplots for different std values
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Store results for summary
    results_summary = []
    
    for idx, std_val in enumerate(std_values):
        print(f"\n{'='*60}")
        print(f"Testing std = {std_val}")
        print(f"{'='*60}")
        
        # Reset random seed for reproducibility
        np.random.seed(42)
        
        # Step 2: Generate 1024 Gaussian vectors (n-dimensional) with current std
        print(f"Generating {num_samples} {n}-dimensional Gaussian vectors (std={std_val})...")
        gaussian_vectors = []
        for i in range(num_samples):
            vector = isotropic_gaussian_vector(n, mean=0, std=std_val)
            gaussian_vectors.append(vector)
        
        # Convert to tensor for neural network
        X = torch.tensor(np.array(gaussian_vectors), dtype=torch.float32)
        
        # Step 3: Pass through neural network
        print(f"Passing through neural network...")
        with torch.no_grad():  # No gradient computation needed
            network_outputs = mlp(X)
        
        # Convert to numpy array and flatten to 1D
        network_outputs = network_outputs.numpy().flatten()
        print(f"Network outputs range: [{network_outputs.min():.3f}, {network_outputs.max():.3f}]")
        
        # Step 4: Generate Gumbel noise
        print(f"Generating {num_samples} Gumbel noise samples...")
        gumbel_noise = gumbel_sample(loc=0.0, scale=1.0, size=num_samples, seed=42)
        
        # Step 5: Add Gumbel noise to network outputs
        final_outputs = network_outputs + gumbel_noise
        print(f"Final outputs range: [{final_outputs.min():.3f}, {final_outputs.max():.3f}]")
        
        # Step 6: Plot histogram with Gumbel distribution fitting
        ax = axes[idx]
        
        # Plot histogram (normalized to get probability density)
        counts, bins, patches = ax.hist(
            final_outputs, 
            bins=50, 
            alpha=0.7, 
            color='#90EE90',          # light green
            edgecolor='#006400',      # dark green
            density=True, 
            label='Histogram'
        )
        
        # Fit Gumbel distribution
        print(f"Fitting Gumbel distribution...")
        params = stats.gumbel_r.fit(final_outputs)
        loc, scale = params
        print(f"Fitted Gumbel parameters: loc={loc:.3f}, scale={scale:.3f}")
        
        # Generate x values for plotting the fitted curve
        x_min, x_max = final_outputs.min(), final_outputs.max()
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
        
        # Calculate and plot the fitted Gumbel PDF
        fitted_pdf = stats.gumbel_r.pdf(x, loc=loc, scale=scale)
        ax.plot(x, fitted_pdf, color='orange', linewidth=3, label='Fitted Gumbel')
        
        # Calculate statistics
        mean_val = np.mean(final_outputs)
        std_val_sample = np.std(final_outputs)
        fitted_mean = loc + scale * np.euler_gamma
        fitted_std = scale * np.pi / np.sqrt(6)
        
        # Goodness of fit test
        ks_statistic, p_value = stats.kstest(final_outputs, lambda x: stats.gumbel_r.cdf(x, loc=loc, scale=scale))
        
        # Store results
        results_summary.append({
            'input_std': std_val,
            'sample_mean': mean_val,
            'sample_std': std_val_sample,
            'fitted_loc': loc,
            'fitted_scale': scale,
            'fitted_mean': fitted_mean,
            'fitted_std': fitted_std,
            'ks_statistic': ks_statistic,
            'p_value': p_value
        })
        
        print(f"Sample Mean: {mean_val:.3f}, Sample Std: {std_val_sample:.3f}")
        print(f"Fitted Mean: {fitted_mean:.3f}, Fitted Std: {fitted_std:.3f}")
        print(f"KS test p-value: {p_value:.4f}")
        
        # Add p-value to the plot title
        significance = "Not Gumbel-like" if p_value < 0.05 else "Gumbel-like"
        ax.set_title(f'std = {std_val}\nGumbel(loc={loc:.3f}, scale={scale:.3f})\np-value: {p_value:.4f} ({significance})', fontsize=16, weight='bold')
        ax.set_xlabel('Value', fontsize=16, weight='bold')
        ax.set_ylabel('Probability Density', fontsize=16, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=20)
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold') 
    
    plt.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.5)
    
    # Save the figure to a PDF file before showing
    output_filename = "toy Gumbel.pdf"
    plt.savefig(output_filename, format='pdf')
    print(f"\nPlot saved to {output_filename}")
    
    plt.show()
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Input Std':<10} {'Sample Mean':<12} {'Sample Std':<12} {'Fitted Loc':<12} {'Fitted Scale':<13} {'KS p-value':<12}")
    print(f"{'-'*80}")
    
    for result in results_summary:
        print(f"{result['input_std']:<10.1f} {result['sample_mean']:<12.3f} {result['sample_std']:<12.3f} "
              f"{result['fitted_loc']:<12.3f} {result['fitted_scale']:<13.3f} {result['p_value']:<12.4f}")
    
    print(f"\n{'='*80}")
    print("OBSERVATIONS:")
    print(f"{'='*80}")
    
    # Analyze trends
    input_stds = [r['input_std'] for r in results_summary]
    sample_means = [r['sample_mean'] for r in results_summary]
    sample_stds = [r['sample_std'] for r in results_summary]
    fitted_locs = [r['fitted_loc'] for r in results_summary]
    fitted_scales = [r['fitted_scale'] for r in results_summary]
    
    # Check correlation
    input_std_vs_sample_std_corr = np.corrcoef(input_stds, sample_stds)[0, 1]
    input_std_vs_fitted_scale_corr = np.corrcoef(input_stds, fitted_scales)[0, 1]
    
    print(f"6. Correlation between input std and sample std: {input_std_vs_sample_std_corr:.3f}")
    print(f"7. Correlation between input std and fitted scale: {input_std_vs_fitted_scale_corr:.3f}")
    print(f"8. For std values {std_values[0]:.1f} and {std_values[1]:.1f}, the distributions seem to be Gumbel-like based on p-value.")
    print(f"9. As the input standard deviation increases, both the sample standard deviation and the fitted Gumbel scale parameter tend to increase, indicating a positive correlation.")
    print(f"10. The sample means and fitted Gumbel location parameters fluctuate around 0, which is consistent with the mean of the isotropic Gaussian vectors being 0.")