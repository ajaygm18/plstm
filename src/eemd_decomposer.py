"""
Signal Decomposition Module using Ensemble Empirical Mode Decomposition (EEMD)
Based on: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"
Implementation of Algorithm 1 from the paper
"""

import numpy as np
import pandas as pd
from PyEMD.EEMD import EEMD as EEMDClass
from PyEMD.EMD import EMD as EMDClass
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

class EEMDDecomposer:
    """
    Implements Ensemble Empirical Mode Decomposition as described in the paper
    Decomposes stock price signals into Intrinsic Mode Functions (IMFs)
    """
    
    def __init__(self, n_trials=100, noise_scale=0.005, max_imf=-1):
        """
        Initialize EEMD decomposer
        
        Parameters:
        n_trials: Number of trials for ensemble
        noise_scale: Standard deviation of Gaussian noise
        max_imf: Maximum number of IMFs (-1 for automatic)
        """
        self.n_trials = n_trials
        self.noise_scale = noise_scale
        self.max_imf = max_imf
        self.eemd = EEMDClass(trials=n_trials, noise_width=noise_scale, ext_EMD=EMDClass())
        
    def decompose_signal(self, signal):
        """
        Decompose signal into IMFs using EEMD
        
        Parameters:
        signal: 1D array of price data
        
        Returns:
        imfs: Array of IMFs
        residue: Residual component
        """
        # Ensure signal is 1D numpy array
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        # Convert to numpy array and ensure it's 1D
        signal = np.array(signal).flatten()
        
        # Check if signal has enough points
        if len(signal) < 10:
            print(f"Warning: Signal too short ({len(signal)} points). Need at least 10 points.")
            return np.array([signal])
        
        # Decompose signal
        try:
            # Use simple EMD if EEMD fails
            try:
                imfs = self.eemd.eemd(signal, max_imf=self.max_imf)
            except:
                print("EEMD failed, falling back to EMD...")
                emd = EMDClass()
                imfs = emd.emd(signal, max_imf=self.max_imf)
            
            if len(imfs) == 0:
                return np.array([signal])
                
            return imfs
        except Exception as e:
            print(f"Error in EEMD decomposition: {str(e)}")
            # Return original signal as single IMF if decomposition fails
            return np.array([signal])
    
    def calculate_sample_entropy(self, time_series, m=2, r_factor=0.2):
        """
        Calculate Sample Entropy for each IMF
        Used to identify the most complex/noisy component
        
        Parameters:
        time_series: 1D array
        m: Pattern length (default=2)
        r_factor: Tolerance factor (default=0.2)
        
        Returns:
        sample_entropy: Float value of sample entropy
        """
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([time_series[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j], m) <= r:
                        C[i] += 1.0
                        
            phi = np.mean(np.log(C / float(N - m + 1.0)))
            return phi
        
        N = len(time_series)
        
        if N < m + 1:
            return float('inf')
            
        r = r_factor * np.std(time_series, ddof=1)
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return float('inf')
    
    def filter_signal(self, signal, remove_highest_entropy=True):
        """
        Filter signal by removing the IMF with highest sample entropy
        As described in the paper methodology
        
        Parameters:
        signal: 1D array of price data
        remove_highest_entropy: Boolean to remove most complex component
        
        Returns:
        filtered_signal: Filtered signal with noise reduced
        imf_entropies: Dictionary of entropy values for each IMF
        removed_imf_index: Index of removed IMF
        """
        # Decompose signal
        imfs = self.decompose_signal(signal)
        
        if imfs is None:
            return signal, {}, -1
        
        # Calculate sample entropy for each IMF
        imf_entropies = {}
        for i, imf in enumerate(imfs):
            entropy = self.calculate_sample_entropy(imf)
            imf_entropies[f'IMF{i+1}'] = entropy
            
        print("Sample Entropy Values:")
        for imf_name, entropy in imf_entropies.items():
            print(f"{imf_name}: {entropy:.4f}")
        
        if remove_highest_entropy and len(imfs) > 1:
            # Find IMF with highest entropy (most noisy)
            entropy_values = [self.calculate_sample_entropy(imf) for imf in imfs[:-1]]  # Exclude residue
            max_entropy_idx = np.argmax(entropy_values)
            
            print(f"Removing IMF{max_entropy_idx+1} with highest entropy: {entropy_values[max_entropy_idx]:.4f}")
            
            # Reconstruct signal without the highest entropy IMF
            filtered_signal = np.sum([imfs[i] for i in range(len(imfs)) if i != max_entropy_idx], axis=0)
            
            return filtered_signal, imf_entropies, max_entropy_idx
        else:
            # Return original signal if no filtering
            return signal, imf_entropies, -1
    
    def visualize_decomposition(self, signal, imfs=None, title="EEMD Decomposition"):
        """
        Visualize the decomposition results
        """
        import matplotlib.pyplot as plt
        
        if imfs is None:
            imfs = self.decompose_signal(signal)
            
        if imfs is None:
            print("Cannot visualize: decomposition failed")
            return
        
        n_imfs = len(imfs)
        fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(12, 2*(n_imfs + 1)))
        
        # Plot original signal
        axes[0].plot(signal, 'b-', linewidth=1)
        axes[0].set_title(f'Original Signal ({title})')
        axes[0].grid(True)
        
        # Plot IMFs
        for i, imf in enumerate(imfs[:-1]):  # Exclude residue
            axes[i+1].plot(imf, 'g-', linewidth=1)
            entropy = self.calculate_sample_entropy(imf)
            axes[i+1].set_title(f'IMF {i+1} (Entropy: {entropy:.4f})')
            axes[i+1].grid(True)
        
        # Plot residue
        if len(imfs) > 1:
            axes[-1].plot(imfs[-1], 'r-', linewidth=1)
            residue_entropy = self.calculate_sample_entropy(imfs[-1])
            axes[-1].set_title(f'Residue (Entropy: {residue_entropy:.4f})')
            axes[-1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def apply_eemd_to_dataset(data, price_column='Close'):
    """
    Apply EEMD filtering to a complete dataset
    
    Parameters:
    data: DataFrame with stock data
    price_column: Column name containing price data
    
    Returns:
    data_with_filtered: DataFrame with additional filtered price column
    """
    decomposer = EEMDDecomposer()
    
    # Get price series
    price_series = data[price_column].values
    
    # Apply EEMD filtering
    filtered_signal, entropies, removed_idx = decomposer.filter_signal(price_series)
    
    # Add filtered signal to dataframe
    data_filtered = data.copy()
    data_filtered['Close_Filtered'] = filtered_signal
    data_filtered['Close_Filtered_Returns'] = pd.Series(filtered_signal).pct_change()
    
    return data_filtered, entropies, removed_idx

if __name__ == "__main__":
    # Example usage with sample data
    from data_collector import StockDataCollector
    
    # Load some sample data
    collector = StockDataCollector()
    raw_data = collector.download_data(start_date='2020-01-01', end_date='2021-12-31')
    
    if 'SP500' in raw_data:
        sp500_data = raw_data['SP500']
        
        # Apply EEMD decomposition
        decomposer = EEMDDecomposer()
        
        # Test with closing prices
        close_prices = sp500_data['Close'].dropna().values
        
        print("Applying EEMD decomposition to S&P 500 closing prices...")
        
        # Decompose signal
        imfs = decomposer.decompose_signal(close_prices)
        
        if imfs is not None:
            print(f"Decomposed into {len(imfs)} components")
            
            # Calculate sample entropies
            print("\nSample Entropy Analysis:")
            for i, imf in enumerate(imfs):
                entropy = decomposer.calculate_sample_entropy(imf)
                component_name = f"IMF{i+1}" if i < len(imfs)-1 else "Residue"
                print(f"{component_name}: {entropy:.4f}")
            
            # Filter signal
            filtered_signal, entropies, removed_idx = decomposer.filter_signal(close_prices)
            
            print(f"\nFiltered signal shape: {filtered_signal.shape}")
            print(f"Removed IMF index: {removed_idx}")
            
            # Visualize if matplotlib is available
            try:
                decomposer.visualize_decomposition(close_prices[:100], title="S&P 500 Sample")
            except ImportError:
                print("Matplotlib not available for visualization")
    else:
        print("No S&P 500 data available for testing")
