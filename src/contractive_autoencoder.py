"""
Contractive Autoencoder for Feature Extraction
Based on: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"
Implementation follows the CAE methodology described in the paper
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ContractiveAutoencoder:
    """
    Implements Contractive Autoencoder for feature extraction
    As described in the paper with Frobenius norm regularization
    """
    
    def __init__(self, input_dim, encoding_dim=20, lambda_reg=1e-4, learning_rate=0.001):
        """
        Initialize Contractive Autoencoder
        
        Parameters:
        input_dim: Number of input features
        encoding_dim: Number of encoded features (reduced dimension)
        lambda_reg: Regularization parameter for contractive term
        learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None
        
    def contractive_loss(self, y_true, y_pred):
        """
        Custom loss function with contractive regularization term
        Loss = MSE + lambda * ||J||_F^2
        where J is the Jacobian matrix of encoder activations
        """
        # Reconstruction loss (MSE)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Get encoder activations (hidden layer)
        encoder_activations = self.encoder(y_true)
        
        # Calculate Jacobian matrix using gradient
        with tf.GradientTape() as tape:
            tape.watch(y_true)
            h = self.encoder(y_true)
        
        # Compute gradients (Jacobian)
        jacobian = tape.gradient(h, y_true)
        
        # Frobenius norm of Jacobian
        frobenius_norm = tf.reduce_sum(tf.square(jacobian))
        
        # Total loss
        total_loss = mse_loss + self.lambda_reg * frobenius_norm
        
        return total_loss
    
    def build_model(self):
        """
        Build the contractive autoencoder architecture
        """
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(
            units=64,
            activation='tanh',
            activity_regularizer=regularizers.l2(1e-5)
        )(input_layer)
        
        encoded = layers.Dense(
            units=32,
            activation='tanh',
            activity_regularizer=regularizers.l2(1e-5)
        )(encoded)
        
        encoded = layers.Dense(
            units=self.encoding_dim,
            activation='tanh',
            name='encoded_features'
        )(encoded)
        
        # Decoder
        decoded = layers.Dense(
            units=32,
            activation='tanh'
        )(encoded)
        
        decoded = layers.Dense(
            units=64,
            activation='tanh'
        )(decoded)
        
        decoded = layers.Dense(
            units=self.input_dim,
            activation='linear'
        )(decoded)
        
        # Full autoencoder
        self.model = Model(input_layer, decoded)
        
        # Encoder model (for feature extraction)
        self.encoder = Model(input_layer, encoded)
        
        # Decoder model
        encoded_input = layers.Input(shape=(self.encoding_dim,))
        decoder_layers = self.model.layers[-3:]
        decoded_output = encoded_input
        for layer in decoder_layers:
            decoded_output = layer(decoded_output)
        self.decoder = Model(encoded_input, decoded_output)
        
        # Compile model with custom loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.contractive_loss,
            metrics=['mse']
        )
        
    def train(self, X_train, X_val=None, epochs=100, batch_size=32, verbose=1):
        """
        Train the contractive autoencoder
        
        Parameters:
        X_train: Training data
        X_val: Validation data (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        verbose: Verbosity level
        """
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        validation_data = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, X_val_scaled)
        
        # Train the model
        self.history = self.model.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose,
            shuffle=True
        )
        
        return self.history
    
    def extract_features(self, X):
        """
        Extract features using the trained encoder
        
        Parameters:
        X: Input data
        
        Returns:
        Encoded features
        """
        if self.encoder is None:
            raise ValueError("Model must be trained before extracting features")
        
        X_scaled = self.scaler.transform(X)
        features = self.encoder.predict(X_scaled, verbose=0)
        
        return features
    
    def reconstruct(self, X):
        """
        Reconstruct input data using the full autoencoder
        """
        if self.model is None:
            raise ValueError("Model must be trained before reconstruction")
        
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        reconstructed = self.scaler.inverse_transform(reconstructed)
        
        return reconstructed
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Contractive Autoencoder Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MSE
        ax2.plot(self.history.history['mse'], label='Training MSE')
        if 'val_mse' in self.history.history:
            ax2.plot(self.history.history['val_mse'], label='Validation MSE')
        ax2.set_title('Mean Squared Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_feature_analysis(self, X, feature_names=None):
        """
        Plot analysis of extracted features
        """
        features = self.extract_features(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot feature distributions
        ax1.boxplot(features.T)
        ax1.set_title('Extracted Feature Distributions')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Feature Value')
        ax1.grid(True)
        
        # Plot feature correlations
        corr_matrix = np.corrcoef(features.T)
        im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title('Feature Correlation Matrix')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Feature Index')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save full model
        self.model.save(f"{filepath}_full_model.h5")
        
        # Save encoder
        self.encoder.save(f"{filepath}_encoder.h5")
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model and scaler
        """
        # Load models
        self.model = keras.models.load_model(f"{filepath}_full_model.h5", 
                                           custom_objects={'contractive_loss': self.contractive_loss})
        self.encoder = keras.models.load_model(f"{filepath}_encoder.h5")
        
        # Load scaler
        import joblib
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        print(f"Model loaded from {filepath}")

def apply_cae_to_dataset(data, feature_columns, target_column=None, encoding_dim=20, 
                        test_size=0.2, epochs=100, batch_size=32):
    """
    Apply Contractive Autoencoder to a complete dataset
    
    Parameters:
    data: DataFrame with features
    feature_columns: List of column names to use as features
    target_column: Target column name (optional)
    encoding_dim: Number of encoded features
    test_size: Proportion of data for validation
    epochs: Training epochs
    batch_size: Batch size
    
    Returns:
    cae: Trained CAE model
    features_train: Extracted training features
    features_val: Extracted validation features (if validation data exists)
    """
    # Prepare features
    X = data[feature_columns].dropna()
    
    # Split data if validation is needed
    if test_size > 0:
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
    else:
        X_train = X
        X_val = None
    
    # Initialize and build CAE
    cae = ContractiveAutoencoder(
        input_dim=len(feature_columns),
        encoding_dim=encoding_dim
    )
    cae.build_model()
    
    print(f"Training CAE with {len(feature_columns)} input features -> {encoding_dim} encoded features")
    
    # Train the model
    history = cae.train(
        X_train.values,
        X_val.values if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Extract features
    features_train = cae.extract_features(X_train.values)
    features_val = None
    if X_val is not None:
        features_val = cae.extract_features(X_val.values)
    
    print(f"Feature extraction completed. Shape: {features_train.shape}")
    
    # Return simple integer indices instead of pandas indices
    train_idx = np.arange(len(X_train))
    val_idx = np.arange(len(X_train), len(X_train) + len(X_val)) if X_val is not None else None
    
    return cae, features_train, features_val, train_idx, val_idx

if __name__ == "__main__":
    # Example usage
    print("Testing Contractive Autoencoder...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 40
    
    # Create correlated features (simulating technical indicators)
    X_sample = np.random.randn(n_samples, n_features)
    for i in range(1, n_features):
        X_sample[:, i] = 0.7 * X_sample[:, i-1] + 0.3 * X_sample[:, i]
    
    # Initialize CAE
    cae = ContractiveAutoencoder(input_dim=n_features, encoding_dim=10)
    cae.build_model()
    
    print("Model architecture:")
    cae.model.summary()
    
    # Split data
    split_idx = int(n_samples * 0.8)
    X_train = X_sample[:split_idx]
    X_val = X_sample[split_idx:]
    
    # Train
    print("\nTraining Contractive Autoencoder...")
    history = cae.train(X_train, X_val, epochs=50, batch_size=32)
    
    # Extract features
    features_train = cae.extract_features(X_train)
    features_val = cae.extract_features(X_val)
    
    print(f"\nOriginal features shape: {X_train.shape}")
    print(f"Encoded features shape: {features_train.shape}")
    print(f"Compression ratio: {features_train.shape[1] / X_train.shape[1]:.2f}")
    
    # Plot results
    try:
        cae.plot_training_history()
        cae.plot_feature_analysis(X_train)
    except Exception as e:
        print(f"Plotting error: {e}")
    
    print("CAE testing completed successfully!")
