"""
Peephole LSTM with Temporal Attention Layer (PLSTM-TAL) Implementation
Based on: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"
This implements the novel architecture described in the paper
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class PeepholeLSTMCell(Layer):
    """
    Custom Peephole LSTM Cell implementation
    Allows gates to access cell state even when output gate is closed
    """
    
    def __init__(self, units, **kwargs):
        super(PeepholeLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]  # [cell_state, hidden_state]
        
    def build(self, input_shape):
        # Input weights
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units * 4),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Recurrent weights
        self.recurrent_kernel = self.add_weight(
            name='recurrent_kernel',
            shape=(self.units, self.units * 4),
            initializer='orthogonal',
            trainable=True
        )
        
        # Peephole weights (cell state to gates)
        self.peephole_f = self.add_weight(
            name='peephole_f',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        self.peephole_i = self.add_weight(
            name='peephole_i',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        self.peephole_o = self.add_weight(
            name='peephole_o',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Bias
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units * 4,),
            initializer='zeros',
            trainable=True
        )
        
        super(PeepholeLSTMCell, self).build(input_shape)
    
    def call(self, inputs, states):
        prev_cell_state = states[0]
        prev_hidden_state = states[1]
        
        # Linear transformation
        z = tf.matmul(inputs, self.kernel)
        z += tf.matmul(prev_hidden_state, self.recurrent_kernel)
        z += self.bias
        
        # Split into gates
        z_f, z_i, z_c, z_o = tf.split(z, 4, axis=1)
        
        # Forget gate with peephole connection
        f = tf.sigmoid(z_f + self.peephole_f * prev_cell_state)
        
        # Input gate with peephole connection
        i = tf.sigmoid(z_i + self.peephole_i * prev_cell_state)
        
        # Candidate values
        c_tilde = tf.tanh(z_c)
        
        # New cell state
        c = f * prev_cell_state + i * c_tilde
        
        # Output gate with peephole connection (uses new cell state)
        o = tf.sigmoid(z_o + self.peephole_o * c)
        
        # New hidden state
        h = o * tf.tanh(c)
        
        return h, [c, h]
    
    def get_config(self):
        config = super(PeepholeLSTMCell, self).get_config()
        config.update({
            "units": self.units,
        })
        return config

class TemporalAttentionLayer(Layer):
    """
    Temporal Attention Layer as described in the paper
    Focuses on relevant temporal information for prediction
    """
    
    def __init__(self, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Attention weight matrix
        self.W_a = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Attention bias
        self.b_a = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        
        # Context vector
        self.u_a = self.add_weight(
            name='context_vector',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(TemporalAttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        
        # Calculate attention scores
        # u_it = tanh(W_a * h_it + b_a)
        u = tf.tanh(tf.tensordot(inputs, self.W_a, axes=1) + self.b_a)
        
        # Calculate attention weights
        # alpha_it = softmax(u_a^T * u_it)
        attention_scores = tf.tensordot(u, self.u_a, axes=1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention weights
        # context = sum(alpha_it * h_it)
        attention_weights = tf.expand_dims(attention_weights, -1)
        attended_output = tf.reduce_sum(attention_weights * inputs, axis=1)
        
        return attended_output
    
    def get_config(self):
        config = super(TemporalAttentionLayer, self).get_config()
        return config

class PLSTM_TAL:
    """
    Main PLSTM-TAL model class
    Combines Peephole LSTM with Temporal Attention Layer
    """
    
    def __init__(self, sequence_length, n_features, lstm_units=64, dropout_rate=0.1):
        """
        Initialize PLSTM-TAL model
        
        Parameters:
        sequence_length: Length of input sequences
        n_features: Number of input features
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def build_model(self):
        """
        Build the PLSTM-TAL architecture
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Peephole LSTM layers
        lstm_out = layers.RNN(
            PeepholeLSTMCell(self.lstm_units),
            return_sequences=True,
            name='peephole_lstm'
        )(inputs)
        
        # Dropout for regularization
        lstm_out = layers.Dropout(self.dropout_rate)(lstm_out)
        
        # Temporal Attention Layer
        attention_out = TemporalAttentionLayer(name='temporal_attention')(lstm_out)
        
        # Dense layers for classification
        dense1 = layers.Dense(32, activation='tanh')(attention_out)
        dense1 = layers.Dropout(self.dropout_rate)(dense1)
        
        dense2 = layers.Dense(16, activation='tanh')(dense1)
        dense2 = layers.Dropout(self.dropout_rate)(dense2)
        
        # Output layer (binary classification)
        outputs = layers.Dense(1, activation='sigmoid', name='prediction')(dense2)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='PLSTM_TAL')
        
        return self.model
    
    def compile_model(self, optimizer='adamax', loss='binary_crossentropy', metrics=None):
        """
        Compile the model with specified optimizer and loss
        """
        if metrics is None:
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def prepare_sequences(self, data, target_col='Target'):
        """
        Prepare sequences for LSTM input
        
        Parameters:
        data: DataFrame with features and target
        target_col: Name of target column
        
        Returns:
        X_sequences: 3D array of sequences
        y_sequences: 1D array of targets
        """
        feature_cols = [col for col in data.columns if col != target_col]
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(data[feature_cols])
        y = data[target_col].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(data)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, 
              batch_size=32, verbose=1, callbacks=None):
        """
        Train the PLSTM-TAL model
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose,
            callbacks=callbacks,
            shuffle=True
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        """
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance with comprehensive metrics
        """
        # Get predictions
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, average='binary'),
            'recall': recall_score(y_test_flat, y_pred, average='binary'),
            'f1_score': f1_score(y_test_flat, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y_test_flat, y_pred_proba.flatten()),
            'mcc': matthews_corrcoef(y_test_flat, y_pred)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test_flat, y_pred)
        
        return metrics, cm, y_pred_proba
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
        
        metrics_to_plot = ['loss', 'accuracy']
        n_metrics = len(metrics_to_plot)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.history.history:
                axes[i].plot(self.history.history[metric], label=f'Training {metric.title()}')
                if f'val_{metric}' in self.history.history:
                    axes[i].plot(self.history.history[f'val_{metric}'], label=f'Validation {metric.title()}')
                axes[i].set_title(f'{metric.title()} Over Time')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.title())
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_confusion_matrix(self, cm, class_names=['Down', 'Up']):
        """
        Plot confusion matrix
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Show all ticks and label them
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_model(self, filepath):
        """
        Save the trained model and scalers
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(f"{filepath}_plstm_tal.h5")
        
        # Save scalers
        import joblib
        joblib.dump(self.scaler_X, f"{filepath}_scaler_X.pkl")
        joblib.dump(self.scaler_y, f"{filepath}_scaler_y.pkl")
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model and scalers
        """
        # Load model with custom objects
        custom_objects = {
            'PeepholeLSTMCell': PeepholeLSTMCell,
            'TemporalAttentionLayer': TemporalAttentionLayer
        }
        
        self.model = keras.models.load_model(f"{filepath}_plstm_tal.h5", 
                                           custom_objects=custom_objects)
        
        # Load scalers
        import joblib
        self.scaler_X = joblib.load(f"{filepath}_scaler_X.pkl")
        self.scaler_y = joblib.load(f"{filepath}_scaler_y.pkl")
        
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Example usage and testing
    print("Testing PLSTM-TAL model...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    sequence_length = 30
    
    # Create sample sequential data
    X_sample = np.random.randn(n_samples, n_features)
    for i in range(1, n_samples):
        X_sample[i] = 0.8 * X_sample[i-1] + 0.2 * np.random.randn(n_features)
    
    # Create sample targets (binary classification)
    y_sample = (X_sample[:, 0] > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data_sample = pd.DataFrame(X_sample, columns=feature_names)
    data_sample['Target'] = y_sample
    
    # Initialize model
    model = PLSTM_TAL(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=32,
        dropout_rate=0.1
    )
    
    # Build and compile model
    model.build_model()
    model.compile_model()
    
    print("Model architecture:")
    model.model.summary()
    
    # Prepare sequences
    X_sequences, y_sequences = model.prepare_sequences(data_sample)
    
    print(f"Sequence shape: {X_sequences.shape}")
    print(f"Target shape: {y_sequences.shape}")
    
    # Split data
    split_idx = int(len(X_sequences) * 0.8)
    X_train = X_sequences[:split_idx]
    y_train = y_sequences[:split_idx]
    X_test = X_sequences[split_idx:]
    y_test = y_sequences[split_idx:]
    
    # Train model
    print("\nTraining PLSTM-TAL model...")
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=10,  # Short training for testing
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    metrics, cm, predictions = model.evaluate_model(X_test, y_test)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Plot results
    try:
        model.plot_training_history()
        model.plot_confusion_matrix(cm)
    except Exception as e:
        print(f"Plotting error: {e}")
    
    print("PLSTM-TAL testing completed successfully!")
