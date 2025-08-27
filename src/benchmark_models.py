"""
Benchmark Models for Comparison
Based on: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"
Implements CNN, LSTM, SVM, and Random Forest models for comparison
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class CNNModel:
    """
    Convolutional Neural Network for sequence classification
    """
    
    def __init__(self, sequence_length, n_features, filters=64, kernel_size=3, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN architecture"""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Convolutional layers
        x = layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, 
                         activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Conv1D(filters=self.filters*2, kernel_size=self.kernel_size, 
                         activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Global pooling and dense layers
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='CNN_Model')
        
        return self.model
    
    def compile_model(self, optimizer='adam', loss='binary_crossentropy'):
        """Compile the model"""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the model"""
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, average='binary'),
            'recall': recall_score(y_test_flat, y_pred, average='binary'),
            'f1_score': f1_score(y_test_flat, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y_test_flat, y_pred_proba.flatten()),
            'mcc': matthews_corrcoef(y_test_flat, y_pred)
        }
        
        cm = confusion_matrix(y_test_flat, y_pred)
        
        return metrics, cm, y_pred_proba

class LSTMModel:
    """
    Standard LSTM model for comparison
    """
    
    def __init__(self, sequence_length, n_features, lstm_units=64, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build LSTM architecture"""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        x = layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)(inputs)
        x = layers.LSTM(self.lstm_units//2, dropout=self.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Model')
        
        return self.model
    
    def compile_model(self, optimizer='adam', loss='binary_crossentropy'):
        """Compile the model"""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the model"""
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, average='binary'),
            'recall': recall_score(y_test_flat, y_pred, average='binary'),
            'f1_score': f1_score(y_test_flat, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y_test_flat, y_pred_proba.flatten()),
            'mcc': matthews_corrcoef(y_test_flat, y_pred)
        }
        
        cm = confusion_matrix(y_test_flat, y_pred)
        
        return metrics, cm, y_pred_proba

class SVMModel:
    """
    Support Vector Machine for sequence classification
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_data(self, X):
        """Prepare sequence data for SVM (flatten sequences)"""
        # Flatten sequences: (samples, timesteps, features) -> (samples, timesteps*features)
        X_flattened = X.reshape(X.shape[0], -1)
        return X_flattened
    
    def train(self, X_train, y_train):
        """Train the SVM model"""
        # Prepare data
        X_train_flat = self.prepare_data(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        # Train model
        self.model.fit(X_train_scaled, y_train.flatten())
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        X_flat = self.prepare_data(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, average='binary'),
            'recall': recall_score(y_test_flat, y_pred, average='binary'),
            'f1_score': f1_score(y_test_flat, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y_test_flat, y_pred_proba.flatten()),
            'mcc': matthews_corrcoef(y_test_flat, y_pred)
        }
        
        cm = confusion_matrix(y_test_flat, y_pred)
        
        return metrics, cm, y_pred_proba

class RandomForestModel:
    """
    Random Forest for sequence classification
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        
    def prepare_data(self, X):
        """Prepare sequence data for Random Forest (flatten sequences)"""
        # Flatten sequences: (samples, timesteps, features) -> (samples, timesteps*features)
        X_flattened = X.reshape(X.shape[0], -1)
        return X_flattened
    
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        # Prepare data
        X_train_flat = self.prepare_data(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        # Train model
        self.model.fit(X_train_scaled, y_train.flatten())
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        X_flat = self.prepare_data(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, average='binary'),
            'recall': recall_score(y_test_flat, y_pred, average='binary'),
            'f1_score': f1_score(y_test_flat, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y_test_flat, y_pred_proba.flatten()),
            'mcc': matthews_corrcoef(y_test_flat, y_pred)
        }
        
        cm = confusion_matrix(y_test_flat, y_pred)
        
        return metrics, cm, y_pred_proba

class BenchmarkComparison:
    """
    Class to compare all benchmark models against PLSTM-TAL
    """
    
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Initialize all benchmark models"""
        self.models = {
            'CNN': CNNModel(self.sequence_length, self.n_features),
            'LSTM': LSTMModel(self.sequence_length, self.n_features),
            'SVM': SVMModel(),
            'RF': RandomForestModel()
        }
        
        # Build and compile neural network models
        for name, model in self.models.items():
            if hasattr(model, 'build_model'):
                model.build_model()
                model.compile_model()
        
        return self.models
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train all benchmark models"""
        print("Training benchmark models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                if name in ['CNN', 'LSTM']:
                    # Neural network models
                    model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32)
                else:
                    # Traditional ML models
                    model.train(X_train, y_train)
                
                print(f"{name} training completed")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
        
        return self.models
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models and compare performance"""
        print("Evaluating all models...")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                metrics, cm, predictions = model.evaluate(X_test, y_test)
                
                self.results[name] = {
                    'metrics': metrics,
                    'confusion_matrix': cm,
                    'predictions': predictions
                }
                
                print(f"{name} evaluation completed")
                
                # Print key metrics
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
        
        return self.results
    
    def compare_with_plstm_tal(self, plstm_tal_metrics):
        """Compare benchmark models with PLSTM-TAL"""
        print("\n" + "="*60)
        print("BENCHMARK COMPARISON WITH PLSTM-TAL")
        print("="*60)
        
        # Add PLSTM-TAL results to comparison
        all_results = self.results.copy()
        all_results['PLSTM-TAL'] = {'metrics': plstm_tal_metrics}
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            model_name: result['metrics'] 
            for model_name, result in all_results.items()
        }).T
        
        print("\nPerformance Comparison:")
        print("-" * 80)
        print(comparison_df.round(4))
        
        # Find best performing model for each metric
        print("\nBest performing model for each metric:")
        print("-" * 40)
        for metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"{metric.upper()}: {best_model} ({best_score:.4f})")
        
        # Calculate ranking
        print("\nOverall Ranking (average rank across all metrics):")
        print("-" * 50)
        ranks = comparison_df.rank(ascending=False, method='average')
        avg_ranks = ranks.mean(axis=1).sort_values()
        
        for i, (model, avg_rank) in enumerate(avg_ranks.items(), 1):
            print(f"{i}. {model}: {avg_rank:.2f}")
        
        return comparison_df, avg_ranks
    
    def plot_comparison(self, plstm_tal_metrics=None):
        """Plot comparison results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data
        results_for_plot = self.results.copy()
        if plstm_tal_metrics is not None:
            results_for_plot['PLSTM-TAL'] = {'metrics': plstm_tal_metrics}
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            model_name: result['metrics'] 
            for model_name, result in results_for_plot.items()
        }).T
        
        # Plot metrics comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in comparison_df.columns:
                data = comparison_df[metric].sort_values(ascending=False)
                
                bars = axes[i].bar(range(len(data)), data.values)
                axes[i].set_title(f'{metric.upper()} Comparison')
                axes[i].set_ylabel('Score')
                axes[i].set_xticks(range(len(data)))
                axes[i].set_xticklabels(data.index, rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Highlight best performer
                best_idx = data.values.argmax()
                bars[best_idx].set_color('red')
                bars[best_idx].set_alpha(0.8)
                
                # Add value labels
                for j, (bar, value) in enumerate(zip(bars, data.values)):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        plt.show()
        
        return fig, comparison_df

def run_benchmark_comparison(X_train, y_train, X_val, y_val, X_test, y_test, 
                           plstm_tal_metrics=None, epochs=50):
    """
    Run complete benchmark comparison
    
    Parameters:
    X_train, y_train: Training data
    X_val, y_val: Validation data
    X_test, y_test: Test data
    plstm_tal_metrics: PLSTM-TAL metrics for comparison
    epochs: Training epochs for neural networks
    
    Returns:
    benchmark: BenchmarkComparison object
    comparison_df: DataFrame with comparison results
    """
    
    print("Running Benchmark Comparison...")
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Initialize benchmark comparison
    benchmark = BenchmarkComparison(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2]
    )
    
    # Initialize models
    benchmark.initialize_models()
    
    # Train all models
    benchmark.train_all_models(X_train, y_train, X_val, y_val, epochs=epochs)
    
    # Evaluate all models
    benchmark.evaluate_all_models(X_test, y_test)
    
    # Compare with PLSTM-TAL if provided
    if plstm_tal_metrics is not None:
        comparison_df, rankings = benchmark.compare_with_plstm_tal(plstm_tal_metrics)
    else:
        comparison_df = pd.DataFrame({
            model_name: result['metrics'] 
            for model_name, result in benchmark.results.items()
        }).T
        rankings = None
    
    # Plot comparison
    benchmark.plot_comparison(plstm_tal_metrics)
    
    return benchmark, comparison_df, rankings

if __name__ == "__main__":
    # Example usage
    print("Testing Benchmark Models...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 30
    n_features = 21  # 20 extracted features + 1 filtered price
    
    X_sample = np.random.randn(n_samples, sequence_length, n_features)
    y_sample = np.random.randint(0, 2, (n_samples,))
    
    # Split data
    train_size = int(n_samples * 0.6)
    val_size = int(n_samples * 0.2)
    
    X_train = X_sample[:train_size]
    y_train = y_sample[:train_size]
    X_val = X_sample[train_size:train_size+val_size]
    y_val = y_sample[train_size:train_size+val_size]
    X_test = X_sample[train_size+val_size:]
    y_test = y_sample[train_size+val_size:]
    
    print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Sample PLSTM-TAL metrics for comparison
    plstm_tal_metrics = {
        'accuracy': 0.8530,
        'precision': 0.8683,
        'recall': 0.8659,
        'f1_score': 0.8671,
        'auc_roc': 0.9312,
        'mcc': 0.6975
    }
    
    # Run benchmark comparison with reduced epochs for testing
    benchmark, comparison_df, rankings = run_benchmark_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test,
        plstm_tal_metrics=plstm_tal_metrics,
        epochs=5  # Reduced for quick testing
    )
    
    print("Benchmark comparison completed!")
