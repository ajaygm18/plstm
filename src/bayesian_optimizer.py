"""
Bayesian Optimization Module for Hyperparameter Tuning
Based on: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"
Implements hyperparameter optimization as described in the paper
"""

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class BayesianOptimizer:
    """
    Bayesian Optimization for PLSTM-TAL hyperparameter tuning
    Optimizes the hyperparameters mentioned in Table 2 of the paper
    """
    
    def __init__(self, model_class, X_train, y_train, X_val, y_val, cv_folds=3):
        """
        Initialize Bayesian Optimizer
        
        Parameters:
        model_class: Model class to optimize
        X_train: Training data
        y_train: Training targets
        X_val: Validation data  
        y_val: Validation targets
        cv_folds: Number of cross-validation folds
        """
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cv_folds = cv_folds
        self.best_params = None
        self.best_score = None
        self.optimization_results = None
        
        # Define search space as per Table 2 in the paper
        self.search_space = [
            Integer(16, 128, name='units'),  # LSTM units: 16, 32, 64, 128
            Categorical(['tanh', 'relu', 'sigmoid'], name='activation'),  # Activation functions
            Categorical(['adam', 'adamax', 'rmsprop', 'sgd'], name='optimizer'),  # Optimizers
            Categorical(['binary_crossentropy'], name='loss'),  # Loss functions
            Real(0.1, 0.4, name='dropout'),  # Dropout: 0.1, 0.2, 0.3, 0.4
            Real(0.001, 0.01, name='learning_rate'),  # Learning rate
            Integer(16, 64, name='batch_size'),  # Batch size
        ]
    
    def objective_function(self, **params):
        """
        Objective function to minimize (negative validation accuracy)
        """
        try:
            # Extract hyperparameters
            units = params['units']
            activation = params['activation']
            optimizer = params['optimizer']
            loss = params['loss']
            dropout = params['dropout']
            learning_rate = params['learning_rate']
            batch_size = params['batch_size']
            
            # Build model with current hyperparameters
            model = self.model_class(
                sequence_length=self.X_train.shape[1],
                n_features=self.X_train.shape[2],
                lstm_units=units,
                dropout_rate=dropout
            )
            
            # Build and compile model
            model.build_model()
            
            # Configure optimizer
            if optimizer == 'adam':
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer == 'adamax':
                opt = keras.optimizers.Adamax(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
            elif optimizer == 'sgd':
                opt = keras.optimizers.SGD(learning_rate=learning_rate)
            
            model.compile_model(optimizer=opt, loss=loss)
            
            # Train model with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train for limited epochs during optimization
            history = model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=50,  # Reduced for faster optimization
                batch_size=batch_size,
                verbose=0,
                callbacks=[early_stopping]
            )
            
            # Get validation accuracy (metric to maximize)
            val_accuracy = max(history.history['val_accuracy'])
            
            # Return negative accuracy (since we minimize)
            return -val_accuracy
            
        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            return 0.0  # Return poor score for failed runs
    
    def optimize(self, n_calls=20, random_state=42):
        """
        Run Bayesian optimization
        
        Parameters:
        n_calls: Number of optimization iterations
        random_state: Random seed for reproducibility
        
        Returns:
        result: Optimization result object
        """
        print("Starting Bayesian optimization...")
        print(f"Search space: {len(self.search_space)} hyperparameters")
        print(f"Total evaluations: {n_calls}")
        
        # Run optimization
        objective_with_args = use_named_args(self.search_space)(self.objective_function)
        
        self.optimization_results = gp_minimize(
            func=objective_with_args,
            dimensions=self.search_space,
            n_calls=n_calls,
            random_state=random_state,
            acq_func='EI',  # Expected Improvement
            n_initial_points=5,
            verbose=True
        )
        
        # Extract best parameters
        self.best_score = -self.optimization_results.fun  # Convert back to positive
        self.best_params = {}
        
        for i, param_name in enumerate([dim.name for dim in self.search_space]):
            self.best_params[param_name] = self.optimization_results.x[i]
        
        print(f"\nOptimization completed!")
        print(f"Best validation accuracy: {self.best_score:.4f}")
        print("Best hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self.optimization_results
    
    def plot_optimization_results(self):
        """
        Plot optimization convergence
        """
        if self.optimization_results is None:
            print("No optimization results to plot")
            return
        
        import matplotlib.pyplot as plt
        
        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Convergence plot
        scores = [-y for y in self.optimization_results.func_vals]
        ax1.plot(scores, 'b-', marker='o')
        ax1.set_title('Bayesian Optimization Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Validation Accuracy')
        ax1.grid(True)
        
        # Best score so far
        best_scores = [max(scores[:i+1]) for i in range(len(scores))]
        ax2.plot(best_scores, 'r-', marker='s', linewidth=2)
        ax2.set_title('Best Score Over Time')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Best Validation Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_optimized_model(self):
        """
        Create model with optimized hyperparameters
        """
        if self.best_params is None:
            raise ValueError("Optimization must be run first")
        
        # Create model with best parameters
        model = self.model_class(
            sequence_length=self.X_train.shape[1],
            n_features=self.X_train.shape[2],
            lstm_units=self.best_params['units'],
            dropout_rate=self.best_params['dropout']
        )
        
        model.build_model()
        
        # Configure optimizer
        optimizer_name = self.best_params['optimizer']
        learning_rate = self.best_params['learning_rate']
        
        if optimizer_name == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'adamax':
            opt = keras.optimizers.Adamax(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        
        model.compile_model(
            optimizer=opt,
            loss=self.best_params['loss']
        )
        
        return model
    
    def save_results(self, filepath):
        """
        Save optimization results
        """
        if self.optimization_results is None:
            print("No results to save")
            return
        
        # Save best parameters
        params_df = pd.DataFrame([self.best_params])
        params_df['best_score'] = self.best_score
        params_df.to_csv(f"{filepath}_best_params.csv", index=False)
        
        # Save optimization history
        history_data = {
            'iteration': list(range(len(self.optimization_results.func_vals))),
            'score': [-y for y in self.optimization_results.func_vals]
        }
        
        # Add parameter values for each iteration
        for i, param_name in enumerate([dim.name for dim in self.search_space]):
            history_data[param_name] = [x[i] for x in self.optimization_results.x_iters]
        
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(f"{filepath}_optimization_history.csv", index=False)
        
        print(f"Optimization results saved to {filepath}")

class GridSearchOptimizer:
    """
    Alternative grid search optimizer for comparison
    """
    
    def __init__(self, model_class, X_train, y_train, X_val, y_val):
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_params = None
        self.best_score = None
        
        # Simplified grid for faster execution
        self.param_grid = {
            'units': [32, 64],
            'activation': ['tanh', 'relu'],
            'optimizer': ['adam', 'adamax'],
            'dropout': [0.1, 0.2],
            'learning_rate': [0.001, 0.005],
            'batch_size': [32]
        }
    
    def optimize(self):
        """
        Run grid search optimization
        """
        print("Starting Grid Search optimization...")
        
        best_score = -np.inf
        best_params = None
        total_combinations = np.prod([len(v) for v in self.param_grid.values()])
        
        print(f"Total parameter combinations: {total_combinations}")
        
        current_iteration = 0
        
        # Generate all parameter combinations
        import itertools
        
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        for param_combination in itertools.product(*param_values):
            current_iteration += 1
            params = dict(zip(param_names, param_combination))
            
            print(f"Iteration {current_iteration}/{total_combinations}: {params}")
            
            try:
                # Build and train model
                model = self.model_class(
                    sequence_length=self.X_train.shape[1],
                    n_features=self.X_train.shape[2],
                    lstm_units=params['units'],
                    dropout_rate=params['dropout']
                )
                
                model.build_model()
                
                # Configure optimizer
                if params['optimizer'] == 'adam':
                    opt = keras.optimizers.Adam(learning_rate=params['learning_rate'])
                elif params['optimizer'] == 'adamax':
                    opt = keras.optimizers.Adamax(learning_rate=params['learning_rate'])
                
                model.compile_model(optimizer=opt, loss='binary_crossentropy')
                
                # Train model
                history = model.train(
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    epochs=30,
                    batch_size=params['batch_size'],
                    verbose=0
                )
                
                # Get validation accuracy
                val_accuracy = max(history.history['val_accuracy'])
                
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = params.copy()
                    print(f"New best score: {best_score:.4f}")
                
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue
        
        self.best_score = best_score
        self.best_params = best_params
        
        print(f"\nGrid Search completed!")
        print(f"Best validation accuracy: {self.best_score:.4f}")
        print("Best hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self.best_params, self.best_score

if __name__ == "__main__":
    # Example usage
    from plstm_tal_model import PLSTM_TAL
    import numpy as np
    
    print("Testing Bayesian Optimization...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    sequence_length = 20
    
    # Create sample data
    X_sample = np.random.randn(n_samples, sequence_length, n_features)
    y_sample = np.random.randint(0, 2, (n_samples,))
    
    # Split data
    split_idx = int(n_samples * 0.8)
    X_train = X_sample[:split_idx]
    y_train = y_sample[:split_idx]
    X_val = X_sample[split_idx:]
    y_val = y_sample[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(
        model_class=PLSTM_TAL,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    # Run optimization (reduced iterations for testing)
    results = optimizer.optimize(n_calls=5)
    
    # Plot results
    try:
        optimizer.plot_optimization_results()
    except Exception as e:
        print(f"Plotting error: {e}")
    
    # Get optimized model
    optimized_model = optimizer.get_optimized_model()
    print("Optimized model created successfully!")
    
    print("Bayesian optimization testing completed!")
