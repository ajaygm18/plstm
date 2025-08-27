"""
Main Pipeline for PLSTM-TAL Stock Market Prediction
Based on: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"
Complete implementation following the paper's methodology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_collector import StockDataCollector
from eemd_decomposer import EEMDDecomposer, apply_eemd_to_dataset
from contractive_autoencoder import ContractiveAutoencoder, apply_cae_to_dataset
from plstm_tal_model import PLSTM_TAL
from bayesian_optimizer import BayesianOptimizer

class PLSTMTALPipeline:
    """
    Complete pipeline for PLSTM-TAL stock market prediction
    Implements the full methodology described in the research paper
    """
    
    def __init__(self, sequence_length=30, encoding_dim=20, test_size=0.2, 
                 validation_size=0.1, random_state=42):
        """
        Initialize the pipeline
        
        Parameters:
        sequence_length: Length of sequences for LSTM input
        encoding_dim: Dimension of encoded features from CAE
        test_size: Proportion of data for testing
        validation_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        # Initialize components
        self.data_collector = StockDataCollector()
        self.eemd_decomposer = EEMDDecomposer()
        self.cae = None
        self.plstm_tal = None
        self.bayesian_optimizer = None
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.decomposed_data = {}
        self.extracted_features = {}
        self.sequences = {}
        
        # Results storage
        self.results = {}
        
    def step1_collect_data(self, start_date='2005-01-01', end_date='2022-03-31'):
        """
        Step 1: Collect stock market data for all four indices
        """
        print("=" * 60)
        print("STEP 1: DATA COLLECTION")
        print("=" * 60)
        
        print("Downloading stock market data...")
        print(f"Period: {start_date} to {end_date}")
        print("Indices: S&P 500, FTSE 100, SSE Composite, Nifty 50")
        
        self.raw_data = self.data_collector.download_data(start_date, end_date)
        
        print(f"\nData collection completed. Downloaded {len(self.raw_data)} datasets.")
        for name, data in self.raw_data.items():
            print(f"  {name}: {len(data)} records")
        
        return self.raw_data
    
    def step2_calculate_technical_indicators(self):
        """
        Step 2: Calculate 40 technical indicators for each dataset
        """
        print("=" * 60)
        print("STEP 2: TECHNICAL INDICATORS CALCULATION")
        print("=" * 60)
        
        for name, data in self.raw_data.items():
            print(f"\nProcessing {name}...")
            
            # Calculate technical indicators
            data_with_indicators = self.data_collector.calculate_technical_indicators(data)
            
            # Create target labels
            data_with_targets = self.data_collector.create_target_labels(data_with_indicators)
            
            # Preprocess data
            processed = self.data_collector.preprocess_data(data_with_targets)
            
            self.processed_data[name] = processed
            
            print(f"  Shape after preprocessing: {processed.shape}")
            print(f"  Features: {processed.shape[1]} columns")
        
        return self.processed_data
    
    def step3_apply_eemd_decomposition(self):
        """
        Step 3: Apply EEMD decomposition to filter closing prices
        """
        print("=" * 60)
        print("STEP 3: EEMD SIGNAL DECOMPOSITION")
        print("=" * 60)
        
        for name, data in self.processed_data.items():
            print(f"\nApplying EEMD to {name}...")
            
            # Apply EEMD decomposition
            decomposed_data, entropies, removed_idx = apply_eemd_to_dataset(data, 'Close')
            
            self.decomposed_data[name] = decomposed_data
            
            print(f"  Original closing price column added")
            print(f"  Filtered closing price column added")
            print(f"  Removed IMF index: {removed_idx}")
            print(f"  Sample entropies calculated for all IMFs")
        
        return self.decomposed_data
    
    def step4_extract_features_with_cae(self, epochs=100, test_size=0.2):
        """
        Step 4: Extract features using Contractive Autoencoder
        """
        print("=" * 60)
        print("STEP 4: FEATURE EXTRACTION WITH CAE")
        print("=" * 60)
        
        for name, data in self.decomposed_data.items():
            print(f"\nExtracting features for {name}...")
            
            # Select feature columns (exclude target and price columns)
            feature_cols = [col for col in data.columns 
                          if col not in ['Target', 'Close', 'Open', 'High', 'Low', 
                                       'Volume', 'Close_Filtered', 'Close_Filtered_Returns',
                                       'Returns', 'Next_Return']]
            
            print(f"  Using {len(feature_cols)} technical indicators as input features")
            
            # Apply CAE
            cae, features_train, features_val, train_idx, val_idx = apply_cae_to_dataset(
                data, feature_cols, 'Target', self.encoding_dim, 
                test_size=test_size, epochs=epochs, batch_size=32
            )
            
            # Store results
            self.extracted_features[name] = {
                'cae_model': cae,
                'features_train': features_train,
                'features_val': features_val,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'feature_cols': feature_cols
            }
            
            print(f"  Feature extraction completed")
            print(f"  Reduced from {len(feature_cols)} to {self.encoding_dim} features")
            print(f"  Training features shape: {features_train.shape}")
            if features_val is not None:
                print(f"  Validation features shape: {features_val.shape}")
        
        return self.extracted_features
    
    def step5_prepare_sequences(self):
        """
        Step 5: Prepare sequences for PLSTM-TAL input
        """
        print("=" * 60)
        print("STEP 5: SEQUENCE PREPARATION")
        print("=" * 60)
        
        for name in self.extracted_features.keys():
            print(f"\nPreparing sequences for {name}...")
            
            # Get data components
            data = self.decomposed_data[name]
            features_info = self.extracted_features[name]
            
            # Combine original features with extracted features and filtered price
            # Get the filtered closing price and add it to features
            filtered_price = data['Close_Filtered'].values
            
            # Normalize filtered price
            price_scaler = MinMaxScaler()
            filtered_price_scaled = price_scaler.fit_transform(filtered_price.reshape(-1, 1))
            
            # Combine extracted features with filtered price
            train_idx = np.array(features_info['train_idx'], dtype=int)
            val_idx = np.array(features_info['val_idx'], dtype=int) if features_info['val_idx'] is not None else None
            
            train_features = np.column_stack([
                features_info['features_train'],
                filtered_price_scaled[train_idx]
            ])
            
            if features_info['features_val'] is not None:
                val_features = np.column_stack([
                    features_info['features_val'],
                    filtered_price_scaled[val_idx]
                ])
            else:
                val_features = None
            
            # Get targets using iloc for integer indices
            train_targets = data.iloc[train_idx]['Target'].values
            val_targets = data.iloc[val_idx]['Target'].values if val_idx is not None else None
            
            # Create sequences
            X_train_seq, y_train_seq = self._create_sequences(train_features, train_targets)
            X_val_seq, y_val_seq = None, None
            
            if val_features is not None and val_targets is not None:
                X_val_seq, y_val_seq = self._create_sequences(val_features, val_targets)
            
            # Check if we have sufficient validation data
            min_val_samples = 10  # Minimum validation samples needed
            
            if X_val_seq is None or len(X_val_seq) < min_val_samples:
                print(f"  Insufficient validation data ({len(X_val_seq) if X_val_seq is not None else 0} samples), creating split from training data")
                # Create validation split from training data
                X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
                    X_train_seq, y_train_seq, test_size=self.test_size, 
                    random_state=self.random_state, stratify=y_train_seq
                )
            
            # Store sequences
            self.sequences[name] = {
                'X_train': X_train_seq,
                'y_train': y_train_seq,
                'X_val': X_val_seq,
                'y_val': y_val_seq,
                'n_features': train_features.shape[1],
                'price_scaler': price_scaler
            }
            
            print(f"  Training sequences: {X_train_seq.shape}")
            print(f"  Validation sequences: {X_val_seq.shape}")
            print(f"  Features per timestep: {train_features.shape[1]}")
            print(f"  Sequence length: {self.sequence_length}")
        
        return self.sequences
    
    def _create_sequences(self, features, targets):
        """
        Helper method to create sequences from features and targets
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(features)):
            X_sequences.append(features[i-self.sequence_length:i])
            y_sequences.append(targets[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def step6_optimize_hyperparameters(self, index_name, n_calls=20):
        """
        Step 6: Optimize hyperparameters using Bayesian Optimization
        """
        print("=" * 60)
        print("STEP 6: HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        
        if index_name not in self.sequences:
            raise ValueError(f"Sequences not prepared for {index_name}")
        
        seq_data = self.sequences[index_name]
        
        print(f"Optimizing hyperparameters for {index_name}...")
        print(f"Using Bayesian Optimization with {n_calls} iterations")
        
        # Initialize Bayesian optimizer
        self.bayesian_optimizer = BayesianOptimizer(
            model_class=PLSTM_TAL,
            X_train=seq_data['X_train'],
            y_train=seq_data['y_train'],
            X_val=seq_data['X_val'],
            y_val=seq_data['y_val']
        )
        
        # Run optimization
        results = self.bayesian_optimizer.optimize(n_calls=n_calls)
        
        print("Hyperparameter optimization completed!")
        
        return self.bayesian_optimizer.best_params
    
    def step7_train_final_model(self, index_name, use_optimized_params=True, epochs=100):
        """
        Step 7: Train final PLSTM-TAL model
        """
        print("=" * 60)
        print("STEP 7: FINAL MODEL TRAINING")
        print("=" * 60)
        
        seq_data = self.sequences[index_name]
        
        print(f"Training final PLSTM-TAL model for {index_name}...")
        
        if use_optimized_params and self.bayesian_optimizer is not None:
            print("Using optimized hyperparameters")
            self.plstm_tal = self.bayesian_optimizer.get_optimized_model()
        else:
            print("Using default hyperparameters")
            self.plstm_tal = PLSTM_TAL(
                sequence_length=self.sequence_length,
                n_features=seq_data['n_features'],
                lstm_units=64,
                dropout_rate=0.1
            )
            self.plstm_tal.build_model()
            self.plstm_tal.compile_model()
        
        # Train model
        history = self.plstm_tal.train(
            seq_data['X_train'], seq_data['y_train'],
            seq_data['X_val'], seq_data['y_val'],
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        print("Final model training completed!")
        
        return history
    
    def step8_evaluate_model(self, index_name):
        """
        Step 8: Evaluate model performance
        """
        print("=" * 60)
        print("STEP 8: MODEL EVALUATION")
        print("=" * 60)
        
        seq_data = self.sequences[index_name]
        
        print(f"Evaluating model performance for {index_name}...")
        
        # Evaluate on validation set
        metrics, cm, predictions = self.plstm_tal.evaluate_model(
            seq_data['X_val'], seq_data['y_val']
        )
        
        # Store results
        self.results[index_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'predictions': predictions,
            'y_true': seq_data['y_val']
        }
        
        # Print results
        print(f"\nPerformance Metrics for {index_name}:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return metrics, cm
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive analysis report
        """
        print("=" * 60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 60)
        
        for index_name, result in self.results.items():
            print(f"\n{index_name} RESULTS:")
            print("-" * 30)
            
            metrics = result['metrics']
            cm = result['confusion_matrix']
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"MCC: {metrics['mcc']:.4f}")
            
            # Calculate additional metrics
            try:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                print(f"Specificity: {specificity:.4f}")
            except ValueError:
                print("Specificity: Cannot calculate (insufficient data for confusion matrix)")
                print(f"Confusion matrix shape: {cm.shape}")
        
        # Compare results across indices
        print(f"\nCOMPARATIVE ANALYSIS:")
        print("-" * 30)
        
        comparison_df = pd.DataFrame({
            index: result['metrics'] 
            for index, result in self.results.items()
        }).T
        
        print(comparison_df.round(4))
        
        return comparison_df
    
    def plot_results(self):
        """
        Generate comprehensive visualizations
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Set up plotting
        n_indices = len(self.results)
        fig = plt.figure(figsize=(20, 5 * n_indices))
        
        for i, (index_name, result) in enumerate(self.results.items()):
            # Confusion Matrix
            ax1 = plt.subplot(n_indices, 4, i*4 + 1)
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title(f'{index_name} - Confusion Matrix')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # ROC Curve
            ax2 = plt.subplot(n_indices, 4, i*4 + 2)
            y_true = result['y_true']
            y_pred_proba = result['predictions'].flatten()
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = auc(fpr, tpr)
            
            ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title(f'{index_name} - ROC Curve')
            ax2.legend()
            ax2.grid(True)
            
            # Prediction Distribution
            ax3 = plt.subplot(n_indices, 4, i*4 + 3)
            ax3.hist(y_pred_proba[y_true == 0], alpha=0.7, label='Down (0)', bins=20)
            ax3.hist(y_pred_proba[y_true == 1], alpha=0.7, label='Up (1)', bins=20)
            ax3.set_xlabel('Prediction Probability')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'{index_name} - Prediction Distribution')
            ax3.legend()
            ax3.grid(True)
            
            # Metrics Comparison
            ax4 = plt.subplot(n_indices, 4, i*4 + 4)
            metrics = result['metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax4.bar(metric_names, metric_values)
            ax4.set_title(f'{index_name} - Performance Metrics')
            ax4.set_ylabel('Score')
            ax4.set_ylim([0, 1])
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_results(self, base_path="../results"):
        """
        Save all results to files
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save processed data
        for name, data in self.processed_data.items():
            data.to_csv(f"{base_path}/{name}_processed_data.csv")
        
        # Save model
        if self.plstm_tal is not None:
            self.plstm_tal.save_model(f"{base_path}/plstm_tal_model")
        
        # Save optimization results
        if self.bayesian_optimizer is not None:
            self.bayesian_optimizer.save_results(f"{base_path}/optimization")
        
        # Save performance results
        results_df = pd.DataFrame({
            index: result['metrics'] 
            for index, result in self.results.items()
        }).T
        results_df.to_csv(f"{base_path}/performance_results.csv")
        
        print(f"Results saved to {base_path}")

def run_complete_pipeline(target_index='SP500', optimize_hyperparams=True, 
                         quick_test=False):
    """
    Run the complete PLSTM-TAL pipeline for a specific index
    
    Parameters:
    target_index: Index to focus on ('SP500', 'FTSE', 'SSE', 'NIFTY')
    optimize_hyperparams: Whether to run Bayesian optimization
    quick_test: Whether to run a quick test with reduced parameters
    """
    
    # Initialize pipeline
    pipeline = PLSTMTALPipeline(sequence_length=30, encoding_dim=20)
    
    try:
        # Step 1: Collect Data
        if quick_test:
            raw_data = pipeline.step1_collect_data('2020-01-01', '2021-12-31')
        else:
            raw_data = pipeline.step1_collect_data('2005-01-01', '2022-03-31')
        
        # Step 2: Calculate Technical Indicators
        processed_data = pipeline.step2_calculate_technical_indicators()
        
        # Step 3: Apply EEMD Decomposition
        decomposed_data = pipeline.step3_apply_eemd_decomposition()
        
        # Step 4: Extract Features with CAE
        if quick_test:
            # Use smaller validation split for quick test to preserve more training data
            extracted_features = pipeline.step4_extract_features_with_cae(epochs=20, test_size=0.1)
        else:
            extracted_features = pipeline.step4_extract_features_with_cae(epochs=100)
        
        # Step 5: Prepare Sequences
        sequences = pipeline.step5_prepare_sequences()
        
        # Step 6: Optimize Hyperparameters (optional)
        best_params = None
        if optimize_hyperparams and target_index in sequences:
            if quick_test:
                best_params = pipeline.step6_optimize_hyperparameters(target_index, n_calls=5)
            else:
                best_params = pipeline.step6_optimize_hyperparameters(target_index, n_calls=20)
        
        # Step 7: Train Final Model
        if target_index in sequences:
            if quick_test:
                history = pipeline.step7_train_final_model(target_index, optimize_hyperparams, epochs=20)
            else:
                history = pipeline.step7_train_final_model(target_index, optimize_hyperparams, epochs=100)
            
            # Step 8: Evaluate Model
            metrics, cm = pipeline.step8_evaluate_model(target_index)
            
            # Generate Report
            report = pipeline.generate_comprehensive_report()
            
            # Plot Results
            pipeline.plot_results()
            
            # Save Results
            pipeline.save_results()
            
            print("=" * 60)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return pipeline, metrics, report
            
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    print("Starting PLSTM-TAL Stock Market Prediction Pipeline")
    print("Based on: Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL")
    
    # Run pipeline for S&P 500 with quick test
    pipeline, metrics, report = run_complete_pipeline(
        target_index='SP500',
        optimize_hyperparams=True,
        quick_test=True  # Set to False for full run
    )
