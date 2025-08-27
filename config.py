"""
Configuration file for PLSTM-TAL Stock Market Prediction
Contains all hyperparameters and settings as mentioned in the research paper
"""

# Stock Market Indices (as mentioned in the paper)
STOCK_INDICES = {
    'SP500': '^GSPC',     # S&P 500 (United States)
    'FTSE': '^FTSE',      # FTSE 100 (United Kingdom)
    'SSE': '000001.SS',   # SSE Composite (China)
    'NIFTY': '^NSEI'      # Nifty 50 (India)
}

# Data Collection Settings
DATA_CONFIG = {
    'start_date': '2005-01-01',  # As mentioned in paper
    'end_date': '2022-03-31',    # As mentioned in paper
    'download_timeout': 30
}

# Technical Indicators Configuration (40 indicators as mentioned in paper)
TECHNICAL_INDICATORS = [
    'Returns', 'Log_Returns',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
    'RSI_14', 'RSI_9', 'RSI_25',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Stoch_K', 'Stoch_D',
    'Williams_R', 'CCI', 'ATR',
    'Volume_SMA_20', 'Volume_Ratio', 'OBV', 'AD', 'MFI',
    'ROC_10', 'ROC_20', 'MOM_10', 'MOM_20',
    'SAR', 'ADX', 'TEMA', 'TRIMA', 'WMA',
    'AROON_UP', 'AROON_DOWN', 'AROONOSC', 'PLUS_DI', 'MINUS_DI', 'DX',
    'High_Low_Ratio', 'Close_Open_Ratio'
]

# EEMD Configuration
EEMD_CONFIG = {
    'n_trials': 100,        # Number of ensemble trials
    'noise_scale': 0.005,   # Gaussian noise standard deviation
    'max_imf': -1,          # Maximum number of IMFs (-1 for automatic)
    'sample_entropy_m': 2,  # Pattern length for sample entropy
    'sample_entropy_r': 0.2 # Tolerance factor for sample entropy
}

# Contractive Autoencoder Configuration
CAE_CONFIG = {
    'encoding_dim': 20,     # Compressed feature dimension
    'lambda_reg': 1e-4,     # Regularization parameter
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2
}

# PLSTM-TAL Model Configuration (Based on Table 2 in paper)
# Optimized hyperparameters as mentioned in the paper
PLSTM_TAL_CONFIG = {
    'sequence_length': 30,   # Input sequence length
    'lstm_units': 64,        # Optimized value from Table 2
    'activation': 'tanh',    # Optimized value from Table 2
    'optimizer': 'adamax',   # Optimized value from Table 2
    'loss': 'binary_crossentropy',  # Optimized value from Table 2
    'dropout': 0.1,          # Optimized value from Table 2
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2
}

# Bayesian Optimization Configuration
BAYESIAN_OPT_CONFIG = {
    'n_calls': 20,          # Number of optimization iterations
    'n_initial_points': 5,  # Initial random points
    'acq_func': 'EI',       # Expected Improvement
    'random_state': 42,
    
    # Search space (as mentioned in Table 2)
    'search_space': {
        'units': [16, 32, 64, 128],
        'activation': ['relu', 'tanh', 'linear', 'sigmoid'],
        'optimizer': ['SGD', 'Adam', 'Adamax', 'RMSprop', 'Adagrad'],
        'loss': ['binary_crossentropy', 'hinge', 'squared_hinge'],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [16, 32, 64]
    }
}

# Benchmark Models Configuration
BENCHMARK_CONFIG = {
    'CNN': {
        'filters': 64,
        'kernel_size': 3,
        'dropout_rate': 0.2,
        'epochs': 50,
        'batch_size': 32
    },
    'LSTM': {
        'lstm_units': 64,
        'dropout_rate': 0.2,
        'epochs': 50,
        'batch_size': 32
    },
    'SVM': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42
    }
}

# Data Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'scale_features': True,
    'remove_outliers': True,
    'outlier_threshold': 3  # Standard deviations
}

# Evaluation Metrics (as mentioned in paper)
EVALUATION_METRICS = [
    'accuracy',
    'precision', 
    'recall',
    'f1_score',
    'auc_roc',
    'pr_auc',
    'mcc'  # Matthews Correlation Coefficient
]

# Results and Output Configuration
OUTPUT_CONFIG = {
    'save_models': True,
    'save_results': True,
    'save_plots': True,
    'results_dir': '../results',
    'models_dir': '../models',
    'data_dir': '../data',
    'plots_dir': '../results/plots'
}

# Quick Test Configuration (for faster testing)
QUICK_TEST_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2021-12-31',
    'cae_epochs': 20,
    'model_epochs': 20,
    'bayesian_calls': 5,
    'sequence_length': 20,
    'encoding_dim': 10
}

# Paper Information
PAPER_INFO = {
    'title': 'Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities',
    'journal': 'Heliyon',
    'year': 2024,
    'doi': '10.1016/j.heliyon.2024.e27747',
    'authors': ['Saima Latif', 'Nadeem Javaid', 'Faheem Aslam', 'Abdulaziz Aldegheishem', 'Nabil Alrajeh', 'Safdar Hussain Bouk'],
    'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10963254/'
}

# Expected Results (as reported in the paper)
EXPECTED_RESULTS = {
    'SP500': {
        'accuracy': 0.85,
        'precision': 0.87,
        'recall': 0.87,
        'f1_score': 0.87,
        'auc_roc': 0.93,
        'mcc': 0.70
    },
    'FTSE': {
        'accuracy': 0.96,
        'precision': 0.97,
        'recall': 0.96,
        'f1_score': 0.96,
        'auc_roc': 0.99,
        'mcc': 0.92
    },
    'SSE': {
        'accuracy': 0.88,
        'precision': 0.89,
        'recall': 0.89,
        'f1_score': 0.89,
        'auc_roc': 0.95,
        'mcc': 0.76
    },
    'NIFTY': {
        'accuracy': 0.85,
        'precision': 0.85,
        'recall': 0.89,
        'f1_score': 0.87,
        'auc_roc': 0.92,
        'mcc': 0.70
    }
}
