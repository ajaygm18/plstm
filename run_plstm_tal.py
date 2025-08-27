"""
Main Execution Script for PLSTM-TAL Stock Market Prediction
Complete implementation of the research paper:
"Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import all required modules
from src.main_pipeline import run_complete_pipeline
from src.benchmark_models import run_benchmark_comparison
from config import *
import pandas as pd
import numpy as np

def print_paper_info():
    """Print information about the research paper being implemented"""
    print("=" * 80)
    print("PLSTM-TAL STOCK MARKET PREDICTION IMPLEMENTATION")
    print("=" * 80)
    print(f"Paper: {PAPER_INFO['title']}")
    print(f"Journal: {PAPER_INFO['journal']} ({PAPER_INFO['year']})")
    print(f"DOI: {PAPER_INFO['doi']}")
    print(f"Authors: {', '.join(PAPER_INFO['authors'])}")
    print(f"URL: {PAPER_INFO['url']}")
    print("=" * 80)
    print()

def print_methodology():
    """Print the methodology being implemented"""
    print("METHODOLOGY IMPLEMENTATION:")
    print("-" * 40)
    print("1. Data Collection: 4 stock indices (S&P 500, FTSE, SSE, Nifty)")
    print("   Period: January 2005 - March 2022")
    print()
    print("2. Technical Indicators: 40 indicators calculated using TA-Lib")
    print("   Including SMA, EMA, RSI, MACD, Bollinger Bands, etc.")
    print()
    print("3. EEMD Decomposition: Signal filtering using Ensemble Empirical Mode Decomposition")
    print("   Removes highest entropy IMF component")
    print()
    print("4. Feature Extraction: Contractive Autoencoder with Frobenius norm regularization")
    print(f"   Reduces {len(TECHNICAL_INDICATORS)} features to {CAE_CONFIG['encoding_dim']} features")
    print()
    print("5. PLSTM-TAL Model: Peephole LSTM with Temporal Attention Layer")
    print("   - Peephole connections allow gates to access cell state")
    print("   - Temporal attention focuses on relevant time steps")
    print()
    print("6. Hyperparameter Optimization: Bayesian optimization")
    print("   Optimizes units, activation, optimizer, loss, dropout")
    print()
    print("7. Benchmark Comparison: CNN, LSTM, SVM, Random Forest")
    print("   Comprehensive evaluation with 7 metrics")
    print("=" * 80)
    print()

def run_quick_demo():
    """Run a quick demonstration with reduced parameters"""
    print("RUNNING QUICK DEMONSTRATION")
    print("=" * 40)
    print("Note: Using reduced parameters for faster execution")
    print("For full implementation, set quick_test=False in run_complete_pipeline()")
    print()
    
    # Run pipeline for S&P 500
    pipeline, metrics, report = run_complete_pipeline(
        target_index='SP500',
        optimize_hyperparams=True,
        quick_test=True  # Use reduced parameters
    )
    
    if pipeline is not None and metrics is not None:
        print("\nQUICK DEMO RESULTS:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return pipeline, metrics, report
    else:
        print("Demo failed. Please check error messages above.")
        return None, None, None

def run_full_implementation():
    """Run the complete implementation as described in the paper"""
    print("RUNNING FULL IMPLEMENTATION")
    print("=" * 40)
    print("This will take significant time. Please be patient.")
    print()
    
    results = {}
    
    # Run for all indices mentioned in the paper
    indices_to_process = ['SP500', 'FTSE', 'SSE', 'NIFTY']
    
    for index in indices_to_process:
        print(f"\nProcessing {index}...")
        print("-" * 30)
        
        try:
            pipeline, metrics, report = run_complete_pipeline(
                target_index=index,
                optimize_hyperparams=True,
                quick_test=False  # Full implementation
            )
            
            if metrics is not None:
                results[index] = metrics
                print(f"{index} completed successfully!")
            else:
                print(f"{index} failed!")
                
        except Exception as e:
            print(f"Error processing {index}: {str(e)}")
            continue
    
    # Print comparative results
    if results:
        print("\nFULL IMPLEMENTATION RESULTS:")
        print("=" * 60)
        
        results_df = pd.DataFrame(results).T
        print(results_df.round(4))
        
        # Compare with expected results from paper
        print("\nCOMPARISON WITH PAPER RESULTS:")
        print("-" * 40)
        
        for index in results:
            if index in EXPECTED_RESULTS:
                print(f"\n{index}:")
                print("Metric    | Our Result | Paper Result | Difference")
                print("-" * 50)
                
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']:
                    if metric in results[index] and metric in EXPECTED_RESULTS[index]:
                        our_result = results[index][metric]
                        paper_result = EXPECTED_RESULTS[index][metric]
                        diff = our_result - paper_result
                        print(f"{metric:9} | {our_result:10.4f} | {paper_result:11.4f} | {diff:+9.4f}")
        
        return results
    else:
        print("No successful results obtained.")
        return None

def demonstrate_individual_components():
    """Demonstrate each component individually"""
    print("DEMONSTRATING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    try:
        # 1. Data Collection
        print("\n1. TESTING DATA COLLECTION:")
        print("-" * 30)
        from src.data_collector import StockDataCollector
        
        collector = StockDataCollector()
        # Use recent data for quick test
        raw_data = collector.download_data('2023-01-01', '2023-12-31')
        
        if raw_data:
            print("✓ Data collection successful")
            for name, data in raw_data.items():
                print(f"  {name}: {len(data)} records")
        else:
            print("✗ Data collection failed")
        
        # 2. Technical Indicators
        print("\n2. TESTING TECHNICAL INDICATORS:")
        print("-" * 30)
        
        if raw_data and 'SP500' in raw_data:
            indicators_data = collector.calculate_technical_indicators(raw_data['SP500'])
            print(f"✓ Technical indicators calculated: {indicators_data.shape[1]} features")
        else:
            print("✗ Technical indicators test skipped (no data)")
        
        # 3. EEMD Decomposition
        print("\n3. TESTING EEMD DECOMPOSITION:")
        print("-" * 30)
        
        try:
            from src.eemd_decomposer import EEMDDecomposer
            
            decomposer = EEMDDecomposer()
            
            if raw_data and 'SP500' in raw_data:
                test_signal = raw_data['SP500']['Close'].dropna().values[:200]  # Small sample
                imfs = decomposer.decompose_signal(test_signal)
                
                if imfs is not None:
                    print(f"✓ EEMD decomposition successful: {len(imfs)} components")
                else:
                    print("✗ EEMD decomposition failed")
            else:
                print("✗ EEMD test skipped (no data)")
                
        except Exception as e:
            print(f"✗ EEMD test failed: {str(e)}")
        
        # 4. Contractive Autoencoder
        print("\n4. TESTING CONTRACTIVE AUTOENCODER:")
        print("-" * 30)
        
        try:
            from src.contractive_autoencoder import ContractiveAutoencoder
            
            # Create sample data
            sample_data = np.random.randn(100, 20)
            
            cae = ContractiveAutoencoder(input_dim=20, encoding_dim=10)
            cae.build_model()
            
            print("✓ CAE model built successfully")
            print(f"  Input dimension: 20, Output dimension: 10")
            
        except Exception as e:
            print(f"✗ CAE test failed: {str(e)}")
        
        # 5. PLSTM-TAL Model
        print("\n5. TESTING PLSTM-TAL MODEL:")
        print("-" * 30)
        
        try:
            from src.plstm_tal_model import PLSTM_TAL
            
            model = PLSTM_TAL(sequence_length=30, n_features=21, lstm_units=32)
            model.build_model()
            model.compile_model()
            
            print("✓ PLSTM-TAL model built successfully")
            print(f"  Architecture: Peephole LSTM + Temporal Attention")
            
        except Exception as e:
            print(f"✗ PLSTM-TAL test failed: {str(e)}")
        
        print("\nComponent testing completed!")
        
    except Exception as e:
        print(f"Error in component testing: {str(e)}")

def main():
    """Main execution function"""
    print_paper_info()
    print_methodology()
    
    print("SELECT EXECUTION MODE:")
    print("1. Quick Demo (reduced parameters, ~10-15 minutes)")
    print("2. Full Implementation (all indices, full parameters, ~2-4 hours)")
    print("3. Component Testing (test individual components)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        print("\nStarting Quick Demo...")
        pipeline, metrics, report = run_quick_demo()
        
        if metrics is not None:
            print("\n✓ Quick demo completed successfully!")
            print("See results above and generated plots.")
        else:
            print("\n✗ Quick demo failed. Check error messages.")
            
    elif choice == '2':
        print("\nStarting Full Implementation...")
        confirm = input("This will take 2-4 hours. Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            results = run_full_implementation()
            
            if results:
                print("\n✓ Full implementation completed successfully!")
                print("Results have been saved to the results directory.")
            else:
                print("\n✗ Full implementation failed.")
        else:
            print("Full implementation cancelled.")
            
    elif choice == '3':
        print("\nStarting Component Testing...")
        demonstrate_individual_components()
        
    elif choice == '4':
        print("Exiting...")
        return
        
    else:
        print("Invalid choice. Please run again and select 1-4.")

if __name__ == "__main__":
    main()
