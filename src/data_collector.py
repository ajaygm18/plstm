"""
Data Collection Module for PLSTM-TAL Stock Market Prediction
Based on: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"
Author: Implementation of research paper
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockDataCollector:
    """
    Collects real stock market data for the four indices mentioned in the paper:
    - S&P 500 (^GSPC) - United States
    - FTSE 100 (^FTSE) - United Kingdom  
    - SSE Composite (000001.SS) - China
    - Nifty 50 (^NSEI) - India
    """
    
    def __init__(self):
        self.tickers = {
            'SP500': '^GSPC',
            'FTSE': '^FTSE', 
            'SSE': '000001.SS',
            'NIFTY': '^NSEI'
        }
        
    def download_data(self, start_date='2005-01-01', end_date='2022-03-31'):
        """
        Download historical data for all four stock indices
        Period: January 01, 2005 – March 31, 2022 (as mentioned in paper)
        """
        data = {}
        
        for name, ticker in self.tickers.items():
            print(f"Downloading data for {name} ({ticker})...")
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not stock_data.empty:
                    data[name] = stock_data
                    print(f"Successfully downloaded {len(stock_data)} records for {name}")
                else:
                    print(f"No data found for {name}")
            except Exception as e:
                print(f"Error downloading data for {name}: {str(e)}")
                
        return data
    
    def calculate_technical_indicators(self, data):
        """
        Calculate 40 technical indicators as mentioned in the paper
        These are the same indicators as used in reference [17] of the paper
        """
        df = data.copy()
        
        # Debug info
        print(f"  Input data shape: {df.shape}")
        print(f"  Input columns: {df.columns.tolist()}")
        
        # Fix MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            print(f"  Fixed columns: {df.columns.tolist()}")
        
        # Convert price columns to numpy arrays for TA-Lib
        # Ensure arrays are 1D and float64
        close = np.array(df['Close'].values, dtype=np.float64).flatten()
        high = np.array(df['High'].values, dtype=np.float64).flatten()
        low = np.array(df['Low'].values, dtype=np.float64).flatten()
        volume = np.array(df['Volume'].values, dtype=np.float64).flatten()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        df['SMA_5'] = talib.SMA(close, timeperiod=5)
        df['SMA_10'] = talib.SMA(close, timeperiod=10)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['SMA_200'] = talib.SMA(close, timeperiod=200)
        
        df['EMA_5'] = talib.EMA(close, timeperiod=5)
        df['EMA_10'] = talib.EMA(close, timeperiod=10)
        df['EMA_20'] = talib.EMA(close, timeperiod=20)
        df['EMA_50'] = talib.EMA(close, timeperiod=50)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        # Handle division by zero and ensure single series
        bb_range = (df['BB_Upper'] - df['BB_Lower']).fillna(1.0)  # Avoid division by zero
        bb_range = bb_range.replace(0, 1.0)
        df['BB_Position'] = ((df['Close'] - df['BB_Lower']) / bb_range).fillna(0)
        
        # RSI (Relative Strength Index)
        df['RSI_14'] = talib.RSI(close, timeperiod=14)
        df['RSI_9'] = talib.RSI(close, timeperiod=9)
        df['RSI_25'] = talib.RSI(close, timeperiod=25)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        
        # Williams %R
        df['Williams_R'] = talib.WILLR(high, low, close)
        
        # Commodity Channel Index
        df['CCI'] = talib.CCI(high, low, close)
        
        # Average True Range
        df['ATR'] = talib.ATR(high, low, close)
        
        # Volume indicators
        df['Volume_SMA_20'] = talib.SMA(volume, timeperiod=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # On-Balance Volume
        df['OBV'] = talib.OBV(close, volume)
        
        # Accumulation/Distribution Line
        df['AD'] = talib.AD(high, low, close, volume)
        
        # Money Flow Index
        df['MFI'] = talib.MFI(high, low, close, volume)
        
        # Price Rate of Change
        df['ROC_10'] = talib.ROC(close, timeperiod=10)
        df['ROC_20'] = talib.ROC(close, timeperiod=20)
        
        # Momentum
        df['MOM_10'] = talib.MOM(close, timeperiod=10)
        df['MOM_20'] = talib.MOM(close, timeperiod=20)
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(high, low)
        
        # Average Directional Index
        df['ADX'] = talib.ADX(high, low, close)
        
        # Triple Exponential Moving Average
        df['TEMA'] = talib.TEMA(close)
        
        # Triangular Moving Average
        df['TRIMA'] = talib.TRIMA(close)
        
        # Weighted Moving Average
        df['WMA'] = talib.WMA(close)
        
        # Additional indicators to reach 40 total
        aroon_up, aroon_down = talib.AROON(high, low)
        df['AROON_UP'] = aroon_up
        df['AROON_DOWN'] = aroon_down
        df['AROONOSC'] = talib.AROONOSC(high, low)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close)
        df['DX'] = talib.DX(high, low, close)
        
        # Price position indicators
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        return df
    
    def create_target_labels(self, data):
        """
        Create binary target labels as described in the paper
        Target variable y_i(t) = 1 if return at t+1 > return at t, else 0
        """
        df = data.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Next_Return'] = df['Returns'].shift(-1)
        df['Target'] = (df['Next_Return'] > df['Returns']).astype(int)
        
        return df
    
    def preprocess_data(self, data):
        """
        Preprocess data: handle missing values and normalize
        """
        # Handle missing values
        data = data.dropna()
        
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        return data
    
    def save_data(self, data_dict, filepath):
        """
        Save processed data to files
        """
        for name, data in data_dict.items():
            filename = f"{filepath}/{name}_processed.csv"
            data.to_csv(filename)
            print(f"Saved {name} data to {filename}")

if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector()
    
    # Download data
    raw_data = collector.download_data()
    
    # Process each dataset
    processed_data = {}
    for name, data in raw_data.items():
        print(f"\nProcessing {name}...")
        
        # Calculate technical indicators
        data_with_indicators = collector.calculate_technical_indicators(data)
        
        # Create target labels
        data_with_targets = collector.create_target_labels(data_with_indicators)
        
        # Preprocess
        processed = collector.preprocess_data(data_with_targets)
        
        processed_data[name] = processed
        print(f"Final shape for {name}: {processed.shape}")
    
    # Save processed data
    collector.save_data(processed_data, "../data")
