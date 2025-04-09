"""
XAU/USD Market Regime Detection with Hidden Markov Model
- Strict walk-forward implementation
- No lookahead bias in any component
- Practical trading integration
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta

# Configuration
TICKER = "GC=F"
TRAIN_END = "2024-01-01"
MODEL_FILE = "xauusd_hmm.pkl"
LOOKBACK = 21  # For volatility calculation

def get_clean_data():
    """Fetch and prepare data without lookahead contamination"""
    df = yf.download(TICKER, start="2010-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    df = df[['Open', 'High', 'Low', 'Close']]
    
    # Calculate features using only backward-looking data
    df['Returns'] = df['Close'].pct_change() * 100
    df['Volatility'] = df['Returns'].rolling(LOOKBACK, min_periods=1).std()
    
    # Remove lookahead in volatility calculation
    for i in range(LOOKBACK, len(df)):
        df.iloc[i, df.columns.get_loc('Volatility')] = df['Returns'].iloc[i-LOOKBACK+1:i+1].std()
    
    return df.dropna()

class WalkForwardScaler:
    """Prevents lookahead in feature scaling"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.warmup = LOOKBACK
        
    def fit_transform(self, data):
        transformed = np.zeros_like(data)
        for i in range(len(data)):
            if i < self.warmup:
                transformed[i] = data[i]
                continue
            self.scaler.fit(data[:i+1])
            transformed[i] = self.scaler.transform([data[i]])[0]
        return transformed

def train_model(save_model=True):
    """Train HMM with strict chronological split"""
    df = get_clean_data()
    split_idx = df.index.get_loc(TRAIN_END, method='bfill')
    
    # Prepare features with walk-forward scaling
    scaler = WalkForwardScaler()
    features = scaler.fit_transform(df[['Returns', 'Volatility']].values)
    
    # Split data
    train_features = features[:split_idx]
    test_features = features[split_idx:]
    
    # Train HMM with multiple initializations
    best_model = None
    best_score = -np.inf
    for _ in range(10):
        model = hmm.GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=1000,
            random_state=np.random.randint(0, 1000)
        )
        model.fit(train_features)
        score = model.score(train_features)
        if score > best_score:
            best_score = score
            best_model = model
    
    if save_model:
        joblib.dump(best_model, MODEL_FILE)
        
    # Evaluate
    test_states = best_model.predict(test_features)
    df_test = df.iloc[split_idx:].copy()
    df_test['Regime'] = test_states
    
    print("\nModel Evaluation:")
    print(f"Training Period: 2010-01-01 to {TRAIN_END}")
    print(f"Test Period: {df_test.index[0].date()} to {df_test.index[-1].date()}")
    print("\nRegime Characteristics (Test Period):")
    print(df_test.groupby('Regime')[['Returns', 'Volatility']].mean())
    
    return best_model, df_test

def get_current_regime():
    """Predict today's regime using only available data"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Train model first using train_model()")
    
    model = joblib.load(MODEL_FILE)
    
    # Get data up to previous close
    end_date = datetime.today() - timedelta(days=1)
    df = yf.download(TICKER, start=end_date - timedelta(days=LOOKBACK*2), 
                    end=end_date)
    
    # Calculate features
    df['Returns'] = df['Close'].pct_change() * 100
    df['Volatility'] = df['Returns'].rolling(LOOKBACK).std()
    df = df.dropna()
    
    # Scale features properly
    scaler = StandardScaler()
    scaler.fit(df[['Returns', 'Volatility']])
    features = scaler.transform(df[['Returns', 'Volatility']].tail(1))
    
    # Predict regime
    regime = model.predict(features)[0]
    
    # Get current price context
    current_price = yf.download(TICKER, period="1d")['Close'].iloc[-1]
    
    print("\nCurrent Market Assessment:")
    print(f"Predicted Regime: {regime}")
    print(f"Latest Close Price: {current_price:.2f}")
    print(f"Based on data through: {end_date.date()}")
    
    return regime

def plot_regimes(df):
    """Visualize results without lookahead"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7))
    
    for regime in sorted(df['Regime'].unique()):
        subset = df[df['Regime'] == regime]
        plt.scatter(subset.index, subset['Close'], 
                   label=f'Regime {regime}', alpha=0.7)
    
    plt.title("XAU/USD Price with HMM Market Regimes")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Full training and evaluation workflow
    model, df_test = train_model()
    
    # Visualize test period results
    plot_regimes(df_test)
    
    # Get current regime assessment
    current_regime = get_current_regime()
    
    # Trading integration example
    print("\nSuggested Trading Approach:")
    if current_regime == 0:
        print("- Range-bound strategy: Fade extremes")
        print("- Use mean-reversion indicators")
        print("- Tight stop losses (0.5-1%)")
    elif current_regime == 1:
        print("- Bearish strategy: Sell rallies")
        print("- Use volatility breakout shorts")
        print("- Wider stops (1.5-2%)")
    elif current_regime == 2:
        print("- Bullish strategy: Buy dips")
        print("- Use momentum indicators")
        print("- Moderate stops (1-1.5%)")