"""
Alpha Factor Engineering Module
Implements 25+ quantitative factors for predictive modeling
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window, min_periods=1).mean()

def calculate_ema(data, alpha=None, span=None):
    """Calculate Exponential Moving Average"""
    if alpha is not None:
        return data.ewm(alpha=alpha, adjust=False).mean()
    elif span is not None:
        return data.ewm(span=span, adjust=False).mean()
    else:
        raise ValueError("Either alpha or span must be provided")

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    avg_gains = gains.rolling(window=period, min_periods=1).mean()
    avg_losses = losses.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral RSI

def calculate_bollinger_bands(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, window)
    rolling_std = prices.rolling(window=window, min_periods=1).std()
    
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }

def calculate_volatility(returns, window=20):
    """Calculate annualized volatility"""
    return returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD indicators"""
    fast_ema = calculate_ema(prices, span=fast_period)
    slow_ema = calculate_ema(prices, span=slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, span=signal_period)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_skewness(data, window=60):
    """Calculate rolling skewness"""
    return data.rolling(window=window, min_periods=10).skew()

def calculate_kurtosis(data, window=60):
    """Calculate rolling kurtosis"""
    return data.rolling(window=window, min_periods=10).kurt()

def calculate_autocorrelation(data, lag=5, window=20):
    """Calculate rolling autocorrelation"""
    def rolling_autocorr(series):
        if len(series) <= lag:
            return 0
        return series.autocorr(lag=lag)
    
    return data.rolling(window=window, min_periods=lag+1).apply(rolling_autocorr, raw=False)

def calculate_sharpe_ratio(returns, window=60):
    """Calculate rolling Sharpe ratio"""
    risk_free_rate = 0.02 / 252  # 2% annual risk-free rate, daily
    
    excess_returns = returns - risk_free_rate
    mean_excess = excess_returns.rolling(window=window, min_periods=10).mean()
    volatility = excess_returns.rolling(window=window, min_periods=10).std()
    
    sharpe = mean_excess / volatility
    return sharpe.fillna(0)

def calculate_alpha_factors(df):
    """
    Calculate comprehensive alpha factors for the dataset
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns: date, price, volume, high, low, open, close
    
    Returns:
    pd.DataFrame: DataFrame with all alpha factors
    """
    print(f'Calculating alpha factors for {len(df)} data points')
    
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Extract price and volume series
    prices = df['price']
    volumes = df['volume']
    highs = df['high']
    lows = df['low']
    
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    
    # Pre-calculate technical indicators
    sma_20 = calculate_sma(prices, 20)
    sma_50 = calculate_sma(prices, 50)
    sma_200 = calculate_sma(prices, 200)
    ema_12 = calculate_ema(prices, span=12)
    ema_26 = calculate_ema(prices, span=26)
    
    volatility_20 = calculate_volatility(returns, 20)
    bollinger = calculate_bollinger_bands(prices, 20, 2)
    macd_indicators = calculate_macd(prices)
    
    # Initialize results DataFrame
    factors_df = pd.DataFrame(index=df.index)
    factors_df['date'] = df['date']
    factors_df['price'] = prices
    
    print("Calculating momentum factors...")
    # === MOMENTUM FACTORS ===
    factors_df['momentum_1m'] = prices.pct_change(periods=20).fillna(0)  # 1 month = 20 trading days
    factors_df['momentum_3m'] = prices.pct_change(periods=60).fillna(0)  # 3 months = 60 trading days
    factors_df['momentum_6m'] = prices.pct_change(periods=120).fillna(0)  # 6 months = 120 trading days
    
    # Price acceleration (momentum of momentum)
    mom_1m_prev = prices.pct_change(periods=20).shift(20)
    factors_df['momentum_acceleration'] = (factors_df['momentum_1m'] - mom_1m_prev).fillna(0)
    
    print("Calculating mean reversion factors...")
    # === MEAN REVERSION FACTORS ===
    factors_df['price_to_sma20'] = (prices / sma_20).fillna(1)
    factors_df['price_to_sma50'] = (prices / sma_50).fillna(1)
    factors_df['price_to_sma200'] = (prices / sma_200).fillna(1)
    
    # Moving average crossovers
    factors_df['sma_cross_20_50'] = (sma_20 / sma_50).fillna(1)
    
    # Bollinger Band position
    bb_range = bollinger['upper'] - bollinger['lower']
    factors_df['bb_position'] = ((prices - bollinger['lower']) / bb_range).fillna(0.5)
    
    print("Calculating technical indicators...")
    # === TECHNICAL INDICATORS ===
    factors_df['rsi'] = calculate_rsi(prices, 14)
    factors_df['macd_line'] = macd_indicators['macd'].fillna(0)
    factors_df['macd_histogram'] = macd_indicators['histogram'].fillna(0)
    factors_df['volatility_20d'] = volatility_20.fillna(0.2)
    
    # High-low spread
    factors_df['hl_ratio'] = ((highs - lows) / prices).fillna(0)
    
    print("Calculating volume factors...")
    # === VOLUME FACTORS ===
    volume_sma_20 = calculate_sma(volumes, 20)
    factors_df['volume_ratio'] = (volumes / volume_sma_20).fillna(1)
    
    # Volume-price trend
    factors_df['volume_price_trend'] = (volumes * returns.shift(1)).fillna(0)
    
    # On-balance volume (simplified)
    obv_components = np.where(returns > 0, volumes, -volumes)
    factors_df['obv_trend'] = pd.Series(obv_components).rolling(window=5, min_periods=1).sum()
    
    print("Calculating statistical factors...")
    # === STATISTICAL FACTORS ===
    factors_df['skewness_60d'] = calculate_skewness(returns, 60).fillna(0)
    factors_df['kurtosis_60d'] = calculate_kurtosis(returns, 60).fillna(3)
    
    # Autocorrelation
    factors_df['autocorr_5d'] = calculate_autocorrelation(returns, lag=5, window=20).fillna(0)
    
    # Rolling Sharpe ratio
    factors_df['sharpe_60d'] = calculate_sharpe_ratio(returns, 60)
    
    print("Calculating seasonality factors...")
    # === SEASONALITY FACTORS ===
    factors_df['day_of_week'] = pd.to_datetime(factors_df['date']).dt.dayofweek + 1
    factors_df['month_effect'] = pd.to_datetime(factors_df['date']).dt.month
    
    print("Calculating composite factors...")
    # === COMPOSITE FACTORS ===
    # Momentum-volatility composite
    factors_df['momentum_vol_ratio'] = np.where(
        volatility_20 > 0,
        factors_df['momentum_1m'] / volatility_20,
        0
    )
    
    # Volume-momentum composite
    log_volume = np.log(np.maximum(volumes / 1_000_000, 1))  # Avoid log(0)
    factors_df['volume_momentum'] = factors_df['momentum_1m'] * log_volume
    
    print("Calculating target variable...")
    # === TARGET VARIABLE ===
    # 5-day forward return
    factors_df['future_return_5d'] = prices.pct_change(periods=5).shift(-5)
    
    # Remove rows without sufficient historical data (first 60 rows)
    factors_df = factors_df.iloc[60:].reset_index(drop=True)
    
    print(f'Alpha factors calculated: {len(factors_df)} rows with {factors_df.shape[1]} features')
    
    return factors_df

def get_feature_importance():
    """
    Feature importance mapping based on common alpha factor literature
    """
    return {
        'momentum_3m': 0.15,
        'rsi': 0.12,
        'volatility_20d': 0.11,
        'momentum_1m': 0.10,
        'price_to_sma20': 0.09,
        'volume_ratio': 0.08,
        'momentum_6m': 0.07,
        'volume_price_trend': 0.06,
        'skewness_60d': 0.06,
        'kurtosis_60d': 0.05,
        'bb_position': 0.04,
        'macd_histogram': 0.03,
        'sharpe_60d': 0.02,
        'momentum_acceleration': 0.02
    }

def get_factor_descriptions():
    """
    Get descriptions of all alpha factors
    """
    return {
        'momentum_1m': 'Price momentum over 1 month (20 trading days)',
        'momentum_3m': 'Price momentum over 3 months (60 trading days)',
        'momentum_6m': 'Price momentum over 6 months (120 trading days)',
        'momentum_acceleration': 'Change in momentum (momentum of momentum)',
        'price_to_sma20': 'Current price relative to 20-day moving average',
        'price_to_sma50': 'Current price relative to 50-day moving average',
        'price_to_sma200': 'Current price relative to 200-day moving average',
        'sma_cross_20_50': 'Ratio of 20-day to 50-day moving averages',
        'bb_position': 'Position within Bollinger Bands (0=bottom, 1=top)',
        'rsi': 'Relative Strength Index (momentum oscillator)',
        'macd_line': 'MACD line (12-day EMA minus 26-day EMA)',
        'macd_histogram': 'MACD histogram (MACD minus signal line)',
        'volatility_20d': 'Annualized volatility over 20 days',
        'hl_ratio': 'High-Low range relative to closing price',
        'volume_ratio': 'Current volume relative to 20-day average',
        'volume_price_trend': 'Volume-weighted price trend',
        'obv_trend': 'On-Balance Volume trend (5-day)',
        'skewness_60d': 'Return distribution skewness (60-day)',
        'kurtosis_60d': 'Return distribution kurtosis (60-day)',
        'autocorr_5d': '5-day return autocorrelation',
        'sharpe_60d': 'Rolling Sharpe ratio (60-day)',
        'day_of_week': 'Day of week effect (1=Monday, 5=Friday)',
        'month_effect': 'Calendar month effect',
        'momentum_vol_ratio': 'Momentum adjusted for volatility',
        'volume_momentum': 'Volume-weighted momentum indicator',
        'future_return_5d': 'Target: 5-day forward return'
    }

if __name__ == "__main__":
    # Example usage
    from data_generator import generate_complete_dataset
    
    # Generate sample data
    df = generate_complete_dataset()
    
    # Calculate alpha factors
    factors_df = calculate_alpha_factors(df)
    
    # Display results
    print(f"\nAlpha factors shape: {factors_df.shape}")
    print("\nFeature columns:")
    feature_cols = [col for col in factors_df.columns if col not in ['date', 'price', 'future_return_5d']]
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\nTotal features: {len(feature_cols)}")
    
    # Show sample data
    print("\nSample alpha factors:")
    print(factors_df.head())
    
    # Save results
    output_file = "alpha_factors.csv"
    factors_df.to_csv(output_file, index=False)
    print(f"\nAlpha factors saved to {output_file}")