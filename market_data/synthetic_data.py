"""
market_data/synthetic_data.py — Synthetic OHLCV data generator
Used by generated strategy scripts when no live data is available.
Produces realistic-looking BTC/ETH price data with trends, volatility regimes.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def generate_ohlcv(
    n_bars: int = 500,
    start_price: float = 30000.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    timeframe_minutes: int = 60,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data.
    
    Parameters:
        n_bars: Number of candles
        start_price: Starting price
        volatility: Per-bar volatility (fraction)
        trend: Per-bar drift
        timeframe_minutes: Minutes per bar
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate close prices with GBM (geometric brownian motion) + volatility regimes
    closes = [start_price]
    vol_regime = volatility

    for i in range(1, n_bars):
        # Occasionally shift volatility regime
        if np.random.random() < 0.03:
            vol_regime = np.random.uniform(volatility * 0.5, volatility * 2.5)

        # Add mean reversion tendency
        drift = trend + (start_price - closes[-1]) / start_price * 0.01

        shock = np.random.normal(drift, vol_regime)

        # Occasional fat-tail events (flash crashes / pumps)
        if np.random.random() < 0.01:
            shock += np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.08)

        new_close = closes[-1] * (1 + shock)
        new_close = max(new_close, start_price * 0.1)  # Floor
        closes.append(new_close)

    closes = np.array(closes)

    # Generate OHLC from close prices
    opens = np.roll(closes, 1)
    opens[0] = start_price

    # High and Low
    bar_range = closes * np.abs(np.random.normal(0, volatility * 0.8, n_bars))
    highs = np.maximum(closes, opens) + bar_range * np.random.uniform(0.3, 1.0, n_bars)
    lows = np.minimum(closes, opens) - bar_range * np.random.uniform(0.3, 1.0, n_bars)
    lows = np.maximum(lows, closes * 0.85)  # Prevent extreme lows

    # Volume (correlated with price moves)
    base_volume = 1000.0
    price_changes = np.abs(np.diff(closes, prepend=closes[0]) / closes)
    volumes = base_volume * (1 + price_changes * 10) * np.random.lognormal(0, 0.5, n_bars)

    # Build timestamps
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=timeframe_minutes * n_bars)
    timestamps = pd.date_range(start=start_time, periods=n_bars, freq=f"{timeframe_minutes}min")

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }, index=timestamps)

    return df


def generate_multi_regime(
    n_bars: int = 1000,
    start_price: float = 30000.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate data with distinct market regimes:
    - Bull trend
    - Bear trend  
    - Sideways/chop
    - High volatility
    """
    if seed is not None:
        np.random.seed(seed)

    segments = []
    price = start_price
    bars_per_segment = n_bars // 4

    regimes = [
        {"trend": 0.0005,  "volatility": 0.015, "label": "bull"},    # Bull
        {"trend": -0.0004, "volatility": 0.020, "label": "bear"},    # Bear
        {"trend": 0.00001, "volatility": 0.008, "label": "chop"},    # Sideways
        {"trend": 0.0001,  "volatility": 0.035, "label": "volatile"}, # High vol
    ]

    for regime in regimes:
        seg = generate_ohlcv(
            n_bars=bars_per_segment,
            start_price=price,
            volatility=regime["volatility"],
            trend=regime["trend"],
        )
        segments.append(seg)
        price = seg["close"].iloc[-1]

    df = pd.concat(segments)
    df.index = pd.date_range(
        end=datetime.utcnow(),
        periods=len(df),
        freq="1h"
    )

    return df


# Convenience functions for generated scripts to import easily
def get_btc_data(n_bars: int = 500) -> pd.DataFrame:
    return generate_multi_regime(n_bars=n_bars, start_price=42000.0, seed=42)


def get_eth_data(n_bars: int = 500) -> pd.DataFrame:
    return generate_ohlcv(n_bars=n_bars, start_price=2500.0, volatility=0.022, seed=123)


def get_alt_data(n_bars: int = 500) -> pd.DataFrame:
    return generate_ohlcv(n_bars=n_bars, start_price=1.0, volatility=0.035, seed=999)


if __name__ == "__main__":
    df = get_btc_data(500)
    print(f"Generated {len(df)} bars")
    print(df.tail())
    print(f"\nPrice range: ${df['low'].min():.0f} — ${df['high'].max():.0f}")
