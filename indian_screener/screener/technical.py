from __future__ import annotations

import numpy as np
import pandas as pd


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=period, min_periods=period).mean()
    avg_loss = down.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_technical_snapshot(df: pd.DataFrame) -> dict[str, float | bool]:
    """
    Expects a DataFrame with columns: ['Close', 'High', 'Low', 'Volume'] indexed by date.
    Returns dict of key technicals for the last available day.
    """
    if df.empty:
        return {}
    close = df["Close"].dropna()
    high = df["High"].dropna()
    low = df["Low"].dropna()
    volume = df["Volume"].dropna()
    last_close = float(close.iloc[-1]) if not close.empty else np.nan

    sma50 = simple_moving_average(close, 50)
    sma200 = simple_moving_average(close, 200)
    rsi14 = rsi(close, 14)
    vol50 = volume.rolling(50, min_periods=10).mean()

    last = {
        "close": last_close,
        "sma50": float(sma50.iloc[-1]) if not sma50.empty else np.nan,
        "sma200": float(sma200.iloc[-1]) if not sma200.empty else np.nan,
        "rsi14": float(rsi14.iloc[-1]) if not rsi14.empty else np.nan,
        "vol": float(volume.iloc[-1]) if not volume.empty else np.nan,
        "vol50": float(vol50.iloc[-1]) if not vol50.empty else np.nan,
        "high_52w": float(high.tail(252).max()) if len(high) >= 20 else np.nan,
        "low_52w": float(low.tail(252).min()) if len(low) >= 20 else np.nan,
    }

    last["price_above_200d"] = (
        np.isfinite(last["close"]) and np.isfinite(last["sma200"]) and last["close"] > last["sma200"]
    )
    last["sma50_above_200d"] = (
        np.isfinite(last["sma50"]) and np.isfinite(last["sma200"]) and last["sma50"] > last["sma200"]
    )
    last["within_10pct_52w_high"] = (
        np.isfinite(last["high_52w"]) and np.isfinite(last["close"]) and last["close"] >= 0.9 * last["high_52w"]
    )
    last["volume_mult_vs_50d"] = (
        last["vol"] / last["vol50"] if np.isfinite(last["vol"]) and np.isfinite(last["vol50"]) and last["vol50"] > 0 else np.nan
    )

    return last