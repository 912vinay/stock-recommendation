from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

NIFTY_CSV_URLS: dict[str, str] = {
    "NIFTY50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}


def _nse_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=10))
def fetch_nifty_constituents(index_name: str) -> pd.DataFrame:
    index_name = index_name.upper()
    if index_name not in NIFTY_CSV_URLS:
        raise ValueError(f"Unsupported index: {index_name}")
    url = NIFTY_CSV_URLS[index_name]
    resp = requests.get(url, headers=_nse_headers(), timeout=20)
    resp.raise_for_status()
    # Some NSE CSV files are in Windows-1252 or Unicode with BOM
    content = resp.content
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_csv(io.BytesIO(content), encoding="latin1")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Expected column 'Symbol'
    if "Symbol" not in df.columns:
        # Some variants use 'SYMBOL'
        sym_col = next((c for c in df.columns if c.lower() == "symbol"), None)
        if sym_col is None:
            raise ValueError("Symbol column not found in index CSV")
        df.rename(columns={sym_col: "Symbol"}, inplace=True)
    return df


def to_yahoo_ticker(nse_symbol: str) -> str:
    return f"{nse_symbol.strip().upper()}.NS"


def build_universe(index_name: str, limit: int | None = None) -> list[str]:
    df = fetch_nifty_constituents(index_name)
    symbols = [to_yahoo_ticker(s) for s in df["Symbol"].astype(str).tolist()]
    if limit is not None:
        symbols = symbols[: int(limit)]
    return symbols