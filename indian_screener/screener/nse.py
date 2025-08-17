from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


@dataclass
class PromoterSnapshot:
    latest_percent: float | None
    prev_percent: float | None
    change_qoq_pct_pts: float | None


def _session_with_nse_cookies() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Referer": "https://www.nseindia.com/",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }
    )
    # Prime cookies by hitting homepage
    try:
        s.get("https://www.nseindia.com/", timeout=15)
    except Exception:
        pass
    return s


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=8))
def fetch_promoter_shareholding_percent(symbol: str) -> PromoterSnapshot | None:
    """
    symbol: NSE root symbol without ".NS", e.g. "RELIANCE".
    Returns latest and previous quarter promoter percentage if available.
    """
    session = _session_with_nse_cookies()
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}&section=shareholding"
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    # Parse structure
    try:
        shareholding = data.get("shareholding", {})
        categories = shareholding.get("data", [])
        # Find promoter category time series
        promoter_series = None
        for cat in categories:
            if str(cat.get("category"))[:8].lower().startswith("promoter"):
                promoter_series = cat.get("data", [])
                break
        if not promoter_series:
            return None
        # promoter_series is list of dicts with 'percent' and 'quarterBeginDate'
        promoter_series_sorted = sorted(
            promoter_series,
            key=lambda x: x.get("quarterBeginDate") or x.get("quarterEndDate") or "",
        )
        percents = [p.get("percent") for p in promoter_series_sorted if p.get("percent") is not None]
        if not percents:
            return None
        latest = float(percents[-1])
        prev = float(percents[-2]) if len(percents) >= 2 else None
        change = (latest - prev) if prev is not None else None
        return PromoterSnapshot(latest_percent=latest, prev_percent=prev, change_qoq_pct_pts=change)
    except Exception:
        return None