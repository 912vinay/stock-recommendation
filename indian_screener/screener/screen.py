from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from .config import ScreenerConfig
from .fundamentals import compute_fundamental_snapshot
from .nse import fetch_promoter_shareholding_percent
from .technical import compute_technical_snapshot


def _score_buy_row(row: pd.Series, cfg: ScreenerConfig) -> float:
    score = 0.0
    # Valuation
    if row.get("pe") and cfg.valuation.min_pe <= row["pe"] <= cfg.valuation.max_pe:
        score += 1.0
    if row.get("pb") and row["pb"] <= cfg.valuation.max_pb:
        score += 1.0
    if row.get("ev_ebitda") and cfg.valuation.max_ev_ebitda and row["ev_ebitda"] <= cfg.valuation.max_ev_ebitda:
        score += 1.0
    # Growth
    if row.get("revenue_cagr_3y") and row["revenue_cagr_3y"] >= cfg.growth.min_revenue_cagr_3y:
        score += 1.0
    if row.get("eps_cagr_3y") and row["eps_cagr_3y"] >= cfg.growth.min_eps_cagr_3y:
        score += 1.0
    # Quality
    if row.get("roe") and row["roe"] >= cfg.quality.min_roe:
        score += 1.0
    if row.get("roce") and row["roce"] >= cfg.quality.min_roce:
        score += 1.0
    if row.get("debt_to_equity") is not None and row["debt_to_equity"] <= cfg.quality.max_debt_to_equity:
        score += 1.0
    if row.get("interest_coverage") and row["interest_coverage"] >= cfg.quality.min_interest_coverage:
        score += 1.0
    # Promoter
    if row.get("promoter_change_qoq_pct_pts") is not None and row["promoter_change_qoq_pct_pts"] >= cfg.promoter.min_promoter_change_qoq_pct_pts:
        score += 1.0
    # Technicals
    if row.get("price_above_200d"):
        score += 1.0
    if row.get("sma50_above_200d"):
        score += 1.0
    if row.get("within_pct_52w_high") is not None and row["within_pct_52w_high"] <= cfg.technical.within_pct_52w_high:
        score += 1.0
    if row.get("rsi14") and cfg.technical.rsi_min <= row["rsi14"] <= cfg.technical.rsi_max:
        score += 0.5
    if row.get("volume_mult_vs_50d") and row["volume_mult_vs_50d"] >= cfg.technical.min_volume_multiple_vs_50d:
        score += 0.5
    return score


def _download_history_per_symbol(symbols: List[str], cfg: ScreenerConfig) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(
                tickers=sym,
                period=f"{cfg.run.lookback_days}d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                results[sym] = df.copy()
        except Exception:
            pass
        time.sleep(max(0.0, cfg.run.download_pause_sec))
    return results


def run_screen(symbols: List[str], cfg: ScreenerConfig, for_buy: bool = True) -> pd.DataFrame:
    # Fetch price history
    if cfg.run.single_download:
        hist_all = _download_history_per_symbol(symbols, cfg)
    else:
        hist_all: Dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), cfg.run.batch_size):
            batch = symbols[i : i + cfg.run.batch_size]
            try:
                hist = yf.download(
                    tickers=batch,
                    period=f"{cfg.run.lookback_days}d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )
            except Exception:
                hist = pd.DataFrame()
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                if set(["Open", "High", "Low", "Close", "Adj Close", "Volume"]).issubset(hist.columns):
                    hist_all[batch[0]] = hist.copy()
                else:
                    for t in batch:
                        try:
                            df_t = hist[t].copy()
                            hist_all[t] = df_t
                        except Exception:
                            pass

    # Build technical-only rows first
    tech_rows: List[Dict] = []
    for yticker in symbols:
        base: Dict[str, float | int | str | bool | None] = {"ticker": yticker}
        tech_df = hist_all.get(yticker, pd.DataFrame())
        tech = compute_technical_snapshot(tech_df) if isinstance(tech_df, pd.DataFrame) and not tech_df.empty else {}
        base.update(tech)
        for key in [
            "close","high_52w","low_52w","sma50","sma200","rsi14","vol","vol50","price_above_200d","sma50_above_200d","volume_mult_vs_50d",
        ]:
            base.setdefault(key, None)
        if base.get("close") and base.get("high_52w"):
            base["within_pct_52w_high"] = (1.0 - (base["close"] / base["high_52w"])) * 100.0
        else:
            base["within_pct_52w_high"] = None
        tech_rows.append(base)

    df = pd.DataFrame(tech_rows)

    # Prefilter on technicals only if buy screen
    if for_buy and not df.empty:
        tc = cfg.technical
        mask = pd.Series(True, index=df.index)
        if tc.price_above_200d:
            mask &= (df.get("price_above_200d", False) == True)  # noqa: E712
        if tc.sma50_above_200d:
            mask &= (df.get("sma50_above_200d", False) == True)  # noqa: E712
        mask &= (df.get("within_pct_52w_high", np.nan) <= tc.within_pct_52w_high) | df.get("within_pct_52w_high").isna()
        mask &= (df.get("rsi14", np.nan).between(tc.rsi_min, tc.rsi_max)) | df.get("rsi14").isna()
        mask &= (df.get("volume_mult_vs_50d", np.nan) >= tc.min_volume_multiple_vs_50d) | df.get("volume_mult_vs_50d").isna()
        df = df[mask]

    # Limit number of symbols for fundamentals/promoter to reduce API hits
    symbols_for_funda = df["ticker"].tolist() if not df.empty else []
    if cfg.run.fundamentals_max_symbols is not None:
        symbols_for_funda = symbols_for_funda[: cfg.run.fundamentals_max_symbols]

    # Attach fundamentals and promoter (best-effort)
    enriched_rows: List[Dict] = []
    tech_map = {r["ticker"]: r for r in tech_rows}
    for yticker in symbols_for_funda:
        base = dict(tech_map.get(yticker, {"ticker": yticker}))
        f = compute_fundamental_snapshot(yticker, use_yahoo_info_fallback=cfg.run.use_yahoo_info_fallback)
        base.update({
            "market_cap": f.market_cap,
            "pe": f.pe,
            "pb": f.pb,
            "roe": f.roe,
            "roce": f.roce,
            "debt_to_equity": f.debt_to_equity,
            "interest_coverage": f.interest_coverage,
            "revenue_cagr_3y": f.revenue_cagr_3y,
            "eps_cagr_3y": f.eps_cagr_3y,
            "ev_ebitda": f.ev_ebitda,
        })
        try:
            root = yticker.replace(".NS", "")
            prom = fetch_promoter_shareholding_percent(root)
        except Exception:
            prom = None
        if prom is not None:
            base["promoter_latest_pct"] = prom.latest_percent
            base["promoter_prev_pct"] = prom.prev_percent
            base["promoter_change_qoq_pct_pts"] = prom.change_qoq_pct_pts
        else:
            base["promoter_latest_pct"] = None
            base["promoter_prev_pct"] = None
            base["promoter_change_qoq_pct_pts"] = None
        enriched_rows.append(base)

    df = pd.DataFrame(enriched_rows) if enriched_rows else pd.DataFrame(tech_rows)

    # Ensure expected columns
    expected_cols = [
        "market_cap","pe","pb","roe","roce","debt_to_equity","interest_coverage",
        "revenue_cagr_3y","eps_cagr_3y","ev_ebitda",
        "promoter_change_qoq_pct_pts","promoter_latest_pct","promoter_prev_pct",
        "price_above_200d","sma50_above_200d","within_pct_52w_high","rsi14","volume_mult_vs_50d",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Apply hard filters for buy
    if for_buy and not df.empty:
        mask = pd.Series(True, index=df.index)
        # Valuation
        vc = cfg.valuation
        mask &= (df["pe"].between(vc.min_pe, vc.max_pe)) | df["pe"].isna()
        mask &= (df["pb"] <= vc.max_pb) | df["pb"].isna()
        if vc.max_ev_ebitda is not None:
            mask &= (df["ev_ebitda"] <= vc.max_ev_ebitda) | df["ev_ebitda"].isna()
        # Growth
        gc = cfg.growth
        mask &= (df["revenue_cagr_3y"] >= gc.min_revenue_cagr_3y) | df["revenue_cagr_3y"].isna()
        mask &= (df["eps_cagr_3y"] >= gc.min_eps_cagr_3y) | df["eps_cagr_3y"].isna()
        # Quality
        qc = cfg.quality
        mask &= (df["roe"] >= qc.min_roe) | df["roe"].isna()
        mask &= (df["roce"] >= qc.min_roce) | df["roce"].isna()
        mask &= (df["debt_to_equity"] <= qc.max_debt_to_equity) | df["debt_to_equity"].isna()
        mask &= (df["interest_coverage"] >= qc.min_interest_coverage) | df["interest_coverage"].isna()
        # Promoter (only if available)
        pc = cfg.promoter
        mask &= (df["promoter_change_qoq_pct_pts"] >= pc.min_promoter_change_qoq_pct_pts) | df["promoter_change_qoq_pct_pts"].isna()
        # Technicals (already applied earlier, but keep for safety)
        tc = cfg.technical
        if tc.price_above_200d:
            mask &= (df["price_above_200d"] == True)  # noqa: E712
        if tc.sma50_above_200d:
            mask &= (df["sma50_above_200d"] == True)  # noqa: E712
        mask &= (df["within_pct_52w_high"] <= tc.within_pct_52w_high) | df["within_pct_52w_high"].isna()
        mask &= (df["rsi14"].between(tc.rsi_min, tc.rsi_max)) | df["rsi14"].isna()
        mask &= (df["volume_mult_vs_50d"] >= tc.min_volume_multiple_vs_50d) | df["volume_mult_vs_50d"].isna()
        df = df[mask]

    # Score and sort
    if not df.empty:
        scores = df.apply(lambda r: _score_buy_row(r, cfg), axis=1)
        df = df.assign(score=scores).sort_values(["score", "market_cap"], ascending=[False, False])

    return df.reset_index(drop=True)