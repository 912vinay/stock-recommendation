from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class FundamentalSnapshot:
    market_cap: float | None
    pe: float | None
    pb: float | None
    roe: float | None
    roce: float | None
    debt_to_equity: float | None
    interest_coverage: float | None
    revenue_cagr_3y: float | None
    eps_cagr_3y: float | None
    ev_ebitda: float | None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        if math.isfinite(f):
            return f
        return None
    except Exception:
        return None


def _compute_cagr(first: float, last: float, years: float) -> float | None:
    if first is None or last is None:
        return None
    try:
        if first <= 0 or last <= 0 or years <= 0:
            return None
        return (last / first) ** (1.0 / years) * 100.0 - 100.0
    except Exception:
        return None


def _get_financials_safe(t: yf.Ticker) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    # yfinance has different fields depending on version; try multiple
    try:
        data["financials_annual"] = t.financials if isinstance(t.financials, pd.DataFrame) else pd.DataFrame()
    except Exception:
        data["financials_annual"] = pd.DataFrame()
    try:
        data["balance_annual"] = t.balance_sheet if isinstance(t.balance_sheet, pd.DataFrame) else pd.DataFrame()
    except Exception:
        data["balance_annual"] = pd.DataFrame()
    try:
        data["cashflow_annual"] = t.cashflow if isinstance(t.cashflow, pd.DataFrame) else pd.DataFrame()
    except Exception:
        data["cashflow_annual"] = pd.DataFrame()

    # Quarterly variants
    try:
        data["financials_quarterly"] = t.quarterly_financials if isinstance(t.quarterly_financials, pd.DataFrame) else pd.DataFrame()
    except Exception:
        data["financials_quarterly"] = pd.DataFrame()
    try:
        data["balance_quarterly"] = t.quarterly_balance_sheet if isinstance(t.quarterly_balance_sheet, pd.DataFrame) else pd.DataFrame()
    except Exception:
        data["balance_quarterly"] = pd.DataFrame()
    try:
        data["cashflow_quarterly"] = t.quarterly_cashflow if isinstance(t.quarterly_cashflow, pd.DataFrame) else pd.DataFrame()
    except Exception:
        data["cashflow_quarterly"] = pd.DataFrame()

    return data


def compute_fundamental_snapshot(ticker: str, use_yahoo_info_fallback: bool = False) -> FundamentalSnapshot:
    t = yf.Ticker(ticker)

    # Fast info for basic ratios
    market_cap = _safe_float(getattr(t, "fast_info", {}).get("market_cap") if hasattr(t, "fast_info") else None)
    pe = _safe_float(getattr(t, "fast_info", {}).get("pe_ratio") if hasattr(t, "fast_info") else None)
    pb = _safe_float(getattr(t, "fast_info", {}).get("price_to_book") if hasattr(t, "fast_info") else None)

    # Optional fallback via .info (slower) if fast_info fails
    if use_yahoo_info_fallback and (market_cap is None or pe is None or pb is None):
        try:
            info: Dict[str, Any] = t.info  # may be slow / sometimes blocked
            market_cap = market_cap or _safe_float(info.get("marketCap"))
            pe = pe or _safe_float(info.get("trailingPE"))
            pb = pb or _safe_float(info.get("priceToBook"))
        except Exception:
            pass

    fins = _get_financials_safe(t)

    # ROE approximation: Net Income / Average Equity
    roe = None
    try:
        fin = fins["financials_annual"]
        bal = fins["balance_annual"]
        if not fin.empty and not bal.empty:
            net_income = fin.loc["Net Income"].dropna()
            total_equity = bal.loc["Stockholders Equity"].dropna()
            if len(net_income) >= 2 and len(total_equity) >= 2:
                ni_last = float(net_income.iloc[0])
                eq_last = float(total_equity.iloc[0])
                eq_prev = float(total_equity.iloc[1])
                avg_eq = (eq_last + eq_prev) / 2.0 if (eq_last and eq_prev) else None
                if avg_eq and avg_eq != 0:
                    roe = (ni_last / avg_eq) * 100.0
    except Exception:
        pass

    # ROCE approximation: EBIT / (Total Assets - Current Liabilities)
    roce = None
    try:
        fin = fins["financials_annual"]
        bal = fins["balance_annual"]
        if not fin.empty and not bal.empty:
            ebit = fin.loc.get("Ebit")
            total_assets = bal.loc.get("Total Assets")
            current_liab = bal.loc.get("Current Liabilities")
            if ebit is not None and total_assets is not None and current_liab is not None:
                ebit_last = float(ebit.dropna().iloc[0]) if not ebit.dropna().empty else None
                assets_last = float(total_assets.dropna().iloc[0]) if not total_assets.dropna().empty else None
                curr_liab_last = float(current_liab.dropna().iloc[0]) if not current_liab.dropna().empty else None
                capital_employed = (assets_last - curr_liab_last) if (assets_last is not None and curr_liab_last is not None) else None
                if capital_employed and capital_employed != 0 and ebit_last is not None:
                    roce = (ebit_last / capital_employed) * 100.0
    except Exception:
        pass

    # Debt to Equity
    debt_to_equity = None
    try:
        bal = fins["balance_annual"]
        if not bal.empty:
            total_debt = bal.loc.get("Total Debt")
            total_equity = bal.loc.get("Stockholders Equity")
            if total_debt is not None and total_equity is not None:
                d = float(total_debt.dropna().iloc[0]) if not total_debt.dropna().empty else None
                e = float(total_equity.dropna().iloc[0]) if not total_equity.dropna().empty else None
                if d is not None and e and e != 0:
                    debt_to_equity = d / e
    except Exception:
        pass

    # Interest coverage: EBIT / Interest Expense
    interest_coverage = None
    try:
        fin = fins["financials_annual"]
        if not fin.empty:
            ebit = fin.loc.get("Ebit")
            interest = fin.loc.get("Interest Expense")
            if ebit is not None and interest is not None:
                ebit_last = float(ebit.dropna().iloc[0]) if not ebit.dropna().empty else None
                interest_last = float(interest.dropna().iloc[0]) if not interest.dropna().empty else None
                if ebit_last is not None and interest_last and interest_last != 0:
                    interest_coverage = ebit_last / abs(interest_last)
    except Exception:
        pass

    # Revenue CAGR 3y and EPS CAGR 3y
    revenue_cagr_3y = None
    eps_cagr_3y = None
    try:
        fin = fins["financials_annual"]
        if not fin.empty:
            revenue = fin.loc.get("Total Revenue")
            net_income = fin.loc.get("Net Income")
            if revenue is not None and len(revenue.dropna()) >= 4:
                rev_vals = revenue.dropna().iloc[:4].iloc[::-1].values  # oldest to latest
                revenue_cagr_3y = _compute_cagr(rev_vals[0], rev_vals[-1], 3.0)
            # EPS CAGR: use Net Income per share approximation if shares outstanding available
            shares_out = None
            try:
                shares_out = _safe_float(getattr(t, "fast_info", {}).get("shares_outstanding") if hasattr(t, "fast_info") else None)
            except Exception:
                shares_out = None
            if shares_out and net_income is not None and len(net_income.dropna()) >= 4:
                ni_vals = net_income.dropna().iloc[:4].iloc[::-1].values
                eps_first = ni_vals[0] / shares_out
                eps_last = ni_vals[-1] / shares_out
                eps_cagr_3y = _compute_cagr(eps_first, eps_last, 3.0)
    except Exception:
        pass

    # EV/EBITDA
    ev_ebitda = None
    try:
        info = {}
        if use_yahoo_info_fallback:
            try:
                info = t.info
            except Exception:
                info = {}
        mcap = market_cap or _safe_float(info.get("marketCap"))
        total_debt = None
        cash = None
        bal = fins["balance_annual"]
        if not bal.empty:
            td = bal.loc.get("Total Debt")
            ca = bal.loc.get("Cash") or bal.loc.get("Cash And Cash Equivalents")
            total_debt = float(td.dropna().iloc[0]) if td is not None and not td.dropna().empty else None
            cash = float(ca.dropna().iloc[0]) if ca is not None and not ca.dropna().empty else None
        # EBITDA
        fin = fins["financials_annual"]
        ebitda = None
        if not fin.empty:
            ebitda_row = fin.loc.get("Ebitda") or fin.loc.get("EBITDA")
            if ebitda_row is not None and not ebitda_row.dropna().empty:
                ebitda = float(ebitda_row.dropna().iloc[0])
        if mcap is not None and ebitda and ebitda != 0:
            enterprise_value = mcap + (total_debt or 0.0) - (cash or 0.0)
            ev_ebitda = enterprise_value / ebitda
    except Exception:
        pass

    return FundamentalSnapshot(
        market_cap=market_cap,
        pe=pe,
        pb=pb,
        roe=roe,
        roce=roce,
        debt_to_equity=debt_to_equity,
        interest_coverage=interest_coverage,
        revenue_cagr_3y=revenue_cagr_3y,
        eps_cagr_3y=eps_cagr_3y,
        ev_ebitda=ev_ebitda,
    )