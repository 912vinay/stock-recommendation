# Indian Stock Screener (MVP)

This MVP scans Indian equities for a confluence of valuation, growth, quality, promoter activity, and technical factors. It prioritizes free/public data sources.

## Data sources
- Prices/volume and basic ratios: Yahoo Finance via `yfinance`
- Universe: NIFTY500 list from NSE indices CSV
- Promoter activity: NSE quote shareholding JSON (best-effort; gracefully degrades if blocked)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (quick smoke for 20 stocks)
```bash
python run_screener.py --limit 20 --out results.csv --buy
```

## Full run
```bash
python run_screener.py --universe NIFTY500 --out results.csv --buy
```

## Notes
- NSE endpoints may rate-limit/block automated calls. The tool retries and degrades to skip promoter filters when data is unavailable.
- All thresholds are configurable in `screener/config.py`.