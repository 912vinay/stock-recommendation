import argparse
import os

import pandas as pd

from screener.config import DEFAULT_CONFIG, ScreenerConfig, UniverseConfig
from screener.universe import build_universe
from screener.screen import run_screen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Indian Stock Screener")
    p.add_argument("--universe", type=str, default=DEFAULT_CONFIG.universe.name, help="Universe: NIFTY50|NIFTY200|NIFTY500")
    p.add_argument("--limit", type=int, default=DEFAULT_CONFIG.universe.limit or 0, help="Limit tickers for quick run (0=all)")
    p.add_argument("--buy", action="store_true", help="Run buy-side screen (default)")
    p.add_argument("--sell", action="store_true", help="Run sell-side screen (TODO)")
    p.add_argument("--out", type=str, default="results.csv", help="Output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None

    cfg = DEFAULT_CONFIG
    cfg = ScreenerConfig(
        universe=UniverseConfig(name=args.universe.upper(), limit=limit),
        valuation=cfg.valuation,
        growth=cfg.growth,
        quality=cfg.quality,
        promoter=cfg.promoter,
        technical=cfg.technical,
        run=cfg.run,
    )

    symbols = build_universe(cfg.universe.name, limit=cfg.universe.limit)
    df = run_screen(symbols, cfg, for_buy=(not args.sell))

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()