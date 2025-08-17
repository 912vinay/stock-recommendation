from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseConfig:
    name: str = "NIFTY500"  # NIFTY50, NIFTY200, NIFTY500
    limit: int | None = None  # cap number of symbols for quick runs


@dataclass(frozen=True)
class ValuationConfig:
    min_pe: float = 5.0
    max_pe: float = 35.0
    max_pb: float = 6.0
    max_ev_ebitda: float | None = 20.0  # may be missing for many; ignored if None


@dataclass(frozen=True)
class GrowthConfig:
    min_revenue_cagr_3y: float = 10.0
    min_eps_cagr_3y: float = 10.0


@dataclass(frozen=True)
class QualityConfig:
    min_roe: float = 12.0
    min_roce: float = 15.0
    max_debt_to_equity: float = 0.8
    min_interest_coverage: float = 3.0


@dataclass(frozen=True)
class PromoterConfig:
    min_promoter_change_qoq_pct_pts: float = 0.1  # +0.1 percentage points or more
    max_pledge_percent: float | None = 5.0  # unavailable for most free sources; ignored if None


@dataclass(frozen=True)
class TechnicalConfig:
    price_above_200d: bool = True
    sma50_above_200d: bool = True
    within_pct_52w_high: float = 10.0
    rsi_min: float = 45.0
    rsi_max: float = 70.0
    min_volume_multiple_vs_50d: float = 1.3


@dataclass(frozen=True)
class RunConfig:
    # Fetch 400 trading days to compute 200DMA/RSI etc.
    lookback_days: int = 420
    # Batch size for yfinance multi-download to avoid overload
    batch_size: int = 50
    # HTTP timeout (seconds)
    http_timeout: int = 20
    # When True, download per symbol with throttling to reduce 429s
    single_download: bool = True
    # Pause between per-symbol downloads (seconds)
    download_pause_sec: float = 0.8
    # Avoid heavy Yahoo .info fallback to reduce 429s
    use_yahoo_info_fallback: bool = False
    # Limit number of symbols for fundamentals/promoter after technical prefilter (None = no limit)
    fundamentals_max_symbols: int | None = 150


@dataclass(frozen=True)
class ScreenerConfig:
    universe: UniverseConfig = UniverseConfig()
    valuation: ValuationConfig = ValuationConfig()
    growth: GrowthConfig = GrowthConfig()
    quality: QualityConfig = QualityConfig()
    promoter: PromoterConfig = PromoterConfig()
    technical: TechnicalConfig = TechnicalConfig()
    run: RunConfig = RunConfig()


DEFAULT_CONFIG = ScreenerConfig()