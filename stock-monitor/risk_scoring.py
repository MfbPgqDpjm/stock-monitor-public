"""Public-safe risk scoring helper.

Private risk thresholds, lookback windows, score weights, and action levels are
redacted from this public copy.
"""

from typing import Any, Dict, Iterable, Optional


def simple_ai_risk_filter(*args: Any, **kwargs: Any) -> str:
    return "Public copy redacts private risk parameters"


def ai_bubble_risk_signal(df: Any, fundamentals: Optional[dict] = None, qqq_df: Any = None) -> Dict[str, Any]:
    return {
        "risk_score": None,
        "risk_flags": ["Public copy redacts private risk parameters."],
        "action": simple_ai_risk_filter(),
    }


def build_risk_score_for_holding(
    ticker: str,
    latest_price: Any = None,
    config: Optional[dict] = None,
    signal_ticker: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "ticker": str(ticker or "").strip().upper(),
        "score": None,
        "action": simple_ai_risk_filter(),
        "reason": "Public copy redacts private risk parameters.",
    }


def build_risk_scores_for_holdings(
    tickers: Iterable[str],
    latest_prices: Optional[Dict[str, Any]] = None,
    config: Optional[dict] = None,
    signal_tickers: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    return {str(t or "").strip().upper(): build_risk_score_for_holding(str(t or "")) for t in tickers if str(t or "").strip()}
