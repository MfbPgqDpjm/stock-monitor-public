from private_rules import private_rule, private_parameter, positive_window
import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz
import yfinance as yf

from state_manager import get_data_path

logger = logging.getLogger(__name__)

ET_TIMEZONE = pytz.timezone("America/New_York")
MACRO_CYCLE_FILE = "macro_cycle_dashboard.json"
MACRO_TICKERS = ["SPY", "TLT", "GLD", "BIL"]
DIAGNOSIS_VERSION = 2

RATIO_DEFS = [
    ("SPY_TLT", "股债比 SPY/TLT", "增长 vs 衰退", "SPY", "TLT"),
    ("GLD_TLT", "金债比 GLD/TLT", "通胀 vs 通缩", "GLD", "TLT"),
    ("GLD_SPY", "金股比 GLD/SPY", "避险 vs 风险", "GLD", "SPY"),
    ("SPY_BIL", "股现比 SPY/BIL", "风险偏好", "SPY", "BIL"),
    ("TLT_BIL", "债现比 TLT/BIL", "降息预期", "TLT", "BIL"),
    ("GLD_BIL", "金现比 GLD/BIL", "货币贬值/避险", "GLD", "BIL"),
]


def _is_finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def _macro_file_path() -> str:
    return get_data_path(MACRO_CYCLE_FILE)


def _save_snapshot(snapshot: Dict[str, Any]) -> None:
    path = _macro_file_path()
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"保存宏观周期诊断快照失败：{e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_macro_cycle_snapshot() -> Dict[str, Any]:
    path = _macro_file_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"读取宏观周期诊断快照失败：{e}")
        return {}


def _snapshot_is_fresh(snapshot: Dict[str, Any], max_age_hours: int) -> bool:
    if snapshot.get("diagnosis_version") != DIAGNOSIS_VERSION:
        return False

    updated = snapshot.get("last_updated") if isinstance(snapshot, dict) else None
    if not updated:
        return False
    try:
        dt = datetime.fromisoformat(str(updated))
        if dt.tzinfo is None:
            dt = ET_TIMEZONE.localize(dt)
        return datetime.now(ET_TIMEZONE) - dt.astimezone(ET_TIMEZONE) < timedelta(hours=max_age_hours)
    except Exception:
        return False


def _extract_close_frame(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    closes: Dict[str, pd.Series] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            series = None
            for key in [(ticker, "Close"), ("Close", ticker)]:
                if key in data.columns:
                    series = data[key]
                    break
            if series is not None:
                closes[ticker] = pd.to_numeric(series, errors="coerce")
    else:
        if len(tickers) == 1 and "Close" in data.columns:
            closes[tickers[0]] = pd.to_numeric(data["Close"], errors="coerce")

    if not closes:
        return pd.DataFrame()

    return pd.DataFrame(closes).dropna()


def _download_macro_closes(period: str) -> pd.DataFrame:
    data = yf.download(
        MACRO_TICKERS,
        period=period,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
    )
    return _extract_close_frame(data, MACRO_TICKERS)


def _build_ratio_signals(df: pd.DataFrame, fast: int, slow: int) -> Dict[str, Dict[str, Any]]:
    ratios = pd.DataFrame(index=df.index)
    for key, _, _, numerator, denominator in RATIO_DEFS:
        ratios[key] = df[numerator] / df[denominator]

    signals: Dict[str, Dict[str, Any]] = {}
    for key, label, meaning, _, _ in RATIO_DEFS:
        ratio = pd.to_numeric(ratios[key], errors="coerce").dropna()
        latest = ratio.iloc[-1] if not ratio.empty else None
        ma_fast = ratio.rolling(fast).mean().iloc[-1] if len(ratio) >= fast else None
        ma_slow = ratio.rolling(slow).mean().iloc[-1] if len(ratio) >= slow else None
        trend = "强" if _is_finite(ma_fast) and _is_finite(ma_slow) and float(ma_fast) > float(ma_slow) else "弱"
        signals[key] = {
            "key": key,
            "label": label,
            "meaning": meaning,
            "trend": trend,
            "latest_ratio": float(latest) if _is_finite(latest) else None,
            "fast_ma": float(ma_fast) if _is_finite(ma_fast) else None,
            "slow_ma": float(ma_slow) if _is_finite(ma_slow) else None,
        }
    return signals


def _diagnose(signals: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Private calculation supplied explicitly by the deployment."""
    return private_rule("_diagnose", signals=signals)


def refresh_macro_cycle_dashboard(period: str = "2y", fast: Optional[int] = None, slow: Optional[int] = None) -> Dict[str, Any]:
    fast = fast if fast is not None else positive_window("macro_fast_window")
    slow = slow if slow is not None else positive_window("macro_slow_window")
    if isinstance(fast, bool) or isinstance(slow, bool) or not isinstance(fast, int) or not isinstance(slow, int) or fast <= 0 or slow <= 0:
        raise ValueError("Positive integer windows required")
    df = _download_macro_closes(period)
    if df.empty:
        raise RuntimeError("宏观周期诊断数据获取失败")
    if len(df) < slow:
        raise RuntimeError(f"宏观周期诊断数据不足：需要至少 {slow} 根，当前 {len(df)} 根")

    signals = _build_ratio_signals(df, fast=fast, slow=slow)
    diagnosis = _diagnose(signals)
    latest_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d")
    now_et = datetime.now(ET_TIMEZONE)

    snapshot = {
        "diagnosis_version": DIAGNOSIS_VERSION,
        "last_updated": now_et.isoformat(),
        "latest_date": latest_date,
        "period": period,
        "fast_window": fast,
        "slow_window": slow,
        "tickers": MACRO_TICKERS,
        "signals": signals,
        "conclusion": diagnosis["conclusion"],
        "action": diagnosis["action"],
    }
    _save_snapshot(snapshot)
    return snapshot


def get_macro_cycle_dashboard(max_age_hours: int = 6) -> Dict[str, Any]:
    snapshot = load_macro_cycle_snapshot()
    if snapshot and _snapshot_is_fresh(snapshot, max_age_hours=max_age_hours):
        return snapshot

    try:
        return refresh_macro_cycle_dashboard()
    except Exception as e:
        logger.error(f"刷新宏观周期诊断失败：{e}")
        if snapshot:
            stale_snapshot = snapshot.copy()
            stale_snapshot["refresh_error"] = str(e)
            return stale_snapshot
        return {"refresh_error": str(e), "signals": {}}
