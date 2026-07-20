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
    spy_tlt = signals["SPY_TLT"]["trend"]
    gld_tlt = signals["GLD_TLT"]["trend"]
    gld_spy = signals["GLD_SPY"]["trend"]
    spy_bil = signals["SPY_BIL"]["trend"]
    tlt_bil = signals["TLT_BIL"]["trend"]
    gld_bil = signals["GLD_BIL"]["trend"]

    if spy_bil == "弱" and tlt_bil == "弱" and gld_bil == "弱":
        return {
            "conclusion": "🔴 流动性危机 / 现金为王",
            "action": "股票、长债、黄金全部跑输现金，所有风险资产承压，优先防守。",
        }
    if gld_spy == "强" and gld_tlt == "强" and gld_bil == "强" and spy_bil == "弱":
        return {
            "conclusion": "🟡 滞胀预警 / 黄金主导",
            "action": "黄金强于股票、债券和现金，市场在交易通胀、货币贬值或避险。",
        }
    if spy_tlt == "强" and gld_tlt == "强" and gld_spy == "强":
        return {
            "conclusion": "🟡 滞胀预警 / 黄金主导",
            "action": "黄金强于股票、债券和现金，市场在交易通胀、货币贬值或避险。",
        }
    if spy_tlt == "强" and gld_tlt == "强" and gld_spy == "弱":
        return {
            "conclusion": "🟢 再通胀 / 牛市中后期",
            "action": "股票仍强于黄金，黄金强于债券，说明经济仍强但通胀压力开始抬头。",
        }
    if spy_tlt == "强" and spy_bil == "强" and gld_spy == "弱":
        return {
            "conclusion": "🟢 低通胀扩张 / 牛市早中期",
            "action": "股票强于债券、黄金和现金，是权益资产最舒服的环境。",
        }
    if spy_tlt == "弱" and tlt_bil == "强":
        return {
            "conclusion": "🔴 衰退预期 / 降息交易",
            "action": "长债强于股票和现金，市场开始交易经济放缓和降息。",
        }
    if gld_bil == "强" and spy_bil == "弱":
        return {
            "conclusion": "🟡 避险升温",
            "action": "黄金跑赢现金，股票跑输现金，风险资产压力增大。",
        }
    if spy_bil == "强" and tlt_bil == "弱" and gld_bil == "弱":
        return {
            "conclusion": "🟢 风险偏好扩张",
            "action": "股票跑赢现金，债券和黄金跑输现金，市场明显偏向风险资产。",
        }
    return {
        "conclusion": "🔵 周期过渡 / 多空混沌",
        "action": "信号不一致，不用宏观判断主动加仓，回归原机械策略。",
    }


def refresh_macro_cycle_dashboard(period: str = "2y", fast: int = 20, slow: int = 60) -> Dict[str, Any]:
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
