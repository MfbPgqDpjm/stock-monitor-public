import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

from state_manager import get_data_path
from strategy import batch_get_data

logger = logging.getLogger(__name__)

ET_TIMEZONE = pytz.timezone("America/New_York")

MAGA8_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO"]
EXTRA_TICKERS = ["QQQ", "VOO", "TQQQ", "^KS11"]
DISPLAY_NAMES = {"^KS11": "KOSPI"}
YTD_FILE = "ytd_performance.json"


def _is_finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def _close_series(data: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if data is None or data.empty or "Close" not in data.columns:
        return None
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    return close if not close.empty else None


def _ytd_return_from_data(data: Optional[pd.DataFrame], now_et: datetime) -> Optional[Dict[str, Any]]:
    close = _close_series(data)
    if close is None:
        return None

    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    jan1 = pd.Timestamp(datetime(now_et.year, 1, 1))
    before_year = close[close.index < jan1]
    in_year = close[close.index >= jan1]
    if in_year.empty:
        return None

    start_close = before_year.iloc[-1] if not before_year.empty else in_year.iloc[0]
    latest_close = in_year.iloc[-1]
    if not (_is_finite(start_close) and _is_finite(latest_close)) or float(start_close) <= 0:
        return None

    latest_date = in_year.index[-1]
    return {
        "ytd_pct": (float(latest_close) / float(start_close) - 1.0) * 100.0,
        "latest_close": float(latest_close),
        "start_close": float(start_close),
        "latest_date": latest_date.strftime("%Y-%m-%d"),
    }


def _save_ytd_snapshot(snapshot: Dict[str, Any]) -> None:
    path = get_data_path(YTD_FILE)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"保存 YTD 涨幅失败：{e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_ytd_snapshot() -> Dict[str, Any]:
    path = get_data_path(YTD_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"读取 YTD 涨幅失败：{e}")
        return {}


def refresh_ytd_performance(now_et: Optional[datetime] = None, force_reload: bool = False) -> Dict[str, Any]:
    """更新 MAGA8 成分股、QQQ、VOO、TQQQ、KOSPI 的 YTD 涨幅快照。"""
    if now_et is None:
        now_et = datetime.now(ET_TIMEZONE)
    elif now_et.tzinfo is None:
        now_et = ET_TIMEZONE.localize(now_et)
    else:
        now_et = now_et.astimezone(ET_TIMEZONE)

    tickers = MAGA8_TICKERS + EXTRA_TICKERS
    data_cache = batch_get_data(
        tickers,
        period="3y",
        now_et=now_et,
        force_reload=tickers if force_reload else None,
    )

    ticker_results: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        result = _ytd_return_from_data(data_cache.get(ticker), now_et)
        if result:
            ticker_results[ticker] = result
        else:
            logger.warning(f"[YTD] {ticker} 缺少有效 YTD 数据")

    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        if ticker in ticker_results:
            rows.append({
                "name": DISPLAY_NAMES.get(ticker, ticker),
                "ticker": ticker,
                "ytd_pct": ticker_results[ticker]["ytd_pct"],
                "latest_close": ticker_results[ticker]["latest_close"],
                "latest_date": ticker_results[ticker]["latest_date"],
                "group": "MAGA8" if ticker in MAGA8_TICKERS else "INDEX",
            })

    rows.sort(key=lambda item: item.get("ytd_pct", float("-inf")), reverse=True)
    snapshot = {
        "last_updated": now_et.isoformat(),
        "targets": rows,
        "tickers": ticker_results,
    }
    _save_ytd_snapshot(snapshot)
    return snapshot
