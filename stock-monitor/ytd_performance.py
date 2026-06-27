import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

from state_manager import get_data_path, load_config
from strategy import batch_get_data

logger = logging.getLogger(__name__)

ET_TIMEZONE = pytz.timezone("America/New_York")

YTD_FILE = "ytd_performance.json"
YTD_CACHE_TTL_SECONDS = 900
DEFAULT_YTD_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "QQQ", "VOO", "TQQQ", "SPCX", "^KS11", "VUAA.L", "SOXL"]
DEFAULT_YTD_DISPLAY_NAMES = {"^KS11": ".KOSPI"}


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


def _normalize_ytd_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def get_ytd_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """YTD 涨幅标的完全来自运行时 config.json。"""
    cfg = config if isinstance(config, dict) else load_config()
    configured = cfg.get("ytd_tickers")
    raw_tickers = configured if isinstance(configured, list) else DEFAULT_YTD_TICKERS

    tickers: List[str] = []
    for item in raw_tickers:
        ticker = _normalize_ytd_ticker(item)
        if ticker and ticker not in tickers:
            tickers.append(ticker)

    display_names = DEFAULT_YTD_DISPLAY_NAMES.copy()
    configured_display = cfg.get("ytd_display_names")
    if isinstance(configured_display, dict):
        for ticker, display_name in configured_display.items():
            ticker_key = _normalize_ytd_ticker(ticker)
            display_value = str(display_name or "").strip()
            if ticker_key and display_value:
                display_names[ticker_key] = display_value

    return {
        "tickers": tickers,
        "display_names": display_names,
    }


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
    daily_pct = None
    if len(in_year) >= 2:
        previous_close = in_year.iloc[-2]
        if _is_finite(previous_close) and float(previous_close) > 0:
            daily_pct = (float(latest_close) / float(previous_close) - 1.0) * 100.0

    return {
        "ytd_pct": (float(latest_close) / float(start_close) - 1.0) * 100.0,
        "daily_pct": daily_pct,
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


def _snapshot_age_seconds(snapshot: Dict[str, Any], now_et: datetime) -> Optional[float]:
    updated = snapshot.get("last_updated") if isinstance(snapshot, dict) else None
    if not updated:
        return None
    try:
        dt = datetime.fromisoformat(str(updated))
        if dt.tzinfo is None:
            dt = ET_TIMEZONE.localize(dt)
        return (now_et - dt.astimezone(ET_TIMEZONE)).total_seconds()
    except Exception:
        return None


def _is_current_ytd_snapshot(snapshot: Dict[str, Any], config: Optional[Dict[str, Any]], now_et: datetime) -> bool:
    rows = snapshot.get("targets") if isinstance(snapshot, dict) else None
    if not isinstance(rows, list):
        return False

    expected_tickers = get_ytd_config(config)["tickers"]
    snapshot_tickers = snapshot.get("configured_tickers")
    if not isinstance(snapshot_tickers, list):
        return False
    if snapshot_tickers != expected_tickers:
        return False
    if not all(isinstance(row, dict) and "daily_pct" in row for row in rows):
        return False

    age_seconds = _snapshot_age_seconds(snapshot, now_et)
    return age_seconds is not None and 0 <= age_seconds <= YTD_CACHE_TTL_SECONDS


def get_ytd_performance(
    now_et: Optional[datetime] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    页面读取入口：通过 data_cache.json 的 15 分钟持久缓存策略刷新 YTD。

    batch_get_data() 会先读 data_cache.json；缓存未过期时不请求外部，
    缓存过期时才请求 Yahoo，并把最新日线写回 data_cache.json。
    """
    if now_et is None:
        now_et = datetime.now(ET_TIMEZONE)
    elif now_et.tzinfo is None:
        now_et = ET_TIMEZONE.localize(now_et)
    else:
        now_et = now_et.astimezone(ET_TIMEZONE)

    snapshot = load_ytd_snapshot()
    if _is_current_ytd_snapshot(snapshot, config, now_et):
        logger.info("[YTD] 使用本地持久缓存: last_updated=%s", snapshot.get("last_updated"))
        return snapshot

    try:
        logger.info("[YTD] 本地缓存过期或配置变化，刷新 Yahoo 日线并写入 data_cache.json")
        return refresh_ytd_performance(now_et=now_et, force_reload=True, config=config)
    except Exception as exc:
        logger.error(f"刷新 YTD 涨幅失败，使用本地快照兜底：{exc}")
        return load_ytd_snapshot()


def refresh_ytd_performance(
    now_et: Optional[datetime] = None,
    force_reload: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """按 PROD config.json 的 ytd_tickers 更新 YTD 涨幅快照。"""
    if now_et is None:
        now_et = datetime.now(ET_TIMEZONE)
    elif now_et.tzinfo is None:
        now_et = ET_TIMEZONE.localize(now_et)
    else:
        now_et = now_et.astimezone(ET_TIMEZONE)

    ytd_config = get_ytd_config(config)
    tickers = ytd_config["tickers"]
    display_names = ytd_config["display_names"]
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
                "name": display_names.get(ticker, ticker),
                "ticker": ticker,
                "ytd_pct": ticker_results[ticker]["ytd_pct"],
                "daily_pct": ticker_results[ticker]["daily_pct"],
                "latest_close": ticker_results[ticker]["latest_close"],
                "latest_date": ticker_results[ticker]["latest_date"],
            })

    rows.sort(key=lambda item: item.get("ytd_pct", float("-inf")), reverse=True)
    snapshot = {
        "last_updated": now_et.isoformat(),
        "configured_tickers": tickers,
        "display_names": display_names,
        "targets": rows,
        "tickers": ticker_results,
    }
    _save_ytd_snapshot(snapshot)
    return snapshot
