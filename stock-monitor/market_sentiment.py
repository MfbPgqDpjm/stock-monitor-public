import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import requests
import yfinance as yf

from strategy import _df_from_cache_entry, _load_data_cache, _save_data_cache

logger = logging.getLogger(__name__)

CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
MARKET_SENTIMENT_CACHE_TTL_SECONDS = 900
CNN_FEAR_GREED_CACHE_KEY = "__CNN_FEAR_GREED__"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
}


def _diagnostic_log(level: str, message: str) -> None:
    local_data_dir = os.path.join(os.path.dirname(__file__), "data")
    log_path = os.path.join(local_data_dir, "latest_scan.log")
    try:
        os.makedirs(local_data_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} [{level}] {message}\n")
    except Exception:
        logger.exception("[市场情绪] 写入诊断日志失败")


def write_market_sentiment_diagnostic(level: str, message: str) -> None:
    _diagnostic_log(level, message)


def _is_finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def _now_iso() -> str:
    return datetime.now().isoformat()


def _cached_at(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _is_fresh_metric(metric: Any, max_age_seconds: int = MARKET_SENTIMENT_CACHE_TTL_SECONDS) -> bool:
    if not isinstance(metric, dict):
        return False
    cached_at = _cached_at(metric.get("cached_at"))
    if cached_at is None:
        return False
    return datetime.now() - cached_at < timedelta(seconds=max_age_seconds)


def _entry_cached_at(entry: Any) -> Optional[datetime]:
    if not isinstance(entry, dict):
        return None
    return _cached_at(entry.get("cached_at") or entry.get("last_updated"))


def _is_fresh_cache_entry(entry: Any, max_age_seconds: int = MARKET_SENTIMENT_CACHE_TTL_SECONDS) -> bool:
    cached_at = _entry_cached_at(entry)
    if cached_at is None:
        return False
    return datetime.now() - cached_at < timedelta(seconds=max_age_seconds)


def _close_series(data: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if data is None or data.empty:
        return None

    close = None
    if isinstance(data.columns, pd.MultiIndex):
        for column in data.columns:
            if isinstance(column, tuple) and "Close" in column:
                close = data[column]
                break
    elif "Close" in data.columns:
        close = data["Close"]

    if close is None:
        return None
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    return close if not close.empty else None


def _vix_metric_from_cache_entry(entry: Any, source: str) -> Dict[str, Any]:
    df = _df_from_cache_entry(entry) if isinstance(entry, dict) else None
    close = _close_series(df)
    if close is None:
        return {"value": None, "error": "VIX 缓存为空"}

    latest = float(close.iloc[-1])
    previous = float(close.iloc[-2]) if len(close) >= 2 else None
    delta = latest - previous if _is_finite(previous) else None
    delta_pct = (latest / previous - 1.0) * 100.0 if _is_finite(previous) and previous else None
    return {
        "value": latest,
        "delta": delta,
        "delta_pct": delta_pct,
        "date": close.index[-1].strftime("%Y-%m-%d") if hasattr(close.index[-1], "strftime") else "",
        "cached_at": entry.get("cached_at") if isinstance(entry, dict) else None,
        "source": source,
    }


def _save_ohlcv_to_data_cache(ticker: str, data: pd.DataFrame, cached_at: str) -> None:
    if data is None or data.empty:
        return
    if isinstance(data.columns, pd.MultiIndex):
        flattened = {}
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            for column in data.columns:
                if isinstance(column, tuple) and field in column:
                    flattened[field] = pd.to_numeric(data[column], errors="coerce")
                    break
        data = pd.DataFrame(flattened, index=data.index)

    rows_by_date = {}
    for idx, row in data.iterrows():
        date_value = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        values = []
        valid = False
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            value = row[field] if field in data.columns else 0.0
            if _is_finite(value):
                valid = True
                values.append(float(value))
            else:
                values.append(0.0)
        if valid:
            rows_by_date[date_value] = [date_value] + values

    if not rows_by_date:
        return

    cache = _load_data_cache()
    existing = (cache.get("data") or {}).get(ticker, {})
    for row in existing.get("data", []) if isinstance(existing, dict) else []:
        if isinstance(row, list) and row:
            rows_by_date.setdefault(str(row[0]), row)
    cache.setdefault("data", {})[ticker] = {
        "columns": ["Open", "High", "Low", "Close", "Volume"],
        "data": [rows_by_date[key] for key in sorted(rows_by_date.keys())],
        "market": "US",
        "market_status": "closed",
        "cached_at": cached_at,
    }
    _save_data_cache(cache)


def get_vix_metric() -> Dict[str, Any]:
    cache = _load_data_cache()
    cached_entry = (cache.get("data") or {}).get("^VIX")
    cached_metric = _vix_metric_from_cache_entry(cached_entry, "cache")
    if _is_fresh_cache_entry(cached_entry) and _is_finite(cached_metric.get("value")):
        logger.info("[市场情绪] VIX 使用 data_cache.json 缓存: value=%.2f", float(cached_metric["value"]))
        return cached_metric

    try:
        data = yf.download("^VIX", period="3y", progress=False, auto_adjust=False)
        close = _close_series(data)
        if close is not None:
            _save_ohlcv_to_data_cache("^VIX", data, _now_iso())
            entry = (_load_data_cache().get("data") or {}).get("^VIX")
            metric = _vix_metric_from_cache_entry(entry, "live")
            if _is_finite(metric.get("value")):
                return metric
        if _is_finite(cached_metric.get("value")):
            cached_metric["error"] = "VIX 数据为空"
            logger.warning("[市场情绪] VIX 实时数据为空，使用 data_cache.json 兜底: value=%.2f", float(cached_metric["value"]))
            return cached_metric
        return {"value": None, "error": "VIX 数据为空"}
    except Exception as exc:
        logger.warning(f"获取 VIX 指标失败：{exc}")
        if _is_finite(cached_metric.get("value")):
            cached_metric["error"] = str(exc)
            logger.warning("[市场情绪] VIX 使用 data_cache.json 兜底: value=%.2f error=%s", float(cached_metric["value"]), exc)
            return cached_metric
        return {"value": None, "error": str(exc)}


def _timestamp_to_label(value: Any) -> str:
    try:
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _parse_cnn_fear_greed(payload: Dict[str, Any]) -> Dict[str, Any]:
    current = payload.get("fear_and_greed") if isinstance(payload, dict) else None
    if not isinstance(current, dict):
        return {"score": None, "error": "CNN Fear & Greed 数据格式异常"}

    score = current.get("score")
    previous = current.get("previous_close")
    delta = float(score) - float(previous) if _is_finite(score) and _is_finite(previous) else None
    delta_pct = (float(score) / float(previous) - 1.0) * 100.0 if _is_finite(score) and _is_finite(previous) and float(previous) else None
    return {
        "score": float(score) if _is_finite(score) else None,
        "rating": str(current.get("rating") or "").strip(),
        "delta": delta,
        "delta_pct": delta_pct,
        "date": _timestamp_to_label(current.get("timestamp")),
        "cached_at": _now_iso(),
        "source": "live",
    }


def _cnn_metric_from_cache_entry(entry: Any, source: str) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {"score": None, "error": "CNN Fear & Greed 缓存为空"}
    rows = entry.get("data") or []
    if not rows:
        return {"score": None, "error": "CNN Fear & Greed 缓存为空"}
    row = rows[-1]
    try:
        date_value = row[0]
        score = float(row[1])
        previous = float(row[2]) if len(row) > 2 and _is_finite(row[2]) else None
        delta = score - previous if _is_finite(previous) else None
        delta_pct = (score / previous - 1.0) * 100.0 if _is_finite(previous) and previous else None
        return {
            "score": score,
            "rating": str(row[5] if len(row) > 5 else entry.get("rating", "") or "").strip(),
            "delta": delta,
            "delta_pct": delta_pct,
            "date": str(date_value),
            "cached_at": entry.get("cached_at"),
            "source": source,
        }
    except Exception as exc:
        return {"score": None, "error": f"CNN Fear & Greed 缓存解析失败: {exc}"}


def _save_cnn_metric_to_data_cache(metric: Dict[str, Any]) -> None:
    if not _is_finite(metric.get("score")):
        return
    cache = _load_data_cache()
    data = cache.setdefault("data", {})
    date_value = metric.get("date") or datetime.now().strftime("%Y-%m-%d")
    score = float(metric["score"])
    delta = float(metric.get("delta")) if _is_finite(metric.get("delta")) else 0.0
    previous = score - delta
    delta_pct = float(metric.get("delta_pct")) if _is_finite(metric.get("delta_pct")) else 0.0
    entry = data.get(CNN_FEAR_GREED_CACHE_KEY, {})
    existing_rows = entry.get("data") if isinstance(entry, dict) else []
    rows_by_date = {}
    for row in existing_rows or []:
        if isinstance(row, list) and row:
            rows_by_date[str(row[0])] = row
    rows_by_date[str(date_value)] = [
        str(date_value),
        score,
        previous,
        delta,
        delta_pct,
        str(metric.get("rating", "") or ""),
    ]
    data[CNN_FEAR_GREED_CACHE_KEY] = {
        "columns": ["Score", "PreviousClose", "Delta", "DeltaPct", "Rating"],
        "data": [rows_by_date[key] for key in sorted(rows_by_date.keys())],
        "market": "US",
        "market_status": "closed",
        "cached_at": _now_iso(),
    }
    _save_data_cache(cache)


def get_cnn_fear_greed_metric() -> Dict[str, Any]:
    cache = _load_data_cache()
    cached_entry = (cache.get("data") or {}).get(CNN_FEAR_GREED_CACHE_KEY, {})
    cached_metric = _cnn_metric_from_cache_entry(cached_entry, "cache")
    if _is_fresh_cache_entry(cached_entry) and _is_finite(cached_metric.get("score")):
        logger.info(
            "[市场情绪] CNN Fear & Greed 使用 data_cache.json 缓存: score=%.2f rating=%s",
            float(cached_metric["score"]),
            cached_metric.get("rating", ""),
        )
        return cached_metric

    try:
        resp = requests.get(CNN_FEAR_GREED_URL, headers=REQUEST_HEADERS, timeout=8)
        resp.raise_for_status()
        metric = _parse_cnn_fear_greed(resp.json())
        if _is_finite(metric.get("score")):
            _save_cnn_metric_to_data_cache(metric)
            logger.info(
                "[市场情绪] CNN Fear & Greed 实时获取成功: score=%.2f rating=%s date=%s",
                float(metric["score"]),
                metric.get("rating", ""),
                metric.get("date", ""),
            )
            _diagnostic_log(
                "INFO",
                (
                    "[市场情绪] CNN Fear & Greed 实时获取成功: "
                    f"score={float(metric['score']):.2f} rating={metric.get('rating', '')} "
                    f"date={metric.get('date', '')}"
                ),
            )
        else:
            message = f"[市场情绪] CNN Fear & Greed 实时数据无有效分数: {metric}"
            logger.warning(message)
            _diagnostic_log("WARNING", message)
        return metric
    except Exception as exc:
        logger.warning(f"获取 CNN Fear & Greed 指标失败：{exc}")
        _diagnostic_log("WARNING", f"[市场情绪] CNN Fear & Greed 实时获取失败: error={exc}")
        if _is_finite(cached_metric.get("score")):
            cached_metric["error"] = str(exc)
            logger.warning(
                "[市场情绪] CNN Fear & Greed 使用 data_cache.json 兜底: score=%.2f rating=%s error=%s",
                float(cached_metric["score"]),
                cached_metric.get("rating", ""),
                exc,
            )
            _diagnostic_log(
                "WARNING",
                (
                    "[市场情绪] CNN Fear & Greed 使用 data_cache.json 兜底: "
                    f"score={float(cached_metric['score']):.2f} rating={cached_metric.get('rating', '')} "
                    f"error={exc}"
                ),
            )
            return cached_metric
        logger.error(
            "[市场情绪] CNN Fear & Greed 无法展示: live_error=%s cache=%s",
            exc,
            cached_entry,
        )
        _diagnostic_log(
            "ERROR",
            (
                "[市场情绪] CNN Fear & Greed 无法展示: "
                f"live_error={exc} cache={cached_entry}"
            ),
        )
        return {"score": None, "error": str(exc)}
