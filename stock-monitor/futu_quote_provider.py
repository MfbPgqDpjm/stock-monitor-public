from __future__ import annotations

import logging
import os
import pwd
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FUTU_HOST = "127.0.0.1"
DEFAULT_FUTU_PORT = 11111


def _ensure_futu_runtime_env() -> None:
    """Futu SDK expects HOME to exist even when Streamlit is launched by scripts."""
    if os.environ.get("HOME"):
        return
    try:
        os.environ["HOME"] = pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        os.environ["HOME"] = os.path.expanduser("~") or "/tmp"


@dataclass(frozen=True)
class FutuRealtimeQuote:
    ticker: str
    futu_code: str
    price: float
    prev_close: Optional[float]
    price_field: str
    session_label: str
    update_time: str
    source: str = "futu"


def to_futu_code(ticker: object) -> Optional[str]:
    """Map project tickers to Futu quote codes for US holdings."""
    raw = str(ticker or "").strip().upper()
    if not raw:
        return None
    if raw in {"VUAA", "VUAA.L"}:
        return "US.VOO"
    if raw.startswith("US."):
        return raw
    if "." in raw:
        return None
    return f"US.{raw}"


def _clean_price(value: object) -> Optional[float]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return price


def _price_for_market_state(row: Any, market_state: str) -> tuple[Optional[float], str, str]:
    state = str(market_state or "").upper()
    if not state:
        now_et = datetime.now(ZoneInfo("America/New_York")).time()
        if now_et >= time(20, 0) or now_et < time(4, 0):
            state = "OVERNIGHT"
        elif time(4, 0) <= now_et < time(9, 30):
            state = "PRE_MARKET_BEGIN"
        elif time(9, 30) <= now_et < time(16, 0):
            state = "AFTERNOON"
        elif time(16, 0) <= now_et < time(20, 0):
            state = "AFTER_HOURS_BEGIN"
    if state == "OVERNIGHT":
        candidates = [
            ("overnight_price", "富途夜盘"),
            ("last_price", "富途最新"),
        ]
    elif "PRE_MARKET" in state:
        candidates = [
            ("pre_price", "富途盘前"),
            ("last_price", "富途最新"),
        ]
    elif "AFTER_HOURS" in state:
        candidates = [
            ("after_price", "富途盘后"),
            ("last_price", "富途最新"),
        ]
    elif state == "AFTERNOON":
        candidates = [
            ("last_price", "富途盘中"),
        ]
    else:
        candidates = [
            ("last_price", "富途最新"),
            ("overnight_price", "富途夜盘"),
            ("pre_price", "富途盘前"),
            ("after_price", "富途盘后"),
        ]

    for field, label in candidates:
        price = _clean_price(row.get(field))
        if price is not None:
            return price, field, label
    return None, "", ""


def fetch_futu_realtime_quotes(
    tickers: Iterable[object],
    host: str = DEFAULT_FUTU_HOST,
    port: int = DEFAULT_FUTU_PORT,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch Futu realtime snapshot quotes for US holdings.

    Returns a mapping keyed by original project ticker. Unsupported tickers are
    omitted so callers can keep their existing scan-price fallback.
    """
    requested: Dict[str, str] = {}
    for ticker in tickers:
        original = str(ticker or "").strip().upper()
        futu_code = to_futu_code(original)
        if original and futu_code:
            requested[original] = futu_code

    if not requested:
        return {}

    logger.info("[富途实时] 请求持仓报价: %s", ", ".join(sorted(requested.keys())))

    _ensure_futu_runtime_env()
    try:
        from futu import OpenQuoteContext, RET_OK
    except Exception as exc:
        logger.info("Futu SDK unavailable, realtime holding quotes skipped: %s", exc)
        return {}

    quote_ctx = None
    try:
        quote_ctx = OpenQuoteContext(host=host, port=int(port))
        futu_codes = sorted(set(requested.values()))

        ret_state, state_data = quote_ctx.get_market_state(futu_codes)
        state_by_code: Dict[str, str] = {}
        if ret_state == RET_OK:
            for _, row in state_data.iterrows():
                state_by_code[str(row.get("code", "")).upper()] = str(row.get("market_state", ""))
        else:
            logger.warning("Futu market state failed: %s", state_data)

        ret, data = quote_ctx.get_market_snapshot(futu_codes)
        if ret != RET_OK:
            logger.warning("Futu market snapshot failed: %s", data)
            return {}

        rows_by_code = {
            str(row.get("code", "")).upper(): row
            for _, row in data.iterrows()
        }

        quotes: Dict[str, Dict[str, Any]] = {}
        for original, futu_code in requested.items():
            row = rows_by_code.get(futu_code)
            if row is None:
                logger.warning("[富途实时] 未返回报价: %s -> %s", original, futu_code)
                continue

            price, price_field, session_label = _price_for_market_state(
                row,
                state_by_code.get(futu_code, ""),
            )
            if price is None:
                logger.warning("[富途实时] 无有效价格字段: %s -> %s", original, futu_code)
                continue

            quote = FutuRealtimeQuote(
                ticker=original,
                futu_code=futu_code,
                price=price,
                prev_close=_clean_price(row.get("prev_close_price")),
                price_field=price_field,
                session_label=session_label,
                update_time=str(row.get("update_time", "") or ""),
            )
            quotes[original] = quote.__dict__.copy()
        if quotes:
            summary = ", ".join(
                f"{ticker}={quote['session_label']}@{quote['price']:.2f}"
                for ticker, quote in sorted(quotes.items())
            )
            logger.info("[富途实时] 获取成功 %d/%d: %s", len(quotes), len(requested), summary)
        else:
            logger.warning("[富途实时] 未获取到任何有效报价，持仓收益将使用缓存")
        return quotes
    except Exception as exc:
        logger.warning("Futu realtime quote fetch failed: %s", exc)
        return {}
    finally:
        if quote_ctx is not None:
            try:
                quote_ctx.close()
            except Exception:
                pass


def _calc_position_pnl(entry_price: object, latest_price: object) -> Optional[float]:
    try:
        ep = float(entry_price)
        lp = float(latest_price)
        if ep <= 0:
            return None
        return lp / ep - 1
    except (TypeError, ValueError):
        return None


def _fmt_signed_pct_ratio(value: object) -> str:
    try:
        pct = float(value) * 100
    except (TypeError, ValueError):
        return "—"
    if pd.isna(pct):
        return "—"
    return f"{pct:+.2f}%"


def _fallback_close_for_ticker(ticker: object, signals: dict, explicit_price: object = None) -> object:
    if explicit_price not in (None, ""):
        return explicit_price

    key = str(ticker or "").strip().upper()
    candidates = [key]
    if key == "VUAA.L":
        candidates.append("VUAA")
    elif key == "VUAA":
        candidates.append("VUAA.L")

    for candidate in candidates:
        row = signals.get(candidate, {}) if isinstance(signals, dict) else {}
        if not isinstance(row, dict):
            continue
        for field in ("trading_close", "close"):
            value = row.get(field)
            if isinstance(value, (int, float)) and pd.notna(value):
                return value
    return explicit_price


def _realtime_price_for_ticker(
    ticker: object,
    quote: Optional[dict],
    fallback_price: object = None,
) -> tuple[object, str]:
    if not quote:
        return fallback_price, "缓存"

    key = str(ticker or "").strip().upper()
    price = quote.get("price")
    if key in {"VUAA", "VUAA.L"} and quote.get("futu_code") == "US.VOO":
        prev_close = quote.get("prev_close")
        try:
            if fallback_price not in (None, "") and prev_close and float(prev_close) > 0:
                proxy_price = float(fallback_price) * (float(price) / float(prev_close))
                return proxy_price, f"{quote.get('session_label', '富途实时')}·VOO代理"
        except (TypeError, ValueError):
            return fallback_price, "缓存"

    if isinstance(price, (int, float)) and pd.notna(price):
        return price, str(quote.get("session_label") or "富途实时")
    return fallback_price, "缓存"


def _split_realtime_cache_pnl(
    entry_price: object,
    price: object,
    source_label: str,
) -> tuple[str, str]:
    pnl = _calc_position_pnl(entry_price, price)
    pnl_str = _fmt_signed_pct_ratio(pnl)
    if source_label == "缓存":
        return "", pnl_str
    return pnl_str, ""


def _tagged_pnl(entry_price: object, price: object, source_label: str) -> str:
    pnl = _calc_position_pnl(entry_price, price)
    pnl_str = _fmt_signed_pct_ratio(pnl)
    if pnl_str == "—":
        return pnl_str
    source_tag = "缓存" if source_label == "缓存" else "实时"
    return f"{pnl_str} {source_tag}"


def prepare_holding_realtime_view(
    ticker: object,
    entry_price: object,
    signals: dict,
    realtime_quotes: dict,
    fallback_price: object = None,
) -> Dict[str, Any]:
    """
    Resolve holding display price and realtime/cache PnL columns.

    If Futu is unavailable or a specific symbol cannot be priced, this falls
    back to local cached close data from signals/momentum results.
    """
    key = str(ticker or "").strip().upper()
    cached_price = _fallback_close_for_ticker(key, signals, fallback_price)
    quote_price, quote_source = _realtime_price_for_ticker(
        key,
        realtime_quotes.get(key) if isinstance(realtime_quotes, dict) else None,
        cached_price,
    )
    pnl_realtime, pnl_cache = _split_realtime_cache_pnl(entry_price, quote_price, quote_source)
    pnl = _calc_position_pnl(entry_price, quote_price)
    is_cache = quote_source == "缓存"
    return {
        "ticker": key,
        "price": quote_price,
        "source": quote_source,
        "source_type": "缓存" if is_cache else "实时",
        "pnl_realtime": pnl_realtime,
        "pnl_cache": pnl_cache,
        "pnl": _fmt_signed_pct_ratio(pnl),
        "pnl_display": _tagged_pnl(entry_price, quote_price, quote_source),
        "is_cache": is_cache,
    }
