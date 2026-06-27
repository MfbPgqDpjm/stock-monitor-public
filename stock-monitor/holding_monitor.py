from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

import pandas as pd

from futu_quote_provider import prepare_holding_realtime_view
from position_state import load_position_state
from strategy import _df_from_cache_entry, _load_data_cache
from state_manager import get_momentum_ticker_configs, load_config


HOLDING_MONITOR_COLUMNS = [
    "持仓类型",
    "标的",
    "市场",
    "持仓状态",
    "买入价",
    "买入日期",
    "最新价",
    "持仓收益",
    "持仓最高收益",
    "持仓最大回撤",
    "指标",
    "指标线",
    "持股天数",
]


def collect_holding_tickers(us_configs: Iterable[dict], momentum_result: dict) -> Set[str]:
    tickers: Set[str] = {
        str(mc.get("buy_ticker", "")).strip().upper()
        for mc in us_configs
        if isinstance(mc, dict) and mc.get("buy_ticker")
    }

    position_audit = momentum_result.get("position_audit", {}) if isinstance(momentum_result, dict) else {}
    positions = position_audit.get("positions", [])
    if isinstance(positions, list):
        tickers.update(
            str(pos.get("ticker", "")).strip().upper()
            for pos in positions
            if isinstance(pos, dict) and pos.get("ticker")
        )
    elif isinstance(positions, dict) and positions.get("ticker"):
        tickers.add(str(positions.get("ticker")).strip().upper())

    return {ticker for ticker in tickers if ticker}


def build_holding_monitor_rows(
    us_configs: Iterable[dict],
    signals: dict,
    momentum_result: dict,
    realtime_quotes: dict,
) -> List[Dict[str, Any]]:
    momentum_configs = _load_momentum_config_by_ticker()
    market_rows = _build_market_holding_rows(us_configs, signals, realtime_quotes)
    existing_market_tickers = {row.get("标的", "") for row in market_rows}
    momentum_rows = _build_momentum_holding_rows(
        momentum_result,
        signals,
        realtime_quotes,
        existing_market_tickers=existing_market_tickers,
        momentum_configs=momentum_configs,
    )
    rows: List[Dict[str, Any]] = []
    rows.extend(market_rows)
    rows.extend(momentum_rows)
    return rows


def holding_monitor_source_suffix(rows: Iterable[Dict[str, Any]]) -> str:
    sources = {
        str(row.get("_数据来源", "")).strip()
        for row in rows
        if isinstance(row, dict) and row.get("_数据来源")
    }
    if sources == {"实时"}:
        return "实时"
    if sources == {"缓存"}:
        return "缓存"
    if sources:
        return "实时/缓存"
    return "缓存"


def _build_market_holding_rows(
    us_configs: Iterable[dict],
    signals: dict,
    realtime_quotes: dict,
) -> List[Dict[str, Any]]:
    try:
        pos_state = load_position_state()
        strategies = pos_state.get("strategies", {}) if isinstance(pos_state, dict) else {}
    except Exception:
        strategies = {}

    rows: List[Dict[str, Any]] = []
    for mc in us_configs:
        if not isinstance(mc, dict):
            continue
        ticker = str(mc.get("buy_ticker", "")).strip().upper()
        if not ticker:
            continue

        strategy_state = strategies.get(ticker, {}) if isinstance(strategies, dict) else {}
        if not isinstance(strategy_state, dict) or not strategy_state.get("in_position"):
            continue

        signal = signals.get(ticker, {}) if isinstance(signals, dict) else {}
        if not isinstance(signal, dict):
            signal = {}

        entry_price = strategy_state.get("entry_price")
        fallback_price = signal.get("trading_close") if signal.get("trading_close") is not None else signal.get("close")
        holding_view = prepare_holding_realtime_view(
            ticker,
            entry_price,
            signals,
            realtime_quotes,
            fallback_price=fallback_price,
        )

        sell_reason = signal.get("sell_reason") or ""
        status = f"待卖出：{sell_reason}" if signal.get("signal") == "卖出" and sell_reason else "待卖出" if signal.get("signal") == "卖出" else "持有"
        indicator, indicator_line = _market_indicator(signal, strategy_state)
        metrics = _market_position_metrics(
            ticker=ticker,
            entry_date=strategy_state.get("entry_date"),
            entry_price=entry_price,
            latest_price=holding_view.get("price"),
        )
        hold_days = _hold_days(strategy_state.get("entry_date"))

        rows.append({
            "持仓类型": "大盘",
            "标的": ticker,
            "市场": str(signal.get("market") or strategy_state.get("market") or "—"),
            "持仓状态": status,
            "买入价": _to_float(entry_price),
            "买入日期": str(strategy_state.get("entry_date") or ""),
            "最新价": _to_float(holding_view.get("price")),
            "持仓收益": _position_pnl_pct(entry_price, holding_view.get("price")),
            "持仓最高收益": metrics.get("max_return"),
            "持仓最大回撤": metrics.get("max_drawdown"),
            "指标": indicator,
            "指标线": indicator_line,
            "持股天数": hold_days,
            "_数据来源": str(holding_view.get("source_type") or "缓存"),
        })

    return rows


def _build_momentum_holding_rows(
    momentum_result: dict,
    signals: dict,
    realtime_quotes: dict,
    existing_market_tickers: set[str],
    momentum_configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    position_audit = momentum_result.get("position_audit", {}) if isinstance(momentum_result, dict) else {}
    if not isinstance(position_audit, dict) or position_audit.get("status") in {"无持仓", "数据获取失败"}:
        return []

    positions = position_audit.get("positions", [])
    if not isinstance(positions, list):
        positions = [position_audit]

    rows: List[Dict[str, Any]] = []
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        ticker = str(pos.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        if ticker in existing_market_tickers:
            continue

        buy_price = pos.get("buy_price")
        holding_view = prepare_holding_realtime_view(
            ticker,
            buy_price,
            signals,
            realtime_quotes,
            fallback_price=pos.get("latest_price"),
        )
        latest_price = holding_view.get("price")
        indicator_info = _momentum_indicator_info(ticker, pos, momentum_configs)
        trend_indicator = indicator_info["label"]
        trend_ema = indicator_info["line"]
        sell_reason = str(pos.get("sell_reason", "") or "")
        status = f"待卖出：{sell_reason}" if pos.get("action_plan") == "待卖出" and sell_reason else "待卖出" if pos.get("action_plan") == "待卖出" else "持有"
        holding_type = "大盘" if ticker in {"VUAA", "VUAA.L"} else "个股动量"

        rows.append({
            "持仓类型": holding_type,
            "标的": ticker,
            "市场": "美股" if _looks_like_us_ticker(ticker) else "",
            "持仓状态": status,
            "买入价": _to_float(buy_price),
            "买入日期": str(pos.get("buy_date", "") or ""),
            "最新价": _to_float(latest_price),
            "持仓收益": _position_pnl_pct(buy_price, latest_price),
            "持仓最高收益": _pct_points(pos.get("max_return")),
            "持仓最大回撤": _drawdown_pct_points(pos.get("max_drawdown")),
            "指标": str(trend_indicator),
            "指标线": _to_float(trend_ema),
            "持股天数": _int_or_none(pos.get("hold_days")),
            "_数据来源": str(holding_view.get("source_type") or "缓存"),
        })

    return rows


def _load_momentum_config_by_ticker() -> Dict[str, Dict[str, Any]]:
    try:
        config = load_config(reload=True)
        return {
            str(row.get("ticker", "")).upper(): row
            for row in get_momentum_ticker_configs(config)
            if row.get("ticker")
        }
    except Exception:
        return {}


def _momentum_indicator_info(
    ticker: str,
    position: Dict[str, Any],
    momentum_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ticker_u = str(ticker or "").strip().upper()
    cfg = momentum_configs.get(ticker_u, {})
    signal_ticker = str(
        cfg.get("signal_ticker")
        or position.get("signal_ticker")
        or ticker_u
    ).strip().upper()
    indicator = str(
        cfg.get("indicator")
        or position.get("trend_indicator")
        or "EMA50"
    ).strip().upper()
    ema_window = cfg.get("ema_window") or position.get("trend_ema_window")
    if not ema_window and indicator.startswith("EMA"):
        try:
            ema_window = int(indicator[3:])
        except (TypeError, ValueError):
            ema_window = None
    try:
        ema_window_i = int(ema_window)
    except (TypeError, ValueError):
        ema_window_i = 50
        indicator = "EMA50"

    line = _ema_from_cache(signal_ticker, ema_window_i)
    if line is None and str(position.get("signal_ticker") or ticker_u).strip().upper() == signal_ticker:
        line = position.get("trend_ema", position.get("ema100", position.get("ema50")))

    return {
        "label": f"{signal_ticker}+{indicator}",
        "line": line,
    }


def _fmt_price(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value or "")


def _pct_points(value: object) -> Any:
    try:
        pct = float(value) * 100
    except (TypeError, ValueError):
        return None
    if pd.isna(pct):
        return None
    return pct


def _drawdown_pct_points(value: object) -> Any:
    try:
        pct = -abs(float(value) * 100)
    except (TypeError, ValueError):
        return None
    if pd.isna(pct):
        return None
    return pct


def _position_pnl_pct(entry_price: object, latest_price: object) -> Any:
    ep = _to_float(entry_price)
    lp = _to_float(latest_price)
    if ep is None or lp is None or ep <= 0:
        return None
    return (lp / ep - 1) * 100


def _market_position_metrics(
    ticker: str,
    entry_date: object,
    entry_price: object,
    latest_price: object,
) -> Dict[str, Any]:
    try:
        ep = float(entry_price)
        if ep <= 0:
            return {"max_return": None, "max_drawdown": None}
    except (TypeError, ValueError):
        return {"max_return": None, "max_drawdown": None}

    hist = _history_from_cache(ticker)
    if hist is None or hist.empty:
        lp = _to_float(latest_price)
        max_return = (lp / ep - 1) * 100 if lp is not None else None
        return {"max_return": max_return, "max_drawdown": None}

    try:
        entry_ts = pd.to_datetime(str(entry_date)).tz_localize(None)
    except Exception:
        entry_ts = hist.index.min()

    window = hist.loc[hist.index >= entry_ts].copy()
    if window.empty:
        window = hist.copy()

    high = pd.to_numeric(window.get("High"), errors="coerce") if "High" in window else None
    close = pd.to_numeric(window.get("Close"), errors="coerce") if "Close" in window else None
    latest_float = _to_float(latest_price)

    high_values = high.dropna().tolist() if high is not None else []
    if latest_float is not None:
        high_values.append(latest_float)

    max_high = max(high_values) if high_values else None
    max_return = (max_high / ep - 1) * 100 if max_high is not None else None

    max_drawdown = None
    if high is not None and close is not None:
        pairs = pd.DataFrame({"High": high, "Close": close}).dropna()
        if latest_float is not None and not pairs.empty:
            pairs.iloc[-1, pairs.columns.get_loc("Close")] = latest_float
            if latest_float > pairs.iloc[-1]["High"]:
                pairs.iloc[-1, pairs.columns.get_loc("High")] = latest_float
        elif latest_float is not None:
            pairs = pd.DataFrame({"High": [latest_float], "Close": [latest_float]})

        if not pairs.empty:
            running_peak = pairs["High"].cummax()
            drawdowns = (running_peak - pairs["Close"]) / running_peak
            drawdowns = drawdowns.replace([float("inf"), -float("inf")], pd.NA).dropna()
            if not drawdowns.empty:
                max_drawdown = -abs(float(drawdowns.max()) * 100)

    return {"max_return": max_return, "max_drawdown": max_drawdown}


def _history_from_cache(ticker: str) -> pd.DataFrame | None:
    try:
        cache = _load_data_cache()
        data = cache.get("data", {}) if isinstance(cache, dict) else {}
        raw = data.get(str(ticker or "").strip().upper()) if isinstance(data, dict) else None
        df = _df_from_cache_entry(raw)
        if df is None or df.empty:
            return None
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def _ema_from_cache(ticker: str, window: int) -> float | None:
    hist = _history_from_cache(ticker)
    if hist is None or hist.empty or "Close" not in hist:
        return None
    try:
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if close.empty:
            return None
        value = float(close.ewm(span=int(window), adjust=False).mean().iloc[-1])
    except Exception:
        return None
    if pd.isna(value):
        return None
    return value


def _to_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _hold_days(entry_date: object) -> int | None:
    raw = str(entry_date or "").strip()
    if not raw:
        return None
    try:
        start = pd.to_datetime(raw).date()
        return (pd.Timestamp.now(tz="America/New_York").date() - start).days + 1
    except Exception:
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _market_indicator(signal: dict, strategy_state: dict) -> tuple[str, Any]:
    benchmark = str(signal.get("benchmark") or strategy_state.get("benchmark") or "").strip()
    threshold = signal.get("threshold")
    return benchmark, _to_float(threshold)


def _looks_like_us_ticker(ticker: str) -> bool:
    return "." not in ticker or ticker.endswith(".L")
