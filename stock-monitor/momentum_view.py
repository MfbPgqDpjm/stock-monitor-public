from __future__ import annotations

from typing import Any, Dict, List

from momentum_scorer import format_rs_pct


MOMENTUM_DECISION_COLUMNS = [
    "标的",
    "日期",
    "收盘价",
    "HH20",
    "RS120",
    "状态",
    "趋势年龄",
    "乖离率",
    "标记",
]

PENDING_OPERATION_COLUMNS = ["操作", "标的", "信号日期", "RS120", "原因", "执行时间"]


def build_momentum_decision_view(momentum_result: dict) -> Dict[str, Any]:
    buy_signal = momentum_result.get("buy_signal", {}) if isinstance(momentum_result, dict) else {}
    scanned_stocks = buy_signal.get("scanned_stocks", []) if isinstance(buy_signal, dict) else []
    if not isinstance(scanned_stocks, list) or not scanned_stocks:
        return {"status": "no_scan", "rows": [], "limited_history": ""}

    eligible_stocks = [stock for stock in scanned_stocks if isinstance(stock, dict) and stock.get("eligible")]
    if not eligible_stocks:
        return {
            "status": "no_eligible",
            "rows": [],
            "limited_history": _limited_history_summary(scanned_stocks),
        }

    eligible_stocks.sort(key=lambda stock: -(stock.get("rs120") or 0))
    rows = [_decision_row(stock, index) for index, stock in enumerate(eligible_stocks)]
    return {
        "status": "ok",
        "rows": rows,
        "limited_history": _limited_history_summary(scanned_stocks),
    }


def build_pending_operations(momentum_result: dict) -> List[Dict[str, str]]:
    if not isinstance(momentum_result, dict):
        return []

    pending_operations: List[Dict[str, str]] = []
    position_audit = momentum_result.get("position_audit", {})

    positions = position_audit.get("positions", []) if isinstance(position_audit, dict) else []
    if isinstance(positions, list):
        for pos in positions:
            if isinstance(pos, dict) and pos.get("action_plan") == "待卖出":
                pending_operations.append(_sell_operation(pos))
    elif isinstance(position_audit, dict) and position_audit.get("action_plan") == "待卖出":
        pending_operations.append(_sell_operation(position_audit))

    pending_signals = momentum_result.get("pending_buy_signals")
    if not isinstance(pending_signals, list):
        pending_signal = momentum_result.get("pending_buy_signal", {})
        pending_signals = [pending_signal] if isinstance(pending_signal, dict) and pending_signal.get("ticker") else []

    for pending_signal in pending_signals:
        if not isinstance(pending_signal, dict):
            continue
        pending_operations.append({
            "操作": "🟢 买入",
            "标的": str(pending_signal.get("ticker", "") or ""),
            "信号日期": str(pending_signal.get("signal_date", "") or ""),
            "RS120": format_rs_pct(pending_signal.get("rs120")) if isinstance(pending_signal.get("rs120"), (int, float)) else "",
            "原因": str(pending_signal.get("reason", "") or ""),
            "执行时间": "次日开盘",
        })

    return pending_operations


def _decision_row(stock: dict, index: int) -> Dict[str, Any]:
    tags = []
    if stock.get("is_position"):
        tags.append("✅ 持仓中")
    if index == 0 and stock.get("rs120") is not None:
        tags.append("📊 RS120最高")

    data_tags = []
    if stock.get("limited_history"):
        rs_used = stock.get("rs_window_used")
        hhv_used = stock.get("hhv_window_used")
        history_bars = stock.get("history_bars")
        if isinstance(rs_used, (int, float)):
            data_tags.append(f"RS{int(rs_used)}")
        if isinstance(hhv_used, (int, float)):
            data_tags.append(f"HHV{int(hhv_used)}")
        if isinstance(history_bars, (int, float)):
            data_tags.append(f"{int(history_bars)}根")
    if data_tags:
        tags.append("短历史｜" + "｜".join(data_tags))

    latest_price = stock.get("latest_price")
    hh20_prev = stock.get("hh20_prev")
    trend_ema = stock.get("trend_ema", stock.get("ema100", stock.get("ema50")))
    ema_deviation_pct = stock.get("ema_deviation_pct")
    if not isinstance(ema_deviation_pct, (int, float)) and isinstance(latest_price, (int, float)) and isinstance(trend_ema, (int, float)) and trend_ema > 0:
        ema_deviation_pct = (latest_price / trend_ema - 1) * 100
    rs120 = stock.get("rs120")

    return {
        "标的": str(stock.get("ticker", "") or ""),
        "日期": str(stock.get("latest_date", "") or ""),
        "收盘价": latest_price if isinstance(latest_price, (int, float)) else None,
        "HH20": hh20_prev if isinstance(hh20_prev, (int, float)) else None,
        "RS120": rs120 * 100 if isinstance(rs120, (int, float)) else None,
        "状态": str(stock.get("reason", "") or ""),
        "趋势年龄": _trend_age_text(stock),
        "乖离率": ema_deviation_pct if isinstance(ema_deviation_pct, (int, float)) else None,
        "标记": " | ".join(tags) if tags else "",
    }


def _trend_age_text(stock: dict) -> str:
    trend_age = stock.get("trend_age")
    stage = str(stock.get("trend_age_stage", "") or "")
    position_hint = str(stock.get("position_size_hint", "") or "")
    if not isinstance(trend_age, (int, float)) or not stage or not position_hint:
        return ""
    suffix = "（历史不足估算）" if stock.get("trend_age_estimated") else ""
    return f"{int(trend_age):03d}｜{stage}，{position_hint}{suffix}"


def _sell_operation(position: dict) -> Dict[str, str]:
    return {
        "操作": "🔴 卖出",
        "标的": str(position.get("ticker", "") or ""),
        "信号日期": str(position.get("latest_date", "") or ""),
        "RS120": "",
        "原因": str(position.get("sell_reason", "") or ""),
        "执行时间": "次日开盘",
    }


def _limited_history_summary(scanned_stocks: list) -> str:
    limited_stocks = [
        stock for stock in scanned_stocks
        if isinstance(stock, dict) and stock.get("limited_history")
    ]
    if not limited_stocks:
        return ""

    bits = []
    for stock in limited_stocks[:10]:
        ticker = stock.get("ticker", "")
        rs_used = stock.get("rs_window_used")
        history_bars = stock.get("history_bars")
        if isinstance(rs_used, (int, float)) and isinstance(history_bars, (int, float)):
            bits.append(f"{ticker}(RS{int(rs_used)}/{int(history_bars)}根)")
        else:
            bits.append(str(ticker))
    more = " ..." if len(limited_stocks) > 10 else ""
    return f"⚠️ 短历史回退计算：{', '.join(bits)}{more}"
