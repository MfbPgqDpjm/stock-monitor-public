"""
持仓管理模块 - 处理调仓相关的后端逻辑
"""

import json
import os
from datetime import date
from typing import Dict, Any, Optional, Tuple, List

from state_manager import (
    get_data_path,
    EXECUTION_LOG_MAX,
    trade_execution_timestamp,
    load_config,
    parse_momentum_ticker_entry,
)
from position_state import (
    load_position_state,
    save_position_state_atomic,
    get_strategy_state,
    apply_strategy_entry,
    apply_strategy_exit,
)


HIFO_METHOD = "HIFO"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_quantity(value: float) -> float:
    rounded = round(float(value), 6)
    return int(rounded) if abs(rounded - int(rounded)) < 1e-9 else rounded


def _normalize_lots(position: Dict[str, Any]) -> List[Dict[str, Any]]:
    lots = position.get("lots")
    clean: List[Dict[str, Any]] = []
    if isinstance(lots, list):
        for lot in lots:
            if not isinstance(lot, dict):
                continue
            quantity = _as_float(lot.get("quantity"))
            price = _as_float(lot.get("buy_price"))
            if quantity <= 0 or price <= 0:
                continue
            clean.append({
                "buy_date": str(lot.get("buy_date") or position.get("buy_date") or ""),
                "buy_price": round(price, 4),
                "quantity": _round_quantity(quantity),
            })
    if clean:
        return clean

    quantity = _as_float(position.get("quantity"))
    price = _as_float(position.get("buy_price"))
    if quantity <= 0 or price <= 0:
        return []
    return [{
        "buy_date": str(position.get("buy_date") or ""),
        "buy_price": round(price, 4),
        "quantity": _round_quantity(quantity),
    }]


def _weighted_avg_cost(lots: List[Dict[str, Any]]) -> Optional[float]:
    quantity = sum(_as_float(lot.get("quantity")) for lot in lots)
    if quantity <= 0:
        return None
    total_cost = sum(_as_float(lot.get("quantity")) * _as_float(lot.get("buy_price")) for lot in lots)
    return total_cost / quantity


def _position_from_lots(
    ticker: str,
    lots: List[Dict[str, Any]],
    existing: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    buy_reason: str = "手动买入(HIFO lot)",
) -> Optional[Dict[str, Any]]:
    lots = [lot for lot in lots if _as_float(lot.get("quantity")) > 0 and _as_float(lot.get("buy_price")) > 0]
    if not lots:
        return None

    avg_cost = _weighted_avg_cost(lots)
    if avg_cost is None:
        return None

    total_quantity = sum(_as_float(lot.get("quantity")) for lot in lots)
    buy_dates = [str(lot.get("buy_date") or "") for lot in lots if lot.get("buy_date")]
    earliest_buy_date = min(buy_dates) if buy_dates else ""

    out = dict(existing or {})
    out.update({
        "ticker": ticker,
        "buy_price": round(avg_cost, 4),
        "buy_date": earliest_buy_date,
        "quantity": _round_quantity(total_quantity),
        "buy_reason": buy_reason,
        "lots": lots,
    })
    if name and not out.get("name"):
        out["name"] = name

    peak = _as_float(out.get("peak_high_t"), avg_cost)
    out["peak_high_t"] = round(max(peak, avg_cost), 4)
    return out


def _consume_hifo_lots(lots: List[Dict[str, Any]], sell_quantity: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    remaining = [_copy_lot(lot) for lot in lots]
    matched: List[Dict[str, Any]] = []
    qty_left = float(sell_quantity)

    remaining.sort(key=lambda lot: (-_as_float(lot.get("buy_price")), str(lot.get("buy_date") or "")))
    for lot in remaining:
        if qty_left <= 1e-9:
            break
        available = _as_float(lot.get("quantity"))
        if available <= 0:
            continue
        take = min(available, qty_left)
        matched.append({
            "buy_date": str(lot.get("buy_date") or ""),
            "buy_price": round(_as_float(lot.get("buy_price")), 4),
            "quantity": _round_quantity(take),
        })
        lot["quantity"] = _round_quantity(available - take)
        qty_left -= take

    if qty_left > 1e-6:
        raise ValueError(f"HIFO lots 数量不足，缺少 {qty_left:g}")

    remaining = [lot for lot in remaining if _as_float(lot.get("quantity")) > 1e-9]
    remaining.sort(key=lambda lot: (str(lot.get("buy_date") or ""), _as_float(lot.get("buy_price"))))
    return matched, remaining


def _copy_lot(lot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "buy_date": str(lot.get("buy_date") or ""),
        "buy_price": round(_as_float(lot.get("buy_price")), 4),
        "quantity": _round_quantity(_as_float(lot.get("quantity"))),
    }


def _matched_cost(matched_lots: List[Dict[str, Any]]) -> float:
    return sum(_as_float(lot.get("quantity")) * _as_float(lot.get("buy_price")) for lot in matched_lots)


def _append_trade_record(
    state: Dict[str, Any],
    side: str,
    ticker: str,
    price: float,
    quantity: float,
    trade_date: date,
    amount: float,
    name: Optional[str] = None,
    source: str = "调仓弹窗",
) -> None:
    records = state.setdefault("trade_records", [])
    if not isinstance(records, list):
        records = []
        state["trade_records"] = records

    next_id = max(
        (int(r.get("id")) for r in records if isinstance(r, dict) and isinstance(r.get("id"), int)),
        default=0,
    ) + 1
    timestamp = trade_execution_timestamp(trade_date, "卖出" if side == "sell" else "买入")
    records.append({
        "side": side,
        "ticker": ticker,
        "name": name or ticker,
        "price": round(float(price), 4),
        "ts": timestamp,
        "amount": round(float(amount), 2),
        "source": source,
        "id": next_id,
        "quantity": _round_quantity(quantity),
    })


def process_trade(
    trade_type: str,
    ticker: str,
    price: float,
    quantity: float = 1,
    trade_date: Optional[date] = None
) -> Tuple[bool, str]:
    """
    处理调仓操作

    Args:
        trade_type: 操作类型，"买入" 或 "卖出"
        ticker: 标的代码
        price: 交易价格
        quantity: 数量（买入/卖出都会写入状态文件）
        trade_date: 交易日期，默认为今天

    Returns:
        (success: bool, message: str)
    """
    if not ticker or price <= 0:
        return False, "请填写完整信息"
    ticker = _configured_momentum_ticker_for(ticker)

    if trade_date is None:
        trade_date = date.today()

    # 加载状态文件
    state_path = get_data_path("momentum_state.json")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        return False, f"加载状态文件失败: {str(e)}"

    current_positions = state.get("current_positions", [])
    history = state.get("history", [])
    execution_logs = state.get("execution_logs", [])

    if trade_type == "买入":
        return _handle_buy(
            state, current_positions, history, execution_logs, state_path,
            ticker, price, quantity, trade_date
        )
    else:  # 卖出
        return _handle_sell(
            state, current_positions, history, execution_logs, state_path,
            ticker, price, quantity, trade_date
        )


def _configured_momentum_ticker_for(ticker: str) -> str:
    """把展示/旧 ticker 规范化为配置中的行情 ticker，例如 VUAA -> VUAA.L。"""
    ticker_u = str(ticker or "").strip().upper()
    if not ticker_u:
        return ""

    try:
        config = load_config()
    except Exception:
        return ticker_u

    configured = {
        parsed.get("ticker")
        for parsed in (parse_momentum_ticker_entry(t) for t in config.get("tickers", []))
        if parsed.get("ticker")
    }
    if ticker_u in configured:
        return ticker_u

    for cfg_ticker in configured:
        if "." in cfg_ticker and cfg_ticker.split(".", 1)[0] == ticker_u:
            return cfg_ticker

    return ticker_u


def _handle_buy(
    state: Dict[str, Any],
    current_positions: list,
    history: list,
    execution_logs: list,
    state_path: str,
    ticker: str,
    price: float,
    quantity: float,
    trade_date: date
) -> Tuple[bool, str]:
    """处理买入操作"""
    position = next((p for p in current_positions if p.get("ticker") == ticker), None)
    new_lot = {
        "buy_date": trade_date.isoformat(),
        "buy_price": round(float(price), 4),
        "quantity": _round_quantity(quantity),
    }

    if position:
        lots = _normalize_lots(position)
        lots.append(new_lot)
        updated = _position_from_lots(ticker, lots, existing=position, buy_reason="手动买入(HIFO lot)")
        current_positions = [updated if p is position else p for p in current_positions]
        state["current_positions"] = current_positions
        message = f"✅ 已记录买入 {ticker} @ {price:.2f} * {quantity}股（HIFO lot，当前平均成本：{updated['buy_price']:.2f}）"
    else:
        current_positions.append(_position_from_lots(ticker, [new_lot], buy_reason="手动买入(HIFO lot)"))
        state["current_positions"] = current_positions
        message = f"✅ 已记录买入 {ticker} @ {price:.2f} * {quantity}股"

    # 添加执行日志
    timestamp = trade_execution_timestamp(trade_date, "买入")
    execution_logs.append({
        "timestamp": timestamp,
        "log": f"手动买入 {ticker} @ {price:.2f}"
    })
    state["execution_logs"] = execution_logs[-EXECUTION_LOG_MAX:]
    _append_trade_record(
        state,
        side="buy",
        ticker=ticker,
        price=price,
        quantity=quantity,
        trade_date=trade_date,
        amount=-(float(price) * float(quantity)),
    )

    # 保存状态文件
    _save_state(state, state_path)
    sync_position_state_on_trade(ticker, "买入", price, trade_date)

    return True, message


def _handle_sell(
    state: Dict[str, Any],
    current_positions: list,
    history: list,
    execution_logs: list,
    state_path: str,
    ticker: str,
    price: float,
    quantity: float,
    trade_date: date
) -> Tuple[bool, str]:
    """处理卖出操作"""
    # 查找持仓
    position_to_sell = None
    for p in current_positions:
        if p.get("ticker") == ticker:
            position_to_sell = p
            break

    if not position_to_sell:
        return False, f"❌ 未找到 {ticker} 的持仓"

    position_quantity = float(position_to_sell.get("quantity", 1))
    sell_quantity = float(quantity or position_quantity)
    if sell_quantity <= 0:
        return False, "卖出数量必须大于 0"
    if sell_quantity > position_quantity:
        return False, f"❌ {ticker} 持仓数量不足（当前 {position_quantity:g}，卖出 {sell_quantity:g}）"

    lots = _normalize_lots(position_to_sell)
    matched_lots, remaining_lots = _consume_hifo_lots(lots, sell_quantity)
    cost_basis = _matched_cost(matched_lots)
    avg_sold_cost = cost_basis / sell_quantity
    proceeds = float(price) * sell_quantity
    realized_pnl = proceeds - cost_basis
    total_return = realized_pnl / cost_basis if cost_basis else 0.0

    buy_dates = [lot.get("buy_date") for lot in matched_lots if lot.get("buy_date")]
    buy_date_str = min(buy_dates) if buy_dates else position_to_sell.get("buy_date")
    try:
        earliest_buy_date = date.fromisoformat(str(buy_date_str))
        hold_days = (trade_date - earliest_buy_date).days
    except Exception:
        hold_days = None

    # 添加到历史记录
    history.append({
        "ticker": ticker,
        "buy_price": round(avg_sold_cost, 4),
        "buy_date": buy_date_str,
        "sell_price": price,
        "sell_date": trade_date.isoformat(),
        "sell_reason": "手动卖出",
        "quantity": sell_quantity,
        "cost_basis_method": HIFO_METHOD,
        "matched_lots": matched_lots,
        "realized_pnl": round(realized_pnl, 2),
        "total_return": round(total_return, 4),
        "hold_days": hold_days
    })
    state["history"] = history[-100:]

    updated_position = _position_from_lots(
        ticker,
        remaining_lots,
        existing=position_to_sell,
        buy_reason=str(position_to_sell.get("buy_reason") or "手动买入(HIFO lot)"),
    )
    if updated_position:
        state["current_positions"] = [updated_position if p is position_to_sell else p for p in current_positions]
    else:
        current_positions = [p for p in current_positions if p.get("ticker") != ticker]
        state["current_positions"] = current_positions

    # 添加执行日志
    timestamp = trade_execution_timestamp(trade_date, "卖出")
    execution_logs.append({
        "timestamp": timestamp,
        "log": f"手动卖出 {ticker} @ {price:.2f}, 收益：{total_return*100:.2f}%"
    })
    state["execution_logs"] = execution_logs[-EXECUTION_LOG_MAX:]
    _append_trade_record(
        state,
        side="sell",
        ticker=ticker,
        price=price,
        quantity=sell_quantity,
        trade_date=trade_date,
        amount=proceeds,
    )

    # 保存状态文件
    _save_state(state, state_path)
    if not updated_position:
        sync_position_state_on_trade(ticker, "卖出", price, trade_date)

    return True, f"✅ 已记录卖出 {ticker} @ {price:.2f}, 收益：{total_return*100:.2f}%"


def _resolve_position_state_targets(ticker: str, config: Dict[str, Any]) -> list:
    """匹配美股大盘配置，返回 (state_key, benchmark, market) 列表。"""
    from strategy import parse_market_configs

    t = ticker.strip().upper()
    hits = []
    seen = set()

    for mc in parse_market_configs(config.get("us_stocks", "")):
        state_key = mc["buy_ticker"]
        if t != state_key or state_key in seen:
            continue
        seen.add(state_key)
        hits.append((state_key, mc["benchmark"], "US"))

    return hits


def sync_position_state_on_trade(
    ticker: str,
    trade_type: str,
    price: float,
    trade_date: date,
) -> None:
    """调仓时同步 position_state（美股大盘）。"""
    try:
        config = load_config()
        targets = _resolve_position_state_targets(ticker, config)
        if not targets:
            return

        ps = load_position_state()
        entry_date = trade_date.isoformat()

        for state_key, benchmark, market in targets:
            st = get_strategy_state(ps, state_key, benchmark, market, create_if_missing=True)
            if trade_type == "买入":
                apply_strategy_entry(st, entry_date, price)
            else:
                apply_strategy_exit(st)

        save_position_state_atomic(ps)
    except Exception:
        # 动量记账已成功；position_state 同步失败不阻断主流程
        pass


def _save_state(state: Dict[str, Any], state_path: str):
    """保存状态文件（原子写入）"""
    temp_path = state_path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, state_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
