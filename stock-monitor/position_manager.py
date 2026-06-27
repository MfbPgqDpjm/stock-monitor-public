"""
持仓管理模块 - 处理调仓相关的后端逻辑
"""

import json
import os
from datetime import date
from typing import Dict, Any, Optional, Tuple

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
    # 检查是否已有该标的持仓
    existing_positions = [p for p in current_positions if p.get("ticker") == ticker]
    
    if existing_positions:
        # 合并持仓，计算平均成本
        total_cost = sum(p["buy_price"] * p["quantity"] for p in existing_positions) + (price * quantity)
        total_quantity = sum(p["quantity"] for p in existing_positions) + quantity
        avg_price = total_cost / total_quantity
        
        # 更新持仓
        new_positions = [p for p in current_positions if p.get("ticker") != ticker]
        new_positions.append({
            "ticker": ticker,
            "buy_price": round(avg_price, 4),
            "buy_date": existing_positions[0]["buy_date"],  # 保留最早买入日期
            "quantity": total_quantity,
            "buy_reason": "手动买入(合并)",
            "peak_high_t": round(avg_price, 4),
        })
        state["current_positions"] = new_positions
        message = f"✅ 已记录买入 {ticker} @ {price:.2f} * {quantity}股（已合并持仓，平均成本：{avg_price:.2f}）"
    else:
        # 新建持仓
        current_positions.append({
            "ticker": ticker,
            "buy_price": price,
            "buy_date": trade_date.isoformat(),
            "quantity": quantity,
            "buy_reason": "手动买入",
            "peak_high_t": price
        })
        state["current_positions"] = current_positions
        message = f"✅ 已记录买入 {ticker} @ {price:.2f} * {quantity}股"
    
    # 添加执行日志
    timestamp = trade_execution_timestamp(trade_date, "买入")
    execution_logs.append({
        "timestamp": timestamp,
        "log": f"手动买入 {ticker} @ {price:.2f}"
    })
    state["execution_logs"] = execution_logs[-EXECUTION_LOG_MAX:]
    
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

    # 计算收益
    buy_price = position_to_sell["buy_price"]
    buy_date_str = position_to_sell["buy_date"]
    
    try:
        buy_date = date.fromisoformat(buy_date_str)
    except Exception:
        buy_date = trade_date
    
    total_return = (price - buy_price) / buy_price
    hold_days = (trade_date - buy_date).days
    
    # 添加到历史记录
    history.append({
        "ticker": ticker,
        "buy_price": buy_price,
        "buy_date": buy_date_str,
        "sell_price": price,
        "sell_date": trade_date.isoformat(),
        "sell_reason": "手动卖出",
        "quantity": sell_quantity,
        "total_return": round(total_return, 4),
        "hold_days": hold_days
    })
    state["history"] = history[-100:]
    
    # 更新持仓：部分卖出保留原平均成本，全部卖出移除持仓
    remaining_quantity = position_quantity - sell_quantity
    if remaining_quantity > 1e-9:
        for p in current_positions:
            if p.get("ticker") == ticker:
                p["quantity"] = round(remaining_quantity, 6)
                break
        state["current_positions"] = current_positions
    else:
        current_positions = [p for p in current_positions if p.get("ticker") != ticker]
        state["current_positions"] = current_positions
    
    # 添加执行日志
    timestamp = trade_execution_timestamp(trade_date, "卖出")
    execution_logs.append({
        "timestamp": timestamp,
        "log": f"手动卖出 {ticker} @ {price:.2f} (开盘价), 收益：{total_return*100:.2f}%"
    })
    state["execution_logs"] = execution_logs[-EXECUTION_LOG_MAX:]
    
    # 保存状态文件
    _save_state(state, state_path)
    if remaining_quantity <= 1e-9:
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
