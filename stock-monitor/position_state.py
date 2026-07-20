from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from state_manager import get_data_path


POSITION_STATE_VERSION = 1
EXECUTION_LOG_MAX = 200


def default_position_state(now: Optional[datetime] = None) -> Dict[str, Any]:
    if now is None:
        now = datetime.now()
    return {
        "version": POSITION_STATE_VERSION,
        "strategies": {},
        "execution_logs": [],
        "last_updated": now.isoformat(timespec="seconds"),
    }


def _default_strategy_state(buy_ticker: str, benchmark: str, market: str) -> Dict[str, Any]:
    return {
        "buy_ticker": buy_ticker,
        "benchmark": benchmark,
        "market": market,
        "in_position": False,
        "signal_date": None,
        "entry_date": None,
        "entry_price": None,
        "peak_high": None,
        "peak_date": None,
        "notes": "建仓时 peak_high=成本价；扫描自动抬升为持仓期最高 High",
    }


def apply_strategy_entry(st: Dict[str, Any], entry_date: str, entry_price: float) -> None:
    """建仓：峰值初始化为成本价，后续扫描再抬升。"""
    ep = float(entry_price)
    st["in_position"] = True
    st["entry_date"] = entry_date
    st["entry_price"] = ep
    st["peak_high"] = ep
    st["peak_date"] = entry_date


def apply_strategy_exit(st: Dict[str, Any]) -> None:
    """平仓：清空峰值；entry_date/entry_price 保留作记录。"""
    st["in_position"] = False
    st["peak_high"] = None
    st["peak_date"] = None


def resolve_peak_high_for_hold(
    strategy_state: Dict[str, Any],
) -> Tuple[Optional[float], bool]:
    """
    解析持仓峰值：优先 peak_high；缺失时用 entry_price 兜底。
    返回 (peak_high, used_entry_fallback)。
    """
    ph = strategy_state.get("peak_high")
    if ph not in (None, "", "-"):
        try:
            return float(ph), False
        except (TypeError, ValueError):
            pass
    ep = strategy_state.get("entry_price")
    if ep not in (None, "", "-"):
        try:
            return float(ep), True
        except (TypeError, ValueError):
            pass
    return None, False


def append_execution_log(
    state: Dict[str, Any],
    log: str,
    timestamp: Optional[str] = None,
    max_entries: int = EXECUTION_LOG_MAX,
) -> None:
    """向 position_state 追加一条执行日志并截断保留条数。"""
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds").replace("T", " ")[:19]
    logs = state.get("execution_logs")
    if not isinstance(logs, list):
        logs = []
    logs.append({"timestamp": timestamp, "log": log})
    state["execution_logs"] = logs[-max_entries:]
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")


def load_position_state(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = get_data_path("position_state.json")

    if not os.path.exists(path):
        state = default_position_state()
        save_position_state_atomic(state, path)
        return state

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = default_position_state()
        data.setdefault("version", POSITION_STATE_VERSION)
        data.setdefault("strategies", {})
        data.setdefault("execution_logs", [])
        data.setdefault("last_updated", None)
        if not isinstance(data.get("strategies"), dict):
            data["strategies"] = {}
        return data
    except Exception:
        # 读失败不直接抛异常，避免整次扫描崩；由上层返回 ERROR
        return default_position_state()


def save_position_state_atomic(state: Dict[str, Any], path: Optional[str] = None) -> None:
    if path is None:
        path = get_data_path("position_state.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = dict(state)
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")

    payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def get_strategy_state(
    state: Dict[str, Any],
    buy_ticker: str,
    benchmark: str,
    market: str,
    create_if_missing: bool = True,
) -> Dict[str, Any]:
    strategies = state.setdefault("strategies", {})
    if buy_ticker not in strategies:
        if not create_if_missing:
            return {}
        strategies[buy_ticker] = _default_strategy_state(buy_ticker, benchmark, market)
    s = strategies[buy_ticker]

    # 轻量补字段：不覆盖用户手动字段
    if isinstance(s, dict):
        s.setdefault("buy_ticker", buy_ticker)
        s.setdefault("benchmark", benchmark)
        s.setdefault("market", market)
        s.setdefault("in_position", False)
        s.setdefault("signal_date", None)
        s.setdefault("entry_date", None)
        s.setdefault("entry_price", None)
        s.setdefault("peak_high", None)
        s.setdefault("peak_date", None)
        s.setdefault("notes", "建仓时 peak_high=成本价；扫描自动抬升为持仓期最高 High")
        return s

    # 极端：被用户写坏成非 dict，重新初始化但不覆盖整个 strategies
    fresh = _default_strategy_state(buy_ticker, benchmark, market)
    strategies[buy_ticker] = fresh
    return fresh


def validate_in_position_state_fields(strategy_state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    required = ["in_position", "entry_date", "peak_high", "peak_date"]
    missing = [k for k in required if k not in strategy_state]
    if missing:
        return False, f"状态缺字段: {', '.join(missing)}"

    if strategy_state.get("entry_date") in (None, "", "-"):
        return False, "持仓状态 entry_date 为空（请手动填 YYYY-MM-DD）"
    if strategy_state.get("peak_high") in (None, "", "-"):
        ep = strategy_state.get("entry_price")
        if ep in (None, "", "-"):
            return False, "持仓状态 peak_high 为空且无 entry_price（请调仓买入或填写成本价）"
        try:
            float(ep)
        except (TypeError, ValueError):
            return False, f"持仓状态 entry_price 不是数字: {ep}"
    if strategy_state.get("peak_date") in (None, "", "-"):
        return False, "持仓状态 peak_date 为空（请手动填 YYYY-MM-DD）"

    try:
        float(strategy_state.get("peak_high"))
    except Exception:
        return False, f"持仓状态 peak_high 不是数字: {strategy_state.get('peak_high')}"

    return True, None


def maybe_update_peak_high(
    state: Dict[str, Any],
    buy_ticker: str,
    high_t: float,
    data_date: str,
) -> bool:
    """
    仅在 in_position=True 时，peak_high = max(peak_high, high_t)；peak_date 跟随更新。
    返回：是否发生变更（用于决定是否写回文件）。
    """
    strategies = state.get("strategies", {})
    s = strategies.get(buy_ticker)
    if not isinstance(s, dict):
        return False
    if not s.get("in_position"):
        return False

    try:
        prev_peak = float(s.get("peak_high"))
        high_f = float(high_t)
    except Exception:
        return False

    if high_f > prev_peak:
        s["peak_high"] = high_f
        s["peak_date"] = data_date
        state["last_updated"] = datetime.now().isoformat(timespec="seconds")
        return True

    return False
