#!/usr/bin/env python3
"""
动量系统 V3.0：个股突破 + RS120 排名策略
- 美元市值 >= MIN_MARKET_CAP
- 20 日收盘突破入场，候选按 RS120 排名
- 最多 3 只等权持仓，持仓期间不因 RS 变化换仓
- 卖出：Close_t < 配置趋势 EMA_t（默认 EMA50）
- 信号收盘判定，次日开盘执行
"""
import json
import math
import os
from datetime import datetime, date
from typing import Dict, Any, Optional, List
import pandas as pd
import pytz
import logging
import yfinance as yf

from strategy import (
    _apply_time_slice,
    _df_from_cache_entry,
    _load_data_cache,
    fetch_with_retry,
    _get_market_status,
)
from state_manager import (
    get_data_path,
    EXECUTION_LOG_MAX,
    trade_execution_timestamp,
    parse_momentum_ticker_entry,
    get_momentum_ticker_configs,
)

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ET_TIMEZONE = pytz.timezone("America/New_York")
DEFAULT_MIN_MARKET_CAP = 1_000_000_000
DEFAULT_MAX_POSITIONS = 3
RS_WINDOW = 120
MOMENTUM_EMA_WINDOW = 50
TREND_AGE_FIRST_THRESHOLD = 20
TREND_AGE_MID_THRESHOLD = 60
MARKET_CAP_EXEMPT_TICKERS = {"159915.SZ"}
DEFAULT_MARKET_CAP_FX_RATES = {
    "USD": 1.0,
    "HKD": 0.128,
    "CNY": 0.139,
    "CNH": 0.139,
    "TWD": 0.031,
    "JPY": 0.0064,
    "KRW": 0.00073,
    "EUR": 1.08,
    "GBP": 1.27,
    "CAD": 0.73,
    "AUD": 0.66,
    "CHF": 1.11,
    "SGD": 0.74,
}
MARKET_CAP_WARNING_SEEN = set()


def format_score_pct(score: Optional[float], decimals: int = 2) -> str:
    """兼容旧调用：Close/HH20 评分转为突破幅度。V3 主路径使用 format_rs_pct。"""
    if score is None:
        return ""
    try:
        pct = (float(score) - 1.0) * 100.0
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(pct):
        return ""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.{decimals}f}%"


def format_rs_pct(rs: Optional[float], decimals: int = 2) -> str:
    """RS120 转为百分比展示，如 0.2534 → +25.34%。"""
    if rs is None:
        return ""
    try:
        pct = float(rs) * 100.0
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(pct):
        return ""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.{decimals}f}%"


def _format_log_num(value: Any, decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(num):
        return "-"
    return f"{prefix}{num:.{decimals}f}{suffix}"


def _format_log_bool(value: Any) -> str:
    return "是" if bool(value) else "否"


def top_eligible_scored_stock(scanned_stocks: List[Dict]) -> Optional[Dict[str, Any]]:
    """兼容旧 import：V3 返回 eligible 候选中 RS120 最高者。"""
    return top_eligible_rs_stock(scanned_stocks)


def top_eligible_rs_stock(scanned_stocks: List[Dict]) -> Optional[Dict[str, Any]]:
    """符合买入条件的候选中 RS120 最高者；无候选则 None。"""
    best: Optional[Dict[str, Any]] = None
    best_rs: Optional[float] = None
    for stock in scanned_stocks or []:
        if not stock.get("eligible"):
            continue
        rs120 = stock.get("rs120")
        if rs120 is None or not isinstance(rs120, (int, float)) or not math.isfinite(float(rs120)):
            continue
        rs = float(rs120)
        if best_rs is None or rs > best_rs:
            best_rs = rs
            best = {"ticker": stock.get("ticker", ""), "rs120": rs}
    return best if best and best.get("ticker") else None


class MomentumScorer:
    def __init__(self, config: Dict[str, Any], signals: Dict[str, Any] = None, data_cache: Dict[str, pd.DataFrame] = None):
        self.config = config
        self.signals = signals or {}
        self.data_cache = data_cache or {}
        self.hhv_window = int(config.get("HHV_WINDOW", 20))
        self.min_market_cap = float(config.get("MIN_MARKET_CAP_USD", config.get("MIN_MARKET_CAP", DEFAULT_MIN_MARKET_CAP)))
        self.market_caps = {
            str(k).upper(): v for k, v in (config.get("MARKET_CAPS") or {}).items()
        }
        self.market_cap_fx_rates = DEFAULT_MARKET_CAP_FX_RATES.copy()
        for k, v in (config.get("MARKET_CAP_FX_RATES") or {}).items():
            try:
                self.market_cap_fx_rates[str(k).upper()] = float(v)
            except (TypeError, ValueError):
                logger.warning(f"[市值过滤] MARKET_CAP_FX_RATES 配置无效: {k}={v}")
        self._fx_rate_cache: Dict[str, Optional[float]] = {}
        self._market_cap_cache: Dict[str, Dict[str, Any]] = {}
        self._market_cap_warning_seen = set()
        self.max_positions = int(config.get("MAX_MOMENTUM_POSITIONS", DEFAULT_MAX_POSITIONS))
        self.ticker_configs = get_momentum_ticker_configs(config)
        self.tickers = [t["ticker"] for t in self.ticker_configs]
        
        # 统一状态存储
        self.state = self._load_state()
        self._state_dirty_on_load = False
        # 支持多持仓模式
        self.positions = self.state.get("current_positions", [])
        # 兼容旧的单持仓模式
        if not self.positions and "current_position" in self.state and self.state["current_position"]:
            self.positions = [self.state["current_position"]]
        self.positions = self._normalize_ticker_records(self.positions, "current_positions")
        self.state["current_positions"] = self.positions
        if isinstance(self.state.get("current_position"), dict) and self.state["current_position"].get("ticker"):
            current_position = self._normalize_ticker_record(self.state["current_position"], "current_position")
            self.state["current_position"] = current_position
        self.history = self.state.get("history", [])
        self.history = self._normalize_ticker_records(self.history, "history")
        self.state["history"] = self.history
        self.pending_buy_signals = self._normalize_pending_buy_signals()
        self.pending_buy_signal = self.pending_buy_signals[0] if self.pending_buy_signals else {}
        
        # 执行日志
        self.execution_logs = self.state.get("execution_logs", [])
        if self._state_dirty_on_load:
            self._save_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """加载统一状态文件 momentum_state.json"""
        state_path = get_data_path("momentum_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {
            "current_positions": [],
            "cooling_off": {},
            "history": [],
            "pending_buy_signal": {},
            "pending_buy_signals": [],
            "execution_logs": [],
        }

    def _normalize_pending_buy_signals(self) -> List[Dict[str, Any]]:
        """读取 V3 pending 列表；兼容旧版单个 pending_buy_signal。"""
        raw_list = self.state.get("pending_buy_signals")
        if isinstance(raw_list, list):
            clean = self._normalize_ticker_records(
                [p for p in raw_list if isinstance(p, dict) and p.get("ticker")],
                "pending_buy_signals",
            )
            self.state["pending_buy_signals"] = clean
            self.state["pending_buy_signal"] = clean[0] if clean else {}
            return clean
        legacy = self.state.get("pending_buy_signal")
        if isinstance(legacy, dict) and legacy.get("ticker"):
            clean = [self._normalize_ticker_record(legacy, "pending_buy_signal")]
            self.state["pending_buy_signals"] = clean
            self.state["pending_buy_signal"] = clean[0]
            return clean
        self.state["pending_buy_signals"] = []
        self.state["pending_buy_signal"] = {}
        return []

    def _normalize_ticker_record(self, record: Dict[str, Any], context: str) -> Dict[str, Any]:
        ticker = record.get("ticker")
        configured_ticker = self._configured_ticker_for(ticker)
        if configured_ticker and ticker and configured_ticker != str(ticker).strip().upper():
            normalized = record.copy()
            normalized["ticker"] = configured_ticker
            self._state_dirty_on_load = True
            logger.info(f"[状态规范化] {context}: {ticker} -> {configured_ticker}")
            return normalized
        return record

    def _normalize_ticker_records(self, records: Any, context: str) -> List[Dict[str, Any]]:
        if not isinstance(records, list):
            return []
        normalized = []
        for record in records:
            if isinstance(record, dict):
                normalized.append(self._normalize_ticker_record(record, context))
        return normalized
    
    def _save_state(self):
        """保存统一状态文件（原子写入）"""
        state_path = get_data_path("momentum_state.json")
        temp_path = state_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, state_path)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _update_positions(self, positions: List[Dict[str, Any]]):
        """更新持仓列表"""
        self.positions = positions
        self.state["current_positions"] = positions
        self._save_state()
    
    def _add_position(self, position: Dict[str, Any]):
        """添加持仓"""
        self.positions.append(position)
        self.state["current_positions"] = self.positions
        self._save_state()
    
    def _remove_position(self, ticker: str):
        """移除指定持仓"""
        self.positions = [p for p in self.positions if p.get("ticker") != ticker]
        self.state["current_positions"] = self.positions
        self._save_state()
    
    def _update_pending_buy_signals(self, pending_buy_signals: List[Dict[str, Any]]):
        """更新待买入信号列表，并维护旧字段兼容。"""
        clean = [p for p in pending_buy_signals if isinstance(p, dict) and p.get("ticker")]
        self.pending_buy_signals = clean
        self.pending_buy_signal = clean[0] if clean else {}
        self.state["pending_buy_signals"] = clean
        self.state["pending_buy_signal"] = self.pending_buy_signal
        self._save_state()

    def _update_pending_buy_signal(self, pending_buy_signal: Optional[Dict[str, Any]]):
        """兼容旧调用：更新为单个 pending。"""
        self._update_pending_buy_signals([pending_buy_signal] if pending_buy_signal else [])
    
    def _add_history(self, record: Dict[str, Any]):
        """添加历史记录"""
        self.history.append(record)
        self.state["history"] = self.history[-100:]
        self._save_state()
    
    def _add_execution_log(
        self,
        log: str,
        trade_date: Optional[date] = None,
        trade_type: Optional[str] = None,
    ):
        """添加执行日志；提供 trade_date+trade_type 时使用 09:30:00/09:30:01。"""
        if trade_date is not None and trade_type is not None:
            timestamp = trade_execution_timestamp(trade_date, trade_type)
        else:
            timestamp = datetime.now(ET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
        self.execution_logs.append({"timestamp": timestamp, "log": log})
        self.execution_logs = self.execution_logs[-EXECUTION_LOG_MAX:]
        # 保存到状态文件
        self.state["execution_logs"] = self.execution_logs
        self._save_state()
    
    def _load_momentum_result(self) -> Optional[Dict[str, Any]]:
        """加载动量评分结果"""
        result_path = get_data_path("momentum_result.json")
        if not os.path.exists(result_path):
            return None
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"Failed to load momentum result: {e}")
            return None
    
    def _save_momentum_result(self, result: Dict[str, Any]):
        """保存动量评分结果"""
        result_path = get_data_path("momentum_result.json")
        temp_path = result_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, result_path)
        except Exception as e:
            logger.error(f"Failed to save momentum result: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _is_market_open(self) -> bool:
        """检查美股是否处于 strategy 定义的交易时段（与全站扫描一致）。"""
        return _get_market_status(datetime.now(ET_TIMEZONE), market="US") == "open"
    
    def _get_market_data(self, ticker: str, period: str = "1y", now_et: Optional[datetime] = None, purpose: str = "scan") -> Optional[pd.DataFrame]:
        """
        获取市场数据
        
        Args:
            ticker: 股票代码
            period: 数据周期
            now_et: 当前时间
            purpose: 用途（scan=扫描，execute=执行）
        """
        original_ticker = str(ticker or "").strip().upper()
        ticker = self._configured_ticker_for(ticker)
        if ticker and original_ticker and ticker != original_ticker:
            logger.info(f"[数据获取] {original_ticker} 使用配置行情代码 {ticker}")

        logger.debug(f"[数据获取] 尝试获取 {ticker} 数据 (purpose={purpose})")
        
        # 优先从 data_cache 中获取数据
        if ticker in self.data_cache:
            logger.debug(f"[数据获取] 从缓存中找到 {ticker} 数据")
            data = self.data_cache[ticker]
            
            if data is not None and not data.empty:
                logger.debug(f"[数据获取] {ticker} 数据长度：{len(data)} 最新日期：{data.index[-1].date()}")
                
                # 应用时间切片逻辑，处理未收盘的情况
                if now_et is not None:
                    logger.debug(f"[数据获取] 应用时间切片，当前时间：{now_et}")
                    # 对 Close 列应用时间切片
                    close_series = data["Close"].squeeze()
                    
                    if isinstance(close_series, pd.Series):
                        logger.debug(f"[数据获取] Close 系列长度：{len(close_series)}")
                        sliced_close = _apply_time_slice(close_series, now_et, market="US", purpose=purpose)
                        logger.debug(f"[数据获取] 切片后长度：{len(sliced_close)}")
                        
                        # 如果切片后数据为空，则返回 None
                        if sliced_close.empty:
                            logger.warning(f"[数据获取] {ticker} 切片后为空，跳过")
                            return None
                        # 实际替换为切片后的数据
                        data = data.loc[sliced_close.index]
                        logger.debug(f"[数据获取] 切片后数据长度：{len(data)}")
                return data
            else:
                logger.warning(f"[数据获取] {ticker} 缓存中数据为空")
        else:
            logger.debug(f"[数据获取] 缓存中不存在 {ticker}，将尝试下载")
        
        # 回退：当缓存中没有数据时，尝试直接下载
        try:
            logger.debug(f"[数据获取] 尝试直接下载 {ticker} 数据")
            data = fetch_with_retry(ticker, period=period, now_et=now_et, market="US")
            
            if data is not None and not data.empty:
                logger.debug(f"[数据获取] 直接下载成功 {ticker}，长度：{len(data)}")
                
                # 应用时间切片逻辑，处理未收盘的情况
                if now_et is not None:
                    # 对 Close 列应用时间切片
                    close_series = data["Close"].squeeze()
                    
                    if isinstance(close_series, pd.Series):
                        sliced_close = _apply_time_slice(close_series, now_et, market="US", purpose=purpose)
                        
                        # 如果切片后数据为空，则返回 None
                        if sliced_close.empty:
                            logger.warning(f"[数据获取] {ticker} 下载后切片为空，跳过")
                            return None
                        # 实际替换为切片后的数据
                        data = data.loc[sliced_close.index]
                        logger.debug(f"[数据获取] 切片后数据长度：{len(data)}")
                return data
            else:
                logger.warning(f"[数据获取] 直接下载失败或为空：{ticker}")
        except Exception as e:
            logger.error(f"[数据获取] 直接下载出错：{str(e)}")

        cached_data = self._load_cached_market_data(ticker, now_et=now_et, purpose=purpose)
        if cached_data is not None:
            logger.warning(f"[数据获取] {ticker} 直接下载失败，使用磁盘缓存 {cached_data.index[-1].date()}")
            return cached_data
        
        return None

    def _load_cached_market_data(self, ticker: str, now_et: Optional[datetime] = None, purpose: str = "scan") -> Optional[pd.DataFrame]:
        try:
            cache = _load_data_cache()
            entry = (cache.get("data") or {}).get(ticker)
            data = _df_from_cache_entry(entry)
            if data is None or data.empty:
                return None

            if now_et is not None:
                close_series = data["Close"].squeeze()
                if isinstance(close_series, pd.Series):
                    sliced_close = _apply_time_slice(close_series, now_et, market="US", purpose=purpose)
                    if sliced_close.empty:
                        logger.warning(f"[数据获取] {ticker} 磁盘缓存切片后为空")
                        return None
                    data = data.loc[sliced_close.index]
            return data
        except Exception as exc:
            logger.warning(f"[数据获取] {ticker} 读取磁盘缓存失败: {exc}")
            return None
    
    def _indicator_for_ticker(self, ticker: str) -> Dict[str, Any]:
        ticker_u = self._configured_ticker_for(ticker)
        for cfg in self.ticker_configs:
            if cfg.get("ticker") == ticker_u:
                return cfg
        return parse_momentum_ticker_entry(ticker_u)

    def _signal_ticker_for(self, ticker: str) -> str:
        cfg = self._indicator_for_ticker(ticker)
        return str(cfg.get("signal_ticker") or cfg.get("ticker") or ticker).strip().upper()

    def _configured_ticker_for(self, ticker: str) -> str:
        """Resolve display/legacy tickers to configured Yahoo symbols."""
        ticker_u = str(ticker or "").strip().upper()
        if not ticker_u:
            return ""

        configured = {str(cfg.get("ticker", "")).upper() for cfg in self.ticker_configs}
        if ticker_u in configured:
            return ticker_u

        # Some state/UI rows use a display symbol such as VUAA while config uses VUAA.L.
        for cfg_ticker in configured:
            if "." in cfg_ticker and cfg_ticker.split(".", 1)[0] == ticker_u:
                return cfg_ticker

        return ticker_u

    def _compute_indicators(self, data: pd.DataFrame, ema_window: int = MOMENTUM_EMA_WINDOW) -> Dict[str, Any]:
        """
        计算指标 - 严格对齐索引，短历史标的回退到可用窗口
        
        公式：
        - HHV20_{t-1}: 使用 hhv20.iloc[-2] (截止到昨天的 20 日最高价)
        - HHV20_{t-2}: 使用 hhv20.iloc[-3] (截止到前天的 20 日最高价)
        - CLOSE_t: 使用 close.iloc[-1] (今日收盘价)
        - CLOSE_{t-1}: 使用 close.iloc[-2] (昨日收盘价)
        - EMA_t: 使用 close.ewm(span=配置窗口).mean().iloc[-1]，默认 EMA50
        - RS120: Close_t / Close_{t-120} - 1
        若历史不足完整窗口，则使用已有的尽可能长历史：
        - HHV 使用可覆盖 t-1/t-2 的最大可用窗口
        - RS 使用最早可用收盘价作为基准
        - EMA 仍使用配置 span，但标记为短历史估算
        """
        if data is None or data.empty:
            return {}
        
        # 确保 close 是一个 Series
        close = data["Close"].squeeze()

        if len(close) < 3:
            logger.debug("[指标计算] 数据长度不足 3，无法计算突破所需的 t/t-1/t-2")
            return {}

        available_len = len(close)
        ema_window = int(ema_window or MOMENTUM_EMA_WINDOW)
        hhv_window_used = min(self.hhv_window, available_len - 2)
        rs_window_used = min(RS_WINDOW, available_len - 1)
        limited_history = (
            hhv_window_used < self.hhv_window
            or rs_window_used < RS_WINDOW
            or available_len < ema_window
        )

        hhv20 = close.rolling(window=hhv_window_used).max()
        trend_ema = close.ewm(span=ema_window, adjust=False).mean()
        rs_base = float(close.iloc[-(rs_window_used + 1)])
        trend_indicator = f"EMA{ema_window}"
        latest_close = float(close.iloc[-1])
        latest_trend_ema = float(trend_ema.iloc[-1])
        
        indicators = {
            "close": latest_close,
            "close_prev": float(close.iloc[-2]),
            "hhv20_prev": float(hhv20.iloc[-2]),
            "hhv20_prev_prev": float(hhv20.iloc[-3]),
            "ema100": latest_trend_ema,
            "trend_ema": latest_trend_ema,
            "trend_indicator": trend_indicator,
            "trend_ema_window": ema_window,
            "rs120": (latest_close / rs_base - 1) if rs_base > 0 else None,
            "ema_deviation_pct": (latest_close / latest_trend_ema - 1) * 100 if latest_trend_ema > 0 else None,
            "limited_history": limited_history,
            "history_bars": available_len,
            "hhv_window_used": hhv_window_used,
            "rs_window_used": rs_window_used,
        }
        
        logger.debug(
            f"[指标计算] close_t={indicators['close']:.4f}, close_t1={indicators['close_prev']:.4f}, "
            f"hhv20_t1={indicators['hhv20_prev']:.4f}, hhv20_t2={indicators['hhv20_prev_prev']:.4f}, "
            f"{trend_indicator}={indicators['trend_ema']:.4f}, rs120={format_rs_pct(indicators['rs120'])}, "
            f"history_bars={available_len}, hhv_window_used={hhv_window_used}, "
            f"rs_window_used={rs_window_used}, limited_history={limited_history}"
        )
        
        return indicators

    def _classify_trend_age(self, data: pd.DataFrame, ema_window: int = MOMENTUM_EMA_WINDOW) -> Dict[str, Any]:
        """计算今天买入信号相对最近一次卖出后首次买入的趋势年龄。"""
        if data is None or data.empty or "Close" not in data:
            return {}

        close = data["Close"].squeeze()
        if len(close) < 3:
            return {}

        ema_window = int(ema_window or MOMENTUM_EMA_WINDOW)
        hhv_window_used = min(self.hhv_window, len(close) - 2)
        hhv = close.rolling(window=hhv_window_used).max()
        trend_ema = close.ewm(span=ema_window, adjust=False).mean()

        buy_signal = (close > trend_ema) & (close > hhv.shift(1)) & (close.shift(1) <= hhv.shift(2))
        sell_signal = close < trend_ema
        latest_pos = len(close) - 1

        if not bool(buy_signal.iloc[latest_pos]):
            return {}

        prior_sell_positions = [
            pos for pos, has_signal in enumerate(sell_signal.iloc[:latest_pos])
            if bool(has_signal)
        ]
        trend_age_estimated = False
        last_sell_pos = None
        if prior_sell_positions:
            last_sell_pos = prior_sell_positions[-1]
            buy_after_sell_positions = [
                pos for pos in range(last_sell_pos + 1, latest_pos + 1)
                if bool(buy_signal.iloc[pos])
            ]
        else:
            trend_age_estimated = True
            buy_after_sell_positions = [
                pos for pos in range(0, latest_pos + 1)
                if bool(buy_signal.iloc[pos])
            ]
        if not buy_after_sell_positions:
            return {}

        first_buy_pos = buy_after_sell_positions[0]
        trend_age = latest_pos - first_buy_pos
        if trend_age <= TREND_AGE_FIRST_THRESHOLD:
            trend_age_stage = "首次趋势"
            position_size_hint = "正常仓位"
        elif trend_age <= TREND_AGE_MID_THRESHOLD:
            trend_age_stage = "中期趋势"
            position_size_hint = "半仓"
        else:
            trend_age_stage = "老趋势"
            position_size_hint = "观察仓"

        return {
            "trend_age": trend_age,
            "trend_age_stage": trend_age_stage,
            "position_size_hint": position_size_hint,
            "trend_age_estimated": trend_age_estimated,
            "trend_age_basis": "可用历史内首次买入" if trend_age_estimated else "最近卖出后首次买入",
            "last_sell_signal_date": close.index[last_sell_pos].date().isoformat() if last_sell_pos is not None else None,
            "first_buy_after_sell_date": close.index[first_buy_pos].date().isoformat(),
            "latest_buy_signal_date": close.index[latest_pos].date().isoformat(),
            "trend_age_first_threshold": TREND_AGE_FIRST_THRESHOLD,
            "trend_age_mid_threshold": TREND_AGE_MID_THRESHOLD,
        }
    
    def _warn_market_cap_once(self, ticker: str, message: str) -> None:
        """同一 ticker 的市值 warning 只打一次，避免 Streamlit/扫描重复刷屏。"""
        ticker_u = str(ticker).upper()
        if ticker_u in self._market_cap_warning_seen or ticker_u in MARKET_CAP_WARNING_SEEN:
            return
        self._market_cap_warning_seen.add(ticker_u)
        MARKET_CAP_WARNING_SEEN.add(ticker_u)
        logger.warning(message)

    def _convert_market_cap_to_usd(self, value: Any, currency: Optional[str], source: Optional[str] = None) -> Dict[str, Any]:
        """把市值统一折算为美元；返回原币值、币种、汇率和美元值。"""
        try:
            local_value = float(value)
        except (TypeError, ValueError):
            return {"market_cap": None, "market_cap_currency": currency, "market_cap_fx_rate": None, "market_cap_usd": None, "market_cap_source": source}
        if not math.isfinite(local_value):
            return {"market_cap": None, "market_cap_currency": currency, "market_cap_fx_rate": None, "market_cap_usd": None, "market_cap_source": source}

        raw_currency = str(currency or "USD").strip()
        currency_code = raw_currency.upper()
        fx_rate = self._fx_rate_to_usd(raw_currency)
        market_cap_usd = local_value * fx_rate if fx_rate is not None else None
        return {
            "market_cap": local_value,
            "market_cap_currency": raw_currency or currency_code,
            "market_cap_fx_rate": fx_rate,
            "market_cap_usd": market_cap_usd,
            "market_cap_source": source,
        }

    def _fx_rate_to_usd(self, currency: str) -> Optional[float]:
        """获取 1 单位 currency 折合多少 USD；配置优先，yfinance 兜底。"""
        raw_currency = str(currency or "USD").strip()
        currency_code = raw_currency.upper()
        if currency_code in ("USD", "US DOLLAR"):
            return 1.0
        if raw_currency == "GBp" or currency_code in ("GBX", "GBPENCE"):
            gbp_rate = self._fx_rate_to_usd("GBP")
            return gbp_rate / 100.0 if gbp_rate is not None else None
        if currency_code in self.market_cap_fx_rates:
            return self.market_cap_fx_rates[currency_code]
        if currency_code in self._fx_rate_cache:
            return self._fx_rate_cache[currency_code]

        pair = f"{currency_code}USD=X"
        try:
            fx_data = yf.download(pair, period="5d", progress=False, auto_adjust=True)
            if fx_data is None or fx_data.empty or "Close" not in fx_data.columns:
                self._fx_rate_cache[currency_code] = None
                return None
            close = fx_data["Close"].squeeze().dropna()
            if close.empty:
                self._fx_rate_cache[currency_code] = None
                return None
            rate = float(close.iloc[-1])
            self._fx_rate_cache[currency_code] = rate if math.isfinite(rate) else None
            return self._fx_rate_cache[currency_code]
        except Exception as e:
            logger.warning(f"[市值过滤] 汇率获取失败 {pair}: {e}")
            self._fx_rate_cache[currency_code] = None
            return None

    def _get_market_cap(self, ticker: str) -> Dict[str, Any]:
        """获取市值并统一美元口径；优先使用 config.MARKET_CAPS 覆盖。"""
        ticker_u = str(ticker).upper()
        if ticker_u in self._market_cap_cache:
            return self._market_cap_cache[ticker_u]

        configured = self.market_caps.get(ticker_u)
        if configured is not None:
            if isinstance(configured, dict):
                value = configured.get("value", configured.get("market_cap"))
                currency = configured.get("currency", "USD")
            else:
                value = configured
                currency = "USD"
            cap = self._convert_market_cap_to_usd(value, currency, source="config")
            if cap["market_cap_usd"] is None:
                self._warn_market_cap_once(ticker_u, f"[市值过滤] {ticker_u} MARKET_CAPS 配置无效: {configured}")
            self._market_cap_cache[ticker_u] = cap
            return cap

        try:
            ticker_obj = yf.Ticker(ticker_u)
            fast_info = getattr(ticker_obj, "fast_info", None)
            market_cap = None
            currency = None
            source = None
            if fast_info is not None:
                try:
                    market_cap = fast_info.get("market_cap")
                    currency = fast_info.get("currency")
                except AttributeError:
                    market_cap = getattr(fast_info, "market_cap", None)
                    currency = getattr(fast_info, "currency", None)
                if market_cap is not None:
                    source = "fast_info.market_cap"
            if market_cap is None or not currency:
                info = ticker_obj.info
                if market_cap is None:
                    market_cap = info.get("marketCap")
                    source = "marketCap" if market_cap is not None else source
                if market_cap is None:
                    for key in ("totalAssets", "netAssets", "fundTotalAssets"):
                        val = info.get(key)
                        if val is not None:
                            market_cap = val
                            source = key
                            break
                currency = currency or info.get("currency") or info.get("financialCurrency") or "USD"
            cap = self._convert_market_cap_to_usd(market_cap, currency, source=source)
            if cap["market_cap_usd"] is None:
                self._warn_market_cap_once(ticker_u, f"[市值过滤] {ticker_u} 市值或资产规模无效: cap={market_cap}, currency={currency}")
            self._market_cap_cache[ticker_u] = cap
            return cap
        except Exception as e:
            self._warn_market_cap_once(ticker_u, f"[市值过滤] {ticker_u} 市值获取失败: {e}")
            cap = {"market_cap": None, "market_cap_currency": None, "market_cap_fx_rate": None, "market_cap_usd": None, "market_cap_source": None}
            self._market_cap_cache[ticker_u] = cap
            return cap

    def _check_market_cap_filter(self, ticker: str) -> Dict[str, Any]:
        """市值过滤：美元市值 >= MIN_MARKET_CAP。"""
        ticker_u = str(ticker or "").strip().upper()
        if ticker_u in MARKET_CAP_EXEMPT_TICKERS:
            return {
                "market_cap": None,
                "market_cap_currency": None,
                "market_cap_fx_rate": None,
                "market_cap_usd": None,
                "market_cap_source": "exempt",
                "market_cap_ok": True,
                "market_cap_reason": "资产规模检查豁免",
            }

        cap = self._get_market_cap(ticker)
        market_cap_usd = cap.get("market_cap_usd")
        ok = market_cap_usd is not None and market_cap_usd >= self.min_market_cap
        reason = "市值通过" if ok else "市值低于阈值或获取失败"
        return {
            **cap,
            "market_cap_ok": ok,
            "market_cap_reason": reason,
        }

    def _log_scan_stock_details(
        self,
        scanned_stocks: List[Dict[str, Any]],
        selected_candidates: List[Dict[str, Any]],
    ) -> None:
        """逐票打印买入决策，重点解释为什么选中或未选中。"""
        if not scanned_stocks:
            return

        active_positions = [p for p in self.positions if not p.get("sell_flag")]
        pending_set = {
            self._configured_ticker_for(p.get("ticker"))
            for p in self.pending_buy_signals
        }
        available_slots = max(0, self.max_positions - len(active_positions) - len(self.pending_buy_signals))

        rs_rank_by_ticker: Dict[str, int] = {}
        rs_rank = 0
        for stock in scanned_stocks:
            rs120 = stock.get("rs120")
            if not isinstance(rs120, (int, float)) or not math.isfinite(float(rs120)):
                continue
            rs_rank += 1
            rs_rank_by_ticker[str(stock.get("ticker", "")).upper()] = rs_rank

        candidate_rank_by_ticker = {
            str(stock.get("ticker", "")).upper(): idx
            for idx, stock in enumerate(selected_candidates, start=1)
        }

        logger.info(
            "[买入扫描][决策] 逐票原因，共 %s 只，可用仓位=%s",
            len(scanned_stocks),
            available_slots,
        )
        for stock in scanned_stocks:
            ticker = str(stock.get("ticker", "") or "-")
            ticker_u = ticker.upper()
            trend_indicator = str(stock.get("trend_indicator") or f"EMA{stock.get('trend_ema_window') or ''}").strip()
            if not trend_indicator or trend_indicator == "EMA":
                trend_indicator = "趋势EMA"

            close_above_trend = bool(stock.get("close_above_trend_ema"))
            breakout = bool(stock.get("breakout"))
            market_cap_ok = bool(stock.get("market_cap_ok"))
            candidate_rank = candidate_rank_by_ticker.get(ticker_u)
            is_pending = ticker_u in pending_set
            is_position = bool(stock.get("is_position"))
            is_eligible = bool(stock.get("eligible"))

            if is_eligible and candidate_rank is not None and candidate_rank <= available_slots:
                decision = "选中"
                decision_reason = "符合买入条件且在可用仓位内"
            elif is_eligible and is_position:
                decision = "未选中"
                decision_reason = "已持仓，不重复买入"
            elif is_eligible and is_pending:
                decision = "未选中"
                decision_reason = "已有待买信号，不重复标记"
            elif is_eligible and available_slots <= 0:
                decision = "未选中"
                decision_reason = "持仓/待买已占满可用仓位"
            elif is_eligible:
                decision = "未选中"
                decision_reason = f"候选排名 {candidate_rank or '-'} 超过可用仓位 {available_slots}"
            else:
                decision = "未选中"
                decision_reason = stock.get("reason") or "-"

            block_bits = []
            if not close_above_trend:
                block_bits.append(f"未站上{trend_indicator}")
            if close_above_trend and not breakout:
                block_bits.append("未突破HH20")
            if close_above_trend and breakout and not market_cap_ok:
                block_bits.append(stock.get("market_cap_reason") or "市值/资产规模未通过")
            block_text = "；".join(block_bits) if block_bits else decision_reason

            trend_age_bits = []
            if stock.get("trend_age") is not None:
                trend_age_bits.extend([
                    f"趋势年龄={stock.get('trend_age')}日",
                    f"阶段={stock.get('trend_age_stage') or '-'}",
                ])
                if stock.get("trend_age_estimated"):
                    trend_age_bits.append("历史不足估算")
            trend_age_text = " ".join(trend_age_bits)
            history_text = " 短历史=是" if stock.get("limited_history") else ""
            trend_age_suffix = f" {trend_age_text}" if trend_age_text else ""

            logger.info(
                "[买入扫描][决策] %s %s | 信号=%s | RS#%s 候选#%s | "
                "交易价=%s 信号收盘=%s %s=%s 偏离=%s HH20前高=%s RS120=%s | "
                "趋势=%s 突破=%s 市值=%s 持仓=%s 待买=%s%s%s | 原因=%s",
                ticker,
                decision,
                stock.get("signal_ticker") or ticker,
                rs_rank_by_ticker.get(ticker_u, "-"),
                candidate_rank or "-",
                _format_log_num(stock.get("latest_price"), 4),
                _format_log_num(stock.get("signal_latest_price"), 4),
                trend_indicator,
                _format_log_num(stock.get("trend_ema"), 4),
                _format_log_num(stock.get("ema_deviation_pct"), 2, suffix="%"),
                _format_log_num(stock.get("hh20_prev"), 4),
                format_rs_pct(stock.get("rs120")) or "-",
                _format_log_bool(close_above_trend),
                _format_log_bool(breakout),
                "是" if market_cap_ok else "否/未检查",
                _format_log_bool(is_position),
                _format_log_bool(is_pending),
                history_text,
                trend_age_suffix,
                block_text,
            )

    def _open_for_trade_date(self, data: Optional[pd.DataFrame], trade_date: date) -> Optional[float]:
        """取 trade_date 当日日 K 的 Open；缺行或无效则返回 None（禁止回退 iloc[-1]）。"""
        if data is None or data.empty or "Open" not in data.columns:
            return None
        opens = data["Open"].squeeze()
        for i in range(len(data) - 1, -1, -1):
            idx = data.index[i]
            bar_date = idx.date() if hasattr(idx, "date") else pd.Timestamp(idx).date()
            if bar_date != trade_date:
                continue
            val = float(opens.iloc[i] if isinstance(opens, pd.Series) else opens[i])
            return val if math.isfinite(val) else None
        return None

    def _fetch_open_on_trade_date(self, ticker: str, trade_date: date, now: datetime) -> Optional[float]:
        """执行前直连数据源拉日 K，按执行日取开盘价（绕过 data_cache 末行=T-1）。"""
        data = fetch_with_retry(ticker, period="10d", now_et=now, market="US")
        return self._open_for_trade_date(data, trade_date)
    
    def _execute_pending_orders(self, now: datetime):
        """
        执行引擎：处理待执行的买入/卖出订单
        
        逻辑：
        1. 检查市场状态，只有开盘时才执行操作
        2. 检查当前持仓是否有 sell_flag，若有则执行卖出
        3. 检查 pending_buy_signals，若存在且已过 T+1 日，则执行买入
        """
        # 检查市场状态，只有开盘时才执行操作
        market_status = _get_market_status(now, market="US")
        if market_status != "open":
            logger.debug(f"[执行引擎] 市场未开盘（状态：{market_status}），跳过执行")
            return
        
        current_date = now.date()
        
        # === 执行卖出 ===
        # 遍历所有持仓检查卖出信号
        for position in list(self.positions):
            if position.get("sell_flag"):
                ticker = position.get("ticker")
                sell_flag_date_str = position.get("sell_flag_date")
                
                if sell_flag_date_str:
                    sell_flag_date = datetime.fromisoformat(sell_flag_date_str).date()
                    days_since_flag = (current_date - sell_flag_date).days
                    
                    logger.debug(f"[执行引擎] 检查待卖出持仓：{ticker}, 标记日期：{sell_flag_date}, 已过天数：{days_since_flag}")
                    
                    # T+1 日或之后，执行卖出
                    if days_since_flag >= 1:
                        logger.debug(f"[执行引擎] 执行卖出：{ticker}")
                        open_price = self._fetch_open_on_trade_date(ticker, current_date, now)
                        if open_price is None:
                            logger.warning(
                                f"[执行引擎] {ticker} 执行日 {current_date} 无有效开盘价，延后至下次扫描"
                            )
                            continue

                        sell_date = current_date.isoformat()
                        buy_price = position.get("buy_price")
                        total_return = (open_price / buy_price) - 1

                        history_record = {
                            "ticker": ticker,
                            "buy_price": buy_price,
                            "buy_date": position.get("buy_date"),
                            "sell_price": open_price,
                            "sell_date": sell_date,
                            "sell_reason": position.get("sell_reason"),
                            "total_return": total_return,
                            "hold_days": (current_date -
                                         datetime.strptime(position.get("buy_date"), "%Y-%m-%d").date()).days + 1
                        }
                        self._add_history(history_record)
                        self._remove_position(ticker)
                        self._add_execution_log(
                            f"自动卖出 {ticker} @ {open_price:.2f} (开盘价), 收益：{total_return*100:.2f}%",
                            trade_date=current_date,
                            trade_type="卖出",
                        )
                        logger.info(
                            f"[执行引擎] 卖出成功：{ticker} 执行日={sell_date} @ {open_price:.2f}, "
                            f"收益：{total_return*100:.2f}%"
                        )
        
        # === 执行买入 ===
        remaining_pending: List[Dict[str, Any]] = []
        for pending in list(self.pending_buy_signals):
            pending_ticker = pending.get("ticker")
            signal_date_str = pending.get("signal_date")
            
            if signal_date_str:
                signal_date = datetime.fromisoformat(signal_date_str).date()
                days_since_signal = (current_date - signal_date).days
                
                logger.debug(f"[执行引擎] 检查待买入信号：{pending_ticker}, 信号日期：{signal_date}, 已过天数：{days_since_signal}")
                
                # T+1 日或之后，执行买入
                if days_since_signal >= 1:
                    if len(self.positions) >= self.max_positions:
                        logger.info(f"[执行引擎] 持仓已满，保留待买入信号：{pending_ticker}")
                        remaining_pending.append(pending)
                        continue

                    logger.debug(f"[执行引擎] 执行买入：{pending_ticker}")
                    open_price = self._fetch_open_on_trade_date(pending_ticker, current_date, now)
                    if open_price is None:
                        logger.warning(
                            f"[执行引擎] {pending_ticker} 执行日 {current_date} 无有效开盘价，延后至下次扫描"
                        )
                        remaining_pending.append(pending)
                        continue

                    buy_date = current_date.isoformat()
                    new_position = {
                        "ticker": pending_ticker,
                        "buy_price": open_price,
                        "buy_date": buy_date,
                        "buy_reason": pending.get("reason", "动量突破"),
                        "signal_rs120": pending.get("rs120"),
                        "peak_high_t": open_price,
                    }
                    self._add_position(new_position)
                    self._add_execution_log(
                        f"自动买入 {pending_ticker} @ {open_price:.2f} (开盘价), RS120：{format_rs_pct(pending.get('rs120'))}",
                        trade_date=current_date,
                        trade_type="买入",
                    )
                    logger.info(
                        f"[执行引擎] 买入成功：{pending_ticker} 执行日={buy_date} @ {open_price:.2f}"
                    )
                else:
                    remaining_pending.append(pending)
            else:
                remaining_pending.append(pending)

        if remaining_pending != self.pending_buy_signals:
            self._update_pending_buy_signals(remaining_pending)
    
    def _audit_position(self, now: datetime) -> Dict[str, Any]:
        """
        持仓审计 - 支持多持仓。

        卖出规则：Close_t < 配置趋势 EMA_t（默认 EMA50）。
        仍保留 peak_high_t / max_drawdown / total_return 展示字段。
        数据与扫描一致：`_get_market_data` 内已对未收盘日做切片，盘后则保留完整最新 bar。
        """
        if not self.positions:
            return {"status": "无持仓", "positions": []}
        
        audit_results = []
        
        for position in self.positions:
            ticker = position.get("ticker")
            buy_price = position.get("buy_price")
            buy_date = position.get("buy_date")
            indicator_cfg = self._indicator_for_ticker(ticker)
            trend_indicator = indicator_cfg["indicator"]
            ema_window = indicator_cfg["ema_window"]
            signal_ticker = self._signal_ticker_for(ticker)
            
            # 交易标的用于持仓收益/回撤；信号观察标的用于 EMA 卖出判断。
            data = self._get_market_data(ticker, now_et=now)
            if data is None:
                audit_results.append({
                    "status": "数据获取失败",
                    "ticker": ticker
                })
                continue
            signal_data = data if signal_ticker == str(ticker).strip().upper() else self._get_market_data(signal_ticker, now_et=now)
            if signal_data is None:
                audit_results.append({
                    "status": "信号数据获取失败",
                    "ticker": ticker,
                    "signal_ticker": signal_ticker,
                })
                continue
        
            # 切片序列：与扫描口径一致
            close = data["Close"].squeeze()
            high = data["High"].squeeze()
            close_t = float(close.iloc[-1])
            latest_date = close.index[-1].strftime("%Y-%m-%d")
            signal_close = signal_data["Close"].squeeze()
            signal_close_t = float(signal_close.iloc[-1])
            trend_ema = float(signal_close.ewm(span=ema_window, adjust=False).mean().iloc[-1])
            buy_date_obj = datetime.strptime(buy_date, "%Y-%m-%d").date()
            buy_date_idx = None
            for i in range(len(close)):
                if close.index[i].date() == buy_date_obj:
                    buy_date_idx = i
                    break
            
            prev_stored = position.get("peak_high_t")
            
            latest_high = float(high.iloc[-1])
            cost_floor = float(buy_price)
            current_peak_high = float(prev_stored) if prev_stored is not None else None
            if current_peak_high is None:
                logger.warning(f"[持仓审计] {ticker} peak_high_t 不存在，以成本价为底并从 buy_date 抬升")
                current_peak_high = cost_floor
                if buy_date_idx is not None:
                    high_since_buy = high.iloc[buy_date_idx:]
                    current_peak_high = max(cost_floor, float(high_since_buy.max()))
                else:
                    current_peak_high = max(cost_floor, float(high.max()))
            else:
                current_peak_high = max(current_peak_high, cost_floor)
            peak_high_t = max(current_peak_high, latest_high)
            
            logger.debug(
                f"[持仓审计] {ticker} peak_high_t=max(High|{buy_date}:t,切片)={peak_high_t:.2f}, close_t={close_t:.2f}"
            )
            
            if prev_stored is None or abs(peak_high_t - float(prev_stored)) > 1e-6:
                logger.debug(f"[持仓审计] {ticker} 更新 peak_high_t: {prev_stored} -> {peak_high_t:.2f}")
                position["peak_high_t"] = peak_high_t
                self._update_positions(self.positions)
            
            # 计算持股天数（基于数据窗口，包含买入当天）
            if buy_date_idx is not None:
                close_since_buy = close.iloc[buy_date_idx:].astype(float)
                high_since_buy = high.iloc[buy_date_idx:].astype(float)
                hold_days = len(close_since_buy)
            else:
                close_since_buy = None
                high_since_buy = None
                hold_days = (now.date() - buy_date_obj).days + 1  # 包含买入当天
            
            # 持仓期最大回撤：逐日从 running peak high 到当日 close 的最大跌幅。
            # 若数据窗口缺少买入日，退回到最新 close 相对已记录 peak_high_t 的当前回撤，避免伪造历史最大值。
            if peak_high_t <= 0:
                logger.error(f"[持仓审计] {ticker} peak_high_t 非正，无法计算 drawdown")
                max_drawdown = 0.0
            elif close_since_buy is not None and high_since_buy is not None and not close_since_buy.empty:
                running_peak = high_since_buy.cummax().clip(lower=cost_floor)
                drawdown_series = (running_peak - close_since_buy) / running_peak
                max_drawdown = max(0.0, float(drawdown_series.max()))
                logger.debug(
                    f"[持仓审计] {ticker} max_drawdown=max((running_peak_high-Close)/running_peak_high)={max_drawdown:.4f}"
                )
            else:
                max_drawdown = (peak_high_t - close_t) / peak_high_t
            
            total_return = (close_t / buy_price) - 1
            max_return = (peak_high_t / buy_price) - 1
            
            sell_signal = signal_close_t < trend_ema
            sell_reason = f"跌破{trend_indicator}" if sell_signal else ""
            logger.info(
                "[持仓审计][卖出判断] %s %s | 信号=%s 收盘=%s %s=%s | 交易收盘=%s 收益=%s 回撤=%s 持仓天数=%s | 原因=%s",
                ticker,
                "待卖出" if sell_signal else "继续持有",
                signal_ticker,
                _format_log_num(signal_close_t, 4),
                trend_indicator,
                _format_log_num(trend_ema, 4),
                _format_log_num(close_t, 4),
                _format_log_num(total_return * 100, 2, suffix="%"),
                _format_log_num(max_drawdown * 100, 2, suffix="%"),
                hold_days,
                sell_reason or f"未跌破{trend_indicator}",
            )
            
            if sell_signal:
                # 标记卖出 flag（次日执行）
                position["sell_flag"] = True
                # 使用数据的最后一个完整交易日作为信号日期
                signal_date = signal_data.index[-1].date().isoformat()
                position["sell_flag_date"] = signal_date
                position["sell_reason"] = sell_reason
                self._update_positions(self.positions)
                
                logger.info(f"[持仓审计] {ticker} 标记为待卖出，信号日期：{signal_date}")
                
                audit_results.append({
                    "status": "已标记卖出",
                    "action_plan": "待卖出",
                    "ticker": ticker,
                    "buy_price": buy_price,
                    "buy_date": buy_date,
                    "latest_price": close_t,
                    "latest_date": latest_date,
                    "signal_ticker": signal_ticker,
                    "signal_latest_price": signal_close_t,
                    "total_return": total_return,
                    "max_return": max_return,
                    "max_drawdown": max_drawdown,
                    "ema100": trend_ema,
                    "trend_ema": trend_ema,
                    "trend_indicator": trend_indicator,
                    "trend_ema_window": ema_window,
                    "hold_days": hold_days,
                    "sell_reason": sell_reason,
                    "execute_date": "次日开盘"
                })
            else:
                audit_results.append({
                    "status": "继续持有",
                    "action_plan": "持有",
                    "ticker": ticker,
                    "buy_price": buy_price,
                    "buy_date": buy_date,
                    "latest_price": close_t,
                    "latest_date": latest_date,
                    "signal_ticker": signal_ticker,
                    "signal_latest_price": signal_close_t,
                    "total_return": total_return,
                    "max_return": max_return,
                    "max_drawdown": max_drawdown,
                    "ema100": trend_ema,
                    "trend_ema": trend_ema,
                    "trend_indicator": trend_indicator,
                    "trend_ema_window": ema_window,
                    "hold_days": hold_days
                })
        
        return {"status": "有持仓", "positions": audit_results}
    
    def _scan_for_buy(self, now: datetime) -> Dict[str, Any]:
        """
        扫描全部配置标的：
        趋势：Close_t > 配置趋势 EMA_t（默认 EMA50）；
        突破：Close_t > HH20_{t-1} 且 Close_{t-1} <= HH20_{t-2}；
        eligible = 市值 >= MIN_MARKET_CAP 且趋势过滤通过且突破；
        候选按 RS120 从高到低排序。
        """
        scanned_stocks: List[Dict[str, Any]] = []
        selected_candidates: List[Dict[str, Any]] = []

        n_no_data = 0
        n_bad_ind = 0
        n_cap_fail = 0
        n_trend_fail = 0
        n_break_fail = 0
        n_limited_history = 0
        data_failed: List[str] = []
        held_tickers = {self._configured_ticker_for(p.get("ticker")) for p in self.positions}
        pending_tickers = {self._configured_ticker_for(p.get("ticker")) for p in self.pending_buy_signals}

        for ticker_cfg in self.ticker_configs:
            ticker = ticker_cfg["ticker"]
            signal_ticker = str(ticker_cfg.get("signal_ticker") or ticker).upper()
            trend_indicator = ticker_cfg["indicator"]
            ema_window = ticker_cfg["ema_window"]
            ticker_u = str(ticker).upper()
            is_current_position = ticker_u in held_tickers

            data = self._get_market_data(signal_ticker, now_et=now)
            trade_data = data if signal_ticker == ticker_u else self._get_market_data(ticker, now_et=now, purpose="trade")
            if data is None:
                n_no_data += 1
                data_failed.append(signal_ticker)
                scanned_stocks.append({
                    "ticker": ticker,
                    "signal_ticker": signal_ticker,
                    "latest_price": None,
                    "signal_latest_price": None,
                    "latest_date": None,
                    "close_prev": None,
                    "hh20_prev": None,
                    "hh20_prev_prev": None,
                    "rs120": None,
                    "ema100": None,
                    "trend_ema": None,
                    "ema_deviation_pct": None,
                    "trend_indicator": trend_indicator,
                    "trend_ema_window": ema_window,
                    "market_cap": None,
                    "market_cap_currency": None,
                    "market_cap_fx_rate": None,
                    "market_cap_usd": None,
                    "market_cap_source": None,
                    "market_cap_ok": False,
                    "market_cap_reason": "未检查",
                    "limited_history": False,
                    "history_bars": 0,
                    "hhv_window_used": None,
                    "rs_window_used": None,
                    "close_above_ema100": False,
                    "close_above_trend_ema": False,
                    "breakout": False,
                    "eligible": False,
                    "reason": "数据获取失败",
                    "is_position": is_current_position,
                })
                continue

            indicators = self._compute_indicators(data, ema_window=ema_window)
            if not indicators:
                n_bad_ind += 1
                scanned_stocks.append({
                    "ticker": ticker,
                    "signal_ticker": signal_ticker,
                    "latest_price": None,
                    "signal_latest_price": None,
                    "latest_date": data.index[-1].strftime("%Y-%m-%d"),
                    "close_prev": None,
                    "hh20_prev": None,
                    "hh20_prev_prev": None,
                    "rs120": None,
                    "ema100": None,
                    "trend_ema": None,
                    "ema_deviation_pct": None,
                    "trend_indicator": trend_indicator,
                    "trend_ema_window": ema_window,
                    "market_cap": None,
                    "market_cap_currency": None,
                    "market_cap_fx_rate": None,
                    "market_cap_usd": None,
                    "market_cap_source": None,
                    "market_cap_ok": False,
                    "market_cap_reason": "未检查",
                    "limited_history": False,
                    "history_bars": len(data) if data is not None else 0,
                    "hhv_window_used": None,
                    "rs_window_used": None,
                    "close_above_ema100": False,
                    "close_above_trend_ema": False,
                    "breakout": False,
                    "eligible": False,
                    "reason": "指标数据不足",
                    "is_position": is_current_position,
                })
                continue

            c = indicators["close"]
            c1 = indicators["close_prev"]
            h1 = indicators["hhv20_prev"]
            h2 = indicators["hhv20_prev_prev"]
            trend_ema = indicators["trend_ema"]
            rs120 = indicators["rs120"]
            trend_indicator = indicators.get("trend_indicator", trend_indicator)
            limited_history = bool(indicators.get("limited_history"))
            if limited_history:
                n_limited_history += 1
            latest_date = data.index[-1].strftime("%Y-%m-%d")

            close_above_trend_ema = c > trend_ema
            breakout = (c > h1) and (c1 <= h2)
            trend_age_info = {}
            latest_price = c
            if trade_data is not None:
                try:
                    latest_price = float(trade_data["Close"].squeeze().iloc[-1])
                except Exception:
                    latest_price = c
            cap_filter = {
                "market_cap": None,
                "market_cap_currency": None,
                "market_cap_fx_rate": None,
                "market_cap_usd": None,
                "market_cap_source": None,
                "market_cap_ok": False,
                "market_cap_reason": "未检查",
            }

            if not close_above_trend_ema:
                n_trend_fail += 1
                reason = f"未站上{trend_indicator}"
            elif not breakout:
                n_break_fail += 1
                reason = "未满足20日突破"
            else:
                cap_filter = self._check_market_cap_filter(ticker_u)
                if not cap_filter["market_cap_ok"]:
                    n_cap_fail += 1
                    reason = "市值过滤未通过"
                else:
                    trend_age_info = self._classify_trend_age(data, ema_window=ema_window)
                    reason = "符合买入条件"

            eligible = bool(close_above_trend_ema and breakout and cap_filter["market_cap_ok"])
            if limited_history:
                reason = f"{reason}（短历史估算）"

            stock_data = {
                "ticker": ticker,
                "signal_ticker": signal_ticker,
                "latest_price": latest_price,
                "signal_latest_price": c,
                "latest_date": latest_date,
                "date": data.index[-1].date().isoformat(),
                "close_prev": c1,
                "hh20_prev": h1,
                "hh20_prev_prev": h2,
                "ema100": trend_ema,
                "trend_ema": trend_ema,
                "ema_deviation_pct": indicators.get("ema_deviation_pct"),
                "trend_indicator": trend_indicator,
                "trend_ema_window": indicators.get("trend_ema_window", ema_window),
                "rs120": rs120,
                "market_cap": cap_filter["market_cap"],
                "market_cap_currency": cap_filter["market_cap_currency"],
                "market_cap_fx_rate": cap_filter["market_cap_fx_rate"],
                "market_cap_usd": cap_filter["market_cap_usd"],
                "market_cap_source": cap_filter["market_cap_source"],
                "market_cap_ok": cap_filter["market_cap_ok"],
                "market_cap_reason": cap_filter.get("market_cap_reason"),
                "limited_history": limited_history,
                "history_bars": indicators.get("history_bars"),
                "hhv_window_used": indicators.get("hhv_window_used"),
                "rs_window_used": indicators.get("rs_window_used"),
                "close_above_ema100": close_above_trend_ema,
                "close_above_trend_ema": close_above_trend_ema,
                "breakout": breakout,
                "eligible": eligible,
                "reason": reason,
                "trend_age": trend_age_info.get("trend_age"),
                "trend_age_stage": trend_age_info.get("trend_age_stage"),
                "position_size_hint": trend_age_info.get("position_size_hint"),
                "trend_age_estimated": trend_age_info.get("trend_age_estimated"),
                "trend_age_basis": trend_age_info.get("trend_age_basis"),
                "last_sell_signal_date": trend_age_info.get("last_sell_signal_date"),
                "first_buy_after_sell_date": trend_age_info.get("first_buy_after_sell_date"),
                "latest_buy_signal_date": trend_age_info.get("latest_buy_signal_date"),
                "trend_age_first_threshold": trend_age_info.get("trend_age_first_threshold"),
                "trend_age_mid_threshold": trend_age_info.get("trend_age_mid_threshold"),
                "is_position": is_current_position,
            }
            scanned_stocks.append(stock_data)

            if eligible and rs120 is not None and ticker_u not in held_tickers and ticker_u not in pending_tickers:
                selected_candidates.append(stock_data)

        def _sort_key(s: Dict[str, Any]):
            rs = s.get("rs120")
            return (rs is None, -(rs or 0))

        scanned_stocks.sort(key=_sort_key)
        selected_candidates.sort(key=_sort_key)
        self._log_scan_stock_details(scanned_stocks, selected_candidates)

        n_list = len(self.tickers)
        n_cand = sum(1 for s in scanned_stocks if s.get("eligible"))
        top_bits = ""
        ranked = [s for s in scanned_stocks if s.get("rs120") is not None]
        if ranked:
            top = ranked[0]
            top5 = [f"{s['ticker']}:{format_rs_pct(s['rs120'])}" for s in ranked[:5]]
            top_bits = f" RS120首位={top['ticker']}({format_rs_pct(top['rs120'])}) Top5={','.join(top5)}"
        logger.info(
            f"[买入扫描] 汇总 名单={n_list} 美元市值阈值={self.min_market_cap:.0f} "
            f"无数据={n_no_data} 指标不足={n_bad_ind} 未站上趋势EMA={n_trend_fail} "
            f"市值未通过={n_cap_fail} 无突破={n_break_fail} "
            f"短历史估算={n_limited_history} 可买候选={n_cand}{top_bits}"
        )
        if data_failed:
            tail = data_failed[:30]
            more = " …" if len(data_failed) > 30 else ""
            logger.warning(
                f"[买入扫描] 数据获取失败 {len(data_failed)} 只: {','.join(tail)}{more}"
            )

        if selected_candidates:
            return {
                "status": "有买入信号",
                "candidates": selected_candidates,
                "data": selected_candidates[0],
                "scanned_stocks": scanned_stocks,
            }

        return {
            "status": "无买入信号（无符合条件标的）",
            "scanned_stocks": scanned_stocks,
        }
    
    def run(self, data_cache: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        运行动量系统 V3.0
        
        执行流程：
        1. 执行引擎：处理待执行的买入/卖出订单
        2. 持仓审计：检查当前持仓状态
        3. 买入扫描：寻找 Close > 配置趋势 EMA、HH20 突破且市值达标的 RS120 强势候选
        4. 状态更新：按可用仓位写入待买入信号；满仓不换仓
        """
        now = datetime.now(ET_TIMEZONE)
        
        # 更新数据缓存
        if data_cache:
            self.data_cache = data_cache
        
        # Step 0: 执行引擎 - 处理待执行订单
        logger.debug("========== [执行引擎] 开始 ==========")
        self._execute_pending_orders(now)
        logger.debug("========== [执行引擎] 结束 ==========")
        
        # Step 1: 持仓审计
        position_audit = self._audit_position(now)
        
        # Step 2: 买入扫描 - 总是执行，无论是否有持仓
        buy_signal = self._scan_for_buy(now)
        
        # Step 3: 状态更新。已满仓且无待卖出时不换仓；有待卖出时可预留卖出腾出的仓位。
        active_positions = [p for p in self.positions if not p.get("sell_flag")]
        available_slots = max(0, self.max_positions - len(active_positions) - len(self.pending_buy_signals))
        if available_slots > 0 and buy_signal.get("status") == "有买入信号":
            new_pending = list(self.pending_buy_signals)
            selected = buy_signal.get("candidates", [])[:available_slots]

            for stock in selected:
                signal_date = stock.get("date", now.date().isoformat())
                pending = {
                    "ticker": stock.get("ticker"),
                    "signal_date": signal_date,
                    "reason": "动量突破",
                    "rs120": stock.get("rs120"),
                    "latest_price": stock.get("latest_price"),
                }
                new_pending.append(pending)
                logger.info(
                    f"[状态更新] 已标记待买入：{stock.get('ticker')}, "
                    f"信号日期：{signal_date}, RS120={format_rs_pct(stock.get('rs120'))}"
                )
            buy_signal["selected_candidates"] = selected
            self._update_pending_buy_signals(new_pending)
        else:
            if len(active_positions) >= self.max_positions:
                buy_signal["status"] = "无买入信号（持仓已满，等待卖出腾出位置）"
            elif self.pending_buy_signals and available_slots == 0:
                buy_signal["status"] = "无买入信号（待买入已占满可用仓位）"
        
        result = {
            "position_audit": position_audit,
            "buy_signal": buy_signal,
            "pending_buy_signal": self.pending_buy_signal,
            "pending_buy_signals": self.pending_buy_signals,
            "timestamp": now.isoformat(),
        }
        
        # 保存结果
        self._save_momentum_result(result)
        
        return result
