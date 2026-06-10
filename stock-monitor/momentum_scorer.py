#!/usr/bin/env python3
"""
动量系统 V3.0：个股突破 + RS120 排名策略
- 美元市值 >= MIN_MARKET_CAP
- 20 日收盘突破入场，候选按 RS120 排名
- 最多 3 只等权持仓，持仓期间不因 RS 变化换仓
- 卖出：Close_t < EMA50_t
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

from strategy import _apply_time_slice, fetch_with_retry, _get_market_status
from state_manager import get_data_path, EXECUTION_LOG_MAX, trade_execution_timestamp

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ET_TIMEZONE = pytz.timezone("America/New_York")
DEFAULT_MIN_MARKET_CAP = 1_000_000_000
DEFAULT_MAX_POSITIONS = 3
RS_WINDOW = 120
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
        self.tickers = config.get("tickers", [])
        
        # 统一状态存储
        self.state = self._load_state()
        # 支持多持仓模式
        self.positions = self.state.get("current_positions", [])
        # 兼容旧的单持仓模式
        if not self.positions and "current_position" in self.state and self.state["current_position"]:
            self.positions = [self.state["current_position"]]
        self.history = self.state.get("history", [])
        self.pending_buy_signals = self._normalize_pending_buy_signals()
        self.pending_buy_signal = self.pending_buy_signals[0] if self.pending_buy_signals else {}
        
        # 执行日志
        self.execution_logs = self.state.get("execution_logs", [])
    
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
            return [p for p in raw_list if isinstance(p, dict) and p.get("ticker")]
        legacy = self.state.get("pending_buy_signal")
        if isinstance(legacy, dict) and legacy.get("ticker"):
            return [legacy]
        return []
    
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
        
        return None
    
    def _compute_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算指标 - 严格对齐索引
        
        公式：
        - HHV20_{t-1}: 使用 hhv20.iloc[-2] (截止到昨天的 20 日最高价)
        - HHV20_{t-2}: 使用 hhv20.iloc[-3] (截止到前天的 20 日最高价)
        - CLOSE_t: 使用 close.iloc[-1] (今日收盘价)
        - CLOSE_{t-1}: 使用 close.iloc[-2] (昨日收盘价)
        - EMA50_t: 使用 close.ewm(span=50).mean().iloc[-1]
        - RS120: Close_t / Close_{t-120} - 1
        """
        if data is None or data.empty:
            return {}
        
        # 确保 close 是一个 Series
        close = data["Close"].squeeze()
        
        min_len = max(self.hhv_window + 2, RS_WINDOW + 1, 50)
        if len(close) < min_len:
            logger.debug(f"[指标计算] 数据长度不足 {min_len}，无法计算")
            return {}
        
        hhv20 = close.rolling(window=self.hhv_window).max()
        ema50 = close.ewm(span=50, adjust=False).mean()
        close_120 = float(close.iloc[-(RS_WINDOW + 1)])
        
        indicators = {
            "close": float(close.iloc[-1]),
            "close_prev": float(close.iloc[-2]),
            "hhv20_prev": float(hhv20.iloc[-2]),
            "hhv20_prev_prev": float(hhv20.iloc[-3]),
            "ema50": float(ema50.iloc[-1]),
            "rs120": (float(close.iloc[-1]) / close_120 - 1) if close_120 > 0 else None,
        }
        
        logger.debug(
            f"[指标计算] close_t={indicators['close']:.4f}, close_t1={indicators['close_prev']:.4f}, "
            f"hhv20_t1={indicators['hhv20_prev']:.4f}, hhv20_t2={indicators['hhv20_prev_prev']:.4f}, "
            f"ema50={indicators['ema50']:.4f}, rs120={format_rs_pct(indicators['rs120'])}"
        )
        
        return indicators
    
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
        cap = self._get_market_cap(ticker)
        market_cap_usd = cap.get("market_cap_usd")
        ok = market_cap_usd is not None and market_cap_usd >= self.min_market_cap
        reason = "市值通过" if ok else "市值低于阈值或获取失败"
        return {
            **cap,
            "market_cap_ok": ok,
            "market_cap_reason": reason,
        }

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

        卖出规则：Close_t < EMA50_t。
        仍保留 peak_high_t / drawdown / total_return 展示字段。
        数据与扫描一致：`_get_market_data` 内已对未收盘日做切片，盘后则保留完整最新 bar。
        """
        if not self.positions:
            return {"status": "无持仓", "positions": []}
        
        audit_results = []
        
        for position in self.positions:
            ticker = position.get("ticker")
            buy_price = position.get("buy_price")
            buy_date = position.get("buy_date")
            
            # 获取最新数据
            data = self._get_market_data(ticker, now_et=now)
            if data is None:
                audit_results.append({
                    "status": "数据获取失败",
                    "ticker": ticker
                })
                continue
        
            # 切片序列：与扫描口径一致
            close = data["Close"].squeeze()
            high = data["High"].squeeze()
            close_t = float(close.iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            latest_date = close.index[-1].strftime("%Y-%m-%d")
            
            prev_stored = position.get("peak_high_t")
            
            latest_high = float(high.iloc[-1])
            cost_floor = float(buy_price)
            current_peak_high = float(prev_stored) if prev_stored is not None else None
            if current_peak_high is None:
                logger.warning(f"[持仓审计] {ticker} peak_high_t 不存在，以成本价为底并从 buy_date 抬升")
                current_peak_high = cost_floor
                buy_date_obj = datetime.strptime(buy_date, "%Y-%m-%d").date()
                buy_date_idx = None
                for i in range(len(high)):
                    if high.index[i].date() == buy_date_obj:
                        buy_date_idx = i
                        break
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
            buy_date_obj = datetime.strptime(buy_date, "%Y-%m-%d").date()
            buy_date_idx = None
            for i in range(len(close)):
                if close.index[i].date() == buy_date_obj:
                    buy_date_idx = i
                    break
            
            if buy_date_idx is not None:
                hold_days = len(close.iloc[buy_date_idx:])
            else:
                hold_days = (now.date() - buy_date_obj).days + 1  # 包含买入当天
            
            # high-to-close 最大回撤：drawdown_t = (peak_high_t - close_t) / peak_high_t
            # peak_high_t = max(High_entry .. High_t)；分母为持仓期历史最高 High，分子为 peak - 当日收盘
            if peak_high_t <= 0:
                logger.error(f"[持仓审计] {ticker} peak_high_t 非正，无法计算 drawdown")
                max_drawdown = 0.0
            else:
                max_drawdown = (peak_high_t - close_t) / peak_high_t
            
            total_return = (close_t / buy_price) - 1
            max_return = (peak_high_t / buy_price) - 1
            
            sell_signal = close_t < ema50
            sell_reason = "跌破EMA50" if sell_signal else ""
            
            if sell_signal:
                # 标记卖出 flag（次日执行）
                position["sell_flag"] = True
                # 使用数据的最后一个完整交易日作为信号日期
                signal_date = data.index[-1].date().isoformat()
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
                    "total_return": total_return,
                    "max_return": max_return,
                    "max_drawdown": max_drawdown,
                    "ema50": ema50,
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
                    "total_return": total_return,
                    "max_return": max_return,
                    "max_drawdown": max_drawdown,
                    "ema50": ema50,
                    "hold_days": hold_days
                })
        
        return {"status": "有持仓", "positions": audit_results}
    
    def _scan_for_buy(self, now: datetime) -> Dict[str, Any]:
        """
        扫描全部配置标的：
        突破：Close_t > HH20_{t-1} 且 Close_{t-1} <= HH20_{t-2}；
        eligible = 市值 >= MIN_MARKET_CAP 且突破；
        候选按 RS120 从高到低排序。
        """
        scanned_stocks: List[Dict[str, Any]] = []
        selected_candidates: List[Dict[str, Any]] = []

        n_no_data = 0
        n_bad_ind = 0
        n_cap_fail = 0
        n_break_fail = 0
        data_failed: List[str] = []
        held_tickers = {str(p.get("ticker", "")).upper() for p in self.positions}
        pending_tickers = {str(p.get("ticker", "")).upper() for p in self.pending_buy_signals}

        for ticker in self.tickers:
            ticker_u = str(ticker).upper()
            is_current_position = ticker_u in held_tickers

            data = self._get_market_data(ticker, now_et=now)
            if data is None:
                n_no_data += 1
                data_failed.append(ticker)
                scanned_stocks.append({
                    "ticker": ticker,
                    "latest_price": None,
                    "latest_date": None,
                    "hh20_prev": None,
                    "rs120": None,
                    "ema50": None,
                    "market_cap": None,
                    "market_cap_currency": None,
                    "market_cap_fx_rate": None,
                    "market_cap_usd": None,
                    "market_cap_source": None,
                    "market_cap_ok": False,
                    "breakout": False,
                    "eligible": False,
                    "reason": "数据获取失败",
                    "is_position": is_current_position,
                })
                continue

            indicators = self._compute_indicators(data)
            if not indicators:
                n_bad_ind += 1
                scanned_stocks.append({
                    "ticker": ticker,
                    "latest_price": None,
                    "latest_date": data.index[-1].strftime("%Y-%m-%d"),
                    "hh20_prev": None,
                    "rs120": None,
                    "ema50": None,
                    "market_cap": None,
                    "market_cap_currency": None,
                    "market_cap_fx_rate": None,
                    "market_cap_usd": None,
                    "market_cap_source": None,
                    "market_cap_ok": False,
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
            ema50 = indicators["ema50"]
            rs120 = indicators["rs120"]
            latest_date = data.index[-1].strftime("%Y-%m-%d")
            cap_filter = self._check_market_cap_filter(ticker_u)

            breakout = (c > h1) and (c1 <= h2)
            eligible = bool(cap_filter["market_cap_ok"] and breakout)

            if not breakout:
                n_break_fail += 1
                reason = "未满足20日突破"
            elif not cap_filter["market_cap_ok"]:
                n_cap_fail += 1
                reason = "市值过滤未通过"
            else:
                reason = "符合买入条件"

            stock_data = {
                "ticker": ticker,
                "latest_price": c,
                "latest_date": latest_date,
                "date": data.index[-1].date().isoformat(),
                "hh20_prev": h1,
                "ema50": ema50,
                "rs120": rs120,
                "market_cap": cap_filter["market_cap"],
                "market_cap_currency": cap_filter["market_cap_currency"],
                "market_cap_fx_rate": cap_filter["market_cap_fx_rate"],
                "market_cap_usd": cap_filter["market_cap_usd"],
                "market_cap_source": cap_filter["market_cap_source"],
                "market_cap_ok": cap_filter["market_cap_ok"],
                "breakout": breakout,
                "eligible": eligible,
                "reason": reason,
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
            f"无数据={n_no_data} 指标不足={n_bad_ind} 市值未通过={n_cap_fail} 无突破={n_break_fail} "
            f"可买候选={n_cand}{top_bits}"
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
        3. 买入扫描：寻找 HH20 突破且市值达标的 RS120 强势候选
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
