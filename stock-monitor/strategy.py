from private_rules import private_rule, private_parameter, positive_window
import pandas as pd
import numpy as np
import yfinance as yf
import time
import logging
import json
import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import pytz
import math

from state_manager import (
    get_data_path,
    load_momentum_state,
    momentum_ticker_symbol,
    get_momentum_ticker_configs,
)
from position_state import (
    load_position_state,
    save_position_state_atomic,
    get_strategy_state,
    validate_in_position_state_fields,
    resolve_peak_high_for_hold,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ET_TIMEZONE = pytz.timezone("America/New_York")
CN_TIMEZONE = pytz.timezone("Asia/Shanghai")

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16

# 缓存刷新：距上次写入超过此时长则重新拉取（open/closed 均适用）
CACHE_RELOAD_MIN_INTERVAL_SEC = 15 * 60

MIN_CACHE_ROWS_FOR_INCREMENTAL_DOWNLOAD = None


def _is_finite_num(x: Any) -> bool:
    try:
        return bool(math.isfinite(float(x)))
    except Exception:
        return False


def _merge_ohlcv_cache_row(
    existing_vals: Optional[List[Any]],
    new_vals: List[Any],
) -> Optional[List[float]]:
    """
    合并单日 OHLCV：新值仅在 finite 时覆盖，否则保留旧值。
    - 无旧数据且新数据四条价线全无效：返回 None（不写该日期）。
    - Close 仍缺失但 O/H/L 均有效：用 (O+H+L)/3 补 Close（yfinance 常见残缺行）。
    """
    old = list(existing_vals[:5]) if existing_vals and len(existing_vals) >= 5 else None
    new = list(new_vals[:5]) if new_vals else []
    while len(new) < 5:
        new.append(float("nan"))

    out: List[float] = []
    if old is not None:
        for i in range(5):
            nv, ov = new[i], old[i]
            if _is_finite_num(nv):
                out.append(float(nv))
            elif _is_finite_num(ov):
                out.append(float(ov))
            else:
                out.append(float("nan"))
        if not any(_is_finite_num(out[i]) for i in range(4)):
            return [float(x) for x in old[:5]]
    else:
        for i in range(5):
            out.append(float(new[i]) if _is_finite_num(new[i]) else float("nan"))
        if not any(_is_finite_num(out[i]) for i in range(4)):
            return None

    if not _is_finite_num(out[3]) and all(_is_finite_num(out[i]) for i in range(3)):
        out[3] = float((out[0] + out[1] + out[2]) / 3.0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具函数 oo
# ─────────────────────────────────────────────────────────────────────────────

def _slice_batch_download_for_ticker(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """
    从 yf.download 批量结果中取单个标的。yfinance 列顺序随 group_by 变化：
    group_by='column'（默认）为 (Field, Ticker)；group_by='ticker' 为 (Ticker, Field)。
    """
    if data is None or data.empty:
        return None
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()
    for level in (1, 0):
        level_vals = data.columns.get_level_values(level)
        if ticker not in set(level_vals):
            continue
        sliced = data.xs(ticker, axis=1, level=level, drop_level=False)
        if sliced.empty:
            continue
        out = sliced.copy()
        out.columns = out.columns.droplevel(level)
        return out
    return None


def _df_from_cache_entry(cached_data: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """将 data_cache.json 中单个标的的序列化条目还原为 DataFrame。"""
    if not cached_data:
        return None
    rows = cached_data.get("data")
    if not rows:
        return None
    try:
        index = pd.to_datetime([item[0] for item in rows])
        values = [item[1:] for item in rows]
        columns = cached_data.get("columns", ["Open", "High", "Low", "Close", "Volume"])
        return pd.DataFrame(values, index=index, columns=columns)
    except Exception as e:
        logger.error(f"[缓存] 将缓存条目还原为 DataFrame 失败：{e}")
        return None


def batch_get_data(tickers: List[str], period: str = "3y", retries: int = 3, delay: float = 2.0, now_et: Optional[datetime] = None, force_reload: Optional[List[str]] = None) -> dict:
    """
    批量抓取多个标的的数据（支持数据缓存）

    Args:
        tickers: 标的列表
        period: 数据周期
        retries: 重试次数
        delay: 重试延迟
        now_et: 当前时间（ET时区）
        force_reload: 强制重新下载的标的列表

    Returns:
        字典 {ticker: dataframe}
    """
    if now_et is None:
        now_et = datetime.now(ET_TIMEZONE)

    result = {}

    # 去重，确保每个标的只下载一次
    unique_tickers = list(set(tickers))

    # 加载本地数据缓存
    cache = _load_data_cache()

    # 需要重新加载的标的列表
    need_reload = []

    # 强制重新下载的标的直接加入 need_reload
    if force_reload:
        for ticker in force_reload:
            if ticker not in need_reload:
                need_reload.append(ticker)

    # 检查每个标的是否需要重新加载
    loaded_from_cache = 0
    for ticker in unique_tickers:
        if ticker in need_reload:
            continue
        if _should_reload_data(cache, now_et, ticker):
            need_reload.append(ticker)
        else:
            # 从缓存中获取数据
            cached_data = cache.get("data", {}).get(ticker)
            if cached_data:
                df = _df_from_cache_entry(cached_data)
                if df is not None and not df.empty:
                    result[ticker] = df
                    loaded_from_cache += 1
                    logger.debug(f"[缓存] 从本地缓存加载 {ticker}，最新两行: {df.tail(2)}")
                else:
                    need_reload.append(ticker)
    if loaded_from_cache:
        logger.info(f"[缓存] 从本地复用 {loaded_from_cache}/{len(unique_tickers)} 个标的（其余需下载或缺缓存）")

    # 下载需要重新加载的标的
    if need_reload:
        logger.info(f"开始批量抓取 {len(need_reload)} 个标的的数据")

        # 将标的分组：按市场和下载周期分组
        # A股/SZ.SH 结尾的为 CN 市场，其他为 US 市场
        groups = {
            ("CN", "3y"): [],  # A股 强制重新下载
            ("CN", "10d"): [], # A股 增量更新
            ("US", "3y"): [],  # 美股 强制重新下载
            ("US", "10d"): [], # 美股 增量更新
        }

        used_full_period_tickers: List[str] = []
        for ticker in need_reload:
            market = "CN" if any(suffix in ticker for suffix in [".SZ", ".SH"]) else "US"
            cached_rows = 0
            t_entry = (cache.get("data") or {}).get(ticker)
            if isinstance(t_entry, dict):
                cached_rows = len(t_entry.get("data") or [])
            if force_reload and ticker in force_reload:
                download_period = period
            elif cached_rows < positive_window("cache_min_rows"):
                download_period = period
                used_full_period_tickers.append(ticker)
            else:
                download_period = "10d"
            groups[(market, download_period)].append(ticker)
        if used_full_period_tickers:
            tail = used_full_period_tickers[:25]
            more = f" 等共{len(used_full_period_tickers)}只" if len(used_full_period_tickers) > len(tail) else ""
            logger.info(
                f"[缓存] {len(used_full_period_tickers)} 只标的本地<{MIN_CACHE_ROWS_FOR_INCREMENTAL_DOWNLOAD}条，"
                f"改用全量 period={period}：{', '.join(tail)}{more}"
            )

        # 批量下载每组标的
        for (market, download_period), tickers in groups.items():
            if not tickers:
                continue

            logger.info(f"[批量下载] 市场={market}, 周期={download_period}, 标的数量={len(tickers)}")

            try:
                # 批量下载：yfinance 支持一次下载多个标的
                data = yf.download(
                    tickers,
                    period=download_period,
                    progress=False,
                    auto_adjust=True,
                )

                # 解析批量结果
                for ticker in tickers:
                    try:
                        if data is None or data.empty:
                            logger.warning(f"[{ticker}] 批量下载结果为空")
                            continue

                        ticker_data = _slice_batch_download_for_ticker(data, ticker)
                        if ticker_data is None or ticker_data.empty:
                            logger.warning(f"[{ticker}] 批量下载结果中无数据或无法解析列层级")
                            continue

                        result[ticker] = ticker_data
                        logger.debug(f"[{ticker}] 批量下载成功，条数={len(ticker_data)}")

                    except Exception as e:
                        logger.error(f"[{ticker}] 解析批量数据失败: {e}")

                # 批量下载后适当休息，避免被限流
                time.sleep(1)

            except Exception as e:
                logger.error(f"批量下载失败: {e}")
                # 批量失败时回退到逐个下载
                logger.info("回退到逐个下载模式...")
                for ticker in tickers:
                    try:
                        data = fetch_with_retry(ticker, download_period, retries, delay, now_et, market)
                        if data is not None:
                            result[ticker] = data
                    except Exception as ex:
                        logger.error(f"[{ticker}] 下载失败: {ex}")
                    time.sleep(0.5)
    else:
        logger.info("所有标的数据均从缓存加载，无需重新下载")

    # 更新并保存缓存
    if need_reload:

        cache_data: Dict[str, Any] = {}
        merge_row_counts: List[int] = []

        # 获取已有的缓存数据
        existing_cache = cache.get("data", {})

        for ticker, data in result.items():
            if data is not None and not data.empty:
                # 将 DataFrame 转换为可序列化的格式
                try:
                    # 标准化列名
                    standardized_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

                    # 确保数据列顺序正确
                    if isinstance(data.columns, pd.MultiIndex):
                        # 对于多级索引，提取正确的列
                        try:
                            # 重新组织数据，确保列顺序正确
                            organized_data = {
                                'Open': data.xs('Open', level=0, axis=1).iloc[:, 0],
                                'High': data.xs('High', level=0, axis=1).iloc[:, 0],
                                'Low': data.xs('Low', level=0, axis=1).iloc[:, 0],
                                'Close': data.xs('Close', level=0, axis=1).iloc[:, 0],
                                'Volume': data.xs('Volume', level=0, axis=1).iloc[:, 0]
                            }
                            data = pd.DataFrame(organized_data, index=data.index)
                        except Exception as e:
                            logger.error(f"[缓存] 处理多级索引失败：{e}")
                            continue


                    new_data_dict = {}
                    for idx, row in data.iterrows():
                        date_str = idx.strftime("%Y-%m-%d")
                        row_data = []
                        for col in standardized_columns:
                            if col in data.columns:
                                row_data.append(float(row[col]))
                            else:
                                row_data.append(0.0)
                        new_data_dict[date_str] = row_data


                    existing_data = existing_cache.get(ticker, {})
                    existing_df_list = existing_data.get("data", [])

                    # 将现有数据转换为字典
                    existing_data_dict = {}
                    for row in existing_df_list:
                        if len(row) > 0:
                            date_str = row[0]
                            values = row[1:]
                            existing_data_dict[date_str] = values

                    # 合并：新数据仅在 finite 时覆盖；NaN/无效不覆盖已保存的好数据
                    merged_data_dict: Dict[str, List[float]] = dict(existing_data_dict)
                    for date_str, new_row in new_data_dict.items():
                        old_row = merged_data_dict.get(date_str)
                        merged_row = _merge_ohlcv_cache_row(old_row, new_row)
                        if merged_row is None:
                            continue
                        merged_data_dict[date_str] = merged_row


                    df_list = []
                    for date_str in sorted(merged_data_dict.keys()):
                        df_list.append([date_str] + merged_data_dict[date_str])

                    logger.debug(
                        f"[缓存] {ticker} 合并：原{len(existing_data_dict)} + 新{len(new_data_dict)} → {len(df_list)}"
                    )
                    merge_row_counts.append(len(df_list))

                    cache_data[ticker] = {
                        "columns": standardized_columns,
                        "data": df_list
                    }
                except Exception as e:
                    logger.error(f"[缓存] 序列化 {ticker} 数据失败：{e}")

        # 更新缓存
        cache["last_updated"] = now_et.isoformat()

        # 为每个标的单独存储市场状态
        for ticker in cache_data:
            market = _get_ticker_market(ticker)
            cache_data[ticker]["market"] = market
            cache_data[ticker]["market_status"] = _get_market_status(now_et, market)

        # 保留全局市场状态（美股）以保持兼容性
        cache["market_status"] = _get_market_status(now_et, "US")

        # 保留原有缓存中的其他标的数据，只更新本次下载的标的
        existing_data = cache.get("data", {})
        existing_data.update(cache_data)
        cache["data"] = existing_data

        # 保存缓存
        _save_data_cache(cache)
        if merge_row_counts:
            mn, mx = min(merge_row_counts), max(merge_row_counts)
            avg = sum(merge_row_counts) / len(merge_row_counts)
            logger.info(
                f"[缓存] 已写入磁盘 {len(cache_data)} 个标的，合并后行数 min={mn} max={mx} avg={avg:.0f}"
            )
        else:
            logger.info("数据缓存已更新")

        # 增量下载写入 result 的仅为短周期窗口；合并后的全长在 cache 中。回灌内存供扫描使用（否则 market_ma 等会误判行数不足）。
        mem_ok = 0
        for ticker in need_reload:
            merged_df = _df_from_cache_entry(cache.get("data", {}).get(ticker))
            if merged_df is not None and not merged_df.empty:
                result[ticker] = merged_df
                mem_ok += 1
                logger.debug(f"[缓存] {ticker} 回灌内存 {len(merged_df)} 条")
        if need_reload:
            miss = [t for t in need_reload if t not in result or result.get(t) is None or result.get(t).empty]
            if miss:
                logger.warning(f"[缓存] 回灌内存失败 {len(miss)} 只: {', '.join(miss[:40])}{'…' if len(miss) > 40 else ''}")
            else:
                logger.info(f"[缓存] 回灌内存成功 {mem_ok}/{len(need_reload)} 只（与合并结果一致）")

    logger.info(f"批量抓取完成，成功获取 {len(result)} 个标的的数据")
    return result

def _load_data_cache() -> Dict[str, Any]:
    """
    加载本地数据缓存
    """
    cache_path = get_data_path("data_cache.json")
    if not os.path.exists(cache_path):
        return {"last_updated": None, "market_status": None, "data": {}}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载数据缓存失败：{e}")
        return {"last_updated": None, "market_status": None, "data": {}}

def _save_data_cache(cache: Dict[str, Any]):
    """
    保存本地数据缓存
    """
    cache_path = get_data_path("data_cache.json")
    temp_path = cache_path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, cache_path)
    except Exception as e:
        logger.error(f"保存数据缓存失败：{e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def _get_market_status(now_et: datetime, market: str = "US") -> str:
    """
    获取市场状态：open 或 closed

    Args:
        now_et: 当前时间（ET时区）
        market: 市场类型，"US" 或 "CN"
    """
    if market == "US":
        # 美股交易时间：9:30 - 16:00 ET
        is_trading_hour = (MARKET_OPEN_HOUR <= now_et.hour < MARKET_CLOSE_HOUR) or \
                         (now_et.hour == MARKET_OPEN_HOUR and now_et.minute >= MARKET_OPEN_MIN)
        is_weekend = now_et.weekday() >= 5
        if is_weekend or not is_trading_hour:
            return "closed"
        return "open"
    else:  # CN
        # A股交易时间：9:30 - 15:00 北京时间（中午不休息）
        now_bj = now_et.astimezone(pytz.timezone('Asia/Shanghai'))
        is_trading_hour = ((now_bj.hour == 9 and now_bj.minute >= 30) or
                          (10 <= now_bj.hour < 15) or
                          (now_bj.hour == 15 and now_bj.minute == 0))
        if not is_trading_hour:
            return "closed"
        return "open"

def _get_ticker_market(ticker: str) -> str:
    """
    根据标的代码判断市场类型
    """
    return "CN" if any(suffix in ticker for suffix in [".SZ", ".SH"]) else "US"

def _should_reload_data(cache: Dict[str, Any], now_et: datetime, ticker: str) -> bool:
    """
    是否需向数据源重新拉取该标的并写回 data_cache。

    1. 缺该标的缓存 / 缺全局 last_updated / 解析不了 last_updated → 拉
    2. last_updated 与当前扫描不在同一美东自然日 → 拉
    3. 上次写入的 market_status 与当前 open/closed 不一致 → 拉
    4. 距上次写入超过 CACHE_RELOAD_MIN_INTERVAL_SEC（open、closed 均适用）→ 拉
    5. 否则不拉
    """
    if ticker not in cache.get("data", {}):
        return True

    last_updated_str = cache.get("last_updated")
    if not last_updated_str:
        return True

    try:
        last_updated = datetime.fromisoformat(last_updated_str).astimezone(ET_TIMEZONE)
    except Exception:
        return True

    if last_updated.date() != now_et.date():
        logger.debug(f"[缓存检查] {ticker} 跨天({last_updated.date()} -> {now_et.date()})，需要重新加载数据")
        return True

    market = _get_ticker_market(ticker)
    ticker_cache = cache.get("data", {}).get(ticker, {})

    last_market_status = ticker_cache.get("market_status")
    if last_market_status is None:
        last_market_status = cache.get("market_status")

    current_market_status = _get_market_status(now_et, market)
    if last_market_status != current_market_status:
        logger.debug(f"[缓存检查] {ticker} 市场状态变化 ({last_market_status} -> {current_market_status})，需要重新加载数据")
        return True

    age_sec = (now_et - last_updated).total_seconds()
    if age_sec > CACHE_RELOAD_MIN_INTERVAL_SEC:
        logger.debug(
            f"[缓存检查] {ticker} 距上次写入 {age_sec:.0f}s > {CACHE_RELOAD_MIN_INTERVAL_SEC}s，需要重新加载数据"
        )
        return True

    logger.debug(f"[缓存检查] {ticker} 数据可复用")
    return False

def preprocess_data(data_cache: dict, now_et: Optional[datetime] = None) -> dict:
    """
    预处理数据，统一计算各种指标

    Args:
        data_cache: 数据缓存字典
        now_et: 当前时间（ET时区）

    Returns:
        处理后的数据缓存字典
    """
    logger.info("开始预处理数据，计算指标")
    preprocess_ok = 0
    for ticker, data in data_cache.items():
        if data is None or data.empty:
            continue

        try:
            # 计算LEGACY_EMA
            data['legacy_ema'] = data['Close'].ewm(span=positive_window("legacy_ema_window"), adjust=False).mean()


            data['fast_ema'] = data['Close'].ewm(span=positive_window("fast_ema_window"), adjust=False).mean()

            # 计算BREAKOUT_WINDOW
            data['breakout_window'] = data['Close'].rolling(window=positive_window("breakout_window")).max()


            data['market_ma'] = data['Close'].rolling(window=positive_window("market_ma_window")).mean()


            data['daily_gain'] = (data['Close'] / data['Close'].shift(1) - 1) * 100

            preprocess_ok += 1
            logger.debug(f"预处理 {ticker} 完成，添加了 legacy_ema, fast_ema, breakout_window, market_ma, daily_gain 指标")
        except Exception as e:
            logger.error(f"预处理 {ticker} 时出错: {e}")

    logger.info(f"数据预处理完成，成功 {preprocess_ok} 个标的")
    return data_cache

def fetch_with_retry(ticker: str, period: str = "3y", retries: int = 3, delay: float = 2.0, now_et: Optional[datetime] = None, market: str = "US") -> Optional[pd.DataFrame]:
    for attempt in range(retries):
        try:
            # 使用 auto_adjust=True 确保价格一致性
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)

            logger.debug(f"[{ticker}] 获取 yfinance 数据，最新两行: {data.tail(2)}")
            # logger.info(f"[{ticker}] 数据列索引: {data.columns}")

            # 预处理：检查最新一行的开盘价和收盘价是否为nan，如果是则跳过该股票代码的扫描
            if not data.empty:
                latest_row = data.iloc[-1]
                # 处理多级索引的情况
                has_nan = False
                # 尝试获取Open和Close列
                try:
                    # 对于多级索引
                    if isinstance(data.columns, pd.MultiIndex):
                        # 尝试获取Close和Open列
                        # 方式1：直接使用('Close', ticker)和('Open', ticker)
                        close_val = latest_row.get(('Close', ticker))
                        open_val = latest_row.get(('Open', ticker))

                        # 方式2：如果方式1失败，尝试使用第一列的Close和Open
                        if close_val is None or open_val is None:
                            # 遍历所有列，找到Close和Open列
                            for col in data.columns:
                                if col[0] == 'Close':
                                    close_val = latest_row.get(col)
                                elif col[0] == 'Open':
                                    open_val = latest_row.get(col)
                                if close_val is not None and open_val is not None:
                                    break
                    else:
                        # 对于单级索引
                        close_val = latest_row.get('Close')
                        open_val = latest_row.get('Open')

                    # 检查是否有nan值
                    if close_val is not None:
                        if pd.isna(close_val):
                            has_nan = True
                    if open_val is not None:
                        if pd.isna(open_val):
                            has_nan = True
                except Exception as e:
                    logger.warning(f"[{ticker}] 检查nan值时出错: {e}")

                if has_nan:
                    logger.warning(f"[{ticker}] 最新一行含 nan，跳过扫描")
                    return None

            if data is not None and len(data) > 0:
                # 🔥 关键：不要做 tz_convert。直接将索引转为不带时区的 Timestamp
                # 这样 '2026-03-23 00:00:00' 永远是 23号，不会因为 -4小时变成 22号
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)

                # 输出时区转换信息（用于调试）
                # logger.info(f"[{ticker}] 时区转换完成，最新索引: {data.index[-1]}")

                # 如果需要根据 now_et 过滤（回测或同步用）
                if now_et is not None:
                    # 根据市场类型选择不同的时区进行过滤
                    if market == "CN":
                        cutoff = now_et.astimezone(CN_TIMEZONE).replace(tzinfo=None)
                    else:
                        # 对于美股，使用 ET 时间
                        cutoff = now_et.astimezone(ET_TIMEZONE).replace(tzinfo=None)
                    # 只保留 cutoff 之前的数据
                    data = data[data.index <= cutoff]
                    if data.empty:
                        logger.warning(f"No data available before {cutoff} for {ticker}")
                        return None
                return data


            logger.warning(f"Empty data for {ticker}, attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"Error fetching {ticker} attempt {attempt + 1}: {e}")
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    return None

def compute_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def is_premarket_by_market(now_et: datetime, market: str) -> bool:
    if market == "CN":
        now_local = now_et.astimezone(CN_TIMEZONE)
        return now_local.hour * 60 + now_local.minute < 15 * 60
    else:
        now_local = now_et.astimezone(ET_TIMEZONE)
        return now_local.hour * 60 + now_local.minute < 16 * 60

def _apply_time_slice(series: pd.Series, now_et: datetime, market: str = "US", purpose: str = "scan") -> pd.Series:
    """
    时间切片函数

    Args:
        series: 价格序列
        now_et: 当前时间（ET 时区）
        market: 市场类型（US/CN）
        purpose: 用途（scan=扫描信号，execute=执行交易）

    Returns:
        切片后的序列
    """
    if len(series) <= 1:
        return series

    # execute 模式：严禁截断，必须保留完整数据以提取 Open 价格
    if purpose == "execute":
        return series

    # scan 模式：保持原有逻辑
    # 1. 获取当前市场的当地物理时间 (Date)
    now_local = now_et.astimezone(CN_TIMEZONE if market == "CN" else ET_TIMEZONE)
    today_date = now_local.date()


    last_trading_day = series.index[-1].date()

    # 定义判定条件
    close_hour = 15 if market == "CN" else 16
    is_same_day = (last_trading_day == today_date)
    is_before_close = (now_local.hour < close_hour)

    # 3. 判定：如果是今天的临时数据且未收盘，则剔除最后一行
    if is_same_day and is_before_close:
        logger.info(f"[{market}] 物理时间未收盘，截断今日临时数据: {last_trading_day}")
        result = series.iloc[:-1]
        # 保留原始 Series 的名称
        result.name = series.name
        return result

    return series

# ─────────────────────────────────────────────────────────────────────────────
# 策略核心函数
# ─────────────────────────────────────────────────────────────────────────────

def fetch_qqq_data(now_et: Optional[datetime] = None) -> Tuple[Optional[pd.Series], Optional[pd.Series], bool]:
    if now_et is None: now_et = datetime.now(ET_TIMEZONE)
    data = fetch_with_retry("QQQ", period="3y", now_et=now_et, market="US")
    if data is None or len(data) < positive_window("market_min_rows"):
        return None, None, False
    premarket = is_premarket_by_market(now_et, "US")
    close_raw = data["Close"].squeeze()
    close = _apply_time_slice(close_raw, now_et, market="US")
    market_ma = compute_ma(close, positive_window("market_ma_window"))
    return close, market_ma, premarket

def check_individual_stock_signal(
    ticker: str,
    qqq_close: pd.Series,
    qqq_market_ma: pd.Series,
    premarket: bool,
    hhv_period: Optional[int] = None,
    now_et: Optional[datetime] = None,
    data_cache: dict = None,
) -> dict:
    """Private calculation supplied explicitly by the deployment."""
    return private_rule("check_individual_stock_signal", ticker=ticker, qqq_close=qqq_close, qqq_market_ma=qqq_market_ma, premarket=premarket, hhv_period=hhv_period, now_et=now_et, data_cache=data_cache)

def check_benchmark_signal(
    buy_ticker: str,
    benchmark_ticker: str,
    coeff: float,
    days: int,
    drawdown_pct: float,
    now_et: Optional[datetime],
    market: str,
    data_cache: dict = None,
    position_state: Optional[Dict[str, Any]] = None,
    ma_window: Optional[int] = None,
    sell_coeff: Optional[float] = None,
) -> dict:
    """Private calculation supplied explicitly by the deployment."""
    return private_rule("check_benchmark_signal", buy_ticker=buy_ticker, benchmark_ticker=benchmark_ticker, coeff=coeff, days=days, drawdown_pct=drawdown_pct, now_et=now_et, market=market, data_cache=data_cache, position_state=position_state, ma_window=ma_window, sell_coeff=sell_coeff)

# ─────────────────────────────────────────────────────────────────────────────
# 解析与扫描入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_market_configs(config_str: str) -> List[dict]:
    configs = []
    if not config_str: return configs
    for item in config_str.split(","):
        parts = item.strip().split(":")
        if len(parts) >= 3:
            buy_ticker = parts[0].upper()
            benchmark = parts[1].upper()
            ma_window = None
            sell_coeff = None

            if str(parts[2]).strip().upper().startswith("MA") and len(parts) >= 6:
                # 扩展格式参数由私有配置提供；公开版不提供示例数字。
                ma_window = int(str(parts[2]).strip().upper().replace("MA", "", 1))
                coeff = float(parts[3])
                days = int(parts[4])
                sell_coeff = float(parts[5])
                drawdown_idx = 6
            else:
                coeff = float(parts[2])
                days = int(parts[3]) if len(parts) >= 4 else 0
                drawdown_idx = 4


            drawdown_pct = None
            if len(parts) > drawdown_idx:
                tail = parts[-1].strip()
                if tail.endswith("%"):
                    try:
                        drawdown_pct = float(tail.rstrip("%")) / 100.0
                    except Exception:
                        drawdown_pct = None
            if ma_window is None:
                continue
            configs.append({
                "buy_ticker": buy_ticker, "benchmark": benchmark,
                "coeff": coeff, "buy_coeff": coeff, "sell_coeff": sell_coeff if sell_coeff is not None else coeff,
                "days": days, "drawdown_pct": drawdown_pct, "ma_window": ma_window
            })
    return configs

def parse_rebalance_config(config: dict) -> Tuple[str, str]:
    """再平衡：交易标的:观测标的；单独代码则观测=交易。例 VOO:VOO、VOO。"""
    raw = str(config.get("rebalance") or "VOO").strip()
    if not raw:
        raw = "VOO"
    if ":" in raw:
        a, b = raw.split(":", 1)
        a, b = a.strip().upper(), b.strip().upper()
        if a and b:
            return a, b
    u = raw.upper()
    return u, u


def _rebalance_market_code(observe_ticker: str) -> str:
    u = observe_ticker.strip().upper()
    if u.endswith(".SZ") or u.endswith(".SH") or u.endswith(".BJ"):
        return "CN"
    return "US"


REBALANCE_MONTHS = ()
REBALANCE_START_DAY = None
REBALANCE_END_DAY = None
REBALANCE_WINDOW_LABEL = "由私人配置提供"


def _rebalance_window_label(year: int, month: int) -> str:
    start_day = positive_window("rebalance_start_day")
    end_day = positive_window("rebalance_end_day")
    return f"{year}-{month:02d}-{start_day:02d} ~ {year}-{month:02d}-{end_day:02d}"


def _next_rebalance_window(now_et: datetime) -> str:
    months = sorted(private_parameter("rebalance_months"))
    if not months or any(isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12 for month in months):
        raise ValueError("Invalid configured calendar months")
    end_day = positive_window("rebalance_end_day")
    for month in months:
        if now_et.month < month or (now_et.month == month and now_et.day <= end_day):
            return _rebalance_window_label(now_et.year, month)
    return _rebalance_window_label(now_et.year + 1, months[0])


def check_rebalance_reminder(trade_ticker: str, now_et: Optional[datetime] = None) -> dict:
    now_et = now_et or datetime.now(ET_TIMEZONE)
    next_window = _next_rebalance_window(now_et)
    reminder = (
        now_et.month in private_parameter("rebalance_months")
        and positive_window("rebalance_start_day") <= now_et.day <= positive_window("rebalance_end_day")
    )
    return {
        "ticker": trade_ticker, "strategy": "再平衡",
        "signal": "再平衡提醒" if reminder else "观望",
        "next_reminder": next_window, "market": "美股",
        "benchmark": trade_ticker, "close": None, "trading_close": None,
        "data_date": "-", "scan_mode": "已收盘", "reminder": reminder,
        "reminder_window": REBALANCE_WINDOW_LABEL,
    }


def enrich_rebalance_signal(
    base: dict,
    observe_ticker: str,
    trade_ticker: str,
    now_et: datetime,
    data_cache: dict,
) -> dict:
    out = dict(base)
    ob = observe_ticker.strip().upper()
    tr = trade_ticker.strip().upper()
    mcode = _rebalance_market_code(ob)
    out["market"] = "A股" if mcode == "CN" else "美股"
    out["benchmark"] = ob
    pre = is_premarket_by_market(now_et, mcode)
    out["scan_mode"] = "未收盘" if pre else "已收盘"
    if data_cache and ob in data_cache:
        try:
            cr = data_cache[ob]["Close"].squeeze()
            sl = _apply_time_slice(cr, now_et, market=mcode)
            if sl is not None and not sl.empty:
                out["close"] = float(sl.iloc[-1])
                out["data_date"] = str(sl.index[-1].date())
        except Exception:
            pass
    if data_cache and tr in data_cache:
        try:
            d2 = data_cache[tr]["Close"].squeeze()
            s2 = _apply_time_slice(d2, now_et, market=mcode)
            if s2 is not None and not s2.empty:
                out["trading_close"] = float(s2.iloc[-1])
        except Exception:
            pass
    if out.get("trading_close") is None:
        out["trading_close"] = out.get("close")
    return out

def run_full_scan(config: dict, now_et: Optional[datetime] = None) -> tuple:
    signals = {}
    if now_et is None:
        now_et = datetime.now(ET_TIMEZONE)

    # 记录当前处理的时间坐标
    logger.info(f"[SCAN] 开始全量扫描，时间坐标: {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")

    # 添加最后更新时间戳
    signals["last_update"] = now_et.isoformat()

    # 收集所有需要的标的
    all_tickers = []

    # 1. 美股大盘策略标的
    for mc in parse_market_configs(config.get("us_stocks", "")):
        all_tickers.append(mc["buy_ticker"])
        all_tickers.append(mc["benchmark"])

    # 2. 个股动量标的：交易标的 + 信号观察标的
    for ticker_cfg in get_momentum_ticker_configs(config):
        ticker = ticker_cfg.get("ticker")
        signal_ticker = ticker_cfg.get("signal_ticker")
        if ticker:
            all_tickers.append(ticker)
        if signal_ticker and signal_ticker != ticker:
            all_tickers.append(signal_ticker)

    # 3. 再平衡策略标的（交易 + 观测，可不同）
    reb_trade, reb_observe = parse_rebalance_config(config)
    all_tickers.append(reb_trade)
    if reb_observe != reb_trade:
        all_tickers.append(reb_observe)

    # 4. 基准标的
    all_tickers.extend(["QQQ", "VOO"])

    # 5. 当前手动持仓也必须进入本轮缓存。部分持仓（如 VUAA.L）不一定在配置扫描名单里，
    # 但动量持仓审计和持仓监控仍需要它们的历史行情。
    try:
        momentum_state = load_momentum_state()
        current_positions = momentum_state.get("current_positions", [])
        if isinstance(current_positions, list):
            for pos in current_positions:
                if not isinstance(pos, dict):
                    continue
                ticker = momentum_ticker_symbol(pos.get("ticker"))
                if ticker:
                    all_tickers.append(ticker)
    except Exception as exc:
        logger.warning(f"[SCAN] 加载动量持仓标的失败，跳过持仓补充缓存: {exc}")

    # 批量获取所有数据
    data_cache = batch_get_data(all_tickers, now_et=now_et)

    # 加载持仓状态（由用户手动维护 in_position/entry_date；程序仅在持仓时更新 peak_high/peak_date）
    position_state = load_position_state()
    position_state_dirty = False

    # 预处理数据，统一计算指标
    data_cache = preprocess_data(data_cache, now_et=now_et)

    # --- [熔断保护]：基准缺失则全量终止 ---
    if "QQQ" not in data_cache:
        logger.error("❌ 无法获取 QQQ 基准数据，网络可能中断。为防止产生假信号，本次扫描已熔断。")
        return {}, data_cache

    # 1. 处理 QQQ 数据
    qqq_data = data_cache.get("QQQ")
    if qqq_data is None or len(qqq_data) < 202:
        logger.error("❌ QQQ 数据不足，本次扫描已熔断。")
        return {}, data_cache

    # 应用时间切片
    q_close_raw = qqq_data["Close"].squeeze()
    q_close = _apply_time_slice(q_close_raw, now_et, market="US")
    q_ma = None
    us_pre = is_premarket_by_market(now_et, "US")

    # 市场策略配置由私有运行时配置提供。
    for mc in parse_market_configs(config.get("us_stocks", "")):
        res = check_benchmark_signal(
            mc["buy_ticker"],
            mc["benchmark"],
            mc["coeff"],
            mc["days"],
            mc.get("drawdown_pct"),
            now_et,
            "US",
            data_cache,
            position_state,
            mc.get("ma_window"),
            mc.get("sell_coeff", mc.get("coeff")),
        )
        # 拦截：如果大盘标的数据下载失败或有错误，不存入结果
        if res.get("signal") == "ERROR" or res.get("error"):
            logger.warning(f"⚠️ 跳过大盘标的 {mc['buy_ticker']}，原因：{res.get('error', '数据获取失败')}")
            continue
        # 添加 is_market 字段，标记为大盘策略
        res["is_market"] = True
        signals[mc["buy_ticker"]] = res

        if res.get("position_state_updated"):
            position_state_dirty = True

    # 3. 个股策略 - 【增加错误过滤】
    hhv_period = int(config.get("hhv_period") or positive_window("breakout_window"))
    for t in config.get("tickers", []):
        ticker_symbol = momentum_ticker_symbol(t)
        if not ticker_symbol:
            continue

        # 检查数据是否存在
        if ticker_symbol not in data_cache:
            logger.warning(f"⚠️ 跳过个股 {ticker_symbol}，原因：数据获取失败")
            continue

        # 传入预取好的数据
        res = check_individual_stock_signal(
            ticker_symbol,
            q_close,
            q_ma,
            us_pre,
            hhv_period,
            now_et,
            data_cache
        )

        # --- [关键逻辑]：如果个股下载失败（ERROR），跳过该标的，保留 JSON 中的旧值 ---
        if res.get("signal") == "ERROR":
            logger.warning(
                f"⚠️ 跳过个股 {ticker_symbol}，原因：{res.get('error') or '获取历史数据失败'}"
            )
            continue

        # 个股策略评分已由 momentum_scorer.py 接管，此处不再设置评分字段

        res["market"] = "美股"
        res["scan_mode"] = "未收盘" if us_pre else "已收盘"
        signals[ticker_symbol] = res

    # 4. 再平衡策略
    rb_trade, rb_observe = parse_rebalance_config(config)
    rb_base = check_rebalance_reminder(rb_trade, now_et)
    signals[rb_trade] = enrich_rebalance_signal(
        rb_base, rb_observe, rb_trade, now_et, data_cache
    )

    if position_state_dirty:
        save_position_state_atomic(position_state)

    # 添加时间戳
    signals["last_update"] = now_et.isoformat()

    # 记录扫描完成信息
    logger.info(f"[SCAN] 全量扫描完成，处理 {len(signals) - 1} 个标的，时间坐标: {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")

    return signals, data_cache


def sort_by_signal_and_name(df):
    """
    按信号类型和标的名称排序
    排序规则：买入>关注>卖出>观望，相同信号按名字a-z排序
    """
    df_copy = df.copy()
    signal_order = {"买入": 0, "关注": 1, "卖出": 2, "观望": 3}
    df_copy["signal_order"] = df_copy["信号"].apply(lambda x: signal_order.get(x.split()[-1], 3))
    df_sorted = df_copy.sort_values(
        by=["signal_order", "标的"],
        ascending=[True, True]
    )
    return df_sorted
