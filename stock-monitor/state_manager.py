import json
import os
import logging
import re
from datetime import datetime, date
from typing import Any, Optional, Dict, List, Set, Tuple

# 尝试导入python-dotenv库
try:
    from dotenv import load_dotenv
    # 加载项目根目录的 .env 文件
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path)
except ImportError:
    logging.warning("python-dotenv not installed, .env file will not be loaded")

# 配置日志
# 注意：按你的选择，DEV 仍保留本地 latest_scan.log（不随 STOCK_MONITOR_DATA_DIR 迁移）。
_local_data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(_local_data_dir, exist_ok=True)
log_file = os.path.join(_local_data_dir, "latest_scan.log")

# 设置日志格式
log_format = "%(asctime)s [%(levelname)s] %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# 创建文件处理器 (追加模式)
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 配置根 logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# 只在根 logger 没有处理器时才添加文件处理器
if not root_logger.handlers:
    root_logger.addHandler(file_handler)
    # 移除控制台处理器，避免日志重复输出
    # root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

def _resolve_data_dir() -> str:
    override = os.environ.get("STOCK_MONITOR_DATA_DIR") or os.environ.get("stock_monitor_data_dir")
    if override and str(override).strip():
        return os.path.abspath(os.path.expanduser(str(override).strip()))

    env_type = (os.environ.get("ENV_TYPE") or os.environ.get("env_type") or "").strip().upper()
    if env_type == "DEVELOPMENT":
        project_root = os.path.dirname(os.path.dirname(__file__))
        sibling_prod_data = os.path.abspath(
            os.path.join(project_root, "..", "Stock-Monitor-PROD", "stock-monitor", "data")
        )
        if os.path.isdir(sibling_prod_data):
            logging.info(f"DEV 未配置 STOCK_MONITOR_DATA_DIR，自动使用 PROD data: {sibling_prod_data}")
            return sibling_prod_data

    return _local_data_dir

DATA_DIR = _resolve_data_dir()
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")


def refresh_data_dir() -> None:
    """刷新运行时 data 目录，避免长驻 Streamlit 进程拿着旧环境常量。"""
    global DATA_DIR, CONFIG_FILE, _cached_config

    resolved = _resolve_data_dir()
    if resolved == DATA_DIR:
        return

    logger.warning(f"DATA_DIR 已刷新: {DATA_DIR} -> {resolved}")
    DATA_DIR = resolved
    CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
    _cached_config = None


def get_data_path(filename: str) -> str:
    """获取数据文件路径"""
    refresh_data_dir()
    return os.path.join(DATA_DIR, filename)


def trade_execution_timestamp(trade_date: date, trade_type: str) -> str:
    """动量执行日志时间：卖出 09:30:00，买入 09:30:01（同日先卖后买）。"""
    time_part = "09:30:00" if trade_type == "卖出" else "09:30:01"
    return f"{trade_date.isoformat()} {time_part}"

DEFAULT_CONFIG = {
    "bark_key": "",
    "tickers": [],
    "ytd_tickers": [],
    "ytd_display_names": {},
    "us_stocks": "",
    "scan_time": "",
    "hhv_period": None,
    "HHV_WINDOW": None,
    "MIN_MARKET_CAP": None,
    "MAX_MOMENTUM_POSITIONS": None,
    "MARKET_CAPS": {},
    "MARKET_CAP_FX_RATES": {},
    "RISK_VALUATION_METRICS": {},
    "MOMENTUM_TICKER_SETTINGS": {},
    "ENABLE_MOMENTUM_AUTO_TRADE": False,
}

DEFAULT_MOMENTUM_INDICATOR = ""


def _momentum_setting_for(config: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    ticker_u = str(ticker or "").strip().upper()
    if not ticker_u:
        return {}

    settings = config.get("MOMENTUM_TICKER_SETTINGS") or config.get("momentum_ticker_settings") or {}
    setting = settings.get(ticker_u) if isinstance(settings, dict) else None
    out = dict(setting) if isinstance(setting, dict) else {}

    signal_map = config.get("MOMENTUM_SIGNAL_TICKERS") or config.get("momentum_signal_tickers") or {}
    if isinstance(signal_map, dict) and signal_map.get(ticker_u):
        out["signal_ticker"] = signal_map.get(ticker_u)

    indicator_map = config.get("MOMENTUM_INDICATORS") or config.get("momentum_indicators") or {}
    if isinstance(indicator_map, dict) and indicator_map.get(ticker_u):
        out["indicator"] = indicator_map.get(ticker_u)

    return out


def parse_momentum_ticker_entry(entry: Any, setting: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """解析动量标的配置；公开版不提供默认窗口参数。"""
    raw = str(entry or "").strip().upper()
    ticker, _, indicator_raw = raw.partition(":")
    ticker = ticker.strip()
    indicator_raw = indicator_raw.strip()
    setting = setting if isinstance(setting, dict) else {}
    signal_ticker = str(
        setting.get("signal_ticker")
        or setting.get("observe_ticker")
        or setting.get("benchmark_ticker")
        or ticker
    ).strip().upper()

    indicator_override = (
        setting.get("indicator")
        or setting.get("trend_indicator")
        or setting.get("ema")
    )
    if indicator_override:
        indicator_raw = str(indicator_override).strip().upper()

    default_window = None
    if DEFAULT_MOMENTUM_INDICATOR:
        try:
            default_window = int(DEFAULT_MOMENTUM_INDICATOR.replace("EMA", ""))
        except (TypeError, ValueError):
            default_window = None
    window = default_window
    if indicator_raw:
        if indicator_raw.startswith("EMA"):
            indicator_raw = indicator_raw[3:]
        try:
            parsed_window = int(indicator_raw)
            if parsed_window > 0:
                window = parsed_window
        except (TypeError, ValueError):
            window = default_window
    indicator = f"EMA{window}" if window else ""

    return {
        "raw": raw,
        "ticker": ticker,
        "signal_ticker": signal_ticker or ticker,
        "indicator": indicator,
        "ema_window": window,
    }


def get_momentum_ticker_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """返回动量配置：交易标的、信号观察标的、趋势指标。"""
    configs: List[Dict[str, Any]] = []
    for entry in config.get("tickers") or []:
        base = parse_momentum_ticker_entry(entry)
        ticker = base.get("ticker")
        if not ticker:
            continue
        setting = _momentum_setting_for(config, ticker)
        configs.append(parse_momentum_ticker_entry(entry, setting=setting))
    return configs


def momentum_ticker_symbol(entry: Any) -> str:
    """取动量配置里的真实行情代码。"""
    return parse_momentum_ticker_entry(entry).get("ticker", "")


def configured_signal_keys(config: Dict[str, Any]) -> Set[str]:
    """当前配置仍管理的 signals.json 顶层键。"""
    keys: Set[str] = set()

    for item in str(config.get("us_stocks") or "").split(","):
        buy_ticker = item.strip().split(":", 1)[0].strip().upper()
        if buy_ticker:
            keys.add(buy_ticker)

    for entry in get_momentum_ticker_configs(config):
        ticker = entry.get("ticker")
        if ticker:
            keys.add(ticker.upper())

    raw_rebalance = str(config.get("rebalance") or "VOO").strip()
    rebalance_ticker = raw_rebalance.split(":", 1)[0].strip().upper() if raw_rebalance else "VOO"
    if rebalance_ticker:
        keys.add(rebalance_ticker)

    return keys

# 全局配置缓存，防止重复加载日志刷屏
_cached_config = None

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def load_config(reload: bool = False) -> dict:
    """加载配置。reload=False 时使用缓存防止重复打印日志。"""
    global _cached_config
    refresh_data_dir()
    if _cached_config and not reload:
        return _cached_config

    _ensure_data_dir()

    # 1. 基础加载
    if not os.path.exists(CONFIG_FILE):
        logger.warning(f"配置文件不存在，使用默认配置: {CONFIG_FILE}")
        cfg = DEFAULT_CONFIG.copy()
    else:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            logger.exception(f"读取配置文件失败，使用默认配置: {CONFIG_FILE}")
            cfg = DEFAULT_CONFIG.copy()

    # 2. 补全默认字段
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)

    def _get_env(key: str) -> Optional[str]:
        return os.environ.get(key) or os.environ.get(key.lower()) or None

    # --- 环境变量覆盖逻辑 ---
    env_bark_key = _get_env("BARK_KEY")
    if env_bark_key:
        cfg["bark_key"] = env_bark_key

    if not cfg.get("bark_key"):
        logger.error("Secret BARK_KEY 未配置，Bark 推送将被跳过")
    else:
        k = cfg["bark_key"]
        masked = f"{k[:4]}****{k[-4:]}" if len(k) > 8 else "****"
        logger.info(f"Bark 配置已就绪 (Key: {masked})")

    # HHV_PERIOD
    env_hhv = _get_env("HHV_PERIOD")
    if env_hhv:
        try:
            cfg["hhv_period"] = int(env_hhv)
        except ValueError:
            logger.error(f"HHV_PERIOD 环境变量格式错误: {env_hhv}")

    logger.info(f"HHV_PERIOD 当前值: {cfg['hhv_period']}")

    # Tickers
    env_individual = _get_env("INDIVIDUAL_STOCKS")
    if env_individual:
        parsed = [t.strip().upper() for t in env_individual.split(",") if t.strip()]
        if parsed:
            cfg["tickers"] = parsed
            logger.info(f"使用环境变量更新标的列表: {cfg['tickers']}")
    else:
        logger.info(f"使用配置文件的标的列表: {cfg['tickers']} (config={CONFIG_FILE})")

    # 扫描时间等
    for key in ["us_stocks", "scan_time"]:
        env_val = _get_env(key.upper())
        if env_val:
            cfg[key] = env_val.strip()
            logger.info(f"使用环境变量更新 {key.upper()}")

    _cached_config = cfg
    return cfg

def save_config(config: dict):
    _ensure_data_dir()
    try:
        clean_config = config.copy()
        if os.environ.get("BARK_KEY"):
            clean_config["bark_key"] = "ENV_PROTECTED"
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

def load_signals() -> Dict[str, Any]:
    file_path = get_data_path("signals.json")
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 额外检查：确保 json.load 出来的是字典而不是 None 或其他类型
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"Failed to load signals: {e}")
        return {}  # <--- 修改这里：发生错误时返回空字典，确保后续 .get() 不崩溃

def save_signals(signals: Dict[str, Any]):
    file_path = get_data_path("signals.json")
    temp_path = file_path + ".tmp"
    _ensure_data_dir()
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(signals, f, ensure_ascii=False, indent=2)
        # 写入成功后再重命名，保证 signals.json 的完整性
        os.replace(temp_path, file_path)
    except Exception as e:
        logger.error(f"Failed to save signals: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def update_and_save_signals(
    new_signals: Dict[str, Any],
    scan_time: Optional[datetime] = None,
    active_signal_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    【核心优化】增量更新信号。只覆盖本次扫描成功的标的，
    报错或断网跳过的标的在 JSON 中保持原样，防止信号丢失。
    """
    if not new_signals:
        logger.warning("本次扫描未产生任何有效新信号，跳过保存以保护原始数据。")
        return load_signals()

    # 1. 读取旧信号
    current_signals = load_signals()

    # 2. 增量更新 (用新结果覆盖旧结果，但旧结果中没被新结果触碰的标的得以保留)
    # 过滤掉含有 ERROR 的新信号，防止脏数据入库
    # 跳过 last_update 键，因为它是一个字符串时间戳，不是信号对象
    valid_new_signals = {k: v for k, v in new_signals.items() if k != "last_update" and isinstance(v, dict) and v.get("signal") != "ERROR"}

    current_signals.update(valid_new_signals)

    if active_signal_keys is not None:
        active_keys = {str(k).upper() for k in active_signal_keys if str(k).strip()}
        stale_keys = [
            k for k, v in current_signals.items()
            if k != "last_update" and isinstance(v, dict) and str(k).upper() not in active_keys
        ]
        for key in stale_keys:
            current_signals.pop(key, None)
        if stale_keys:
            logger.info(f"已清理不在当前配置中的旧信号: {', '.join(sorted(stale_keys))}")

    # 更新 last_update 字段
    if "last_update" in new_signals:
        current_signals["last_update"] = new_signals["last_update"]

    # 3. 持久化
    save_signals(current_signals)
    return current_signals

def detect_signal_changes(old_signals: Dict[str, Any], new_signals: Dict[str, Any], override_time: Optional[datetime] = None) -> list:
    """对比信号差异（仅用于 history 审计）；Bark 推送请用当次扫描快照，勿依赖本函数。"""
    changes = []
    ts = override_time.isoformat() if override_time else datetime.now().isoformat()

    for ticker, new_data in new_signals.items():
        # 跳过 last_update 键，因为它是一个字符串时间戳，不是信号对象
        if ticker == "last_update":
            continue

        # 确保 new_data 是字典类型
        if not isinstance(new_data, dict):
            logger.warning(f"跳过非字典类型的信号数据: {ticker} = {new_data}")
            continue

        new_signal = new_data.get("signal", "观望")

        # 核心拦截：如果新信号是 ERROR，绝对不作为“变更”处理，直接忽略
        if new_signal == "ERROR":
            continue

        old_signal = old_signals.get(ticker, {}).get("signal", "观望")

        if old_signal != new_signal:
            changes.append({
                "ticker": ticker,
                "old_signal": old_signal,
                "new_signal": new_signal,
                "timestamp": ts,
                "data": new_data,
            })
    return changes

def load_history() -> list:
    file_path = get_data_path("history.json")
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 额外检查：确保 json.load 出来的是列表而不是 None 或其他类型
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return []

def save_history(history: list):
    file_path = get_data_path("history.json")
    temp_path = file_path + ".tmp"
    _ensure_data_dir()
    try:
        limit = 200
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(history[-limit:], f, ensure_ascii=False, indent=2)
        # 写入成功后再重命名，保证 history.json 的完整性
        os.replace(temp_path, file_path)
    except Exception as e:
        logger.error(f"Failed to save history: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def append_history(changes: list):
    history = load_history()
    for change in changes:
        history.append({
            "timestamp": change["timestamp"],
            "ticker": change["ticker"],
            "old_signal": change["old_signal"],
            "new_signal": change["new_signal"],
        })
    save_history(history)


EXECUTION_LOG_MAX = 200


def load_momentum_state() -> Dict[str, Any]:
    """读取动量状态（持仓、execution_logs、history 等）。"""
    file_path = get_data_path("momentum_state.json")
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"Failed to load momentum state: {e}")
        return {}


def _round_price_key(value: Any) -> Optional[float]:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _format_return_pct(value: Any) -> Optional[str]:
    try:
        pct = float(value) * 100
    except (TypeError, ValueError):
        return None
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def _history_return_lookup(state: Dict[str, Any]) -> Dict[Tuple[str, str, float], str]:
    lookup: Dict[Tuple[str, str, float], str] = {}
    history = state.get("history") if isinstance(state, dict) else None
    if not isinstance(history, list):
        return lookup

    for entry in history:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker") or "").strip().upper()
        sell_date = str(entry.get("sell_date") or "").strip()
        price_key = _round_price_key(entry.get("sell_price"))
        return_str = _format_return_pct(entry.get("total_return"))
        if ticker and sell_date and price_key is not None and return_str:
            lookup[(sell_date, ticker, price_key)] = return_str
    return lookup


_MANUAL_SELL_LOG_RE = re.compile(r"手动卖出\s+([A-Z0-9._-]+)\s+@\s+([0-9]+(?:\.[0-9]+)?)")


def _with_history_return(log: str, timestamp: str, return_lookup: Dict[Tuple[str, str, float], str]) -> str:
    if "收益" in log:
        return log
    match = _MANUAL_SELL_LOG_RE.search(log)
    if not match:
        return log

    trade_date = str(timestamp or "")[:10]
    ticker = match.group(1).strip().upper()
    price_key = _round_price_key(match.group(2))
    if not trade_date or price_key is None:
        return log

    return_str = return_lookup.get((trade_date, ticker, price_key))
    if not return_str:
        return log
    return f"{log}, 收益：{return_str}"


def load_merged_execution_logs() -> List[Dict[str, str]]:
    """合并 momentum_state 与 position_state 的执行日志，去重后按时间倒序。"""
    from position_state import load_position_state

    seen = set()
    merged: List[Dict[str, str]] = []
    sources = (
        (load_momentum_state, "个股动量"),
        (load_position_state, "大盘/疯牛"),
    )
    for loader, strategy_label in sources:
        try:
            state = loader()
        except Exception:
            state = {}
        return_lookup = _history_return_lookup(state) if strategy_label == "个股动量" else {}
        logs = state.get("execution_logs") if isinstance(state, dict) else None
        if not isinstance(logs, list):
            continue
        for entry in logs:
            if not isinstance(entry, dict):
                continue
            ts = str(entry.get("timestamp") or "")
            log = _with_history_return(str(entry.get("log") or ""), ts, return_lookup)
            key = (ts, log)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"timestamp": ts, "strategy": strategy_label, "log": log})
    merged.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return merged


def load_scan_status() -> Dict[str, Any]:
    status_file = get_data_path("scan_status.json")
    if not os.path.exists(status_file):
        return {"last_scan": None, "status": "未扫描"}
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 额外检查：确保 json.load 出来的是字典而不是 None 或其他类型
            return data if isinstance(data, dict) else {"last_scan": None, "status": "未扫描"}
    except Exception:
        return {"last_scan": None, "status": "未扫描"}


def save_scan_status(status: str, scan_time: Optional[datetime] = None):
    status_file = get_data_path("scan_status.json")
    temp_path = status_file + ".tmp"
    _ensure_data_dir()
    try:
        data = {
            "last_scan": scan_time.isoformat() if scan_time else datetime.now().isoformat(),
            "status": status,
        }
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 写入成功后再重命名，保证 scan_status.json 的完整性
        os.replace(temp_path, status_file)
    except Exception as e:
        logger.error(f"Failed to save scan status: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
