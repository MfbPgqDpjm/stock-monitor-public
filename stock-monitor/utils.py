"""
通用工具函数模块
"""

from datetime import datetime
import logging
import pytz

from state_manager import load_signals, load_scan_status
from momentum_scorer import ET_TIMEZONE

logger = logging.getLogger(__name__)


def get_signals_timestamp():
    """获取信号的时间戳"""
    signals = load_signals()
    # 尝试从信号中获取 last_update 字段
    if isinstance(signals, dict) and 'last_update' in signals:
        try:
            timestamp = datetime.fromisoformat(signals['last_update']).astimezone(ET_TIMEZONE)
            logger.debug(f"从 signals.json 获取时间戳: {timestamp}")
            return timestamp
        except Exception as e:
            logger.error(f"解析时间戳失败: {e}")
    # 尝试从扫描状态获取时间戳
    scan_status = load_scan_status()
    if scan_status.get('last_scan'):
        try:
            timestamp = datetime.fromisoformat(scan_status['last_scan']).astimezone(ET_TIMEZONE)
            logger.debug(f"从 scan_status.json 获取时间戳: {timestamp}")
            return timestamp
        except Exception as e:
            logger.error(f"解析扫描状态时间戳失败: {e}")
    # 默认返回当前时间
    default_timestamp = datetime.now(ET_TIMEZONE)
    logger.debug(f"使用默认时间戳: {default_timestamp}")
    return default_timestamp


def _fmt_ts(ts_str):
    """格式化时间戳显示"""
    try:
        dt = datetime.fromisoformat(str(ts_str))
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        et = dt.astimezone(ET_TIMEZONE)
        return et.strftime(f"%Y-%m-%d %H:%M:%S {et.strftime('%Z')}")
    except Exception:
        return str(ts_str)
