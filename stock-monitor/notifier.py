import requests
import logging
from momentum_scorer import format_rs_pct, top_eligible_rs_stock
import pytz
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置为 INFO 级别，打印重要日志

BARK_BASE_URL = "https://api.day.app"

# Bark 推送日志文件路径（相对路径）- 与系统扫描日志共用同一文件
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "latest_scan.log")

def write_bark_log(title: str, body: str, success: bool, scan_time: Optional[datetime] = None):
    """
    将 Bark 推送记录写入日志文件
    """
    try:
        # 确保日志目录存在
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        et_tz = pytz.timezone("America/New_York")
        if scan_time:
            et_now = scan_time.astimezone(et_tz) if scan_time.tzinfo else et_tz.localize(scan_time)
        else:
            et_now = datetime.now(pytz.utc).astimezone(et_tz)

        timestamp = et_now.strftime("%Y-%m-%d %H:%M:%S ET")
        status = "成功" if success else "失败"

        log_entry = (
            f"[{timestamp}] {status} - {title}\n"
            f"{body}\n"
            "---\n"
        )

        # 写入日志文件（追加模式）
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)

    except Exception as e:
        logger.error(f"写入 Bark 日志失败: {e}")

# 信号 Emoji 映射
SIGNAL_EMOJI = {
    "买入": "🟢",
    "卖出": "🔴",
    "观望": "⚪",
    "再平衡提醒": "🔔",
}


def _runtime_env_label() -> str:
    env_type = (os.environ.get("ENV_TYPE") or os.environ.get("env_type") or "").strip().upper()
    if env_type == "DEVELOPMENT":
        return "DEV"
    if env_type == "PRODUCTION":
        return "PROD"

    cwd = os.getcwd().upper()
    if "STOCK-MONITOR-DEV" in cwd:
        return "DEV"
    if "STOCK-MONITOR-PROD" in cwd:
        return "PROD"
    return ""


def _with_env_title(title: str) -> str:
    label = _runtime_env_label()
    return f"{label} {title}" if label else title

def get_dual_timestamp(scan_time: Optional[datetime] = None) -> str:
    """
    生成基于扫描时间（美东）的双时区时间戳。
    """
    et_tz = pytz.timezone("America/New_York")
    sh_tz = pytz.timezone("Asia/Shanghai")

    if scan_time:
        # 如果传入的时间没有时区信息，强制指定为美东
        if scan_time.tzinfo is None:
            et_now = et_tz.localize(scan_time)
        else:
            et_now = scan_time.astimezone(et_tz)
    else:
        # 降级：使用当前系统时间并转为美东
        et_now = datetime.now(pytz.utc).astimezone(et_tz)

    # 转换为上海时间
    sh_now = et_now.astimezone(sh_tz)

    return (f"ET: {et_now.strftime('%m-%d %H:%M')}\n"
            f"CN: {sh_now.strftime('%m-%d %H:%M')}")

def send_bark_notification(bark_key: str, title: str, body: str, group: str = "美股监控", scan_time: Optional[datetime] = None) -> bool:
    """
    基础发送函数：自动在 Body 底部附加对应扫描时间的时间戳
    """
    if not bark_key or not bark_key.strip():
        logger.warning("Bark Key 未配置，跳过推送")
        # 即使没有推送，也要记录日志
        write_bark_log(title, body, False, scan_time)
        return False

    # 核心：将 body 与计算出的双时区时间戳合并
    full_body = f"{body}\n\n{get_dual_timestamp(scan_time)}"

    url = f"{BARK_BASE_URL}/{bark_key.strip()}"
    payload = {
        "title": title,
        "body": full_body,
        "group": group,
        "sound": "alarm",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        success = resp.status_code == 200
        # 记录推送日志
        write_bark_log(title, body, success, scan_time)
        return success
    except Exception as e:
        logger.error(f"Bark 推送异常: {e}")
        # 记录推送失败日志
        write_bark_log(title, body, False, scan_time)
        return False

def notify_market_scan_signals(
    bark_key: str,
    signals: Dict[str, Any],
    scan_time: Optional[datetime] = None,
    is_lookback: bool = False,  # 保留与旧调用兼容，当前未使用
    is_debug: bool = False,
    is_manual: bool = False,
) -> int:
    """
    按当次扫描快照推送：凡 is_market 且当前信号为买入/卖出即推送（不依赖与上一轮的 diff）。
    个股等非 is_market 标的不在此推送。
    """
    del is_lookback  # 保留与旧调用兼容，当前未使用
    if not bark_key or not bark_key.strip():
        return 0

    rows: List[Tuple[str, str]] = []
    for ticker, data in signals.items():
        if ticker == "last_update" or not isinstance(data, dict):
            continue
        if not data.get("is_market"):
            continue
        sig = data.get("signal", "观望")
        if sig not in ("买入", "卖出"):
            continue
        close = data.get("close", "N/A")
        close_str = f"{close:.2f}" if isinstance(close, (int, float)) else str(close)
        emoji = SIGNAL_EMOJI.get(sig, "⚠️")
        extra = ""
        if sig == "卖出" and data.get("sell_reason"):
            extra = f" · {_format_market_sell_reason(data.get('sell_reason'), data.get('benchmark'))}"
        rows.append((ticker, f"{emoji} [大盘] {ticker}: 当前 {sig} (${close_str}){extra}"))

    if not rows:
        logger.info("本次扫描无 is_market 的买入/卖出快照，跳过市场信号 Bark。")
        return 0

    rows.sort(key=lambda x: x[0])
    body_lines = [line for _, line in rows]
    logger.info(f"市场信号快照推送: {len(body_lines)} 条（买入/卖出）")

    env_type = os.environ.get("ENV_TYPE", "").upper()
    base_title = f"⚡ 市场信号通知 ({len(body_lines)} 个)"
    if is_debug:
        title = f"🔧 Debug观测点扫描 {base_title}"
    elif is_manual:
        if env_type == "DEVELOPMENT":
            title = f"🔧 DEV实时扫描 {base_title}"
        elif env_type == "PRODUCTION":
            title = f"🔧 PROD实时扫描 {base_title}"
        else:
            title = f"🔧 实时扫描 {base_title}"
    else:
        title = f"⏰ 定时扫描 {base_title}"

    if send_bark_notification(bark_key, title, "\n".join(body_lines), group="市场信号", scan_time=scan_time):
        return 1
    return 0


def _format_market_sell_reason(reason: object, benchmark: object = None) -> str:
    raw = str(reason or "").strip()
    bm = str(benchmark or "观测基准").strip() or "观测基准"
    if raw == "drawdown":
        return f"{bm} High峰值/{bm} Close回撤止损"
    if raw == "threshold_break":
        return f"{bm} Close跌破趋势退出阈值"
    return raw


def notify_voo_rebalance(bark_key: str, scan_time: Optional[datetime] = None, rebalance_ticker: str = "VOO") -> bool:
    """
    再平衡提醒
    """
    title = f"🔔 {rebalance_ticker} 再平衡提醒"
    body = (
        f"{rebalance_ticker} 进入再平衡提醒窗口。\n"
        "提醒窗口：每年5月第3、4周与11月第3、4周。\n"
        f"请检查你的 {rebalance_ticker} 仓位并执行操作。"
    )
    return send_bark_notification(bark_key, title, body, group="资产再平衡", scan_time=scan_time)

def notify_momentum_positions(bark_key: str, positions: List[Dict], scan_time: Optional[datetime] = None) -> bool:
    """
    动量持仓状态提醒
    推送所有持仓的状态，包括"待卖出"和"继续持有"
    """
    if not positions:
        return False

    sell_positions = [p for p in positions if p.get("action_plan") == "待卖出"]
    hold_positions = [p for p in positions if p.get("action_plan") == "持有"]

    if not sell_positions and not hold_positions:
        return False

    if sell_positions:
        title = f"🔴 动量卖出信号 ({len(sell_positions)} 个)"
    else:
        title = f"✅ 动量持仓状态 ({len(hold_positions)} 个)"
    title = _with_env_title(title)

    body_lines = []

    # 先推送卖出信号
    if sell_positions:
        body_lines.append("🔴 建议卖出持仓：")
        for p in sell_positions:
            ticker = p.get("ticker", "")
            sell_reason = p.get("sell_reason", "未知")
            latest_price = p.get("latest_price")
            trend_indicator = p.get("trend_indicator") or "FAST_EMA"
            trend_ema = p.get("trend_ema", p.get("slow_ema", p.get("fast_ema")))
            total_return = p.get("total_return", 0)
            if isinstance(total_return, (int, float)):
                return_str = f"{total_return * 100:.2f}%"
            else:
                return_str = "N/A"
            price_str = f"${latest_price:.2f}" if isinstance(latest_price, (int, float)) else "N/A"
            trend_ema_str = f"{trend_indicator}：${trend_ema:.2f}" if isinstance(trend_ema, (int, float)) else f"{trend_indicator}：N/A"
            body_lines.append(f"  {ticker} | {sell_reason} | {price_str} | {trend_ema_str} | 收益：{return_str}")

    # 再推送持有持仓
    if hold_positions:
        if sell_positions:
            body_lines.append("\n✅ 继续持有：")
        for p in hold_positions:
            ticker = p.get("ticker", "")
            latest_price = p.get("latest_price")
            trend_indicator = p.get("trend_indicator") or "FAST_EMA"
            trend_ema = p.get("trend_ema", p.get("slow_ema", p.get("fast_ema")))
            total_return = p.get("total_return", 0)
            max_drawdown = p.get("max_drawdown", 0)
            if isinstance(total_return, (int, float)):
                return_str = f"{total_return * 100:.2f}%"
            else:
                return_str = "N/A"
            if isinstance(max_drawdown, (int, float)):
                drawdown_str = f"{max_drawdown * 100:.2f}%"
            else:
                drawdown_str = "N/A"
            price_str = f"${latest_price:.2f}" if isinstance(latest_price, (int, float)) else "N/A"
            trend_ema_str = f"{trend_indicator}：${trend_ema:.2f}" if isinstance(trend_ema, (int, float)) else f"{trend_indicator}：N/A"
            body_lines.append(f"  {ticker} | {price_str} | {trend_ema_str} | 收益：{return_str} | 回撤：{drawdown_str}")

    body = "\n".join(body_lines)
    return send_bark_notification(bark_key, title, body, group="动量系统", scan_time=scan_time)

def _momentum_top_rs_line(buy_signal: Dict) -> Optional[str]:
    """可买候选中 REL_STRENGTH 最高一行；无 eligible 则 None。"""
    top = top_eligible_rs_stock(buy_signal.get("scanned_stocks") or [])
    if not top:
        return None
    return f"REL_STRENGTH最高：{top['ticker']} {format_rs_pct(top['relative_strength'])}"


def _momentum_buy_candidates_for_notice(
    buy_signal: Dict,
    pending_buy_signals: Optional[List[Dict]] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """通知口径：优先展示本轮选中的候选；若已转入待买入，则展示待买入列表。"""
    raw_candidates: List[Dict] = []
    for key in ("selected_candidates", "candidates"):
        vals = buy_signal.get(key) or []
        if isinstance(vals, list):
            raw_candidates.extend([v for v in vals if isinstance(v, dict)])

    data = buy_signal.get("data")
    if isinstance(data, dict) and data.get("ticker"):
        raw_candidates.append(data)

    if pending_buy_signals:
        raw_candidates.extend([p for p in pending_buy_signals if isinstance(p, dict)])

    out: List[Dict] = []
    seen = set()
    for item in raw_candidates:
        ticker = str(item.get("ticker") or "").upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def _append_momentum_buy_signal_lines(
    body_lines: List[str],
    buy_signal: Dict,
    buy_status: str,
    pending_buy_signals: Optional[List[Dict]] = None,
    max_buy_count: Optional[int] = None,
) -> int:
    """写入买入信号段；有 eligible 最高 REL_STRENGTH 时附带标注（与 UI 一致）。"""
    top = top_eligible_rs_stock(buy_signal.get("scanned_stocks") or [])
    top_line = _momentum_top_rs_line(buy_signal)
    selected = _momentum_buy_candidates_for_notice(buy_signal, pending_buy_signals, max_buy_count)

    if selected:
        shown_tickers = set()
        for data in selected:
            ticker = data.get("ticker", "")
            if not ticker:
                continue
            shown_tickers.add(ticker)
            relative_strength = data.get("relative_strength")
            latest_price = data.get("latest_price")
            rs_str = format_rs_pct(relative_strength) if isinstance(relative_strength, (int, float)) else str(relative_strength)
            price_str = f"${latest_price:.2f}" if isinstance(latest_price, (int, float)) else str(latest_price)
            suffix = "（REL_STRENGTH最高）" if top and top.get("ticker") == ticker else ""
            body_lines.append(f"🟢 买入信号：{ticker} | REL_STRENGTH：{rs_str} | {price_str}{suffix}")
        if top_line and top and top.get("ticker") not in shown_tickers:
            body_lines.append(top_line)
        return len(shown_tickers)
    else:
        body_lines.append(f"⚪ 买入信号：{buy_status}")
        if top_line:
            body_lines.append(top_line)
        return 0


# 保持向后兼容
def notify_momentum_sell(bark_key: str, positions: List[Dict], scan_time: Optional[datetime] = None) -> bool:
    return notify_momentum_positions(bark_key, positions, scan_time)

def notify_momentum_buy(bark_key: str, buy_signal: Dict, scan_time: Optional[datetime] = None) -> bool:
    """
    动量买入信号提醒
    推送买入信号或无买入信号的原因
    """
    if not buy_signal:
        return False

    status = buy_signal.get("status", "")

    if status == "有买入信号":
        data = buy_signal.get("data", {})
        ticker = data.get("ticker", "")
        if not ticker:
            return False

        relative_strength = data.get("relative_strength")
        latest_price = data.get("latest_price")
        top = top_eligible_rs_stock(buy_signal.get("scanned_stocks") or [])

        title = _with_env_title("🟢 动量买入信号")
        rs_str = format_rs_pct(relative_strength) if isinstance(relative_strength, (int, float)) else str(relative_strength)
        price_str = f"${latest_price:.2f}" if isinstance(latest_price, (int, float)) else str(latest_price)
        suffix = "（REL_STRENGTH最高）" if top and top.get("ticker") == ticker else ""
        body = f"{ticker} | REL_STRENGTH：{rs_str} | {price_str}{suffix}"
        top_line = _momentum_top_rs_line(buy_signal)
        if top_line and top and top.get("ticker") != ticker:
            body = f"{body}\n{top_line}"
        return send_bark_notification(bark_key, title, body, group="动量系统", scan_time=scan_time)
    else:
        title = _with_env_title("⚪ 动量扫描 - 无买入信号")
        body_lines = [f"原因：{status}"]
        top_line = _momentum_top_rs_line(buy_signal)
        if top_line:
            body_lines.append(top_line)
        body = "\n".join(body_lines)
        return send_bark_notification(bark_key, title, body, group="动量系统", scan_time=scan_time)

def notify_momentum_status(bark_key: str, position_audit: Dict, buy_signal: Dict, scan_time: Optional[datetime] = None) -> bool:
    """
    动量系统运行状态确认推送
    推送系统运行状态摘要
    """
    if not bark_key or not bark_key.strip():
        return False

    position_status = position_audit.get("status", "无数据")
    buy_status = buy_signal.get("status", "无数据")
    positions_count = len(position_audit.get("positions", []))

    title = _with_env_title("📊 动量系统运行完成")
    body_lines = [
        f"持仓状态：{position_status}",
        f"持仓数量：{positions_count}",
        f"买入扫描：{buy_status}"
    ]

    body = "\n".join(body_lines)
    return send_bark_notification(bark_key, title, body, group="动量系统", scan_time=scan_time)

def notify_momentum_combined(
    bark_key: str,
    position_audit: Dict,
    buy_signal: Dict,
    scan_time: Optional[datetime] = None,
    pending_buy_signals: Optional[List[Dict]] = None,
    max_positions: Optional[int] = None,
) -> bool:
    """
    合并推送：系统状态 + 持仓状态 + 买卖信号
    将所有动量系统信息合并成一条推送
    """
    if not bark_key or not bark_key.strip():
        return False

    positions = position_audit.get("positions", [])
    buy_status = buy_signal.get("status", "")
    active_position_count = len([p for p in positions if p.get("action_plan") != "待卖出"])
    max_buy_count = None
    if isinstance(max_positions, int):
        max_buy_count = max(0, max_positions - active_position_count)
    buy_notice_candidates = _momentum_buy_candidates_for_notice(
        buy_signal,
        pending_buy_signals,
        max_buy_count,
    )

    # 确定标题
    sell_positions = [p for p in positions if p.get("action_plan") == "待卖出"]
    hold_positions = [p for p in positions if p.get("action_plan") == "持有"]

    if sell_positions:
        title = f"🔴 动量系统 ({len(sell_positions)} 建议卖出)"
    elif buy_notice_candidates:
        title = f"🟢 动量系统 ({len(buy_notice_candidates)} 买入信号)"
    elif hold_positions:
        title = f"✅ 动量系统 ({len(hold_positions)} 持仓)"
    else:
        title = f"📊 动量系统运行完成"
    title = _with_env_title(title)

    # 构建消息体
    body_lines = []

    # 1. 持仓状态
    if positions:
        body_lines.append("📈 持仓状态：")

        # 先显示建议卖出
        if sell_positions:
            for p in sell_positions:
                ticker = p.get("ticker", "")
                sell_reason = p.get("sell_reason", "未知")
                latest_price = p.get("latest_price")
                trend_indicator = p.get("trend_indicator") or "FAST_EMA"
                trend_ema = p.get("trend_ema", p.get("slow_ema", p.get("fast_ema")))
                total_return = p.get("total_return", 0)
                return_str = f"{total_return * 100:.2f}%" if isinstance(total_return, (int, float)) else "N/A"
                price_str = f"${latest_price:.2f}" if isinstance(latest_price, (int, float)) else "N/A"
                trend_ema_str = f"{trend_indicator}：${trend_ema:.2f}" if isinstance(trend_ema, (int, float)) else f"{trend_indicator}：N/A"
                body_lines.append(f"  🔴 建议卖出 {ticker} | {sell_reason} | {price_str} | {trend_ema_str} | 收益：{return_str}")

        # 再显示持有
        if hold_positions:
            for p in hold_positions:
                ticker = p.get("ticker", "")
                latest_price = p.get("latest_price")
                trend_indicator = p.get("trend_indicator") or "FAST_EMA"
                trend_ema = p.get("trend_ema", p.get("slow_ema", p.get("fast_ema")))
                total_return = p.get("total_return", 0)
                max_drawdown = p.get("max_drawdown", 0)
                return_str = f"{total_return * 100:.2f}%" if isinstance(total_return, (int, float)) else "N/A"
                drawdown_str = f"{max_drawdown * 100:.2f}%" if isinstance(max_drawdown, (int, float)) else "N/A"
                price_str = f"${latest_price:.2f}" if isinstance(latest_price, (int, float)) else "N/A"
                trend_ema_str = f"{trend_indicator}：${trend_ema:.2f}" if isinstance(trend_ema, (int, float)) else f"{trend_indicator}：N/A"
                body_lines.append(f"  ✅ {ticker} | {price_str} | {trend_ema_str} | 收益：{return_str} | 回撤：{drawdown_str}")
    else:
        body_lines.append("📈 持仓状态：无持仓")

    # 2. 买入信号
    body_lines.append("")
    _append_momentum_buy_signal_lines(
        body_lines,
        buy_signal,
        buy_status,
        pending_buy_signals=pending_buy_signals,
        max_buy_count=max_buy_count,
    )

    # 3. 系统状态
    body_lines.append("")
    body_lines.append(f"📊 持仓数量：{len(positions)}")

    body = "\n".join(body_lines)
    return send_bark_notification(bark_key, title, body, group="动量系统", scan_time=scan_time)
