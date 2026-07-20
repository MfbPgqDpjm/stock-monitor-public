import html
import os
import sys
import logging
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, date, time, timedelta
import pytz
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

sys.path.insert(0, os.path.dirname(__file__))

from state_manager import (
    load_config,
    load_signals,
    load_scan_status,
    load_merged_execution_logs,
    update_and_save_signals,
    save_scan_status,
)

# 尝试直接从当前目录下的模块导入
try:
    from scheduler import execute_scan, manual_scan_blocked_by_schedule
except ImportError:
    # 如果上面的失败，尝试这种方式（针对某些本地环境配置）
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from scheduler import execute_scan, manual_scan_blocked_by_schedule

from strategy import (
    parse_market_configs,
    parse_rebalance_config,
    run_full_scan,
)
from position_manager import process_trade
from utils import get_signals_timestamp
from ytd_performance import get_ytd_config, get_ytd_performance, refresh_ytd_performance
from market_sentiment import (
    get_cnn_fear_greed_metric,
    get_vix_metric,
    write_market_sentiment_diagnostic,
)
from futu_quote_provider import fetch_futu_realtime_quotes
from holding_monitor import (
    build_holding_monitor_rows,
    collect_holding_tickers,
    holding_monitor_source_suffix,
)
from dip_zone import DIP_ZONE_COLUMNS, build_dip_zone_view
from momentum_view import (
    MOMENTUM_DECISION_COLUMNS,
    PENDING_OPERATION_COLUMNS,
    build_momentum_decision_view,
    build_pending_operations,
)

# 配置日志
logger = logging.getLogger(__name__)

# 定义缓存启动函数
@st.cache_resource
def init_backend():
    logger.debug("正在初始化后端...")
    # start_scheduler()  # 移除，由独立进程负责调度
    logger.debug("后端初始化完成")
    return True

# 调用初始化
# init_backend()  # 移除，由独立进程负责调度


ET_TIMEZONE = pytz.timezone("America/New_York")
CN_TIMEZONE = pytz.timezone("Asia/Shanghai")

@st.dialog("📝 调仓记录")
def trade_dialog():
    """调仓对话框"""
    from position_manager import process_trade

    # 操作类型（单选 radio）
    trade_type = st.radio("操作类型", ["买入", "卖出"], horizontal=True, key="dialog_trade_type")

    # 标的输入
    ticker = st.text_input("标的代码", value="", placeholder="输入标的代码", key="dialog_ticker")

    # 价格输入
    price = st.number_input("交易价格", min_value=0.01, step=0.01, format="%.2f", key="dialog_price")

    # 数量写入状态文件；执行日志和页面历史只展示单价
    quantity = st.number_input("数量（股）", min_value=0.000001, step=1.0, format="%.6f", key="dialog_quantity")

    # 日期选择
    trade_date = st.date_input("交易日期", value=date.today(), key="dialog_date")

    # 提交按钮和取消按钮（宽度相同）
    col_submit, col_cancel = st.columns(2)
    with col_submit:
        submit_btn = st.button("提交", use_container_width=True, type="secondary")
    with col_cancel:
        cancel_btn = st.button("取消", use_container_width=True, type="secondary")

    if cancel_btn:
        st.session_state.dialog_open = False
        st.rerun()

    if submit_btn:
        if not ticker.strip():
            st.error("请输入标的代码")
            return

        # 调用后端处理
        success, message = process_trade(
            trade_type=trade_type,
            ticker=ticker.strip().upper(),
            price=price,
            quantity=quantity,
            trade_date=trade_date
        )

        if success:
            st.success(message)
            st.session_state.dialog_open = False
            # 标记需要刷新动量结果
            st.session_state.need_refresh_momentum = True
            st.rerun()
        else:
            st.error(message)

def _mark_bool(ok: bool) -> str:
    return "✅" if ok else "❌"


def _colorize_trade_log(log: str) -> str:
    """执行日志：买入绿色、卖出红色（HTML 片段）。"""
    s = html.escape(str(log or ""))
    s = s.replace("卖出", '<span style="color:#dc2626;font-weight:600">卖出</span>')
    s = s.replace("买入", '<span style="color:#16a34a;font-weight:600">买入</span>')
    return s


def _render_execution_logs_table(df_logs: pd.DataFrame, max_height: int = 500) -> None:
    rows_html = []
    for _, row in df_logs.iterrows():
        ts = html.escape(str(row.get("时间", "")))
        strategy = html.escape(str(row.get("策略", "")))
        log_cell = _colorize_trade_log(row.get("日志", ""))
        rows_html.append(
            f'<tr><td style="padding:6px 12px;white-space:nowrap;vertical-align:top">{ts}</td>'
            f'<td style="padding:6px 12px;white-space:nowrap;vertical-align:top">{strategy}</td>'
            f'<td style="padding:6px 12px">{log_cell}</td></tr>'
        )
    table = (
        f'<div style="display:block;max-height:{max_height}px;overflow:auto">'
        '<table style="width:100%;border-collapse:collapse;font-size:14px">'
        '<thead><tr>'
        '<th style="text-align:left;padding:6px 12px;border-bottom:1px solid rgba(128,128,128,0.3)">时间</th>'
        '<th style="text-align:left;padding:6px 12px;border-bottom:1px solid rgba(128,128,128,0.3)">策略</th>'
        '<th style="text-align:left;padding:6px 12px;border-bottom:1px solid rgba(128,128,128,0.3)">日志</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)


def _is_finite_num(value: object) -> bool:
    try:
        return pd.notna(value) and float(value) not in (float("inf"), float("-inf"))
    except Exception:
        return False


def _render_ytd_progress_capsule(
    name: str,
    ytd_pct: float,
    daily_pct: object = None,
    ytd_high_drawdown_pct: object = None,
) -> None:
    color = "#2e7d32" if ytd_pct >= 0 else "#d32f2f"
    sign = "+" if ytd_pct > 0 else ""
    fill_percent = min(abs(ytd_pct), 100.0)
    safe_name = html.escape(str(name))
    daily_html = ""
    if _is_finite_num(daily_pct):
        daily_value = float(daily_pct)
        daily_color = "#2e7d32" if daily_value >= 0 else "#d32f2f"
        daily_sign = "+" if daily_value > 0 else ""
        daily_html = (
            f'<span style="color:{daily_color};font-weight:700;'
            f'font-variant-numeric:tabular-nums;">(昨日 {daily_sign}{daily_value:.2f}%)</span>'
        )
    drawdown_html = ""
    if _is_finite_num(ytd_high_drawdown_pct):
        drawdown_value = float(ytd_high_drawdown_pct)
        drawdown_color = "#2e7d32" if drawdown_value >= 0 else "#d32f2f"
        drawdown_sign = "+" if drawdown_value > 0 else ""
        drawdown_html = (
            f'<span style="color:{drawdown_color};font-weight:700;'
            f'font-variant-numeric:tabular-nums;">(YTD最高点 {drawdown_sign}{drawdown_value:.2f}%)</span>'
        )
    meta_parts = [part for part in (daily_html, drawdown_html) if part]
    meta_html = (
        f'<div class="ytd-capsule-meta">{"".join(meta_parts)}</div>'
        if meta_parts
        else ""
    )

    html_code = (
        '<style>'
        '.ytd-capsule-row{display:flex;align-items:center;margin-bottom:10px;width:100%;max-width:940px;}'
        '.ytd-capsule-name{width:60px;flex:0 0 60px;font-weight:650;white-space:nowrap;}'
        '.ytd-capsule-track{flex:1;min-width:72px;background-color:#f1f3f4;height:16px;display:flex;'
        'justify-content:flex-start;border-radius:10px;overflow:hidden;position:relative;}'
        '.ytd-capsule-value{width:82px;flex:0 0 82px;text-align:right;font-weight:800;margin-left:10px;'
        'font-variant-numeric:tabular-nums;}'
        '.ytd-capsule-meta{display:flex;align-items:center;gap:8px;font-size:12px;margin-left:8px;'
        'white-space:nowrap;}'
        '@media (max-width:640px){'
        '.ytd-capsule-row{flex-wrap:wrap;}'
        '.ytd-capsule-meta{flex:0 0 100%;margin-left:60px;margin-top:3px;white-space:normal;'
        'line-height:1.25;row-gap:2px;}'
        '}'
        '</style>'
        '<div class="ytd-capsule-row">'
        f'<div class="ytd-capsule-name">{safe_name}</div>'
        '<div class="ytd-capsule-track">'
        f'<div style="width:{fill_percent:.1f}%;background-color:{color};height:16px;border-radius:0;"></div>'
        '</div>'
        f'<div class="ytd-capsule-value" style="color:{color};">{sign}{ytd_pct:.2f}%</div>'
        f'{meta_html}'
        '</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def _load_market_sentiment_metrics(cache_version: int = 8) -> dict:
    return {
        "vix": get_vix_metric(),
        "fear_greed": get_cnn_fear_greed_metric(),
    }


def _format_metric_delta(delta: object, delta_pct: object = None, suffix: str = "") -> str | None:
    if not _is_finite_num(delta):
        return None
    sign = "+" if float(delta) > 0 else ""
    if _is_finite_num(delta_pct):
        return f"{sign}{float(delta):.2f}{suffix} ({sign}{float(delta_pct):.2f}%)"
    return f"{sign}{float(delta):.2f}{suffix}"


def _fear_greed_delta_pct(metric: dict) -> float | None:
    if _is_finite_num(metric.get("delta_pct")):
        return float(metric["delta_pct"])
    score = metric.get("score")
    delta = metric.get("delta")
    if not (_is_finite_num(score) and _is_finite_num(delta)):
        return None
    previous = float(score) - float(delta)
    if previous == 0:
        return None
    return (float(score) / previous - 1.0) * 100.0


def _render_sentiment_card(
    label: str,
    value: str,
    delta: str | None,
    is_positive: bool,
) -> None:
    bg_color = "#e8f5e9" if is_positive else "#ffebee"
    border_color = "#66bb6a" if is_positive else "#ef5350"
    text_color = "#1b5e20" if is_positive else "#b71c1c"
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_delta = html.escape(delta or "")
    delta_html = (
        f'<div style="font-size:15px;font-weight:750;color:{text_color};'
        f'font-variant-numeric:tabular-nums;margin-top:4px;">{safe_delta}</div>'
        if safe_delta
        else ""
    )
    st.markdown(
        (
            f'<div style="background:{bg_color};border:1px solid {border_color};'
            'border-left-width:7px;border-radius:8px;padding:14px 16px;'
            'min-height:116px;box-shadow:0 1px 2px rgba(0,0,0,0.06);">'
            '<div style="font-size:14px;font-weight:700;color:#263238;line-height:1.3;">'
            f'{safe_label}</div>'
            f'<div style="font-size:32px;font-weight:850;color:{text_color};'
            'line-height:1.15;margin-top:9px;font-variant-numeric:tabular-nums;">'
            f'{safe_value}</div>'
            f'{delta_html}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _vix_card_is_positive(vix: dict, value: object) -> bool:
    delta = vix.get("delta") if isinstance(vix, dict) else None
    if _is_finite_num(delta):
        return float(delta) <= 0
    if _is_finite_num(value):
        return float(value) < 20
    return False


def _fear_greed_card_is_positive(metric: dict, rating: str) -> bool:
    delta_pct = _fear_greed_delta_pct(metric) if isinstance(metric, dict) else None
    if _is_finite_num(delta_pct):
        return float(delta_pct) >= 0
    delta = metric.get("delta") if isinstance(metric, dict) else None
    if _is_finite_num(delta):
        return float(delta) >= 0
    return "greed" in str(rating).lower()


def _render_market_sentiment_metrics() -> None:
    metrics = _load_market_sentiment_metrics(cache_version=8)
    vix = metrics.get("vix") if isinstance(metrics, dict) else {}
    fear_greed = metrics.get("fear_greed") if isinstance(metrics, dict) else {}

    col_vix, col_fear = st.columns(2)
    with col_vix:
        value = vix.get("value") if isinstance(vix, dict) else None
        _render_sentiment_card(
            label=".VIX 标普500波动率指数",
            value=f"{float(value):.2f}" if _is_finite_num(value) else "N/A",
            delta=None,
            is_positive=_vix_card_is_positive(vix, value),
        )
    with col_fear:
        score = fear_greed.get("score") if isinstance(fear_greed, dict) else None
        rating = fear_greed.get("rating", "") if isinstance(fear_greed, dict) else ""
        if not _is_finite_num(score):
            message = f"[市场情绪] CNN Fear & Greed UI 将显示 N/A: metric={fear_greed}"
            logger.error(message)
            write_market_sentiment_diagnostic("ERROR", message)
        rating_suffix = f" {rating.title()}" if rating else ""
        _render_sentiment_card(
            label="CNN Fear & Greed Index",
            value=f"{float(score):.0f}{rating_suffix}" if _is_finite_num(score) else "N/A",
            delta=None,
            is_positive=_fear_greed_card_is_positive(fear_greed, rating),
        )


def _is_current_ytd_snapshot(snapshot: dict, config: dict) -> bool:
    rows = snapshot.get("targets") if isinstance(snapshot, dict) else None
    if not isinstance(rows, list):
        return False

    expected_tickers = get_ytd_config(config)["tickers"]
    snapshot_tickers = snapshot.get("configured_tickers")
    if not isinstance(snapshot_tickers, list):
        snapshot_tickers = [
            str(row.get("ticker", "")).strip().upper()
            for row in rows
            if isinstance(row, dict) and row.get("ticker")
        ]
    has_ytd_schema = all(
        isinstance(row, dict)
        and "daily_pct" in row
        and "ytd_high_drawdown_pct" in row
        for row in rows
    )
    return snapshot_tickers == expected_tickers and has_ytd_schema


def _render_ytd_capsules(snapshot: dict) -> None:
    """APP.py 页面展示：按运行时配置渲染 YTD 胶囊进度条。"""
    _render_market_sentiment_metrics()

    rows = snapshot.get("targets") if isinstance(snapshot, dict) else None
    if not rows:
        st.info("YTD 涨幅暂无数据，请点击「立即扫描」更新。")
        return

    valid_rows = [row for row in rows if _is_finite_num(row.get("ytd_pct"))]
    if not valid_rows:
        st.info("YTD 涨幅暂无数据，请点击「立即扫描」更新。")
        return

    valid_rows = sorted(valid_rows, key=lambda row: float(row.get("ytd_pct", 0.0)), reverse=True)

    updated_label = ""
    updated = snapshot.get("last_updated", "")
    if updated:
        try:
            dt = datetime.fromisoformat(updated).astimezone(ET_TIMEZONE)
            updated_label = f"更新 {dt.strftime('%Y-%m-%d %H:%M ET')}"
        except Exception:
            updated_label = f"更新 {updated}"

    st.markdown("##### YTD 涨幅")
    if updated_label:
        st.caption(updated_label)

    for row in valid_rows:
        name = str(row.get("name", ""))
        ytd_pct = float(row.get("ytd_pct", 0.0))
        _render_ytd_progress_capsule(
            name,
            ytd_pct,
            row.get("daily_pct"),
            row.get("ytd_high_drawdown_pct"),
        )


# 一眼执行表格 [实时信号] — 美股大盘/再平衡 展示列（不含内部排序列）
MARKET_SIGNAL_TABLE_COLS = [
    "策略类型",
    "交易标的",
    "信号",
    "信号产生日期",
    "观测基准",
    "当前价",
    "关键指标",
    "信号情况",
    "数据模式",
]


def _signal_generated_date(signal: str, data_date: object, actionable_signals: set[str]) -> str:
    """Only show a market data date when the row has produced an actionable signal."""
    return str(data_date or "—") if signal in actionable_signals else "—"


@st.cache_data(ttl=20, show_spinner=False)
def _load_futu_quotes_cached(tickers: tuple[str, ...], host: str, port: int) -> dict:
    return fetch_futu_realtime_quotes(tickers, host=host, port=port)


def render_dashboard():
    """渲染仪表盘"""
    # 检查 signals.json 是否为空，如果为空则强制触发一次扫描
    signals = load_signals()
    if not signals:
        logger.info("signals.json 为空，强制触发一次扫描...")
        execute_scan()
        logger.info("强制扫描完成")
        st.rerun()



    st.set_page_config(
        page_title="📈 美股监控系统 V1.6",
        page_icon="📈",
        layout="wide",
    )

    # 首次加载时强制重新读取配置文件
    config = load_config(reload=True)

    # 后续使用缓存的配置
    cfg = config

    # ── 主页面 ──────────────────────────────────────────────────────────────────

    # 显示实时时间
    now_et = datetime.now(ET_TIMEZONE)
    now_cn = datetime.now(CN_TIMEZONE)
    st.caption(f"实时时间 🇺🇸 {now_et.strftime('%Y-%m-%d %H:%M')} | 🇨🇳 {now_cn.strftime('%Y-%m-%d %H:%M')}")

    # 加载个股动量系统结果
    momentum_result = None

    # 检查是否需要刷新动量结果（调仓后）
    need_refresh = st.session_state.get("need_refresh_momentum", False)
    if need_refresh:
        st.session_state.need_refresh_momentum = False

    # 首先尝试加载保存的动量结果
    try:
        from momentum_scorer import MomentumScorer
        # 加载信号数据，避免重复扫描
        signals = load_signals()
        # 移除 last_update 字段
        if isinstance(signals, dict) and 'last_update' in signals:
            signals = {k: v for k, v in signals.items() if k != 'last_update'}

        # 加载动量结果
        scorer = MomentumScorer(cfg, signals=signals)
        momentum_result = scorer._load_momentum_result() or {}

        if need_refresh:
            # 调仓后只从 state 文件刷新持仓信息，不重新下载数据
            logger.info("调仓后从状态文件刷新持仓信息...")
            # 直接从 scorer 的 state 获取持仓（已加载）
            positions = scorer.positions
            # 获取现有的持仓审计数据（保持未操作标的数据不变）
            existing_audit = momentum_result.get("position_audit", {})
            existing_positions = existing_audit.get("positions", [])
            # 构建标的到现有数据的映射
            existing_pos_map = {p.get("ticker"): p for p in existing_positions if p.get("ticker")}

            if positions:
                # 构建新的持仓审计结果
                new_positions = []
                for p in positions:
                    ticker = p.get("ticker")
                    if ticker in existing_pos_map:
                        # 已存在的标的：保留原有数据，只更新可能变化的字段
                        existing = existing_pos_map[ticker].copy()
                        existing.update({
                            "buy_price": p.get("buy_price"),
                            "buy_date": p.get("buy_date"),
                            "quantity": p.get("quantity"),
                            "sell_flag": p.get("sell_flag"),
                            "sell_reason": p.get("sell_reason")
                        })
                        new_positions.append(existing)
                    else:
                        # 新买入的标的：创建新记录（无实时价格数据）
                        new_positions.append({
                            "status": "继续持有",
                            "action_plan": "持有",
                            "ticker": ticker,
                            "buy_price": p.get("buy_price"),
                            "buy_date": p.get("buy_date"),
                            "quantity": p.get("quantity"),
                            "latest_price": None,
                            "latest_date": None,
                            "total_return": None,
                            "max_return": None,
                            "max_drawdown": None,
                            "hold_days": None,
                            "sell_flag": p.get("sell_flag"),
                            "sell_reason": p.get("sell_reason")
                        })

                position_audit = {"status": "有持仓", "positions": new_positions}
            else:
                position_audit = {"status": "无持仓", "positions": []}

            momentum_result["position_audit"] = position_audit
            # 保存更新后的结果
            scorer._save_momentum_result(momentum_result)
    except Exception as e:
        logger.error(f"加载动量评分结果失败：{str(e)}")

    # 立即扫描按钮
    col_scan = st.columns([1])[0]
    with col_scan:
        scan_button = st.button("立即扫描", key="scan_button")

        if scan_button:
            # 检查缓存
            signals = load_signals()
            if isinstance(signals, dict) and 'last_update' in signals:
                try:
                    last_update = datetime.fromisoformat(signals['last_update'])
                    now = datetime.now(ET_TIMEZONE)
                    time_diff = (now - last_update).total_seconds()
                    if time_diff < 30:  # 30s内
                        try:
                            refresh_ytd_performance(now_et=now, force_reload=True, config=cfg)
                        except Exception as e:
                            logger.error(f"YTD 涨幅更新失败：{e}")
                        st.info(f"距离上次扫描仅 {int(time_diff)} 秒，使用缓存数据")
                        st.rerun()
                except Exception as e:
                    logger.warning(f"缓存检查失败：{e}")

            # 防止手动扫描撞上定时档位（避免并发写同一份 data/）
            blocked, msg = manual_scan_blocked_by_schedule()
            if blocked:
                st.warning(msg)
                st.stop()

            # 禁用按钮并显示进度条
            st.info("正在执行扫描...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # 执行扫描
                status_text.text("正在准备数据...")
                progress_bar.progress(0.1)

                # 执行扫描（execute_scan内部已包含动量评分系统运行）
                execute_scan(is_manual=True)
                status_text.text("正在更新 YTD 涨幅...")
                progress_bar.progress(0.9)
                try:
                    refresh_ytd_performance(force_reload=True, config=cfg)
                except Exception as e:
                    logger.error(f"YTD 涨幅更新失败：{e}")
                    st.warning("扫描完成，但 YTD 涨幅更新失败，请稍后再试。")
                progress_bar.progress(1.0)
                status_text.text("扫描完成！")
                st.success("扫描完成！")
            except Exception as e:
                st.error(f"扫描执行失败：{str(e)}")
                logger.error(f"扫描执行失败：{e}")
            finally:
                # 确保进度条完成
                progress_bar.progress(1.0)
                st.rerun()

    st.title("📈 美股监控系统 V1.6.6")
    st.caption("大盘策略（MARKET_MA×系数 连续N日确认）| 个股策略（QQQ>MARKET_MA + FAST_EMA/SLOW_EMA交叉）| VOO（5/11月第3、4周再平衡）")


    # 显示当前观测时间
    current_view_time = get_signals_timestamp()
    st.subheader(f"📊 当前观测坐标: {current_view_time.strftime('%Y-%m-%d %H:%M')} (ET)")

    # 加载扫描状态
    scan_status = load_scan_status()
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        # 显示扫描时间
        last_scan = scan_status.get("last_scan")
        if last_scan:
            try:
                last_dt = datetime.fromisoformat(last_scan).astimezone(ET_TIMEZONE)
                st.info(f"🕐 上次扫描：{last_dt.strftime('%Y-%m-%d %H:%M:%S ET')}")
            except Exception:
                st.info(f"🕐 上次扫描：{last_scan}")
        else:
            st.warning("⏳ 尚未执行扫描")
    with col_s2:
        status = scan_status.get('status', '未知')
        if '成功' in status or '完成' in status:
            st.info(f"📊 {status}")
        else:
            st.error(f"📊 {status}")
    with col_s3:
        st.info(f"定时扫描已配置: {cfg.get('scan_time', 'ET1630,ET0135')}")

    ytd_snapshot = get_ytd_performance(now_et=now_et, config=cfg)
    if not _is_current_ytd_snapshot(ytd_snapshot, cfg):
        logger.warning("[YTD] 本地快照与当前配置/格式不一致；请通过「立即扫描」刷新持久缓存。")
    _render_ytd_capsules(ytd_snapshot)

    st.divider()

    # ── 读取信号 ──────────────────────────────────────────────────────────────────
    signals = load_signals()
    # 移除 last_update 字段，避免影响后续处理
    if isinstance(signals, dict) and 'last_update' in signals:
        signals = {k: v for k, v in signals.items() if k != 'last_update'}
    signals_label = "实时信号"

    if not signals:
        st.info("暂无数据。请等待定时扫描获取最新信号。")
    else:
        SIGNAL_COLOR = {
            "买入": "🟢",
            "卖出": "🔴",
            "观望": "⚪",
            "关注": "🟡",
            "再平衡提醒": "🔔",
        }

        us_configs = parse_market_configs(cfg.get("us_stocks", ""))
        rebalance_trade, rebalance_observe = parse_rebalance_config(cfg)
        market_tickers_us = [mc["buy_ticker"] for mc in us_configs]
        realtime_tickers = collect_holding_tickers(us_configs, momentum_result or {})
        futu_host = str(cfg.get("FUTU_OPEND_HOST") or "127.0.0.1")
        try:
            futu_port = int(cfg.get("FUTU_OPEND_PORT") or 11111)
        except (TypeError, ValueError):
            futu_port = 11111
        realtime_quotes = _load_futu_quotes_cached(tuple(sorted(t for t in realtime_tickers if t)), futu_host, futu_port)

        st.subheader(f"📋 一眼执行表格 [{signals_label}]")
        rows = []

        special_tickers = [mc["buy_ticker"] for mc in us_configs]
        special_tickers.append(rebalance_trade)

        rebalance_data = signals.get(rebalance_trade, {})

        # ── 0: 再平衡策略（固定首块）────────────────────────────────────────────
        if rebalance_data:
            rebalance_signal = rebalance_data.get("signal", "观望")
            rb_market = rebalance_data.get("market", "美股")
            rb_bm = rebalance_data.get("benchmark") or rebalance_observe
            rb_close = rebalance_data.get("close")
            if isinstance(rb_close, (int, float)):
                sym = "$" if rb_market == "美股" else "¥" if rb_market == "A股" else ""
                px = f"{sym}{float(rb_close):.2f}" if sym else f"{float(rb_close):.2f}"
            else:
                px = "N/A"

            next_rem = rebalance_data.get("next_reminder", "N/A")
            reminder_window = rebalance_data.get("reminder_window", "每年5月/11月第3、4周")
            key_indicators_rb = (
                f"提醒窗口: {reminder_window} | 下次提醒：{next_rem}"
            )
            reminder = bool(rebalance_data.get("reminder"))
            in_reminder_signal = rebalance_signal == "再平衡提醒"
            signal_situation_rb = (
                f"提醒窗口内: {_mark_bool(reminder)} | 再平衡提醒: {_mark_bool(in_reminder_signal)}"
            )
            signal_date_rb = _signal_generated_date(
                rebalance_signal,
                rebalance_data.get("data_date"),
                {"再平衡提醒"},
            )
            observe_label_rb = str(rb_bm)

            rows.append({
                "策略类型": "再平衡策略",
                "市场": rb_market,
                "交易标的": rebalance_trade,
                "信号": f"{SIGNAL_COLOR.get(rebalance_signal, '⚪')} {rebalance_signal}",
                "信号产生日期": signal_date_rb,
                "观测基准": observe_label_rb,
                "当前价": px,
                "关键指标": key_indicators_rb,
                "信号情况": signal_situation_rb,
                "数据模式": rebalance_data.get("scan_mode", "N/A"),
                "标的": rebalance_trade,
                "_table_block": 0,
                "_block_order": 0,
                "is_index": 1,
                "score": None,
                "signal_sort": 2,
                "Score": "N/A",
            })

        # ── 1: 美股大盘策略 ─────────────────────────────────────────────────────
        for i, tkr in enumerate(market_tickers_us):
            mc_row = next((m for m in us_configs if m["buy_ticker"] == tkr), None)
            data = signals.get(tkr, {})
            is_index = 1 if tkr in special_tickers else 0

            if not data:
                rows.append({
                    "策略类型": "美股大盘策略",
                    "市场": "N/A",
                    "交易标的": tkr,
                    "信号": "⚪ 无数据",
                    "信号产生日期": "—",
                    "观测基准": "N/A",
                    "当前价": "N/A",
                    "关键指标": "N/A",
                    "信号情况": "N/A",
                    "数据模式": "N/A",
                    "标的": tkr,
                    "_table_block": 1,
                    "_block_order": i,
                    "is_index": is_index,
                    "score": None,
                    "signal_sort": 3,
                    "Score": "N/A",
                })
                continue

            signal = data.get("signal", "观望")
            market = data.get("market", "N/A")
            currency_symbol = "$" if market == "美股" else "¥"

            try:
                confirm_days_data = int(data.get("confirm_days") or 0)
            except (TypeError, ValueError):
                confirm_days_data = 0
            days_cfg = int(mc_row.get("days", confirm_days_data)) if mc_row else confirm_days_data
            dd_pct_cfg = float(mc_row.get("drawdown_pct") or 0) if mc_row else 0
            coeff_f = float(data.get("buy_coeff", data.get("coeff", mc_row.get("coeff", 0)))) if mc_row else float(data.get("buy_coeff", data.get("coeff", 0)))
            sell_coeff_f = float(data.get("sell_coeff", mc_row.get("sell_coeff", coeff_f))) if mc_row else float(data.get("sell_coeff", coeff_f))
            ma_window = data.get("ma_window", mc_row.get("ma_window")) if mc_row else data.get("ma_window")
            ma_label = str(data.get("ma_label") or "MARKET_MA")

            consecutive = int(data.get("consecutive_days_above", 0) or 0)
            close_val = data.get("close")
            ma_val = data.get("ma", data.get("market_ma"))
            entry_threshold_val = data.get("entry_threshold", data.get("threshold"))
            exit_threshold_val = data.get("exit_threshold", entry_threshold_val)

            if isinstance(close_val, (int, float)):
                close_str = f"{currency_symbol}{float(close_val):.2f}"
            else:
                close_str = f"{currency_symbol}{close_val}"

            if isinstance(ma_val, (int, float)) and isinstance(entry_threshold_val, (int, float)) and isinstance(exit_threshold_val, (int, float)):
                dd_str = f"{dd_pct_cfg * 100:.0f}%"
                exit_rule = f"{ma_label}×{sell_coeff_f:g}"
                if abs(sell_coeff_f - coeff_f) < 1e-9:
                    exit_rule = "同入场"
                key_indicators = (
                    f"入 {currency_symbol}{float(entry_threshold_val):.2f} ({ma_label}×{coeff_f:g}) | "
                    f"出 {currency_symbol}{float(exit_threshold_val):.2f} ({exit_rule}) | "
                    f"{days_cfg}日 | DD {dd_str}"
                )
            else:
                key_indicators = "N/A"

            standing_ok = consecutive >= days_cfg
            first_ok = bool(data.get("is_first_trigger"))
            # 与 strategy.check_benchmark_signal 一致：入场站上 entry_threshold；趋势退出跌破 exit_threshold。
            close_gt = (
                isinstance(close_val, (int, float))
                and isinstance(entry_threshold_val, (int, float))
                and float(close_val) > float(entry_threshold_val)
            )
            close_gt_exit = (
                isinstance(close_val, (int, float))
                and isinstance(exit_threshold_val, (int, float))
                and float(close_val) > float(exit_threshold_val)
            )
            in_pos = bool(data.get("in_position"))
            dd = data.get("drawdown")
            if in_pos and isinstance(dd, (int, float)):
                dd_part = f"{float(dd) * 100:.2f}% {_mark_bool(float(dd) >= float(dd_pct_cfg))}"
            else:
                dd_part = "—"

            signal_situation = (
                f"站稳 {consecutive}/{days_cfg}{_mark_bool(standing_ok)} | "
                f"首触{_mark_bool(first_ok)} | "
                f"入{_mark_bool(close_gt)} | "
                f"趋{_mark_bool(close_gt_exit)} | "
                f"DD {dd_part}"
            )

            bm = data.get("benchmark", "N/A")
            observe_label = str(bm)
            signal_date = _signal_generated_date(
                signal,
                data.get("data_date"),
                {"买入", "卖出"},
            )

            score_val = data.get("score")
            score_str = f"{float(score_val):.4f}" if isinstance(score_val, (int, float)) else "N/A"

            rows.append({
                "策略类型": "美股大盘策略",
                "市场": market,
                "交易标的": tkr,
                "信号": f"{SIGNAL_COLOR.get(signal, '⚪')} {signal}",
                "信号产生日期": signal_date,
                "观测基准": observe_label,
                "当前价": close_str,
                "关键指标": key_indicators,
                "信号情况": signal_situation,
                "数据模式": data.get("scan_mode", "N/A"),
                "标的": tkr,
                "_table_block": 1,
                "_block_order": i,
                "is_index": is_index,
                "score": data.get("score"),
                "signal_sort": 0 if signal == "买入" else 0.5 if signal == "关注" else 1 if signal == "卖出" else 2,
                "Score": score_str,
            })

        if rows:
            df = pd.DataFrame(rows)

            # 第一个表格：美股大盘、再平衡
            market_strategies = ["美股大盘策略", "再平衡策略"]
            df_market = df[df["策略类型"].isin(market_strategies)]

            # 显示第一个表格（块顺序：再平衡 → 美股大盘）
            if not df_market.empty:
                st.markdown("##### 📈 美股大盘 / 再平衡")
                df_market_sorted = df_market.sort_values(
                    by=["_table_block", "_block_order"],
                    ascending=[True, True],
                )
                row_count = len(df_market_sorted)
                calc_height = (row_count * 35) + 37
                calc_height = min(calc_height, 600)
                st.dataframe(
                    df_market_sorted[MARKET_SIGNAL_TABLE_COLS],
                    width="stretch",
                    hide_index=True,
                    height=calc_height,
                )

        holding_rows = build_holding_monitor_rows(us_configs, signals, momentum_result or {}, realtime_quotes)
        st.markdown("##### 📌 持仓监控")
        if holding_rows:
            df_holdings = pd.DataFrame(holding_rows)
            holding_display_columns = [
                "标的",
                "持仓收益",
                "最新价",
                "持仓状态",
                "持仓类型",
                "市场",
                "买入价",
                "买入日期",
                "持仓最高收益",
                "持仓最大回撤",
                "持股天数",
                "风险分",
                "风险动作",
                "风险触发",
                "价格/FAST_EMA",
                "价格/MARKET_MA",
            ]
            df_holdings = df_holdings.sort_values(
                by="持仓收益",
                ascending=False,
                na_position="last",
                kind="mergesort",
            )
            source_suffix = holding_monitor_source_suffix(holding_rows)
            latest_price_col = f"最新价（{source_suffix}）"
            pnl_col = f"持仓收益（{source_suffix}）"
            df_holdings_display = df_holdings[holding_display_columns].rename(columns={
                "最新价": latest_price_col,
                "持仓收益": pnl_col,
                "买入价": "成本价",
            })
            row_count = len(df_holdings)
            calc_height = min((row_count * 35) + 37, 600)
            st.dataframe(
                df_holdings_display,
                width="stretch",
                hide_index=True,
                height=calc_height,
                column_config={
                    "成本价": st.column_config.NumberColumn(format="%.2f"),
                    latest_price_col: st.column_config.NumberColumn(format="%.2f"),
                    pnl_col: st.column_config.NumberColumn(format="%+.2f%%"),
                    "风险分": st.column_config.NumberColumn(format="%d"),
                    "价格/MARKET_MA": st.column_config.NumberColumn(format="%+.2f%%"),
                    "价格/FAST_EMA": st.column_config.NumberColumn(format="%+.2f%%"),
                    "持仓最高收益": st.column_config.NumberColumn(format="%.2f%%"),
                    "持仓最大回撤": st.column_config.NumberColumn(format="%.2f%%"),
                    "持股天数": st.column_config.NumberColumn(format="%d"),
                },
            )
        else:
            st.info("当前无持仓")

        # ── 个股动量系统 ───────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📈 个股动量系统 V4.0")

        # 显示已配置的标的和数量
        momentum_tickers = cfg.get('tickers', [])
        num_tickers = len(momentum_tickers)
        if num_tickers > 0:
            tickers_str = ', '.join(momentum_tickers)
            st.caption(f"📋 已配置标的{num_tickers}个：{tickers_str}")

        min_market_cap = cfg.get('MIN_MARKET_CAP')
        max_positions = cfg.get('MAX_MOMENTUM_POSITIONS')
        cap_label = "私有配置" if min_market_cap in (None, "") else "已配置"
        pos_label = "私有配置" if max_positions in (None, "") else str(max_positions)
        st.caption(f"⚙️ 当前配置：市值过滤 {cap_label} | 趋势过滤 私有参数 | 最多持仓 {pos_label}")

        st.markdown("##### 【左侧甜点区】")
        dip_zone_view = build_dip_zone_view(cfg, now_et=now_et)
        if dip_zone_view.get("message"):
            st.caption(dip_zone_view["message"])
        dip_rows = dip_zone_view.get("rows", [])
        if dip_rows:
            df_dip = pd.DataFrame(dip_rows)
            row_count = len(df_dip)
            calc_height = min((row_count * 35) + 37, 900)
            st.dataframe(
                df_dip[DIP_ZONE_COLUMNS],
                width="stretch",
                hide_index=True,
                height=calc_height,
                column_config={
                    "建议仓位(%目标仓位)": st.column_config.NumberColumn(format="%.0f%%"),
                    "低吸分": st.column_config.NumberColumn(format="%d"),
                    "收盘价": st.column_config.NumberColumn(format="%.2f"),
                    "FAST_EMA": st.column_config.NumberColumn(format="%.2f"),
                    "SLOW_EMA": st.column_config.NumberColumn(format="%.2f"),
                    "MARKET_MA": st.column_config.NumberColumn(format="%.2f"),
                    "距SLOW_EMA": st.column_config.NumberColumn(format="%+.2f%%"),
                    "距MARKET_MA": st.column_config.NumberColumn(format="%+.2f%%"),
                    "距长期高点": st.column_config.NumberColumn(format="%+.2f%%"),
                    "RSI14": st.column_config.NumberColumn(format="%.1f"),
                    "ATR14": st.column_config.NumberColumn(format="%.2f"),
                    "ATR/Close": st.column_config.NumberColumn(format="%.2f%%"),
                    "60日成交额": st.column_config.NumberColumn(format="$%.0f"),
                    "REL_STRENGTH": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )
        else:
            st.info("当前无左侧甜点区候选")

        if momentum_result:
            st.markdown("##### 【决策信号】")
            decision_view = build_momentum_decision_view(momentum_result, config=cfg)
            if decision_view.get("limited_history"):
                logger.info(decision_view["limited_history"])

            if decision_view.get("status") == "no_scan":
                st.info("暂无扫描数据，请点击「立即扫描」")
            elif decision_view.get("status") == "no_eligible":
                st.info("无符合买入条件的标的")
            else:
                st.caption("📈 候选池排名（QQQ > MARKET_MA 且 FAST_EMA 上穿 SLOW_EMA，按 REL_STRENGTH 从高到低）")
                df_scored = pd.DataFrame(decision_view.get("rows", []))
                row_count = len(df_scored)
                calc_height = min((row_count * 35) + 37, 500)
                st.dataframe(
                    df_scored[MOMENTUM_DECISION_COLUMNS],
                    width="stretch",
                    hide_index=True,
                    height=calc_height,
                    column_config={
                        "收盘价": st.column_config.NumberColumn(format="%.4f"),
                        "FAST_EMA": st.column_config.NumberColumn(format="%.4f"),
                        "SLOW_EMA": st.column_config.NumberColumn(format="%.4f"),
                        "REL_STRENGTH": st.column_config.NumberColumn(format="%+.2f%%"),
                        "乖离率": st.column_config.NumberColumn(format="%+.2f%%"),
                        "风险分": st.column_config.NumberColumn(format="%d"),
                        "价格/FAST_EMA": st.column_config.NumberColumn(format="%+.2f%%"),
                        "价格/MARKET_MA": st.column_config.NumberColumn(format="%+.2f%%"),
                    },
                )

            st.markdown("##### 【待执行】")
            pending_operations = build_pending_operations(momentum_result)
            if pending_operations:
                st.warning("📌 系统生成以下待执行建议；自动调仓已暂停，请人工确认")
                df_pending = pd.DataFrame(pending_operations)
                row_count = len(df_pending)
                calc_height = min((row_count * 35) + 37, 500)
                st.dataframe(
                    df_pending[PENDING_OPERATION_COLUMNS],
                    width="stretch",
                    hide_index=True,
                    height=calc_height,
                )
            else:
                st.info("当前无待执行操作")
        else:
            st.info("点击 '立即扫描' 按钮以更新个股动量系统数据")

    # ── 执行日志（个股动量 + 大盘，独立于动量 UI）────────────────────────
    st.divider()
    col_log_title, col_log_btn = st.columns([4, 1])
    with col_log_title:
        st.subheader("📒 执行日志")
    with col_log_btn:
        st.write("")
        if st.button("⚙️ 调仓", key="adjust_position_btn"):
            st.session_state.dialog_open = True
    if st.session_state.get("dialog_open", False):
        trade_dialog()
    st.caption(
        "汇总个股动量（momentum_state）与大盘（position_state）买卖流水；"
        "调仓记录写入动量状态。"
    )
    execution_logs = load_merged_execution_logs()
    if execution_logs:
        rows = [
            {
                "时间": log.get("timestamp", ""),
                "策略": log.get("strategy", ""),
                "日志": log.get("log", ""),
            }
            for log in execution_logs
        ]
        df_logs = pd.DataFrame(rows)
        row_count = len(df_logs)
        calc_height = min((row_count * 35) + 37, 500)
        _render_execution_logs_table(df_logs, calc_height)
    else:
        st.info("暂无执行日志")

    st.divider()
    st.markdown("##### 🔗 工具链接")
    st.markdown(
        "- [MM US Economic Cycle Clock](https://en.macromicro.me/charts/64/econ-cycle)\n"
        "- [MM China Economic Cycle Clock](https://en.macromicro.me/charts/328/cn-econ-cycle)"
    )


# 如果直接运行此文件，则渲染仪表盘
if __name__ == "__main__":
    render_dashboard()
