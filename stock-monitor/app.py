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
    parse_madbulls_config,
    parse_rebalance_config,
    run_full_scan,
)
from position_manager import process_trade
from utils import get_signals_timestamp
from position_state import load_position_state
from ytd_performance import load_ytd_snapshot, refresh_ytd_performance

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
    ticker = st.text_input("标的代码", value="", placeholder="如：AAPL, 000001.SZ", key="dialog_ticker")
    
    # 价格输入
    price = st.number_input("交易价格", min_value=0.01, step=0.01, format="%.2f", key="dialog_price")
    
    # 数量输入（买入时需要）
    quantity = 1
    if trade_type == "买入":
        quantity = st.number_input("数量（股）", min_value=1, step=1, key="dialog_quantity")
    
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

def _fmt_position_pnl(entry_price: object, trading_close: object) -> str:
    """持仓收益率：基于手动 entry_price 与信号中 trading_close（交易标的）。"""
    try:
        ep = float(entry_price)
        tc = float(trading_close)
        if ep <= 0:
            return "—"
        return f"{(tc - ep) / ep * 100:+.2f}%"
    except (TypeError, ValueError):
        return "—"


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


def _render_ytd_progress_capsule(name: str, ytd_pct: float, meta: str = "") -> None:
    color = "#2e7d32" if ytd_pct >= 0 else "#d32f2f"
    sign = "+" if ytd_pct > 0 else ""
    fill_percent = min(abs(ytd_pct), 100.0)
    safe_name = html.escape(str(name))
    safe_meta = html.escape(str(meta)) if meta else ""

    html_code = (
        '<div style="display:flex;align-items:center;margin-bottom:10px;width:100%;max-width:760px;">'
        f'<div style="width:86px;font-weight:650;white-space:nowrap;">{safe_name}</div>'
        '<div style="flex:1;background-color:#f1f3f4;height:16px;display:flex;'
        'justify-content:flex-start;border-radius:10px;overflow:hidden;position:relative;">'
        f'<div style="width:{fill_percent:.1f}%;background-color:{color};height:16px;border-radius:0;"></div>'
        '</div>'
        f'<div style="width:82px;text-align:right;font-weight:800;color:{color};margin-left:10px;'
        f'font-variant-numeric:tabular-nums;">{sign}{ytd_pct:.2f}%</div>'
        f'<div style="width:72px;color:#777;font-size:12px;margin-left:8px;white-space:nowrap;">{safe_meta}</div>'
        '</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def _is_current_ytd_snapshot(snapshot: dict) -> bool:
    rows = snapshot.get("targets") if isinstance(snapshot, dict) else None
    if not isinstance(rows, list):
        return False

    expected_names = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "QQQ", "VOO", "TQQQ", "KOSPI"}
    row_names = {str(row.get("name", "")).upper() for row in rows if isinstance(row, dict)}
    return row_names == expected_names


def _render_ytd_capsules(snapshot: dict) -> None:
    """APP.py 页面展示：MAGA8/QQQ/VOO/TQQQ/KOSPI YTD 胶囊进度条。"""
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
        meta = ""
        if row.get("latest_date"):
            meta = str(row["latest_date"])
        _render_ytd_progress_capsule(name, ytd_pct, meta)


# 一眼执行表格 [实时信号] — 美股大盘/再平衡/疯牛 展示列（不含内部排序列）
MARKET_SIGNAL_TABLE_COLS = [
    "策略类型",
    "市场",
    "信号",
    "信号产生日期",
    "观测基准",
    "当前价",
    "关键指标",
    "信号情况",
    "数据模式",
]


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
        from momentum_scorer import MomentumScorer, format_rs_pct
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
                            refresh_ytd_performance(now_et=now, force_reload=True)
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
                    refresh_ytd_performance(force_reload=True)
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
    st.caption("大盘策略（MA200×系数 连续N日确认）| 个股策略（EMA+HHV，锚定 QQQ）| VOO（5/11月第3、4周再平衡）")


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

    ytd_snapshot = load_ytd_snapshot()
    if not _is_current_ytd_snapshot(ytd_snapshot):
        try:
            ytd_snapshot = refresh_ytd_performance(now_et=now_et)
        except Exception as e:
            logger.error(f"初始化 YTD 涨幅失败：{e}")
            ytd_snapshot = {}
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

        hhv_period_cfg = int(cfg.get("hhv_period", 20))
        individual_tickers = [t.strip().upper() for t in cfg.get("tickers", []) if t.strip()]
        us_configs = parse_market_configs(cfg.get("us_stocks", ""))
        rebalance_trade, rebalance_observe = parse_rebalance_config(cfg)
        market_tickers_us = [mc["buy_ticker"] for mc in us_configs]

        st.subheader(f"📋 一眼执行表格 [{signals_label}]")
        rows = []

        special_tickers = [mc["buy_ticker"] for mc in us_configs]
        special_tickers.append(rebalance_trade)

        madbulls_configs = parse_madbulls_config(cfg.get("madbulls", ""))
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
            signal_date_rb = rebalance_data.get("data_date") or "—"
            observe_label_rb = str(rb_bm)

            rows.append({
                "策略类型": "再平衡策略",
                "市场": rb_market,
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
                confirm_days_data = int(data.get("confirm_days", 2))
            except (TypeError, ValueError):
                confirm_days_data = 2
            days_cfg = int(mc_row.get("days", confirm_days_data)) if mc_row else confirm_days_data
            dd_pct_cfg = float(mc_row.get("drawdown_pct", 0.15)) if mc_row else 0.15
            coeff_f = float(data.get("coeff", mc_row.get("coeff", 1.0))) if mc_row else float(data.get("coeff", 1.0))

            consecutive = int(data.get("consecutive_days_above", 0) or 0)
            close_val = data.get("close")
            ma200_val = data.get("ma200")
            threshold_val = data.get("threshold")

            if isinstance(close_val, (int, float)):
                close_str = f"{currency_symbol}{float(close_val):.2f}"
            else:
                close_str = f"{currency_symbol}{close_val}"

            if isinstance(ma200_val, (int, float)) and isinstance(threshold_val, (int, float)):
                dd_str = f"{dd_pct_cfg * 100:.0f}%"
                key_indicators = (
                    f"MA200阈值: {currency_symbol}{float(threshold_val):.2f} "
                    f"({currency_symbol}{float(ma200_val):.2f} * {coeff_f}) | "
                    f"需站稳: {days_cfg}天 | 回撤止损: {dd_str}"
                )
            else:
                key_indicators = "N/A"

            standing_ok = consecutive >= days_cfg
            first_ok = bool(data.get("is_first_trigger"))
            # 与 strategy.check_benchmark_signal 一致：站上为 close>threshold；回撤止损为 drawdown>=drawdown_pct
            close_gt = (
                isinstance(close_val, (int, float))
                and isinstance(threshold_val, (int, float))
                and float(close_val) > float(threshold_val)
            )
            in_pos = bool(data.get("in_position"))
            dd = data.get("drawdown")
            if in_pos and isinstance(dd, (int, float)):
                dd_part = _mark_bool(float(dd) >= float(dd_pct_cfg))
            else:
                dd_part = "—"

            signal_situation = (
                f"已站稳: {consecutive}天 {_mark_bool(standing_ok)} | "
                f"首次触发: {_mark_bool(first_ok)} | "
                f"收盘价>阈值: {_mark_bool(close_gt)} | "
                f"触发回撤止损: {dd_part}"
            )

            bm = data.get("benchmark", "N/A")
            observe_label = str(bm)
            signal_date = data.get("data_date") or "—"

            score_val = data.get("score")
            score_str = f"{float(score_val):.4f}" if isinstance(score_val, (int, float)) else "N/A"

            rows.append({
                "策略类型": "美股大盘策略",
                "市场": market,
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

        # ── 2: A股疯牛策略 ───────────────────────────────────────────────────────
        for j, mc in enumerate(madbulls_configs):
            madbull_ticker = mc["ticker"]
            madbull_data = signals.get(madbull_ticker, {})
            if not madbull_data:
                continue

            madbull_signal = madbull_data.get("signal", "观望")
            market = madbull_data.get("market", "A股")
            is_index = 1

            close_val = madbull_data.get("close", "N/A")
            currency_symbol = "¥" if market == "A股" else "$"
            if isinstance(close_val, (int, float)):
                close_str = f"{currency_symbol}{float(close_val):.2f}"
            else:
                close_str = f"{close_val}"

            daily_gain_val = madbull_data.get("daily_gain")
            ema20_prev_val = madbull_data.get("ema20_prev")
            gain_cfg = float(mc.get("threshold", madbull_data.get("threshold", 3.6)))
            dd_cfg = madbull_data.get("drawdown_pct")
            if dd_cfg is None:
                dd_cfg = mc.get("drawdown_pct")
            dd_cur = madbull_data.get("drawdown")

            ki_parts = []
            if isinstance(ema20_prev_val, (int, float)):
                ki_parts.append(f"昨日EMA20: {currency_symbol}{float(ema20_prev_val):.2f}")
            ki_parts.append(f"涨幅阈值: {gain_cfg:g}%")
            if isinstance(dd_cfg, (int, float)) and 0 < float(dd_cfg) < 1:
                ki_parts.append(f"回撤止损: {float(dd_cfg) * 100:.0f}%")
            key_indicators_str = " | ".join(ki_parts) if ki_parts else "N/A"

            gain_ok = isinstance(daily_gain_val, (int, float)) and float(daily_gain_val) >= gain_cfg
            ema_ok = (
                isinstance(close_val, (int, float))
                and isinstance(ema20_prev_val, (int, float))
                and float(close_val) >= float(ema20_prev_val)
            )
            if isinstance(dd_cfg, (int, float)) and 0 < float(dd_cfg) < 1 and isinstance(dd_cur, (int, float)):
                dd_part = _mark_bool(float(dd_cur) >= float(dd_cfg))
            else:
                dd_part = "—"

            gain_pct_str = (
                f"{float(daily_gain_val):.2f}%"
                if isinstance(daily_gain_val, (int, float))
                else "—"
            )
            # 与 check_madbull_signal：涨幅 daily_gain>=threshold；EMA 破位为 close<ema20_prev；回撤触发 drawdown>=pct
            signal_situation_mb = (
                f"最新单日涨幅: {gain_pct_str} | "
                f"涨幅达标: {_mark_bool(gain_ok)} | "
                f"收盘价>=昨日EMA20: {_mark_bool(ema_ok)} | "
                f"触发回撤止损: {dd_part}"
            )

            benchmark = madbull_data.get("benchmark", "—")
            observe_label_mb = str(benchmark)
            signal_date_mb = madbull_data.get("data_date") or "—"

            rows.append({
                "策略类型": "A股疯牛策略",
                "市场": market,
                "信号": f"{SIGNAL_COLOR.get(madbull_signal, '⚪')} {madbull_signal}",
                "信号产生日期": signal_date_mb,
                "观测基准": observe_label_mb,
                "当前价": close_str,
                "关键指标": key_indicators_str,
                "信号情况": signal_situation_mb,
                "数据模式": madbull_data.get("scan_mode", "N/A"),
                "标的": madbull_ticker,
                "_table_block": 2,
                "_block_order": j,
                "is_index": is_index,
                "score": None,
                "signal_sort": 0 if madbull_signal == "买入" else 1 if madbull_signal == "关注" else 2 if madbull_signal == "卖出" else 3,
                "Score": "N/A",
            })

        if rows:
            df = pd.DataFrame(rows)

            # 第一个表格：美股大盘、再平衡、疯牛
            market_strategies = ["美股大盘策略", "再平衡策略", "A股疯牛策略"]
            df_market = df[df["策略类型"].isin(market_strategies)]

            # 第二个表格：包含美股个股策略
            df_individual = df[df["策略类型"] == "个股策略"]

            # 显示第一个表格（块顺序：再平衡 → 美股大盘 → A股疯牛）
            if not df_market.empty:
                st.markdown("##### 📈 美股大盘 / 再平衡 / 疯牛")
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

                # ── 大盘策略持仓展示（手动维护） ───────────────────────────────
                # 位置：紧贴“执行表格”下方，避免被动量模块的持仓面板淹没。
                try:
                    pos_state = load_position_state()
                    strategies = pos_state.get("strategies", {}) if isinstance(pos_state, dict) else {}
                except Exception:
                    strategies = {}

                market_positions = []
                for mc in us_configs:
                    tkr = mc.get("buy_ticker")
                    if not tkr:
                        continue

                    s = strategies.get(tkr, {}) if isinstance(strategies, dict) else {}
                    in_pos = bool(s.get("in_position")) if isinstance(s, dict) else False
                    if not in_pos:
                        continue

                    sig = signals.get(tkr, {}) if isinstance(signals, dict) else {}
                    bm = (sig.get("benchmark") or s.get("benchmark") or "")
                    entry_date = s.get("entry_date")
                    peak_high = s.get("peak_high")
                    peak_date = s.get("peak_date")

                    latest_close = sig.get("close")
                    latest_price_str = ""
                    try:
                        if isinstance(latest_close, (int, float)):
                            market_tag = sig.get("market")
                            currency = "$" if market_tag == "美股" else "¥" if market_tag == "A股" else ""
                            latest_price_str = f"{currency}{float(latest_close):.2f}" if currency else f"{float(latest_close):.2f}"
                    except Exception:
                        latest_price_str = ""

                    dd = sig.get("drawdown")
                    dd_str = f"{dd*100:.2f}%" if isinstance(dd, (int, float)) else ""

                    sell_reason = sig.get("sell_reason") or ""
                    status = f"🔴 待卖出 ({sell_reason})" if sig.get("signal") == "卖出" else "🟢 持仓中"

                    entry_price = s.get("entry_price")
                    pnl_str = _fmt_position_pnl(entry_price, sig.get("trading_close"))

                    peak_str = ""
                    try:
                        if peak_high is not None and str(peak_high).strip() not in ("", "None"):
                            peak_str = f"{float(peak_high):.2f}"
                    except Exception:
                        peak_str = str(peak_high) if peak_high is not None else ""

                    mkt = (sig.get("market") or "").strip() or "—"
                    sig_date = (sig.get("data_date") or "").strip() or "—"

                    market_positions.append({
                        "交易标的": tkr,
                        "市场": mkt,
                        "信号/状态": status,
                        "信号产生日期": sig_date,
                        "观测基准": bm,
                        "峰值High": peak_str,
                        "峰值日期": peak_date or "",
                        "当前价": latest_price_str,
                        "当前回撤": dd_str,
                        "入场日期": entry_date or "",
                        "持仓收益": pnl_str,
                    })

                mad_rows_cfg = parse_madbulls_config(cfg.get("madbulls", ""))
                for mc in mad_rows_cfg:
                    rt = mc.get("real_ticker")
                    mad_key = mc.get("ticker")
                    if not rt or not mad_key:
                        continue
                    mad_key_u = mad_key.upper()
                    s = strategies.get(mad_key_u, {}) if isinstance(strategies, dict) else {}
                    if not (isinstance(s, dict) and bool(s.get("in_position"))):
                        continue
                    sig = signals.get(mad_key_u, {}) if isinstance(signals, dict) else {}
                    bm = mc.get("benchmark") or (sig.get("benchmark") or s.get("benchmark") or "")
                    entry_date = s.get("entry_date")
                    peak_high = s.get("peak_high")
                    peak_date = s.get("peak_date")

                    latest_close = sig.get("close")
                    latest_price_str = ""
                    try:
                        if isinstance(latest_close, (int, float)):
                            market_tag = sig.get("market")
                            currency = "$" if market_tag == "美股" else "¥" if market_tag == "A股" else ""
                            latest_price_str = f"{currency}{float(latest_close):.2f}" if currency else f"{float(latest_close):.2f}"
                    except Exception:
                        latest_price_str = ""

                    dd = sig.get("drawdown")
                    dd_str = f"{dd*100:.2f}%" if isinstance(dd, (int, float)) else ""

                    sell_reason = sig.get("sell_reason") or ""
                    status = f"🔴 待卖出 ({sell_reason})" if sig.get("signal") == "卖出" else "🟢 持仓中"

                    entry_price = s.get("entry_price")
                    pnl_str = _fmt_position_pnl(entry_price, sig.get("trading_close"))

                    peak_str = ""
                    try:
                        if peak_high is not None and str(peak_high).strip() not in ("", "None"):
                            peak_str = f"{float(peak_high):.2f}"
                    except Exception:
                        peak_str = str(peak_high) if peak_high is not None else ""

                    mkt = (sig.get("market") or "").strip() or "—"
                    sig_date = (sig.get("data_date") or "").strip() or "—"

                    market_positions.append({
                        "交易标的": mad_key_u,
                        "市场": mkt,
                        "信号/状态": status,
                        "信号产生日期": sig_date,
                        "观测基准": bm,
                        "峰值High": peak_str,
                        "峰值日期": peak_date or "",
                        "当前价": latest_price_str,
                        "当前回撤": dd_str,
                        "入场日期": entry_date or "",
                        "持仓收益": pnl_str,
                    })

                if market_positions:
                    st.markdown("###### 📌 大盘持仓（手动）")
                    df_pos = pd.DataFrame(market_positions)
                    df_pos = df_pos[
                        [
                            "交易标的",
                            "市场",
                            "信号/状态",
                            "信号产生日期",
                            "观测基准",
                            "峰值High",
                            "峰值日期",
                            "当前价",
                            "当前回撤",
                            "入场日期",
                            "持仓收益",
                        ]
                    ]
                    st.dataframe(df_pos, width="stretch", hide_index=True)
            
            # 显示第二个表格：个股策略
            # if not df_individual.empty:
            #     st.subheader("📊 个股策略")
            #     df_individual_sorted = sort_individual_strategy(df_individual)
            #     display_columns = [
            #         "标的", "策略类型", "观测基准", "市场", "信号", "Score",
            #         "当前价", "关键指标", "达标情况", "数据模式", "备注"
            #     ]
            #     # 动态计算高度
            #     row_count = len(df_individual_sorted)
            #     calc_height = (row_count * 35) + 37  # 35px 为行高，40px 为表头留白
            #     # 设置高度上限
            #     calc_height = min(calc_height, 1000)
            #     # 应用高度参数，保留use_container_width=True
            #     st.dataframe(df_individual_sorted[display_columns], width="stretch", hide_index=True, height=calc_height)

        # ── 个股动量系统 ───────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📈 个股动量系统 V3.0")
        
        # 显示已配置的标的和数量
        momentum_tickers = cfg.get('tickers', [])
        num_tickers = len(momentum_tickers)
        if num_tickers > 0:
            tickers_str = ', '.join(momentum_tickers)
            st.caption(f"📋 已配置标的{num_tickers}个：{tickers_str}")
        
        hhv_window = cfg.get('HHV_WINDOW', 20)
        min_market_cap = cfg.get('MIN_MARKET_CAP', 1_000_000_000)
        max_positions = cfg.get('MAX_MOMENTUM_POSITIONS', 3)
        st.caption(f"⚙️ 当前配置：美元市值 ≥ ${min_market_cap/1_000_000_000:.1f}B | HHV {hhv_window} | 最多持仓 {max_positions} 只 | 等权")
        
        if momentum_result:
            # 第一部分：持仓信息面板
            st.markdown("##### 【第一部分：持仓监控】")
            
            position_audit = momentum_result.get("position_audit", {})
            
            if position_audit.get("status") == "无持仓":
                st.info("当前无持仓")
            elif position_audit.get("status") == "数据获取失败":
                st.error("数据获取失败，请检查网络连接")
            else:
                # 有持仓（支持多持仓）
                positions = position_audit.get("positions", [])
                if not isinstance(positions, list):
                    # 兼容旧格式（单持仓）
                    positions = [position_audit]
                
                rows = []
                for pos in positions:
                    bp = pos.get("buy_price", "")
                    lp = pos.get("latest_price", "")
                    ema50 = pos.get("ema50", "")
                    hd = pos.get("hold_days", "")
                    rows.append({
                        "标的": pos.get("ticker", ""),
                        "买入价": f"{float(bp):.2f}" if isinstance(bp, (int, float)) else str(bp or ""),
                        "买入日期": pos.get("buy_date", ""),
                        "最新价": f"{float(lp):.2f}" if isinstance(lp, (int, float)) else str(lp or ""),
                        "EMA50": f"{float(ema50):.2f}" if isinstance(ema50, (int, float)) else str(ema50 or ""),
                        "最新日期": pos.get("latest_date", ""),
                        "收益率": f"{pos.get('total_return', 0) * 100:.2f}%" if isinstance(pos.get('total_return'), (int, float)) else "",
                        "最高收益": f"{pos.get('max_return', 0) * 100:.2f}%" if isinstance(pos.get('max_return'), (int, float)) else "",
                        "最大回撤": f"{pos.get('max_drawdown', 0) * 100:.2f}%" if isinstance(pos.get('max_drawdown'), (int, float)) else "",
                        "持股天数": str(hd) if hd not in (None, "") else "",
                        "持仓状态": f"🔴 等待卖出 ({pos.get('sell_reason', '')})" if pos.get("action_plan") == "待卖出" else "🟢 继续持有"
                    })
                df_position = pd.DataFrame(rows)
                # 动态计算高度
                row_count = len(df_position)
                calc_height = (row_count * 35) + 37  # 35px 为行高，40px 为表头留白
                # 设置高度上限
                calc_height = min(calc_height, 500)
                # 应用高度参数，保留use_container_width=True
                st.dataframe(df_position, width="stretch", hide_index=True, height=calc_height)
            
            # 第二部分：RS120 排名和决策信号
            st.markdown("##### 【第二部分：决策信号】")
            buy_signal = momentum_result.get("buy_signal", {})
            
            scanned_stocks = buy_signal.get("scanned_stocks", [])
            eligible_stocks = [s for s in scanned_stocks if s.get("eligible")]
            eligible_stocks.sort(key=lambda s: -(s.get("rs120") or 0))

            if not scanned_stocks:
                st.info("暂无扫描数据，请点击「立即扫描」")
            elif not eligible_stocks:
                st.info("无符合买入条件的标的")
            else:
                st.caption("📈 候选池排名（20日新高突破，按 RS120 从高到低）")

                rows = []
                for i, stock in enumerate(eligible_stocks):
                    tags = []
                    if stock.get("is_position"):
                        tags.append("✅ 持仓中")
                    if i == 0 and stock.get("rs120") is not None:
                        tags.append("📊 RS120最高")
                    lp = stock.get("latest_price")
                    h1 = stock.get("hh20_prev")
                    rs120 = stock.get("rs120")
                    market_cap_usd = stock.get("market_cap_usd", stock.get("market_cap"))
                    market_cap_currency = stock.get("market_cap_currency", "USD")
                    rows.append({
                        "标的": stock.get("ticker", ""),
                        "日期": stock.get("latest_date", "") or "",
                        "收盘价": f"{lp:.4f}" if isinstance(lp, (int, float)) else "",
                        "HH20": f"{h1:.4f}" if isinstance(h1, (int, float)) else "",
                        "RS120": format_rs_pct(rs120) if isinstance(rs120, (int, float)) else "",
                        "市值(USD)": f"${market_cap_usd/1_000_000_000:.2f}B" if isinstance(market_cap_usd, (int, float)) else "",
                        "原币种": str(market_cap_currency or ""),
                        "状态": stock.get("reason", ""),
                        "标记": " | ".join(tags) if tags else "",
                    })
                df_scored = pd.DataFrame(rows)
                row_count = len(df_scored)
                calc_height = min((row_count * 35) + 37, 500)
                st.dataframe(df_scored, width="stretch", hide_index=True, height=calc_height)
            
            # 第三部分：待执行面板
            st.markdown("##### 【第三部分：待执行】")
            # 准备待执行操作数据
            pending_operations = []
            
            # 检查是否有待卖出的持仓（支持多持仓）
            positions = position_audit.get("positions", [])
            if isinstance(positions, list):
                for pos in positions:
                    if pos.get("action_plan") == "待卖出":
                        pending_operations.append({
                            "操作": "🔴 卖出",
                            "标的": pos.get("ticker", ""),
                            "信号日期": pos.get("latest_date", ""),
                            "RS120": "",
                            "原因": pos.get("sell_reason", ""),
                            "执行时间": "次日开盘"
                        })
            else:
                # 兼容旧的单持仓模式
                if position_audit.get("action_plan") == "待卖出":
                    pending_operations.append({
                        "操作": "🔴 卖出",
                        "标的": position_audit.get("ticker", ""),
                        "信号日期": position_audit.get("latest_date", ""),
                        "RS120": "",
                        "原因": position_audit.get("sell_reason", ""),
                        "执行时间": "次日开盘"
                    })
            
            # 检查是否有待买入信号
            pending_signals = momentum_result.get("pending_buy_signals")
            if not isinstance(pending_signals, list):
                pending_signal = momentum_result.get("pending_buy_signal", {})
                pending_signals = [pending_signal] if pending_signal and pending_signal.get("ticker") else []
            for pending_signal in pending_signals:
                pending_operations.append({
                    "操作": "🟢 买入",
                    "标的": pending_signal.get("ticker", ""),
                    "信号日期": pending_signal.get("signal_date", ""),
                    "RS120": format_rs_pct(pending_signal.get("rs120")) if isinstance(pending_signal.get("rs120"), (int, float)) else "",
                    "原因": pending_signal.get("reason", ""),
                    "执行时间": "次日开盘"
                })
            
            if pending_operations:
                st.warning("🔒 系统已锁定以下操作，将于次日开盘自动执行")
                df_pending = pd.DataFrame(pending_operations)
                # 动态计算高度
                row_count = len(df_pending)
                calc_height = (row_count * 35) + 37  # 35px 为行高，40px 为表头留白
                # 设置高度上限
                calc_height = min(calc_height, 500)
                # 应用高度参数，保留use_container_width=True
                st.dataframe(df_pending, width="stretch", hide_index=True, height=calc_height)
            else:
                st.info("当前无待执行操作")
        else:
            st.info("点击 '立即扫描' 按钮以更新个股动量系统数据")

    # ── 执行日志（个股动量 + 大盘/疯牛，独立于动量 UI）────────────────────────
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
        "汇总个股动量（momentum_state）与大盘/疯牛（position_state）买卖流水；"
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


# 如果直接运行此文件，则渲染仪表盘
if __name__ == "__main__":
    render_dashboard()
