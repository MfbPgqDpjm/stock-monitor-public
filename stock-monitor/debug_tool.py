#!/usr/bin/env python3
"""
调试工具：支持命令行触发扫描和时间模拟，同时作为 Streamlit 入口
"""
import argparse
import sys
import os
import logging
from datetime import datetime
import pytz

# 确保日志文件生成
# 获取数据目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)
log_file = os.path.join(data_dir, "latest_scan.log")

# 设置日志格式
log_format = "%(asctime)s [%(levelname)s] %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# 创建文件处理器 (追加模式)
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 获取根 logger
root_logger = logging.getLogger()
# 清除现有的处理器
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
# 添加文件处理器和控制台处理器
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)  # 设置为 INFO 级别，打印重要日志

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入 state_manager
import state_manager
from strategy import run_full_scan
from state_manager import load_config, load_signals, update_and_save_signals, save_scan_status, detect_signal_changes, append_history
from notifier import notify_market_scan_signals

import streamlit as st

ET_TIMEZONE = pytz.timezone("America/New_York")

def parse_time_string(time_str):
    """解析时间字符串为 datetime 对象"""
    try:
        # 支持格式：2024-01-01 10:30:00
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        # 转换为美东时间
        return ET_TIMEZONE.localize(dt, is_dst=None)
    except Exception as e:
        print(f"时间格式错误: {e}")
        print("正确格式: 2024-01-01 10:30:00")
        sys.exit(1)

def manual_test(mock_time_str: str):
    """
    手动触发扫描模拟
    :param mock_time_str: 格式如 "2026-03-27 16:05"
    """
    et_tz = pytz.timezone('US/Eastern')
    try:
        mock_now = datetime.strptime(mock_time_str, "%Y-%m-%d %H:%M")
        mock_now_et = et_tz.localize(mock_now, is_dst=None)
    except Exception as e:
        print(f"[MOCK-SCAN] 时间格式错误: {e}")
        print("[MOCK-SCAN] 正确格式: 2026-03-27 16:05")
        return
    
    print(f"[MOCK-SCAN] 🚀 启动模拟扫描...")
    print(f"[MOCK-SCAN] ⏰ 模拟美东时间: {mock_now_et}")
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    env_path = os.path.join(project_root, ".env")
    print(f"[MOCK-SCAN] 读取配置文件: {env_path}")
    
    # 加载配置
    config = load_config()
    
    # 执行重构后的核心函数
    signals, _ = run_full_scan(config, now_et=mock_now_et)
    
    # 打印结果保存路径
    data_dir = os.path.join(script_dir, "data")
    signals_path = os.path.join(data_dir, "signals.json")
    history_path = os.path.join(data_dir, "history.json")
    status_path = os.path.join(data_dir, "scan_status.json")
    
    # 显示结果
    if not signals:
        print("[MOCK-SCAN] ❌ 未获取到任何信号数据")
        return
    
    # 先加载旧信号（在更新之前）
    old_signals = load_signals()
    
    # 更新本地数据
    updated_signals = update_and_save_signals(signals, scan_time=mock_now_et)
    
    # 检测信号变化（仅写入 history.json，与 Bark 推送解耦）
    changes = detect_signal_changes(old_signals, signals, override_time=mock_now_et)
    
    # 输出信号变化结果
    print(f"\n[MOCK-SCAN] 信号变化结果: 共检测到 {len(changes)} 个信号变更")
    if changes:
        print("[MOCK-SCAN] 信号变化详情:")
        for c in changes:
            print(f"[MOCK-SCAN]   - {c['ticker']}: {c['old_signal']} → {c['new_signal']}")
        append_history(changes)

    bark_key = config.get("bark_key", "")
    if bark_key:
        sent = notify_market_scan_signals(
            bark_key, updated_signals, scan_time=mock_now_et, is_lookback=False, is_debug=True, is_manual=True
        )
        print(f"[MOCK-SCAN] 市场信号快照 Bark 推送 {sent} 条")
    else:
        print("[MOCK-SCAN] Bark Key 未配置，跳过推送")
    
    # 保存扫描状态
    save_scan_status("手动测试模式扫描完成", scan_time=mock_now_et)
    
    logging.info("[MOCK-SCAN] ✅ 模拟任务执行完毕，请检查 Bark 推送和数据文件。")
    logging.info(f"[MOCK-SCAN] 结果保存路径:")
    logging.info(f"[MOCK-SCAN]   信号文件: {signals_path}")
    logging.info(f"[MOCK-SCAN]   历史文件: {history_path}")
    logging.info(f"[MOCK-SCAN]   状态文件: {status_path}")
    
    logging.info(f"\n[MOCK-SCAN] 共处理 {len(signals)} 个标的:")
    logging.info("[MOCK-SCAN] " + "-" * 60)
    
    for ticker, data in signals.items():
        if ticker == "last_update":
            continue
        signal = data.get("signal", "未知")
        error = data.get("error")
        
        if error:
            logging.info(f"[MOCK-SCAN] ⚠️ {ticker}: 错误 - {error}")
        else:
            # 构建指标数据字符串
            indicators = []
            if "close" in data:
                indicators.append(f"收盘价:{data['close']:.4f}")
            if "ma200" in data:
                indicators.append(f"MA200:{data['ma200']:.4f}")
            if "threshold" in data:
                indicators.append(f"阈值:{data['threshold']:.4f}")
            if "consecutive_days_above" in data:
                indicators.append(f"已站稳{data['consecutive_days_above']}天")
            if "ema20" in data:
                indicators.append(f"EMA20:{data['ema20']:.4f}")
            if "ema50" in data:
                indicators.append(f"EMA50:{data['ema50']:.4f}")
            if "high20" in data:
                indicators.append(f"HHV20:{data['high20']:.4f}")
            if "qqq_above_ma200" in data:
                indicators.append(f"QQQ>MA200:{'是' if data['qqq_above_ma200'] else '否'}")
            if "ema20_above_ema50" in data:
                indicators.append(f"EMA20>EMA50:{'是' if data['ema20_above_ema50'] else '否'}")
            if "close_at_20d_high" in data:
                indicators.append(f"价≥20日高:{'是' if data['close_at_20d_high'] else '否'}")
            if "close_prev_below_high20_prev" in data:
                indicators.append(f"价(t-1)<20日高(t-1):{'是' if data['close_prev_below_high20_prev'] else '否'}")
            
            # 获取数据日期
            data_date = data.get("data_date", "未知")
            
            # 在同一行输出所有指标数据
            indicator_str = " | ".join(indicators) if indicators else "无指标数据"
            logging.info(f"[MOCK-SCAN] ✅ {ticker}: {signal} | 数据日期:{data_date} | {indicator_str}")

def main():
    parser = argparse.ArgumentParser(description="调试工具：触发扫描和时间模拟")
    parser.add_argument("--time", type=str, help="模拟的美东时间 (格式: 2024-01-01 10:30:00)")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    parser.add_argument("--manual", type=str, help="手动测试模式，模拟指定时间 (格式: 2026-03-27 16:05)")
    
    args = parser.parse_args()
    
    # 手动测试模式
    if args.manual:
        manual_test(args.manual)
        return
    
    # 加载配置
    config = load_config()
    
    # 确定扫描时间
    if args.time:
        scan_time = parse_time_string(args.time)
        print(f"使用模拟时间: {scan_time.strftime('%Y-%m-%d %H:%M:%S ET')}")
    else:
        scan_time = datetime.now(ET_TIMEZONE)
        print(f"使用当前时间: {scan_time.strftime('%Y-%m-%d %H:%M:%S ET')}")
    
    print("\n开始执行扫描...")
    print("=" * 60)
    
    # 执行扫描
    signals, _ = run_full_scan(config, now_et=scan_time)
    
    print("\n扫描完成！")
    print("=" * 60)
    
    # 显示结果
    if not signals:
        print("❌ 未获取到任何信号数据")
        return
    
    # 先加载旧信号（在更新之前）
    old_signals = load_signals()
    
    # 更新本地数据
    updated_signals = update_and_save_signals(signals, scan_time=scan_time)
    
    # 检测信号变化（仅写入 history.json）
    changes = detect_signal_changes(old_signals, signals, override_time=scan_time)
    
    # 输出信号变化结果
    print(f"\n信号变化结果: 共检测到 {len(changes)} 个信号变更")
    if changes:
        print("信号变化详情:")
        for c in changes:
            print(f"  - {c['ticker']}: {c['old_signal']} → {c['new_signal']}")
        append_history(changes)

    bark_key = config.get("bark_key", "")
    if bark_key:
        sent = notify_market_scan_signals(
            bark_key, updated_signals, scan_time=scan_time, is_lookback=False, is_debug=True, is_manual=True
        )
        print(f"市场信号快照 Bark 推送 {sent} 条")
    else:
        print("Bark Key 未配置，跳过推送")
    
    # 保存扫描状态
    save_scan_status("命令行模式扫描完成", scan_time=scan_time)
    
    logging.info(f"\n共处理 {len(signals)} 个标的:")
    logging.info("-" * 60)
    
    for ticker, data in signals.items():
        if ticker == "last_update":
            continue
        signal = data.get("signal", "未知")
        error = data.get("error")
        if error:
            logging.info(f"  - {ticker}: 错误 - {error}")
        else:
            # 构建指标数据字符串
            indicators = []
            if "close" in data:
                indicators.append(f"收盘价:{data['close']:.4f}")
            if "ma200" in data:
                indicators.append(f"MA200:{data['ma200']:.4f}")
            if "threshold" in data:
                indicators.append(f"阈值:{data['threshold']:.4f}")
            if "consecutive_days_above" in data:
                indicators.append(f"已站稳{data['consecutive_days_above']}天")
            if "ema20" in data:
                indicators.append(f"EMA20:{data['ema20']:.4f}")
            if "ema50" in data:
                indicators.append(f"EMA50:{data['ema50']:.4f}")
            if "high20" in data:
                indicators.append(f"HHV20:{data['high20']:.4f}")
            if "qqq_above_ma200" in data:
                indicators.append(f"QQQ>MA200:{'是' if data['qqq_above_ma200'] else '否'}")
            if "ema20_above_ema50" in data:
                indicators.append(f"EMA20>EMA50:{'是' if data['ema20_above_ema50'] else '否'}")
            if "close_at_20d_high" in data:
                indicators.append(f"价≥20日高:{'是' if data['close_at_20d_high'] else '否'}")
            if "close_prev_below_high20_prev" in data:
                indicators.append(f"价(t-1)<20日高(t-1):{'是' if data['close_prev_below_high20_prev'] else '否'}")
            
            # 获取数据日期
            data_date = data.get("data_date", "未知")
            
            # 在同一行输出所有指标数据
            indicator_str = " | ".join(indicators) if indicators else "无指标数据"
            logging.info(f"  - {ticker}: {signal} | 数据日期:{data_date} | {indicator_str}")

def streamlit_app():
    """Streamlit 应用入口"""
    st.set_page_config(
        page_title="🔧 美股监控系统 - 调试工具",
        page_icon="🛠️",
        layout="wide",
    )
    
    st.title("🔧 美股监控系统 - Debug调试工具")
    # st.caption("时间坐标切换与模拟扫描工具")
    
    # 侧边栏时间选择器
    with st.sidebar:
        st.header("⏳ 时间坐标设置")
        col1, col2 = st.columns(2)
        with col1:
            d = st.date_input("日期", value=datetime.now(ET_TIMEZONE))
        with col2:
            # 创建一个时间对象，设置为 16:05
            from datetime import time
            t = st.time_input("时间", value=time(16, 5))
        
        target_now = ET_TIMEZONE.localize(datetime.combine(d, t))
        
        if st.button("重写观测点", use_container_width=True):
            with st.spinner(f"正在计算 {target_now.strftime('%Y-%m-%d %H:%M')} 的市场状态..."):
                # 执行该时间点的逻辑扫描
                config = load_config()
                signals, _ = run_full_scan(config, now_et=target_now)
                
                # 先加载旧信号（在更新之前）
                old_signals = load_signals()
                
                # 更新本地数据
                updated_signals = update_and_save_signals(signals, scan_time=target_now)
                
                # 检测信号变化（仅写入 history.json）
                changes = detect_signal_changes(old_signals, signals, override_time=target_now)
                
                # 输出信号变化结果
                logging.info(f"信号变化结果: 共检测到 {len(changes)} 个信号变更")
                if changes:
                    logging.info("信号变化详情:")
                    for c in changes:
                        logging.info(f"  - {c['ticker']}: {c['old_signal']} → {c['new_signal']}")
                    append_history(changes)

                bark_key = config.get("bark_key", "")
                if bark_key:
                    sent = notify_market_scan_signals(
                        bark_key,
                        updated_signals,
                        scan_time=target_now,
                        is_lookback=False,
                        is_debug=True,
                        is_manual=True,
                    )
                    logging.info(f"市场信号快照 Bark 推送 {sent} 条")
                else:
                    logging.info("Bark Key 未配置，跳过推送")
                # 保存扫描状态
                save_scan_status("手动时间模拟扫描完成", scan_time=target_now)
            st.success(f"✅ 观测点已更新到 {target_now.strftime('%Y-%m-%d %H:%M')}")
            st.rerun()
    
    # 导入并渲染仪表盘
    import app
    app.render_dashboard()

if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        main()
    else:
        # 作为 Streamlit 应用运行
        streamlit_app()
