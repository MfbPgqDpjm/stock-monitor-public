import logging
import threading
from datetime import datetime
from typing import List, Tuple, Optional
import pytz
import streamlit as st

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from strategy import run_full_scan, parse_rebalance_config
from state_manager import (
    load_config,
    load_signals,
    update_and_save_signals, # 使用新增的增量更新函数
    configured_signal_keys,
    detect_signal_changes,
    append_history,
    save_scan_status,
)
from notifier import (
    notify_market_scan_signals,
    notify_voo_rebalance,
    notify_momentum_combined,
)
# 抑制APScheduler的调试日志
logging.getLogger('apscheduler').setLevel(logging.WARNING)

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置为 INFO 级别，打印重要日志

_scheduler = None
_scheduler_lock = threading.Lock()
_scan_lock = threading.Lock() # 全局锁，确保同一秒只能有一个扫描在跑

ET_TIMEZONE = pytz.timezone("America/New_York")
SCAN_END_BLANK_LINES = 10

_TZ_PREFIX_MAP = {
    "ET": "America/New_York",
    "CN": "Asia/Shanghai",
}


def _log_scan_end_spacing() -> None:
    """扫描结束后追加空白行，方便从日志里区分新旧扫描块。"""
    logger.info("\n" * SCAN_END_BLANK_LINES)

def parse_scan_times(scan_time_str: str) -> List[Tuple[int, int, str, str]]:
    """解析配置字符串，例如 ET0935,CN1700"""
    results = []
    if not scan_time_str or not scan_time_str.strip():
        return results

    for item in scan_time_str.split(","):
        item = item.strip().upper()
        if not item: continue
        try:
            prefix, time_part = item[:2], item[2:]
            if prefix not in _TZ_PREFIX_MAP or len(time_part) < 4:
                continue
            hour, minute = int(time_part[:2]), int(time_part[2:4])
            results.append((hour, minute, _TZ_PREFIX_MAP[prefix], item))
        except Exception as e:
            logger.error(f"解析时间配置失败 {item}: {e}")
    return results


def manual_scan_blocked_by_schedule(
    now: Optional[datetime] = None,
    scan_time_str: Optional[str] = None,
    tolerance_min: int = 2,
) -> Tuple[bool, Optional[str]]:
    """
    手动扫描防撞：若当前时间落在任一定时扫描档位附近（默认±2分钟），则返回 (True, message)。
    仅做时段拦截，不检测 PROD 进程是否在线。
    """
    try:
        if now is None:
            now = datetime.now(ET_TIMEZONE)
        if scan_time_str is None:
            config = load_config()
            scan_time_str = config.get("scan_time", "")

        slots = parse_scan_times(scan_time_str or "")
        if not slots:
            return False, None

        for hour, minute, tz_str, label in slots:
            tz = pytz.timezone(tz_str)
            local_now = now.astimezone(tz)
            target_min = hour * 60 + minute
            now_min = local_now.hour * 60 + local_now.minute
            diff = abs(now_min - target_min)
            if diff <= max(0, int(tolerance_min)):
                msg = (
                    f"当前 {tz_str.split('/')[-1]} {local_now.strftime('%H:%M')} 与定时任务 {label} "
                    f"（{hour:02d}:{minute:02d}）时间重合(±{tolerance_min}min)，已跳过手动扫描。"
                    "请稍后再试，或先停止 PROD 调度器再进行验证。"
                )
                return True, msg

        return False, None
    except Exception as e:
        logger.warning(f"手动扫描时段拦截检查失败：{e}")
        return False, None

def execute_scan(manual_dt: Optional[datetime] = None, is_manual: bool = False):
    """
    全量扫描核心任务
    """
    # 第一步：使用锁机制确保同一秒只能有一个扫描在跑
    if not _scan_lock.acquire(blocking=False):
        logger.warning("⚠️ 上次扫描仍在进行中，跳过本次触发。")
        return {}, []

    try:
        # 计算 now_et
        now_obj = manual_dt if manual_dt else datetime.now(ET_TIMEZONE)
        
        logger.info(f"[SCAN] 🚀 开始全量扫描 @ {now_obj.strftime('%Y-%m-%d %H:%M:%S ET')}")
        save_scan_status("正在扫描数据...", scan_time=now_obj)

        config = load_config()
        
        # 1. 获取旧信号（用于对比变更）
        old_signals = load_signals()

        # 第三步：唯一调用 run_full_scan 获取 new_signals 和 data_cache
        # run_full_scan 内部已做网络熔断和错误拦截
        # 如果断网，new_results 将是空字典 {}
        new_results, data_cache = run_full_scan(config, now_et=now_obj)

        if not new_results:
            logger.error("[SCAN] 扫描未能获取任何有效数据（可能断网），流程终止。")
            save_scan_status("扫描终止：网络或基准数据异常", scan_time=now_obj)
            return {}, []

        # 3. 对比上一轮信号：仅用于 history.json 审计（推送不依赖此项）
        changes = detect_signal_changes(old_signals, new_results, override_time=now_obj)
        logger.info(f"对比完成，检测到 {len(changes)} 个信号变更（仅写入历史）")

        # 输出信号变化的详细信息
        if changes:
            logger.info("信号变化详情:")
            for c in changes:
                logger.info(f"  - {c['ticker']}: {c['old_signal']} → {c['new_signal']}")

        # 第四步：保存到文件
        # 核心：【增量更新】保存到文件
        # 这步确保了没扫到的股票保留旧值，扫到的更新新值
        final_signals = update_and_save_signals(
            new_results,
            scan_time=now_obj,
            active_signal_keys=configured_signal_keys(config),
        )

        # 5. 历史记录（仅在实际发生信号字符串变化时追加）
        if changes:
            append_history(changes)

        # 市场类 Bark：按当次快照，凡 is_market 且为买入/卖出即推送
        if config.get("bark_key"):
            sent = notify_market_scan_signals(
                config.get("bark_key"),
                final_signals,
                scan_time=now_obj,
                is_lookback=False,
                is_debug=False,
                is_manual=is_manual,
            )
            logger.info(f"市场信号快照 Bark 推送 {sent} 条")

        # 特殊提醒：再平衡
        reb_trade, _ = parse_rebalance_config(config)
        rebalance_data = final_signals.get(reb_trade, {})
        if rebalance_data.get("signal") == "再平衡提醒":
            ok = notify_voo_rebalance(config.get("bark_key"), scan_time=now_obj, rebalance_ticker=reb_trade)
            logger.info(f"{reb_trade} 再平衡 Bark 推送 {'成功' if ok else '失败'}")

        # 6. 更新最终状态
        success_count = len([k for k, v in new_results.items() if k != "last_update" and isinstance(v, dict) and v.get("signal") != "ERROR"])
        error_count = len([k for k, v in new_results.items() if k != "last_update" and isinstance(v, dict) and v.get("signal") == "ERROR"])
        status_msg = f"完成: 成功 {success_count} 只" + (f", 失败 {error_count} 只" if error_count > 0 else "")
        save_scan_status(status_msg, scan_time=now_obj)
        logger.info(f"[SCAN] 全量扫描完成 - {status_msg}")
        
        # 运行动量评分系统
        try:
            from momentum_scorer import MomentumScorer
            # 加载信号数据，避免重复扫描
            signals = load_signals()
            # 移除 last_update 字段
            if isinstance(signals, dict) and 'last_update' in signals:
                signals = {k: v for k, v in signals.items() if k != 'last_update'}
            scorer = MomentumScorer(config, signals=signals, data_cache=data_cache)
            momentum_result = scorer.run(data_cache=data_cache)
            logger.info("动量评分系统运行完成")

            # 动量系统 Bark 推送 - 合并推送所有信息
            if config.get("bark_key") and momentum_result:
                position_audit = momentum_result.get("position_audit", {})
                buy_signal = momentum_result.get("buy_signal", {})
                pending_buy_signals = momentum_result.get("pending_buy_signals", [])
                max_positions = int(config.get("MAX_MOMENTUM_POSITIONS", 3))

                ok = notify_momentum_combined(
                    config.get("bark_key"),
                    position_audit,
                    buy_signal,
                    scan_time=now_obj,
                    pending_buy_signals=pending_buy_signals,
                    max_positions=max_positions,
                )
                logger.info(f"动量系统 Bark 合并推送 {'成功' if ok else '失败'}")
        except Exception as e:
            logger.error(f"动量评分系统执行失败：{str(e)}")
        
        return new_results, changes

    except Exception as e:
        logger.error(f"扫描异常失败: {e}", exc_info=True)
        save_scan_status(f"异常失败: {str(e)}", scan_time=now_obj)
        return {}, []
    finally:
        try:
            _log_scan_end_spacing()
        finally:
            # 释放锁
            _scan_lock.release()

@st.cache_resource
def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = BackgroundScheduler(
                timezone=ET_TIMEZONE,
                job_defaults={'max_instances': 1, 'coalesce': True}
            )

        config = load_config()
        scan_time_str = config.get("scan_time", "ET0935,ET1605")
        time_configs = parse_scan_times(scan_time_str)

        if not time_configs:
            logger.warning("scan_time 解析结果为空，使用默认 ET0935, ET1605")
            time_configs = [
                (9, 35, "America/New_York", "ET0935"),
                (16, 5, "America/New_York", "ET1605"),
            ]

        # 检查现有的扫描任务
        existing_jobs = {job.id: job for job in _scheduler.get_jobs() if job.id.startswith("scan_")}

        for i, (hour, minute, tz_str, label) in enumerate(time_configs):
            job_id = f"scan_{i}"
            tz = pytz.timezone(tz_str)
            job_name = f"定时扫描 {label} ({tz_str.split('/')[-1]})"
            
            # 检查任务是否已存在
            if job_id in existing_jobs:
                # 检查任务配置是否相同
                existing_job = existing_jobs[job_id]
                # 比较触发器配置
                trigger = existing_job.trigger
                if (isinstance(trigger, CronTrigger) and 
                    trigger.fields[4].__dict__.get('hour') == hour and 
                    trigger.fields[5].__dict__.get('minute') == minute and 
                    existing_job.name == job_name):
                    # 配置没变，跳过添加
                    logger.debug(f"任务 {job_name} 已存在且配置未变，跳过注册")
                    del existing_jobs[job_id]  # 标记为已处理
                    continue
                else:
                    # 配置已变，删除旧任务
                    _scheduler.remove_job(job_id)
                    logger.info(f"任务 {job_name} 配置已变，重新注册")
            
            # 添加新任务
            _scheduler.add_job(
                execute_scan,
                CronTrigger(hour=hour, minute=minute, timezone=tz),
                id=job_id,
                replace_existing=True,
                name=job_name,
                # 定时任务，manual_dt传None
                kwargs={"manual_dt": None}
            )
            logger.info(f"已注册: {job_name}")

        # 删除多余的任务
        for job_id in existing_jobs:
            _scheduler.remove_job(job_id)
            logger.info(f"删除多余任务: {existing_jobs[job_id].name}")

        logger.info(f"调度器已配置 {len(time_configs)} 个定时任务")
    return _scheduler


def start_scheduler():
    sched = get_scheduler()
    if not sched.running:
        sched.start()
        logger.info("📅 调度器已启动")
    else:
        logger.info("📅 调度器已在运行中，跳过重复启动") # 确保不会起两个
    return sched


def stop_scheduler():
    global _scheduler
    with _scheduler_lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None
            logger.info("调度器已停止")


def get_scheduler_status():
    """
    获取调度器状态及其任务列表
    修复了 NameError: name 'scheduler' is not defined
    修复了 AttributeError: 'Job' object has no attribute 'next_run_time'
    """
    # 关键点：使用 get_scheduler() 确保获取到正确的单例对象 _scheduler
    sched = get_scheduler()
    
    jobs = []
    # 检查调度器是否正在运行
    is_running = sched.running if sched else False
    
    if sched:
        for job in sched.get_jobs():
            # 兼容性处理：优先尝试获取 next_run_time
            # getattr(obj, attr, default) 能安全处理 3.x/4.x 版本差异
            next_run = getattr(job, 'next_run_time', None)
            
            if next_run:
                # 统一显示为美东时间或本地字符串
                next_run_str = next_run.strftime('%Y-%m-%d %H:%M:%S')
            else:
                next_run_str = "已暂停或待定"
                
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run_str
            })
            
    return {
        "running": is_running,
        "jobs": jobs
    }
