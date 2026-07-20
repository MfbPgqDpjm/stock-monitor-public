# main.py
import time
import logging
from scheduler import start_scheduler

# 配置日志
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 启动独立后台扫描进程...")
    sched = start_scheduler()

    try:
        # 保持主线程不退出
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 正在停止后台进程...")