#!/usr/bin/env python3
"""Public-safe dip-zone helper.

Private liquidity thresholds, distance bands, score weights, priority ticker
lists, and position-sizing parameters are redacted from this public copy.
"""

from datetime import datetime
from typing import Any, Dict, Optional


DIP_ZONE_COLUMNS = [
    "标的",
    "观测标的",
    "动作",
    "建议仓位(%目标仓位)",
    "低吸分",
    "日期",
    "触发条件",
]


def build_dip_zone_view(config: Dict[str, Any], now_et: Optional[datetime] = None) -> Dict[str, Any]:
    return {
        "rows": [],
        "candidates": [],
        "error": "Public copy redacts private dip-zone parameters.",
        "as_of": now_et.isoformat() if now_et else None,
    }
