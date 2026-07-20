"""Public-safe legacy scoring placeholder.

The private repository contains numeric filters and scoring formulas. Those
parameters are intentionally redacted from this public copy.
"""

from typing import Any, Dict

import pandas as pd


def calculate_score(ticker_data: pd.DataFrame, qqq_data: pd.DataFrame) -> Dict[str, Any]:
    return {
        "score": None,
        "reason": ["Public copy redacts private numeric scoring parameters."],
        "details": {},
    }
