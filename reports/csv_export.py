"""
CSV/JSON 결과 내보내기
"""

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def export_results(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    metrics: Dict[str, Any],
    signal_type: str,
    suffix: str = "",
):
    """equity curve, trade log, metrics summary 내보내기"""
    tag = f"{signal_type}{suffix}"

    # Equity curve CSV
    eq_path = RESULTS_DIR / f"equity_curve_{tag}.csv"
    equity_df.to_csv(eq_path, index=False)
    print(f"  Equity curve → {eq_path}")

    # Trade log CSV
    trades_path = RESULTS_DIR / f"trades_{tag}.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"  Trade log    → {trades_path}")

    # Metrics JSON
    metrics_path = RESULTS_DIR / f"metrics_{tag}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Metrics      → {metrics_path}")
