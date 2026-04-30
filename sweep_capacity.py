#!/usr/bin/env python3
"""
Capacity sweep — 거래대금 필터 강도 검증

baseline (Conservative + slot=10% + TP=20/SL=3) 고정. min_tv만 가변.

목적:
  - 필터 강화 시 trade 수 / Annual / MDD / Calmar 변화 추적
  - 자본 규모 키워도 견딜 수 있는 capacity 영역 파악
  - 일별 평균 거래대금이 큰 종목만 선별 → 실전 슬리피지 감소
"""
import argparse
import time
from datetime import datetime

import pandas as pd

from run_ath_volume_breakout import (
    load_all_data,
    run_with_params,
    INITIAL_CAPITAL,
)

# ── Fixed params (Phase A baseline) ───────────────────────────

ATH_RATIO = 1.02
VOL_RATIO = 2.0
TOP_N = 3
TP_PCT = 0.20
SL_PCT = 0.03
SLOT_FRACTION = 0.10

# ── Sweep grid: 거래대금 하한 (KRW) ────────────────────────────

MIN_TVS = [
    0,                      # 필터 없음
    1_000_000_000,          # 10억
    3_000_000_000,          # 30억
    5_000_000_000,          # 50억
    10_000_000_000,         # 100억
    30_000_000_000,         # 300억
    50_000_000_000,         # 500억
    100_000_000_000,        # 1,000억 (1조)
]


def calmar(ann_pct: float, mdd_pct: float) -> float:
    if mdd_pct == 0 or pd.isna(mdd_pct):
        return float("inf")
    return ann_pct / abs(mdd_pct)


def fmt_tv(tv: float) -> str:
    if tv == 0:
        return "0"
    if tv >= 1e12:
        return f"{tv/1e12:.0f}T"
    if tv >= 1e9:
        return f"{tv/1e9:.0f}B"
    return f"{tv/1e6:.0f}M"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2026-02-13")
    parser.add_argument("--out", type=str, default="sweep_capacity_results.csv")
    args = parser.parse_args()

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("=" * 80)
    print("  Capacity Sweep — 거래대금 필터 강도 검증")
    print(f"  Fixed: ATH×{ATH_RATIO} | Vol×{VOL_RATIO} | "
          f"slot={SLOT_FRACTION*100:.0f}% | TP={TP_PCT*100:.0f}%/SL={SL_PCT*100:.0f}% | "
          f"top={TOP_N}")
    print(f"  Period: {start_d} ~ {end_d}")
    print(f"  min_TV: {[fmt_tv(v) for v in MIN_TVS]}")
    print("=" * 80)

    t0 = time.time()
    loaded = load_all_data(end_d=end_d, verbose=True)

    print(f"\n[Sweep 시작 — {len(MIN_TVS)} 조합]")
    rows = []
    for min_tv in MIN_TVS:
        t_start = time.time()
        sim, stats = run_with_params(
            loaded,
            ath_ratio=ATH_RATIO, vol_ratio=VOL_RATIO,
            min_tv=min_tv, top_n=TOP_N,
            tp_pct=TP_PCT, sl_pct=SL_PCT,
            slot_fraction=SLOT_FRACTION,
            initial_capital=INITIAL_CAPITAL,
            start_d=start_d, end_d=end_d,
            verbose=False,
        )
        ann = stats.get("ann_return_pct", 0)
        mdd = stats.get("mdd_pct", 0)
        row = {
            "min_tv_b": min_tv / 1e9,
            "trades": stats.get("trades", 0),
            "win_rate": stats.get("win_rate_pct", 0),
            "pf": stats.get("profit_factor", 0),
            "rrr": stats.get("rrr", 0),
            "ann_return": ann,
            "total_return": stats.get("total_return_pct", 0),
            "mdd": mdd,
            "calmar": calmar(ann, mdd),
            "final_equity_b": stats.get("final_equity", 0) / 1e9,
        }
        rows.append(row)
        elapsed = time.time() - t_start
        print(f"  min_TV={fmt_tv(min_tv):>6} → "
              f"trades={row['trades']:>5} WR={row['win_rate']:>5.1f}% "
              f"PF={row['pf']:.2f} Ann={ann:+7.1f}% MDD={mdd:>6.1f}% "
              f"Calmar={row['calmar']:.2f} Final={row['final_equity_b']:>7.2f}B "
              f"({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*80}\n  Summary Table\n{'='*80}")
    cols = ["min_tv_b", "trades", "win_rate", "pf", "rrr",
            "ann_return", "mdd", "calmar", "final_equity_b"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n결과 저장: {args.out}")
    print(f"총 소요시간: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
