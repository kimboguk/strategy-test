#!/usr/bin/env python3
"""
TOP_N × Slot Fraction joint sweep

가설: slot × TOP_N ≈ 100% (자본 완전 활용) 라인이 Calmar sweet spot.

다양한 (TOP_N, slot) 조합으로 검증:
  - 대각선: TOP_N=1/slot=100%, TOP_N=3/slot=33%, TOP_N=5/slot=20%, TOP_N=10/slot=10%
  - 비대각선: 자본 활용도 다양 (under/over)
  - heatmap으로 Calmar 분포 시각화
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

# ── Fixed params ──────────────────────────────────────────────

ATH_RATIO = 1.02
VOL_RATIO = 2.0
MIN_TV = 1_000_000_000
TP_PCT = 0.20
SL_PCT = 0.03

# ── Sweep grid ────────────────────────────────────────────────

TOP_N_VALUES = [1, 2, 3, 5, 7, 10]
SLOT_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.33, 0.50]


def calmar(ann_pct, mdd_pct):
    if mdd_pct == 0 or pd.isna(mdd_pct):
        return float("inf")
    return ann_pct / abs(mdd_pct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2026-02-13")
    parser.add_argument("--out", type=str, default="sweep_topn_slot_results.csv")
    args = parser.parse_args()

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()

    n_combos = len(TOP_N_VALUES) * len(SLOT_FRACTIONS)
    print("=" * 80)
    print("  TOP_N × Slot joint sweep")
    print(f"  Fixed: ATH×{ATH_RATIO} | Vol×{VOL_RATIO} | min_TV={MIN_TV/1e9:.0f}B | "
          f"TP={TP_PCT*100:.0f}%/SL={SL_PCT*100:.0f}%")
    print(f"  Period: {start_d} ~ {end_d}")
    print(f"  TOP_N: {TOP_N_VALUES}")
    print(f"  Slot:  {[f'{s*100:.0f}%' for s in SLOT_FRACTIONS]}")
    print(f"  총 {n_combos} 조합")
    print("=" * 80)

    t0 = time.time()
    loaded = load_all_data(end_d=end_d, verbose=True)

    print(f"\n[Sweep 시작]")
    rows = []
    i = 0
    for top_n in TOP_N_VALUES:
        for slot in SLOT_FRACTIONS:
            i += 1
            t_start = time.time()
            sim, stats = run_with_params(
                loaded,
                ath_ratio=ATH_RATIO, vol_ratio=VOL_RATIO,
                min_tv=MIN_TV, top_n=top_n,
                tp_pct=TP_PCT, sl_pct=SL_PCT,
                slot_fraction=slot,
                initial_capital=INITIAL_CAPITAL,
                start_d=start_d, end_d=end_d,
                verbose=False,
            )
            ann = stats.get("ann_return_pct", 0)
            mdd = stats.get("mdd_pct", 0)
            # 자본 활용도 = min(slot × top_n, 1)
            utilization = min(slot * top_n, 1.0) * 100
            row = {
                "top_n": top_n,
                "slot_pct": slot * 100,
                "utilization_pct": utilization,
                "trades": stats.get("trades", 0),
                "win_rate": stats.get("win_rate_pct", 0),
                "pf": stats.get("profit_factor", 0),
                "rrr": stats.get("rrr", 0),
                "ann_return": ann,
                "mdd": mdd,
                "calmar": calmar(ann, mdd),
                "final_equity_b": stats.get("final_equity", 0) / 1e9,
            }
            rows.append(row)
            elapsed = time.time() - t_start
            print(f"  [{i:>2}/{n_combos}] top={top_n} slot={slot*100:>4.1f}% "
                  f"(util={utilization:>5.1f}%) → "
                  f"trades={row['trades']:>5} PF={row['pf']:.2f} "
                  f"Ann={ann:+7.1f}% MDD={mdd:>6.1f}% "
                  f"Calmar={row['calmar']:.2f} ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # ── Heatmap (top_n × slot) ──
    print(f"\n{'='*80}\n  Calmar Heatmap (rows=TOP_N, cols=Slot)\n{'='*80}")
    pivot_calmar = df.pivot(index="top_n", columns="slot_pct", values="calmar")
    pivot_calmar.index = [f"top={t}" for t in pivot_calmar.index]
    pivot_calmar.columns = [f"slot={s:.0f}%" for s in pivot_calmar.columns]
    print(pivot_calmar.to_string(float_format=lambda x: f"{x:6.2f}"))

    print(f"\n{'='*80}\n  Annual Return Heatmap\n{'='*80}")
    pivot_ann = df.pivot(index="top_n", columns="slot_pct", values="ann_return")
    pivot_ann.index = [f"top={t}" for t in pivot_ann.index]
    pivot_ann.columns = [f"slot={s:.0f}%" for s in pivot_ann.columns]
    print(pivot_ann.to_string(float_format=lambda x: f"{x:+6.1f}"))

    print(f"\n{'='*80}\n  MDD Heatmap\n{'='*80}")
    pivot_mdd = df.pivot(index="top_n", columns="slot_pct", values="mdd")
    pivot_mdd.index = [f"top={t}" for t in pivot_mdd.index]
    pivot_mdd.columns = [f"slot={s:.0f}%" for s in pivot_mdd.columns]
    print(pivot_mdd.to_string(float_format=lambda x: f"{x:6.1f}"))

    # ── Top 10 by Calmar ──
    print(f"\n{'='*80}\n  Top 10 by Calmar\n{'='*80}")
    top = df[df["trades"] >= 100].sort_values("calmar", ascending=False).head(10)
    cols = ["top_n", "slot_pct", "utilization_pct", "trades",
            "win_rate", "pf", "ann_return", "mdd", "calmar",
            "final_equity_b"]
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ── 100% utilization 대각선 라인 ──
    print(f"\n{'='*80}\n  100% utilization line (slot × top_n ≈ 1)\n{'='*80}")
    diag = df[df["utilization_pct"] >= 95].sort_values("calmar", ascending=False)
    print(diag[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n결과 저장: {args.out}")
    print(f"총 소요시간: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
