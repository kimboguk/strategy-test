#!/usr/bin/env python3
"""
Slot Fraction sweep — 슬롯 비중 (총자산 대비) 검증

baseline (Conservative + min_TV=1B + TP=20/SL=3) 고정. slot_fraction만 가변.

목적:
  - 슬롯 작게 → 분산 ↑, MDD ↓, Annual ↓
  - 슬롯 크게 → 집중 ↑, MDD ↑, Annual ↑
  - Calmar 기준 sweet spot 탐색
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
MIN_TV = 1_000_000_000   # 10억 (소자본 기준 baseline)
TOP_N = 3
TP_PCT = 0.20
SL_PCT = 0.03

# ── Sweep grid: slot fraction ─────────────────────────────────

SLOT_FRACTIONS = [
    0.03,   # 3% — 매우 분산 (~33 동시 보유)
    0.05,   # 5%
    0.08,   # 8%
    0.10,   # 10% (current baseline)
    0.12,   # 12%
    0.15,   # 15%
    0.20,   # 20% (~5 동시 보유)
    0.33,   # 33% (~3 동시 보유)
    0.50,   # 50% (~2 동시 보유)
]


def calmar(ann_pct, mdd_pct):
    if mdd_pct == 0 or pd.isna(mdd_pct):
        return float("inf")
    return ann_pct / abs(mdd_pct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2026-02-13")
    parser.add_argument("--out", type=str, default="sweep_slot_results.csv")
    args = parser.parse_args()

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("=" * 80)
    print("  Slot Fraction Sweep — 슬롯 비중 검증")
    print(f"  Fixed: ATH×{ATH_RATIO} | Vol×{VOL_RATIO} | min_TV={MIN_TV/1e9:.0f}B | "
          f"TP={TP_PCT*100:.0f}%/SL={SL_PCT*100:.0f}% | top={TOP_N}")
    print(f"  Period: {start_d} ~ {end_d}")
    print(f"  slot grid: {[f'{s*100:.0f}%' for s in SLOT_FRACTIONS]}")
    print("=" * 80)

    t0 = time.time()
    loaded = load_all_data(end_d=end_d, verbose=True)

    print(f"\n[Sweep 시작 — {len(SLOT_FRACTIONS)} 조합]")
    rows = []
    for slot in SLOT_FRACTIONS:
        t_start = time.time()
        sim, stats = run_with_params(
            loaded,
            ath_ratio=ATH_RATIO, vol_ratio=VOL_RATIO,
            min_tv=MIN_TV, top_n=TOP_N,
            tp_pct=TP_PCT, sl_pct=SL_PCT,
            slot_fraction=slot,
            initial_capital=INITIAL_CAPITAL,
            start_d=start_d, end_d=end_d,
            verbose=False,
        )
        ann = stats.get("ann_return_pct", 0)
        mdd = stats.get("mdd_pct", 0)
        max_concurrent = int(round(1.0 / slot)) if slot > 0 else 0
        row = {
            "slot_pct": slot * 100,
            "max_concurrent": max_concurrent,
            "trades": stats.get("trades", 0),
            "win_rate": stats.get("win_rate_pct", 0),
            "pf": stats.get("profit_factor", 0),
            "rrr": stats.get("rrr", 0),
            "ann_return": ann,
            "total_return": stats.get("total_return_pct", 0),
            "mdd": mdd,
            "calmar": calmar(ann, mdd),
            "final_equity_b": stats.get("final_equity", 0) / 1e9,
            "avg_holding": stats.get("avg_holding_days", 0),
        }
        rows.append(row)
        elapsed = time.time() - t_start
        print(f"  slot={slot*100:>5.1f}% (~{max_concurrent}개 동시) → "
              f"trades={row['trades']:>5} WR={row['win_rate']:>5.1f}% "
              f"PF={row['pf']:.2f} Ann={ann:+7.1f}% MDD={mdd:>6.1f}% "
              f"Calmar={row['calmar']:.2f} Final={row['final_equity_b']:>7.2f}B "
              f"({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*80}\n  Summary Table\n{'='*80}")
    cols = ["slot_pct", "max_concurrent", "trades", "win_rate", "pf",
            "ann_return", "mdd", "calmar", "final_equity_b"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Best by Calmar
    best = df.loc[df["calmar"].idxmax()]
    print(f"\n[최적 (Calmar 기준)]")
    print(f"  slot={best['slot_pct']:.1f}% "
          f"(~{int(best['max_concurrent'])} 동시) → "
          f"Calmar={best['calmar']:.2f} Ann={best['ann_return']:+.1f}% "
          f"MDD={best['mdd']:.1f}%")

    print(f"\n결과 저장: {args.out}")
    print(f"총 소요시간: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
