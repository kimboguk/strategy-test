#!/usr/bin/env python3
"""
TP/SL sweep — 보수적/공격적 두 신호 config 각각 16조합

신호 config는 sweep_signal_params.py 결과에서 선정:
  - Conservative: ath=1.02, vol=2.0, tv=1B
  - Aggressive  : ath=1.05, vol=2.0, tv=1B

전 기간 (2014-01-01 ~ 2026-02-13) 백테스트.
신호 계산은 config당 1회 → TP/SL만 바꿔서 시뮬 16번 재실행.
"""
import argparse
import time
from datetime import date, datetime

import pandas as pd

from run_ath_volume_breakout import (
    load_all_data,
    precompute_signals,
    run_with_params,
    INITIAL_CAPITAL,
)


SIGNAL_CONFIGS = [
    {"name": "conservative",
     "ath_ratio": 1.02, "vol_ratio": 2.0, "min_tv": 1_000_000_000,
     "top_n": 3},
    {"name": "aggressive",
     "ath_ratio": 1.05, "vol_ratio": 2.0, "min_tv": 1_000_000_000,
     "top_n": 3},
]

TP_PCTS = [0.15, 0.20, 0.25, 0.30]
SL_PCTS = [0.03, 0.05, 0.07, 0.10]


def calmar(ann_pct: float, mdd_pct: float) -> float:
    if mdd_pct == 0 or pd.isna(mdd_pct):
        return float("inf")
    return ann_pct / abs(mdd_pct)


def run_grid(loaded, sig_cfg, start_d, end_d):
    print(f"\n{'='*70}\n  [{sig_cfg['name']}] "
          f"ath={sig_cfg['ath_ratio']} vol={sig_cfg['vol_ratio']} "
          f"tv={sig_cfg['min_tv']/1e9:.0f}B"
          f"\n{'='*70}")

    # 신호는 1회만 계산 (TP/SL은 시뮬 단계에서 적용되므로)
    print("  signal precompute ...", end=" ", flush=True)
    t_sig = time.time()
    signals = precompute_signals(
        loaded["ticker_data"],
        ath_ratio=sig_cfg["ath_ratio"],
        vol_ratio=sig_cfg["vol_ratio"],
        min_tv=sig_cfg["min_tv"],
        verbose=False,
    )
    n_sig = sum(len(v) for v in signals.values())
    print(f"{n_sig:,} events ({time.time()-t_sig:.1f}s)")

    rows = []
    for tp in TP_PCTS:
        for sl in SL_PCTS:
            t0 = time.time()
            sim, stats = run_with_params(
                loaded,
                ath_ratio=sig_cfg["ath_ratio"],
                vol_ratio=sig_cfg["vol_ratio"],
                min_tv=sig_cfg["min_tv"],
                top_n=sig_cfg["top_n"],
                tp_pct=tp, sl_pct=sl,
                initial_capital=INITIAL_CAPITAL,
                start_d=start_d, end_d=end_d,
                verbose=False,
                cached_signals=signals,
            )
            ann = stats.get("ann_return_pct", 0)
            mdd = stats.get("mdd_pct", 0)
            row = {
                "config": sig_cfg["name"],
                "tp": tp, "sl": sl,
                "trades": stats.get("trades", 0),
                "win_rate": stats.get("win_rate_pct", 0),
                "pf": stats.get("profit_factor", 0),
                "rrr": stats.get("rrr", 0),
                "ann_return": ann,
                "total_return": stats.get("total_return_pct", 0),
                "mdd": mdd,
                "calmar": calmar(ann, mdd),
                "final_equity": stats.get("final_equity", 0),
            }
            rows.append(row)
            print(f"  TP={tp*100:>4.0f}% SL={sl*100:>4.0f}% → "
                  f"trades={row['trades']:>5} WR={row['win_rate']:>5.1f}% "
                  f"PF={row['pf']:.2f} RRR={row['rrr']:.2f} "
                  f"Ann={ann:+7.1f}% MDD={mdd:>6.1f}% "
                  f"Calmar={row['calmar']:.2f} ({time.time()-t0:.1f}s)")
    return rows


def print_heatmap(df: pd.DataFrame, metric: str, label: str):
    """TP × SL heatmap"""
    pivot = df.pivot(index="tp", columns="sl", values=metric)
    pivot.index = [f"TP={x*100:.0f}%" for x in pivot.index]
    pivot.columns = [f"SL={x*100:.0f}%" for x in pivot.columns]
    print(f"\n  [{label}] {metric}")
    print(pivot.to_string(float_format=lambda x: f"{x:7.2f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2026-02-13")
    parser.add_argument("--out", type=str, default="sweep_tpsl_results.csv")
    args = parser.parse_args()

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("=" * 70)
    print("  TP/SL Sweep")
    print(f"  Period: {start_d} ~ {end_d}")
    print(f"  TP grid : {[f'{t*100:.0f}%' for t in TP_PCTS]}")
    print(f"  SL grid : {[f'{s*100:.0f}%' for s in SL_PCTS]}")
    print(f"  Configs : {len(SIGNAL_CONFIGS)} × {len(TP_PCTS)*len(SL_PCTS)} = "
          f"{len(SIGNAL_CONFIGS)*len(TP_PCTS)*len(SL_PCTS)} runs")
    print("=" * 70)

    t0 = time.time()
    loaded = load_all_data(end_d=end_d, verbose=True)

    all_rows = []
    for cfg in SIGNAL_CONFIGS:
        rows = run_grid(loaded, cfg, start_d, end_d)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)

    # config별 heatmap
    for cfg in SIGNAL_CONFIGS:
        sub = df[df["config"] == cfg["name"]]
        print(f"\n{'='*70}\n  [{cfg['name'].upper()}] heatmap\n{'='*70}")
        print_heatmap(sub, "calmar", "Calmar")
        print_heatmap(sub, "ann_return", "Annual Return %")
        print_heatmap(sub, "mdd", "MDD %")
        print_heatmap(sub, "pf", "PF")

    # config 비교 best
    print(f"\n{'='*70}\n  Config 비교 (Top 5 by Calmar)\n{'='*70}")
    top = df[df["trades"] >= 100].sort_values("calmar", ascending=False).head(10)
    cols = ["config", "tp", "sl", "trades", "win_rate", "pf", "rrr",
            "ann_return", "mdd", "calmar"]
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n결과 저장: {args.out}")
    print(f"총 소요시간: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
