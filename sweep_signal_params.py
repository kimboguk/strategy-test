#!/usr/bin/env python3
"""
ATH ratio × Volume ratio × Min Trading Value sweep
TP/SL은 고정 (20%/5%). 신호 정의 파라미터만 탐색.

권장 in-sample / out-of-sample 분할:
  --is-end 2020-12-31    (in-sample 학습 종료)
  --oos-start 2021-01-01 (out-of-sample 시작)
  --oos-end 2026-02-13   (out-of-sample 종료)

데이터는 한 번만 로드, 모든 파라미터 조합 동일 캐시 재사용.
"""
import argparse
import time
from datetime import date, datetime

import pandas as pd

from run_ath_volume_breakout import (
    load_all_data,
    run_with_params,
    INITIAL_CAPITAL,
)


# ── 탐색 그리드 ─────────────────────────────────────────────────

ATH_RATIOS = [1.00, 1.02, 1.05]
VOL_RATIOS = [2.0, 3.0, 5.0]
MIN_TVS = [0, 1_000_000_000, 5_000_000_000, 10_000_000_000]   # 0/10억/50억/100억
TOP_NS = [3]


def calmar(ann_pct: float, mdd_pct: float) -> float:
    if mdd_pct == 0 or pd.isna(mdd_pct):
        return float("inf")
    return ann_pct / abs(mdd_pct)


def run_one(loaded, params, start_d, end_d):
    sim, stats = run_with_params(
        loaded,
        ath_ratio=params["ath_ratio"],
        vol_ratio=params["vol_ratio"],
        min_tv=params["min_tv"],
        top_n=params["top_n"],
        initial_capital=INITIAL_CAPITAL,
        start_d=start_d, end_d=end_d,
        verbose=False,
    )
    n = stats.get("trades", 0)
    return {
        **params,
        "trades": n,
        "win_rate": stats.get("win_rate_pct", 0),
        "pf": stats.get("profit_factor", 0),
        "rrr": stats.get("rrr", 0),
        "ann_return": stats.get("ann_return_pct", 0),
        "total_return": stats.get("total_return_pct", 0),
        "mdd": stats.get("mdd_pct", 0),
        "calmar": calmar(stats.get("ann_return_pct", 0),
                         stats.get("mdd_pct", 0)),
        "final_equity": stats.get("final_equity", 0),
    }


def sweep(loaded, start_d, end_d, label: str):
    rows = []
    grid = [
        {"ath_ratio": a, "vol_ratio": v, "min_tv": tv, "top_n": n}
        for a in ATH_RATIOS for v in VOL_RATIOS
        for tv in MIN_TVS for n in TOP_NS
    ]
    print(f"\n{'='*70}\n  {label}: {len(grid)} 조합\n{'='*70}")
    t0 = time.time()
    for i, p in enumerate(grid, 1):
        t_start = time.time()
        row = run_one(loaded, p, start_d, end_d)
        elapsed = time.time() - t_start
        print(f"  [{i:>3}/{len(grid)}] ath={p['ath_ratio']:.2f} "
              f"vol={p['vol_ratio']:.1f} tv={p['min_tv']/1e9:>4.0f}B "
              f"top={p['top_n']} → trades={row['trades']:>5} "
              f"PF={row['pf']:.2f} Ann={row['ann_return']:+6.1f}% "
              f"MDD={row['mdd']:>6.1f}% Calmar={row['calmar']:.2f} "
              f"({elapsed:.1f}s)")
        rows.append(row)
    print(f"\n총 {time.time()-t0:.1f}초")
    return pd.DataFrame(rows)


def print_top(df: pd.DataFrame, by: str, top_n: int = 10, label: str = ""):
    """주어진 키 기준 상위 N개 출력"""
    df = df[df["trades"] >= 100].copy()  # 통계 유의성 필터
    if df.empty:
        print(f"\n[{label}] (trade≥100 조건 만족 조합 없음)")
        return
    df = df.sort_values(by, ascending=False).head(top_n)
    print(f"\n[{label}] Top {top_n} by {by}")
    cols = ["ath_ratio", "vol_ratio", "min_tv", "top_n",
            "trades", "win_rate", "pf", "rrr",
            "ann_return", "mdd", "calmar"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-start", type=str, default="2014-01-01")
    parser.add_argument("--is-end", type=str, default="2020-12-31")
    parser.add_argument("--oos-start", type=str, default="2021-01-01")
    parser.add_argument("--oos-end", type=str, default="2026-02-13")
    parser.add_argument("--out", type=str, default="sweep_results.csv")
    parser.add_argument("--no-oos", action="store_true",
                        help="OOS 검증 생략 (in-sample만)")
    args = parser.parse_args()

    is_start = datetime.strptime(args.is_start, "%Y-%m-%d").date()
    is_end = datetime.strptime(args.is_end, "%Y-%m-%d").date()
    oos_start = datetime.strptime(args.oos_start, "%Y-%m-%d").date()
    oos_end = datetime.strptime(args.oos_end, "%Y-%m-%d").date()

    print("=" * 70)
    print("  Signal Parameter Sweep")
    print(f"  In-sample : {is_start} ~ {is_end}")
    if not args.no_oos:
        print(f"  Out-sample: {oos_start} ~ {oos_end}")
    print(f"  Grid: ATH×{len(ATH_RATIOS)}, Vol×{len(VOL_RATIOS)}, "
          f"TV×{len(MIN_TVS)}, TopN×{len(TOP_NS)} = "
          f"{len(ATH_RATIOS)*len(VOL_RATIOS)*len(MIN_TVS)*len(TOP_NS)} 조합")
    print("=" * 70)

    # 데이터 한 번만 로드 (OOS 끝까지)
    full_end = oos_end if not args.no_oos else is_end
    loaded = load_all_data(end_d=full_end, verbose=True)

    # In-sample sweep
    is_df = sweep(loaded, is_start, is_end, label="In-sample")
    is_df["phase"] = "IS"

    print_top(is_df, by="calmar", label="In-sample by Calmar")
    print_top(is_df, by="ann_return", label="In-sample by Ann.Return")
    print_top(is_df, by="pf", label="In-sample by PF")

    if not args.no_oos:
        # OOS는 같은 그리드 그대로
        oos_df = sweep(loaded, oos_start, oos_end, label="Out-of-sample")
        oos_df["phase"] = "OOS"

        print_top(oos_df, by="calmar", label="OOS by Calmar")

        # IS-OOS 비교 (top by IS Calmar)
        is_top = is_df[is_df["trades"] >= 100].sort_values("calmar", ascending=False).head(10)
        print("\n[IS Top10 → OOS 결과]")
        merge_cols = ["ath_ratio", "vol_ratio", "min_tv", "top_n"]
        compare = is_top.merge(oos_df, on=merge_cols, suffixes=("_IS", "_OOS"))
        cols = merge_cols + [
            "trades_IS", "pf_IS", "ann_return_IS", "mdd_IS", "calmar_IS",
            "trades_OOS", "pf_OOS", "ann_return_OOS", "mdd_OOS", "calmar_OOS",
        ]
        print(compare[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        all_df = pd.concat([is_df, oos_df], ignore_index=True)
    else:
        all_df = is_df

    all_df.to_csv(args.out, index=False)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
