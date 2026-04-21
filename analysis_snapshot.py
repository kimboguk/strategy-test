#!/usr/bin/env python3
"""
Snapshot 기반 신고가 돌파 vs 수익률 상위 종목 — Forward Performance 분석

특정 시점에서:
  A) 가격 기반: 종가가 ATH/52주 신고가에 가까운 상위 100종목
  B) 수익률 기반: daily_returns_snapshot 연환산 수익률 상위 100종목

→ 2주/1개월 후 성과 비교

실행:
  APP_ENV=dev conda run -n financial_modeling python -m strategy_backtest.analysis_snapshot
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import date, timedelta

from strategy_backtest.config.settings import DATABASE_URL, KRX_MARKETS

engine = create_engine(DATABASE_URL)
TOP_N = 100
FORWARD_WINDOWS = {'2주': 10, '1개월': 21}

# ============================================================================
# 분석 기준일 (거래일)
# ============================================================================
ANALYSIS_DATES = {
    '2020': date(2020, 1, 2),
    '2025': date(2025, 1, 2),
}


def get_filtered_product_ids():
    """financial_modeling 필터링 종목 (asset_quality.is_selected=TRUE & KRX)"""
    markets = ','.join(f"'{m}'" for m in KRX_MARKETS)
    query = f"""
        SELECT aq.product_id, p.ticker, p.name
        FROM asset_quality aq
        JOIN products p ON aq.product_id = p.product_id
        WHERE aq.is_selected = TRUE AND p.market IN ({markets})
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_snapshot_returns(snapshot_date: date, product_ids: list, lookback: int = 252):
    """daily_returns_snapshot에서 연환산 수익률 조회"""
    ids_str = ','.join(str(i) for i in product_ids)
    query = f"""
        SELECT product_id, ticker,
               annual_mean_return::float AS ann_return,
               annual_volatility::float AS ann_vol,
               sharpe_ratio::float AS sharpe,
               sortino_ratio::float AS sortino
        FROM daily_returns_snapshot
        WHERE snapshot_date = '{snapshot_date}'
          AND lookback_days = {lookback}
          AND product_id IN ({ids_str})
        ORDER BY annual_mean_return DESC NULLS LAST
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_price_history(product_ids: list, start_date: date, end_date: date):
    """OHLCV 가격 데이터 조회"""
    ids_str = ','.join(str(i) for i in product_ids)
    query = f"""
        SELECT product_id, trade_date,
               high::float, close::float
        FROM market_data
        WHERE product_id IN ({ids_str})
          AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY product_id, trade_date
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn, parse_dates=['trade_date'])


def compute_high_proximity(prices_df: pd.DataFrame, ref_date: date):
    """
    기준일 종가가 ATH/52주 신고가에 얼마나 가까운지 계산
    proximity = close / high_max  (1.0 = 정확히 신고가)
    """
    results = []
    for pid, grp in prices_df.groupby('product_id'):
        grp = grp.sort_values('trade_date')
        # 기준일 이전 데이터만
        mask = grp['trade_date'].dt.date <= ref_date
        hist = grp[mask]
        if len(hist) < 252:
            continue

        close_on_date = hist.iloc[-1]['close']
        if close_on_date <= 0 or np.isnan(close_on_date):
            continue

        # ATH (전체 최고가)
        ath = hist['high'].max()
        ath_proximity = close_on_date / ath if ath > 0 else 0

        # 52주 신고가 (최근 252일)
        h52 = hist.tail(252)['high'].max()
        h52_proximity = close_on_date / h52 if h52 > 0 else 0

        results.append({
            'product_id': pid,
            'close': close_on_date,
            'ath': ath,
            'ath_proximity': ath_proximity,
            'high_52w': h52,
            'h52_proximity': h52_proximity,
        })

    return pd.DataFrame(results)


def compute_forward_returns(prices_df: pd.DataFrame, ref_date: date, product_ids: list):
    """기준일 종가 대비 N일 후 종가 수익률"""
    results = []
    for pid in product_ids:
        grp = prices_df[prices_df['product_id'] == pid].sort_values('trade_date')
        future = grp[grp['trade_date'].dt.date > ref_date]
        past = grp[grp['trade_date'].dt.date <= ref_date]
        if len(past) == 0 or len(future) == 0:
            continue

        base_close = past.iloc[-1]['close']
        if base_close <= 0 or np.isnan(base_close):
            continue

        row = {'product_id': pid, 'base_close': base_close}
        for label, days in FORWARD_WINDOWS.items():
            if len(future) >= days:
                fwd_close = future.iloc[days - 1]['close']
                row[f'fwd_return_{label}'] = (fwd_close / base_close - 1) * 100
            else:
                row[f'fwd_return_{label}'] = np.nan
        results.append(row)

    return pd.DataFrame(results)


def analyze_one_date(label: str, ref_date: date):
    """단일 기준일 분석"""
    print(f"\n{'='*70}")
    print(f"  분석 기준일: {label} ({ref_date})")
    print(f"{'='*70}")

    # 1) 필터링 종목
    filtered = get_filtered_product_ids()
    pids = filtered['product_id'].tolist()
    ticker_map = dict(zip(filtered['product_id'], filtered['ticker']))
    name_map = dict(zip(filtered['product_id'], filtered['name']))
    print(f"\n  필터링 종목: {len(pids)}")

    # 2) snapshot 수익률 (252일 lookback)
    snap = get_snapshot_returns(ref_date, pids, lookback=252)
    print(f"  Snapshot 종목: {len(snap)}")

    # 3) 가격 히스토리 (기준일 -2년 ~ +2개월)
    price_start = ref_date - timedelta(days=800)  # ATH 계산용 충분한 히스토리
    price_end = ref_date + timedelta(days=60)
    prices = get_price_history(pids, price_start, price_end)
    print(f"  가격 데이터: {len(prices):,} rows")

    # 4) 신고가 근접도
    proximity = compute_high_proximity(prices, ref_date)
    proximity = proximity.merge(
        filtered[['product_id', 'ticker']], on='product_id', how='left'
    )
    print(f"  신고가 근접도 계산: {len(proximity)} 종목")

    # 5) Forward returns
    fwd = compute_forward_returns(prices, ref_date, pids)

    # === 그룹 구성 ===
    groups = {}

    # A-1) ATH 근접도 상위 100
    ath_top = proximity.nlargest(TOP_N, 'ath_proximity')
    groups['ATH근접_Top100'] = set(ath_top['product_id'])

    # A-2) 52주 신고가 근접도 상위 100
    h52_top = proximity.nlargest(TOP_N, 'h52_proximity')
    groups['52W근접_Top100'] = set(h52_top['product_id'])

    # B) 연환산 수익률 상위 100
    ret_top = snap.nlargest(TOP_N, 'ann_return')
    groups['수익률_Top100'] = set(ret_top['product_id'])

    # C) Sortino 상위 100
    sortino_top = snap.dropna(subset=['sortino']).nlargest(TOP_N, 'sortino')
    groups['Sortino_Top100'] = set(sortino_top['product_id'])

    # === 그룹 간 겹침 ===
    print(f"\n{'─'*50}")
    print(f"  그룹 간 겹침 (종목 수)")
    print(f"{'─'*50}")
    group_names = list(groups.keys())
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i+1:]:
            overlap = len(groups[g1] & groups[g2])
            print(f"  {g1} ∩ {g2}: {overlap}")

    # === Forward Performance 비교 ===
    print(f"\n{'─'*50}")
    print(f"  Forward Performance 비교")
    print(f"{'─'*50}")

    header = f"  {'그룹':<20}"
    for wlabel in FORWARD_WINDOWS:
        header += f" {'평균'+wlabel:>10} {'중위'+wlabel:>10} {'승률'+wlabel:>10}"
    print(header)
    print(f"  {'─'*80}")

    for gname, pids_set in groups.items():
        g_fwd = fwd[fwd['product_id'].isin(pids_set)]
        row = f"  {gname:<20}"
        for wlabel in FORWARD_WINDOWS:
            col = f'fwd_return_{wlabel}'
            vals = g_fwd[col].dropna()
            if len(vals) == 0:
                row += f" {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            else:
                avg = vals.mean()
                med = vals.median()
                win = (vals > 0).sum() / len(vals) * 100
                row += f" {avg:>+9.2f}% {med:>+9.2f}% {win:>8.1f}%"
        print(row)

    # === 각 그룹 상위 10 종목 상세 ===
    for gname, pids_set in groups.items():
        print(f"\n{'─'*50}")
        print(f"  {gname} — 상위 10 종목")
        print(f"{'─'*50}")

        if gname.startswith('ATH'):
            detail = ath_top.head(10)
            for _, r in detail.iterrows():
                pid = r['product_id']
                tk = r['ticker']
                nm = name_map.get(pid, '')[:10]
                fwd_row = fwd[fwd['product_id'] == pid]
                fwd_2w = fwd_row['fwd_return_2주'].values[0] if len(fwd_row) else float('nan')
                fwd_1m = fwd_row['fwd_return_1개월'].values[0] if len(fwd_row) else float('nan')
                print(f"  {tk:>8} {nm:<12} "
                      f"근접도:{r['ath_proximity']:.4f}  "
                      f"2주:{fwd_2w:>+7.2f}%  1개월:{fwd_1m:>+7.2f}%")
        elif gname.startswith('52W'):
            detail = h52_top.head(10)
            for _, r in detail.iterrows():
                pid = r['product_id']
                tk = r['ticker']
                nm = name_map.get(pid, '')[:10]
                fwd_row = fwd[fwd['product_id'] == pid]
                fwd_2w = fwd_row['fwd_return_2주'].values[0] if len(fwd_row) else float('nan')
                fwd_1m = fwd_row['fwd_return_1개월'].values[0] if len(fwd_row) else float('nan')
                print(f"  {tk:>8} {nm:<12} "
                      f"근접도:{r['h52_proximity']:.4f}  "
                      f"2주:{fwd_2w:>+7.2f}%  1개월:{fwd_1m:>+7.2f}%")
        elif gname.startswith('수익률'):
            detail = ret_top.head(10)
            for _, r in detail.iterrows():
                pid = r['product_id']
                tk = r['ticker']
                nm = name_map.get(pid, '')[:10]
                fwd_row = fwd[fwd['product_id'] == pid]
                fwd_2w = fwd_row['fwd_return_2주'].values[0] if len(fwd_row) else float('nan')
                fwd_1m = fwd_row['fwd_return_1개월'].values[0] if len(fwd_row) else float('nan')
                print(f"  {tk:>8} {nm:<12} "
                      f"연수익률:{r['ann_return']:>+7.2%}  "
                      f"2주:{fwd_2w:>+7.2f}%  1개월:{fwd_1m:>+7.2f}%")
        else:  # Sortino
            detail = sortino_top.head(10)
            for _, r in detail.iterrows():
                pid = r['product_id']
                tk = r['ticker']
                nm = name_map.get(pid, '')[:10]
                fwd_row = fwd[fwd['product_id'] == pid]
                fwd_2w = fwd_row['fwd_return_2주'].values[0] if len(fwd_row) else float('nan')
                fwd_1m = fwd_row['fwd_return_1개월'].values[0] if len(fwd_row) else float('nan')
                print(f"  {tk:>8} {nm:<12} "
                      f"Sortino:{r['sortino']:>+7.2f}  "
                      f"2주:{fwd_2w:>+7.2f}%  1개월:{fwd_1m:>+7.2f}%")

    # === 전체 시장 (baseline) ===
    print(f"\n{'─'*50}")
    print(f"  전체 시장 Baseline (필터링 전 종목)")
    print(f"{'─'*50}")
    row = f"  {'전체':<20}"
    for wlabel in FORWARD_WINDOWS:
        col = f'fwd_return_{wlabel}'
        vals = fwd[col].dropna()
        if len(vals) == 0:
            row += f" {'N/A':>10} {'N/A':>10} {'N/A':>10}"
        else:
            avg = vals.mean()
            med = vals.median()
            win = (vals > 0).sum() / len(vals) * 100
            row += f" {avg:>+9.2f}% {med:>+9.2f}% {win:>8.1f}%"
    print(row)


def main():
    print("=" * 70)
    print("  KRX 신고가 근접도 vs 수익률 — Forward Performance 분석")
    print("  그룹: ATH근접 Top100 / 52W근접 Top100 / 수익률 Top100 / Sortino Top100")
    print("  Forward: 2주(10일) / 1개월(21일)")
    print("=" * 70)

    for label, ref_date in ANALYSIS_DATES.items():
        analyze_one_date(label, ref_date)

    print(f"\n\n{'='*70}")
    print("  완료.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
