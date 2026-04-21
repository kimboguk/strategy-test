#!/usr/bin/env python3
"""임계값별 동시 보유 종목 수 확인 — 근접도 필터 + BS순위 Top 1"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import date

from strategy_backtest.config.settings import DATABASE_URL, KRX_MARKETS
from strategy_backtest.config import settings
from strategy_backtest.engine.position import Position

engine = create_engine(DATABASE_URL)
markets = ','.join(f"'{m}'" for m in KRX_MARKETS)

TOP_N = 1
ALLOC_PER_STOCK = 1_000_000
BUY_COMM = settings.BUY_COMMISSION
ROLLING_W = 252

with engine.connect() as conn:
    filtered = pd.read_sql(f"""
        SELECT aq.product_id, p.ticker
        FROM asset_quality aq JOIN products p ON aq.product_id = p.product_id
        WHERE aq.is_selected = TRUE AND p.market IN ({markets})
    """, conn)
pids = filtered['product_id'].tolist()
ids_str = ','.join(str(i) for i in pids)
ticker_map = dict(zip(filtered['product_id'], filtered['ticker']))


def precompute(sd, ed, prox_type):
    with engine.connect() as conn:
        prices = pd.read_sql(f"""
            SELECT product_id, trade_date, open::float, high::float, low::float, close::float
            FROM market_data WHERE product_id IN ({ids_str})
              AND trade_date <= '{ed}'
            ORDER BY product_id, trade_date
        """, conn, parse_dates=['trade_date'])

    ohlcv_lookup = {}
    signal_by_ticker = {}

    for pid, grp in prices.groupby('product_id'):
        tk = ticker_map.get(pid)
        if not tk:
            continue
        grp = grp.sort_values('trade_date').reset_index(drop=True)
        tk_lookup = {}
        tk_sig = {}
        highs = grp['high'].values
        closes = grp['close'].values
        dates_arr = grp['trade_date'].values
        n = len(grp)

        if prox_type == 'ath':
            running_max = 0.0
            for i in range(n):
                d = pd.Timestamp(dates_arr[i]).date()
                h = highs[i] if not np.isnan(highs[i]) else 0
                c = closes[i] if not np.isnan(closes[i]) else 0
                running_max = max(running_max, h)
                tk_lookup[d] = {'open': grp.iloc[i]['open'], 'high': h,
                                'low': grp.iloc[i]['low'], 'close': c}
                tk_sig[d] = (c / running_max) if running_max > 0 and c > 0 else 0.0
        else:
            for i in range(n):
                d = pd.Timestamp(dates_arr[i]).date()
                h = highs[i] if not np.isnan(highs[i]) else 0
                c = closes[i] if not np.isnan(closes[i]) else 0
                tk_lookup[d] = {'open': grp.iloc[i]['open'], 'high': h,
                                'low': grp.iloc[i]['low'], 'close': c}
                if i >= ROLLING_W:
                    wh = np.nanmax(highs[i - ROLLING_W:i])
                    tk_sig[d] = (c / wh) if wh > 0 and c > 0 else 0.0

        ohlcv_lookup[tk] = tk_lookup
        signal_by_ticker[tk] = tk_sig

    bs_by_date = {}
    with engine.connect() as conn:
        bs = pd.read_sql(f"""
            SELECT snapshot_date, product_id, annual_expected_return::float AS bs_return
            FROM expected_returns_snapshot
            WHERE estimation_method = 'bayes_stein' AND lookback_days = 252
              AND product_id IN ({ids_str})
              AND snapshot_date BETWEEN '{sd}' AND '{ed}'
        """, conn, parse_dates=['snapshot_date'])
    for _, row in bs.iterrows():
        d = row['snapshot_date'].date() if hasattr(row['snapshot_date'], 'date') else row['snapshot_date']
        tk = ticker_map.get(row['product_id'])
        if tk is None or np.isnan(row['bs_return']):
            continue
        if d not in bs_by_date:
            bs_by_date[d] = {}
        bs_by_date[d][tk] = row['bs_return']

    all_dates = set()
    for d in ohlcv_lookup.values():
        all_dates.update(d.keys())
    calendar = sorted(d for d in all_dates if sd <= d <= ed)
    return ohlcv_lookup, signal_by_ticker, bs_by_date, calendar


def run_with_tracking(ohlcv_lookup, signal_by_ticker, bs_by_date, calendar, threshold):
    positions = {}
    init_shares = {}  # {ticker: 초기 주식 수}
    max_pos = 0
    pos_counts = []       # 단순 종목 수
    weighted_counts = []  # 잔여비중 가중 종목 수
    pending_entry = []

    for today in calendar:
        if pending_entry:
            for tk in pending_entry:
                if tk in positions:
                    continue
                bar = ohlcv_lookup.get(tk, {}).get(today)
                if not bar or bar['open'] <= 0 or np.isnan(bar['open']):
                    continue
                op = bar['open']
                shares = int(ALLOC_PER_STOCK / (op * (1 + BUY_COMM)))
                if shares <= 0:
                    continue
                entry_cost = shares * op * (1 + BUY_COMM)
                positions[tk] = Position(
                    ticker=tk, entry_date=today, entry_price=op,
                    shares=shares, entry_cost=entry_cost)
                init_shares[tk] = shares
            pending_entry = []

        closed = []
        for tk, pos in list(positions.items()):
            bar = ohlcv_lookup.get(tk, {}).get(today)
            if bar is None:
                continue
            pos.check_and_execute(today, bar['high'], bar['low'], bar['close'])
            if pos.is_closed:
                closed.append(tk)
        for tk in closed:
            del positions[tk]
            del init_shares[tk]

        cur = len(positions)
        max_pos = max(max_pos, cur)
        pos_counts.append(cur)

        # 잔여비중 가중 합산 (잔여주식/초기주식)
        w_sum = sum(pos.shares / init_shares[tk] for tk, pos in positions.items())
        weighted_counts.append(w_sum)

        day_bs = bs_by_date.get(today, {})
        candidates = []
        for tk, sig_data in signal_by_ticker.items():
            if tk in positions:
                continue
            prox = sig_data.get(today, 0)
            if prox >= threshold and tk in day_bs:
                candidates.append((tk, day_bs[tk]))
        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_entry = [tk for tk, _ in candidates[:TOP_N]]

    avg_pos = np.mean(pos_counts) if pos_counts else 0
    avg_weighted = np.mean(weighted_counts) if weighted_counts else 0

    return max_pos, avg_pos, avg_weighted


thresholds = [round(0.90 + i * 0.01, 2) for i in range(11)]
years = [
    (2020, date(2020, 1, 1), date(2020, 12, 31)),
    (2025, date(2025, 1, 1), date(2025, 12, 31)),
]

sep = '=' * 70
print(sep)
print('  임계값별 동시 보유 종목 수 — 근접도 필터 + BS순위 Top 1')
print('  가중: 잔여주식/초기주식 비율 합산')
print(sep)

for prox_type, prox_label in [('ath', 'ATH')]:
    print(f'\n  ── {prox_label} 근접 + BS순위 Top 1 ──')

    for year, sd, ed in years:
        print(f'\n  [{year}년]')
        print(f'  {"임계값":>8} {"최대":>6} {"평균(단순)":>10} {"평균(가중)":>10}')
        print('  ' + '-' * 42)

        ohlcv, signals, bs_data, cal = precompute(sd, ed, prox_type)

        for th in thresholds:
            mx, avg, avg_w = run_with_tracking(ohlcv, signals, bs_data, cal, th)
            print(f'  {th:>8.2f} {mx:>6} {avg:>9.1f} {avg_w:>9.1f}')

print('\n' + sep)
