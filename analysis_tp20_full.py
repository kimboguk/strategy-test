#!/usr/bin/env python3
"""
TP 20% 전액 청산 vs 현재(10% 시작 50% 부분청산) 비교
- ATH 근접 + BS순위 Top 1, 임계값 0.93
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import date

from strategy_backtest.config.settings import DATABASE_URL, KRX_MARKETS
from strategy_backtest.config import settings

# TP 20% 전액 청산으로 오버라이드
import strategy_backtest.engine.position as pos_module
pos_module.TP_START_PCT = 0.20
pos_module.TP_CLOSE_RATIO = 1.0

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


def precompute(sd, ed):
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

        running_max = 0.0
        for i in range(n):
            d = pd.Timestamp(dates_arr[i]).date()
            h = highs[i] if not np.isnan(highs[i]) else 0
            c = closes[i] if not np.isnan(closes[i]) else 0
            running_max = max(running_max, h)
            tk_lookup[d] = {'open': grp.iloc[i]['open'], 'high': h,
                            'low': grp.iloc[i]['low'], 'close': c}
            tk_sig[d] = (c / running_max) if running_max > 0 and c > 0 else 0.0

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


def run_backtest(ohlcv_lookup, signal_by_ticker, bs_by_date, calendar, threshold):
    positions = {}
    all_trades = []
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
            pending_entry = []

        closed = []
        for tk, pos in list(positions.items()):
            bar = ohlcv_lookup.get(tk, {}).get(today)
            if bar is None:
                continue
            _, trades = pos.check_and_execute(today, bar['high'], bar['low'], bar['close'])
            all_trades.extend(trades)
            if pos.is_closed:
                closed.append(tk)
        for tk in closed:
            del positions[tk]

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

    last_day = calendar[-1]
    for tk, pos in list(positions.items()):
        bar = ohlcv_lookup.get(tk, {}).get(last_day)
        if bar:
            _, trade = pos.force_close(last_day, bar['close'])
            if trade:
                all_trades.append(trade)

    return all_trades


def calc_metrics(trades):
    if not trades:
        return None
    pnls = np.array([t.pnl for t in trades])
    weights = np.array([t.shares * t.entry_price for t in trades])
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    total_weight = weights.sum()
    wavg = (pnl_pcts * weights).sum() / total_weight if total_weight > 0 else 0

    n = len(trades)
    win_rate = (pnls > 0).sum() / n * 100
    avg_hold = np.array([t.holding_days for t in trades]).mean()
    sl_count = sum(1 for t in trades if t.exit_reason == 'SL')
    sl_pct = sl_count / n * 100
    tp_count = sum(1 for t in trades if t.exit_reason.startswith('TP'))
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum()) if (pnls <= 0).any() else 1e-10
    pf = gp / gl

    return {'n': n, 'win_rate': win_rate, 'wavg_pnl': wavg,
            'avg_hold': avg_hold, 'sl_pct': sl_pct, 'tp_count': tp_count,
            'pf': pf}


years = [
    (2020, date(2020, 1, 1), date(2020, 12, 31)),
    (2025, date(2025, 1, 1), date(2025, 12, 31)),
]
thresholds = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]

sep = '=' * 86
print(sep)
print('  TP 20% 전액 청산 — ATH 근접 + BS순위 Top 1')
print('  SL: -20% / TP: +20% 전액 청산')
print(sep)

for year, sd, ed in years:
    print(f'\n  [{year}년]')
    print(f'  {"임계값":>8} {"거래수":>8} {"승률":>8} {"가중PnL%":>10} '
          f'{"평균보유일":>10} {"SL비율":>8} {"PF":>8}')
    print('  ' + '-' * 74)

    ohlcv, signals, bs_data, cal = precompute(sd, ed)

    for th in thresholds:
        trades = run_backtest(ohlcv, signals, bs_data, cal, th)
        m = calc_metrics(trades)
        if m is None:
            print(f'  {th:>8.2f} {"N/A":>8}')
            continue
        print(f'  {th:>8.2f} {m["n"]:>8,} {m["win_rate"]:>7.1f}% {m["wavg_pnl"]:>+9.2f}% '
              f'{m["avg_hold"]:>9.1f}일 {m["sl_pct"]:>7.1f}% {m["pf"]:>7.3f}')

print('\n' + sep)
