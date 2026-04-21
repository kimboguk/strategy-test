#!/usr/bin/env python3
"""
청산 시 수익률 기반 전략 비교
- 전일 종가 판단 → 다음날 시가 진입, 종목당 100만원
- ATH근접 / 52W근접 / BS기대수익률 / BS+ATH근접 / BS+52W근접
- 거래 횟수, 승률, 가중PnL%(진입비용 가중), 평균보유일, SL비율, PF
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import date

from strategy_backtest.config.settings import DATABASE_URL, KRX_MARKETS
from strategy_backtest.config import settings
from strategy_backtest.engine.position import Position

engine = create_engine(DATABASE_URL)
markets = ','.join(f"'{m}'" for m in KRX_MARKETS)

TOP_N = 10
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


def run_backtest(sd, ed, mode):
    """
    mode: 'ath', '52w', 'bs', 'bs_ath', 'bs_52w'
    - ath: ATH 근접도 >= 0.95, 근접도 순 Top N
    - 52w: 52W 근접도 >= 0.95, 근접도 순 Top N
    - bs: BS 기대수익률 상위 5%, 수익률 순 Top N
    - bs_ath: ATH 근접도 >= 0.95 필터 + BS 기대수익률 순 Top N
    - bs_52w: 52W 근접도 >= 0.95 필터 + BS 기대수익률 순 Top N
    """
    needs_proximity = mode in ('ath', '52w', 'bs_ath', 'bs_52w')
    needs_bs = mode in ('bs', 'bs_ath', 'bs_52w')
    prox_type = 'ath' if mode in ('ath', 'bs_ath') else '52w'

    with engine.connect() as conn:
        prices = pd.read_sql(f"""
            SELECT product_id, trade_date, open::float, high::float, low::float, close::float
            FROM market_data WHERE product_id IN ({ids_str})
              AND trade_date <= '{ed}'
            ORDER BY product_id, trade_date
        """, conn, parse_dates=['trade_date'])

    ohlcv_lookup = {}
    signal_by_ticker = {}  # {ticker: {date: proximity}}
    bs_by_date = {}        # {date: [(ticker, bs_return), ...]}

    # OHLCV 룩업 + 근접도 계산
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

        if needs_proximity and prox_type == 'ath':
            running_max = 0.0
            for i in range(n):
                d = pd.Timestamp(dates_arr[i]).date()
                h = highs[i] if not np.isnan(highs[i]) else 0
                c = closes[i] if not np.isnan(closes[i]) else 0
                running_max = max(running_max, h)
                tk_lookup[d] = {'open': grp.iloc[i]['open'], 'high': h,
                                'low': grp.iloc[i]['low'], 'close': c}
                tk_sig[d] = (c / running_max) if running_max > 0 and c > 0 else 0.0
        elif needs_proximity and prox_type == '52w':
            for i in range(n):
                d = pd.Timestamp(dates_arr[i]).date()
                h = highs[i] if not np.isnan(highs[i]) else 0
                c = closes[i] if not np.isnan(closes[i]) else 0
                tk_lookup[d] = {'open': grp.iloc[i]['open'], 'high': h,
                                'low': grp.iloc[i]['low'], 'close': c}
                if i >= ROLLING_W:
                    wh = np.nanmax(highs[i - ROLLING_W:i])
                    tk_sig[d] = (c / wh) if wh > 0 and c > 0 else 0.0
        else:
            for row in grp.itertuples(index=False):
                d = row.trade_date.date() if hasattr(row.trade_date, 'date') else row.trade_date
                tk_lookup[d] = {'open': row.open, 'high': row.high,
                                'low': row.low, 'close': row.close}

        ohlcv_lookup[tk] = tk_lookup
        if needs_proximity:
            signal_by_ticker[tk] = tk_sig

    # BS 기대수익률 로드
    if needs_bs:
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

    positions = {}
    all_trades = []
    pending_entry = []

    for today in calendar:
        # 1) 전일 신호 종목 시가 진입
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

        # 2) SL/TP 체크
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

        # 3) 종가 기준 신호 → 다음날 진입
        if mode == 'ath' or mode == '52w':
            # 근접도 조건 충족 종목 전부 진입
            rankings = []
            for tk, sig_data in signal_by_ticker.items():
                if tk in positions:
                    continue
                prox = sig_data.get(today, 0)
                if prox >= 0.95:
                    rankings.append((tk, prox))
            rankings.sort(key=lambda x: x[1], reverse=True)
            pending_entry = [tk for tk, _ in rankings]

        elif mode == 'bs':
            # 순수 BS 기대수익률 (상위 5% 임계값)
            day_bs = bs_by_date.get(today, {})
            if day_bs:
                all_rets = list(day_bs.values())
                threshold = np.percentile(all_rets, 95)
                candidates = [(tk, r) for tk, r in day_bs.items()
                              if r >= threshold and tk not in positions]
                candidates.sort(key=lambda x: x[1], reverse=True)
                pending_entry = [tk for tk, _ in candidates[:TOP_N]]
            else:
                pending_entry = []

        elif mode in ('bs_ath', 'bs_52w'):
            # 근접도 >= 0.95 필터 + BS 기대수익률 순위
            day_bs = bs_by_date.get(today, {})
            candidates = []
            for tk, sig_data in signal_by_ticker.items():
                if tk in positions:
                    continue
                prox = sig_data.get(today, 0)
                if prox >= 0.95 and tk in day_bs:
                    candidates.append((tk, day_bs[tk]))
            candidates.sort(key=lambda x: x[1], reverse=True)
            pending_entry = [tk for tk, _ in candidates[:TOP_N]]

    # 잔여 포지션 강제 청산
    last_day = calendar[-1]
    for tk, pos in list(positions.items()):
        bar = ohlcv_lookup.get(tk, {}).get(last_day)
        if bar:
            _, trade = pos.force_close(last_day, bar['close'])
            if trade:
                all_trades.append(trade)

    return all_trades


configs = [
    ('ATH근접', 'ath'),
    ('52W근접', '52w'),
]
years = [
    (2020, date(2020, 1, 1), date(2020, 12, 31)),
    (2025, date(2025, 1, 1), date(2025, 12, 31)),
]

sep = '=' * 82
print(sep)
print('  청산 시 수익률 기반 비교 (전일 종가 판단 -> 다음날 시가, 종목당 100만원)')
print('  SL: -20% / TP: +10% 시작 +5% 간격 50% 부분청산')
print('  근접도 >= 0.95 조건 충족 종목 전부 진입 (Top N 제한 없음)')
print('  가중PnL%: 진입비용(shares×entry_price) 가중 — PF와 수학적 일관')
print(sep)

for year, sd, ed in years:
    print(f'\n  [{year}년]')
    print(f'  {"전략":<14} {"거래수":>8} {"승률":>8} {"가중PnL%":>10} '
          f'{"평균보유일":>10} {"SL비율":>8} {"PF":>8}')
    print('  ' + '-' * 74)

    for label, mode in configs:
        trades = run_backtest(sd, ed, mode)
        if not trades:
            print(f'  {label:<14} {"N/A":>8}')
            continue

        pnls = np.array([t.pnl for t in trades])
        hold_days = np.array([t.holding_days for t in trades])
        # 진입비용 가중 평균 PnL% — PF와 일관
        weights = np.array([t.shares * t.entry_price for t in trades])
        pnl_pcts = np.array([t.pnl_pct for t in trades])
        total_weight = weights.sum()
        wavg_pnl_pct = (pnl_pcts * weights).sum() / total_weight if total_weight > 0 else 0

        n = len(trades)
        wins = pnls > 0
        win_rate = wins.sum() / n * 100
        avg_hold = hold_days.mean()
        sl_count = sum(1 for t in trades if t.exit_reason == 'SL')
        sl_pct = sl_count / n * 100
        gp = pnls[pnls > 0].sum()
        gl = abs(pnls[pnls <= 0].sum()) if (pnls <= 0).any() else 1e-10
        pf = gp / gl

        print(f'  {label:<14} {n:>8,} {win_rate:>7.1f}% {wavg_pnl_pct:>+9.2f}% '
              f'{avg_hold:>9.1f}일 {sl_pct:>7.1f}% {pf:>7.3f}')

print('\n' + sep)
