#!/usr/bin/env python3
"""
포지션 단위 수익률 비교: 부분청산 vs 전액청산
- 별도 Position 클래스로 TP 20% 전액 구현
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import date
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

from strategy_backtest.config.settings import DATABASE_URL, KRX_MARKETS
from strategy_backtest.config import settings
from strategy_backtest.engine.position import Position, TradeRecord

engine = create_engine(DATABASE_URL)
markets = ','.join(f"'{m}'" for m in KRX_MARKETS)

TOP_N = 1
ALLOC_PER_STOCK = 1_000_000
BUY_COMM = settings.BUY_COMMISSION
SELL_COMM = settings.SELL_COMMISSION
SELL_TAX = settings.SELL_TAX
ROLLING_W = 252


@dataclass
class PositionFullTP:
    """TP 20% 전액 청산 버전"""
    ticker: str
    entry_date: date
    entry_price: float
    shares: int
    entry_cost: float

    sl_price: float = 0.0
    tp_price: float = 0.0
    realized_trades: List[TradeRecord] = field(default_factory=list)

    tp_pct: float = 0.20

    def __post_init__(self):
        self.sl_price = self.entry_price * (1 - 0.20)
        self.tp_price = self.entry_price * (1 + self.tp_pct)

    @property
    def is_closed(self) -> bool:
        return self.shares <= 0

    def check_and_execute(self, current_date, high, low, close):
        if self.is_closed:
            return 0.0, []

        holding_days = (current_date - self.entry_date).days

        # SL
        if low <= self.sl_price:
            exit_price = self.sl_price
            sell_shares = self.shares
            gross = sell_shares * exit_price
            commission = gross * (SELL_COMM + SELL_TAX)
            net = gross - commission
            cost_basis = sell_shares * self.entry_price * (1 + BUY_COMM)
            pnl = net - cost_basis
            pnl_pct = (net / cost_basis - 1) * 100

            trade = TradeRecord(
                ticker=self.ticker, entry_date=self.entry_date,
                entry_price=self.entry_price, exit_date=current_date,
                exit_price=exit_price, shares=sell_shares,
                pnl=pnl, pnl_pct=pnl_pct, exit_reason='SL',
                commission=commission + sell_shares * self.entry_price * BUY_COMM,
                holding_days=holding_days)
            self.realized_trades.append(trade)
            self.shares = 0
            return net, [trade]

        # TP 20% 전액
        if high >= self.tp_price:
            exit_price = self.tp_price
            sell_shares = self.shares
            gross = sell_shares * exit_price
            commission = gross * (SELL_COMM + SELL_TAX)
            net = gross - commission
            cost_basis = sell_shares * self.entry_price * (1 + BUY_COMM)
            pnl = net - cost_basis
            pnl_pct = (net / cost_basis - 1) * 100

            trade = TradeRecord(
                ticker=self.ticker, entry_date=self.entry_date,
                entry_price=self.entry_price, exit_date=current_date,
                exit_price=exit_price, shares=sell_shares,
                pnl=pnl, pnl_pct=pnl_pct, exit_reason='TP',
                commission=commission + sell_shares * self.entry_price * BUY_COMM,
                holding_days=holding_days)
            self.realized_trades.append(trade)
            self.shares = 0
            return net, [trade]

        return 0.0, []

    def force_close(self, current_date, close_price):
        if self.is_closed:
            return 0.0, None
        sell_shares = self.shares
        gross = sell_shares * close_price
        commission = gross * (SELL_COMM + SELL_TAX)
        net = gross - commission
        cost_basis = sell_shares * self.entry_price * (1 + BUY_COMM)
        pnl = net - cost_basis
        pnl_pct = (net / cost_basis - 1) * 100
        holding_days = (current_date - self.entry_date).days

        trade = TradeRecord(
            ticker=self.ticker, entry_date=self.entry_date,
            entry_price=self.entry_price, exit_date=current_date,
            exit_price=close_price, shares=sell_shares,
            pnl=pnl, pnl_pct=pnl_pct, exit_reason='FORCE_CLOSE',
            commission=commission + sell_shares * self.entry_price * BUY_COMM,
            holding_days=holding_days)
        self.realized_trades.append(trade)
        self.shares = 0
        return net, trade


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


def run_backtest(ohlcv_lookup, signal_by_ticker, bs_by_date, calendar, threshold, use_full_tp):
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
                if use_full_tp == 'tp30':
                    positions[tk] = PositionFullTP(
                        ticker=tk, entry_date=today, entry_price=op,
                        shares=shares, entry_cost=entry_cost, tp_pct=0.30)
                elif use_full_tp:
                    positions[tk] = PositionFullTP(
                        ticker=tk, entry_date=today, entry_price=op,
                        shares=shares, entry_cost=entry_cost, tp_pct=0.20)
                else:
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


def analyze(trades):
    if not trades:
        return None

    # 포지션 그룹핑
    pos_groups = defaultdict(list)
    for t in trades:
        key = (t.ticker, t.entry_date)
        pos_groups[key].append(t)

    num_positions = len(pos_groups)
    num_trades = len(trades)

    # 총 실현 손익
    total_pnl = sum(t.pnl for t in trades)
    total_entry_cost = sum(t.shares * t.entry_price for t in trades)
    wavg_pnl = (total_pnl / total_entry_cost) * 100 if total_entry_cost > 0 else 0

    # 포지션 승률
    pos_wins = 0
    for key, group in pos_groups.items():
        if sum(t.pnl for t in group) > 0:
            pos_wins += 1
    pos_win_rate = pos_wins / num_positions * 100

    # PF
    pnls = np.array([t.pnl for t in trades])
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum()) if (pnls <= 0).any() else 1e-10
    pf = gp / gl

    # 평균 보유일 (포지션 기준)
    hold_days = []
    for key, group in pos_groups.items():
        last_exit = max(t.exit_date for t in group)
        hold_days.append((last_exit - key[1]).days)
    avg_hold = np.mean(hold_days)

    return {
        'positions': num_positions,
        'trades': num_trades,
        'trades_per_pos': num_trades / num_positions,
        'pos_win_rate': pos_win_rate,
        'wavg_pnl': wavg_pnl,
        'avg_hold': avg_hold,
        'pf': pf,
        'total_pnl': total_pnl,
    }


years = [
    (2020, date(2020, 1, 1), date(2020, 12, 31)),
    (2025, date(2025, 1, 1), date(2025, 12, 31)),
]
threshold = 0.93

sep = '=' * 100
print(sep)
print('  포지션 단위 실현 손익 비교 — ATH 근접 0.93 + BS순위 Top 1, 종목당 100만원')
print(sep)

for year, sd, ed in years:
    print(f'\n  [{year}년]')
    print(f'  {"전략":<22} {"포지션수":>8} {"청산거래":>8} {"거래/포지션":>10} '
          f'{"포지션승률":>10} {"가중PnL%":>10} {"보유일":>8} {"PF":>8} {"총실현손익":>12}')
    print('  ' + '-' * 100)

    ohlcv, signals, bs_data, cal = precompute(sd, ed)

    for label, use_full in [('부분청산(10%+5%×50%)', False), ('전액청산(20%)', True), ('전액청산(30%)', 'tp30')]:
        trades = run_backtest(ohlcv, signals, bs_data, cal, threshold, use_full)
        m = analyze(trades)
        if m is None:
            print(f'  {label:<22} N/A')
            continue
        print(f'  {label:<22} {m["positions"]:>8} {m["trades"]:>8} '
              f'{m["trades_per_pos"]:>9.1f} {m["pos_win_rate"]:>9.1f}% '
              f'{m["wavg_pnl"]:>+9.2f}% {m["avg_hold"]:>7.1f}일 '
              f'{m["pf"]:>7.3f} {m["total_pnl"]:>+12,.0f}원')

print('\n' + sep)
