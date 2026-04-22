#!/usr/bin/env python3
"""
TP 전략 비교 — ATH 근접 0.90 + 거래량 300% 급등 + 기대샤프(BS/vol504) Top 1
4가지 TP 설정 × 전체 기간 (1982~2026.03)
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import date
from pathlib import Path
from sqlalchemy import create_engine

from strategy_backtest.config.settings import DATABASE_URL, KRX_MARKETS
from strategy_backtest.config import settings as _settings
from strategy_backtest.engine.position import Position

engine = create_engine(DATABASE_URL)
markets = ','.join(f"'{m}'" for m in KRX_MARKETS)

THRESHOLD = 0.90
VOL_SURGE_RATIO = 3.0   # 전일 거래량 >= 전전일 × 3
TOP_N = 1
ALLOC_PER_STOCK = 1_000_000

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# TP 전략 정의: (label, tp_start, tp_step, tp_close_ratio)
TP_STRATEGIES = [
    ("10|5|50",  0.10, 0.05, 0.50),
    ("10|10|50", 0.10, 0.10, 0.50),
    ("20|10|50", 0.20, 0.10, 0.50),
    ("30|0|100", 0.30, 0.00, 1.00),
]

PERIODS = [
    ("전체", date(2013, 8, 1), date(2026, 3, 31)),
]

# --- 필터링 종목 ---
with engine.connect() as conn:
    filtered = pd.read_sql(f"""
        SELECT aq.product_id, p.ticker
        FROM asset_quality aq JOIN products p ON aq.product_id = p.product_id
        WHERE aq.is_selected = TRUE AND p.market IN ({markets})
    """, conn)
pids = filtered['product_id'].tolist()
ids_str = ','.join(str(i) for i in pids)
ticker_map = dict(zip(filtered['product_id'], filtered['ticker']))
pid_map = dict(zip(filtered['ticker'], filtered['product_id']))  # 역방향


def precompute(sd, ed):
    """OHLCV(+volume) + ATH 근접도 + 거래량 급등 + BS 기대수익률 + 변동성504 사전계산"""

    # 1) OHLCV + volume
    print('    OHLCV 로드...')
    with engine.connect() as conn:
        prices = pd.read_sql(f"""
            SELECT product_id, trade_date,
                   open::float, high::float, low::float, close::float,
                   volume::float
            FROM market_data WHERE product_id IN ({ids_str})
              AND trade_date <= '{ed}'
            ORDER BY product_id, trade_date
        """, conn, parse_dates=['trade_date'])

    ohlcv_lookup = {}       # {ticker: {date: {open,high,low,close,volume}}}
    signal_by_ticker = {}   # {ticker: {date: proximity}}
    vol_surge_by_ticker = {}  # {ticker: {date: bool}}

    for pid, grp in prices.groupby('product_id'):
        tk = ticker_map.get(pid)
        if not tk:
            continue
        grp = grp.sort_values('trade_date').reset_index(drop=True)
        tk_lookup = {}
        tk_sig = {}
        tk_vol_surge = {}
        highs = grp['high'].values
        closes = grp['close'].values
        volumes = grp['volume'].values
        dates_arr = grp['trade_date'].values
        n = len(grp)

        running_max = 0.0
        for i in range(n):
            d = pd.Timestamp(dates_arr[i]).date()
            h = highs[i] if not np.isnan(highs[i]) else 0
            c = closes[i] if not np.isnan(closes[i]) else 0
            v = volumes[i] if not np.isnan(volumes[i]) else 0
            running_max = max(running_max, h)
            tk_lookup[d] = {'open': grp.iloc[i]['open'], 'high': h,
                            'low': grp.iloc[i]['low'], 'close': c,
                            'volume': v}
            tk_sig[d] = (c / running_max) if running_max > 0 and c > 0 else 0.0

            # 거래량 급등: 전일(i) vs 전전일(i-1)
            if i >= 1:
                prev_v = volumes[i - 1] if not np.isnan(volumes[i - 1]) else 0
                tk_vol_surge[d] = (v >= prev_v * VOL_SURGE_RATIO) if prev_v > 0 else False
            else:
                tk_vol_surge[d] = False

        ohlcv_lookup[tk] = tk_lookup
        signal_by_ticker[tk] = tk_sig
        vol_surge_by_ticker[tk] = tk_vol_surge

    # 2) BS 기대수익률 (252 lookback)
    print('    BS 기대수익률 로드...')
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

    # 3) 변동성 504 lookback (daily_returns_snapshot)
    print('    변동성(504) 로드...')
    vol_by_date = {}
    with engine.connect() as conn:
        vol = pd.read_sql(f"""
            SELECT snapshot_date, product_id, annual_volatility::float AS vol
            FROM daily_returns_snapshot
            WHERE lookback_days = 504
              AND product_id IN ({ids_str})
              AND snapshot_date BETWEEN '{sd}' AND '{ed}'
        """, conn, parse_dates=['snapshot_date'])
    for _, row in vol.iterrows():
        d = row['snapshot_date'].date() if hasattr(row['snapshot_date'], 'date') else row['snapshot_date']
        tk = ticker_map.get(row['product_id'])
        if tk is None or np.isnan(row['vol']) or row['vol'] <= 0:
            continue
        if d not in vol_by_date:
            vol_by_date[d] = {}
        vol_by_date[d][tk] = row['vol']

    all_dates = set()
    for d in ohlcv_lookup.values():
        all_dates.update(d.keys())
    calendar = sorted(d for d in all_dates if sd <= d <= ed)

    return ohlcv_lookup, signal_by_ticker, vol_surge_by_ticker, bs_by_date, vol_by_date, calendar


def run_sim(ohlcv_lookup, signal_by_ticker, vol_surge_by_ticker,
            bs_by_date, vol_by_date, calendar):
    """근접도 0.90 + 거래량 급등 + 기대샤프 Top 1 시뮬레이션"""
    positions = {}
    all_trades = []
    pending_entry = []
    max_weighted_exposure = 0.0

    for today in calendar:
        if pending_entry:
            for tk in pending_entry:
                if tk in positions:
                    continue
                bar = ohlcv_lookup.get(tk, {}).get(today)
                if not bar or bar['open'] <= 0 or np.isnan(bar['open']):
                    continue
                op = bar['open']
                shares = int(ALLOC_PER_STOCK / (op * (1 + _settings.BUY_COMMISSION)))
                if shares <= 0:
                    continue
                entry_cost = shares * op * (1 + _settings.BUY_COMMISSION)
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

        # 당일 종가 기준 가중 보유량 계산
        daily_exposure = 0.0
        for tk, pos in positions.items():
            if pos.is_closed:
                continue
            bar = ohlcv_lookup.get(tk, {}).get(today)
            if bar:
                daily_exposure += pos.shares * bar['close']
        max_weighted_exposure = max(max_weighted_exposure, daily_exposure)

        # 근접도 >= 0.90 + 거래량 급등 + 기대샤프 순위
        day_bs = bs_by_date.get(today, {})
        day_vol = vol_by_date.get(today, {})
        candidates = []
        for tk, sig_data in signal_by_ticker.items():
            if tk in positions:
                continue
            prox = sig_data.get(today, 0)
            if prox < THRESHOLD:
                continue
            # 거래량 급등 필터
            if not vol_surge_by_ticker.get(tk, {}).get(today, False):
                continue
            # 기대샤프 = BS 기대수익률 / 변동성(504)
            if tk not in day_bs or tk not in day_vol:
                continue
            expected_sharpe = day_bs[tk] / day_vol[tk]
            candidates.append((tk, expected_sharpe))

        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_entry = [tk for tk, _ in candidates[:TOP_N]]

    last_day = calendar[-1]
    for tk, pos in list(positions.items()):
        bar = ohlcv_lookup.get(tk, {}).get(last_day)
        if bar:
            _, trade = pos.force_close(last_day, bar['close'])
            if trade:
                all_trades.append(trade)

    return all_trades, max_weighted_exposure


def calc_metrics(trades, max_weighted_exposure=0.0):
    if not trades:
        return None
    pnls = np.array([t.pnl for t in trades])
    weights = np.array([t.shares * t.entry_price for t in trades])
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    total_weight = weights.sum()
    wavg = (pnl_pcts * weights).sum() / total_weight if total_weight > 0 else 0

    n = len(trades)
    # 총 진입 횟수 (고유 포지션 수)
    entries = len({(t.ticker, t.entry_date) for t in trades})
    wins = (pnls > 0).sum()
    win_rate = wins / n * 100

    # 가중 승률: 투자금액(shares x entry_price) 기준
    win_mask = pnls > 0
    weighted_win_rate = (weights[win_mask].sum() / total_weight * 100) if total_weight > 0 else 0

    avg_hold = np.array([t.holding_days for t in trades]).mean()
    med_hold = np.median([t.holding_days for t in trades])
    sl_count = sum(1 for t in trades if t.exit_reason == 'SL')
    sl_pct = sl_count / n * 100
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum()) if (pnls <= 0).any() else 1e-10
    pf = gp / gl
    total_pnl = pnls.sum()
    avg_pnl = pnls.mean()
    avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_loss = pnls[pnls <= 0].mean() if (pnls <= 0).any() else 0

    # 청산 사유 분포
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        'entries': entries,
        'trades': n,
        'wins': int(wins),
        'losses': n - int(wins),
        'win_rate': round(win_rate, 1),
        'weighted_win_rate': round(weighted_win_rate, 1),
        'wavg_pnl_pct': round(wavg, 2),
        'total_pnl': round(float(total_pnl)),
        'avg_pnl': round(float(avg_pnl)),
        'avg_win': round(float(avg_win)),
        'avg_loss': round(float(avg_loss)),
        'profit_factor': round(pf, 3),
        'avg_hold_days': round(avg_hold, 1),
        'median_hold_days': round(med_hold, 1),
        'sl_count': sl_count,
        'sl_pct': round(sl_pct, 1),
        'max_weighted_exposure': round(max_weighted_exposure),
        'exit_reasons': exit_reasons,
    }


# === MAIN ===
sep = '=' * 90
print(sep)
print('  TP 전략 비교 — ATH>=0.90 + 거래량300%급등 + 기대샤프(BS/vol504) Top 1')
print(f'  SL: -20% | 종목당 100만원 | 일별 Top 1')
print(sep)

all_results = {}
all_trades_by_strategy = {}

for period_label, sd, ed in PERIODS:
    print(f'\n  데이터 로드: {period_label} ({sd} ~ {ed})...')
    ohlcv, signals, vol_surge, bs_data, vol_data, cal = precompute(sd, ed)
    print(f'  거래일: {len(cal)}일, 종목: {len(ohlcv)}')

    period_results = {}

    for label, tp_start, tp_step, tp_close in TP_STRATEGIES:
        _settings.TP_START_PCT = tp_start
        _settings.TP_STEP_PCT = tp_step
        _settings.TP_CLOSE_RATIO = tp_close

        trades, max_exp = run_sim(ohlcv, signals, vol_surge, bs_data, vol_data, cal)
        m = calc_metrics(trades, max_exp)
        period_results[label] = m
        all_trades_by_strategy[label] = trades

        if m:
            print(f'\n  [{period_label}] TP {label}:')
            print(f'    거래: {m["trades"]:,} | 가중승률: {m["weighted_win_rate"]}% | '
                  f'가중PnL: {m["wavg_pnl_pct"]:+.2f}% | PF: {m["profit_factor"]:.3f}')
            print(f'    총PnL: {m["total_pnl"]:+,}원 | 평균보유: {m["avg_hold_days"]}일 | '
                  f'SL: {m["sl_count"]}회({m["sl_pct"]}%) | 최대보유: {m["max_weighted_exposure"]/10000:,.0f}만원')
        else:
            print(f'\n  [{period_label}] TP {label}: 거래 없음')

    all_results[period_label] = period_results

# 비교표 출력
print(f'\n\n{sep}')
print('  비교 요약')
print(sep)

for period_label, _, _ in PERIODS:
    print(f'\n  [{period_label}]')
    print(f'  {"TP전략":<12} {"진입":>6} {"거래":>6} {"가중승률":>8} {"가중PnL%":>10} '
          f'{"총PnL(만원)":>12} {"PF":>7} {"보유일":>7} {"SL%":>6} {"최대보유(만원)":>14}')
    print('  ' + '-' * 98)

    for label, _, _, _ in TP_STRATEGIES:
        m = all_results[period_label].get(label)
        if m is None:
            print(f'  {label:<12} {"N/A":>6}')
            continue
        print(f'  {label:<12} {m["entries"]:>6,} {m["trades"]:>6,} {m["weighted_win_rate"]:>7.1f}% '
              f'{m["wavg_pnl_pct"]:>+9.2f}% {m["total_pnl"]/10000:>+11,.0f} '
              f'{m["profit_factor"]:>7.3f} {m["avg_hold_days"]:>6.1f} {m["sl_pct"]:>5.1f}% '
              f'{m["max_weighted_exposure"]/10000:>13,.0f}')

# === 연도별 Breakdown ===
print(f'\n\n{sep}')
print('  연도별 Breakdown')
print(sep)

all_years = set()
for label, trades in all_trades_by_strategy.items():
    for t in trades:
        all_years.add(t.entry_date.year)
all_years = sorted(all_years)

yearly_metrics = {}
for label, trades in all_trades_by_strategy.items():
    by_year = defaultdict(list)
    for t in trades:
        by_year[t.entry_date.year].append(t)
    yearly_metrics[label] = {}
    for yr in sorted(by_year):
        m = calc_metrics(by_year[yr])
        if m:
            yearly_metrics[label][yr] = m

labels = [lb for lb, _, _, _ in TP_STRATEGIES]
for lb in labels:
    print(f'\n  TP {lb}')
    print(f'  {"연도":>6} {"진입":>6} {"거래":>6} {"가중승률":>8} {"가중PnL%":>10} '
          f'{"총PnL(만원)":>11} {"PF":>7} {"보유일":>7} {"SL%":>6}')
    print('  ' + '-' * 76)
    for yr in all_years:
        m = yearly_metrics.get(lb, {}).get(yr)
        if m:
            print(f'  {yr:>6} {m["entries"]:>6,} {m["trades"]:>6,} {m["weighted_win_rate"]:>7.1f}% '
                  f'{m["wavg_pnl_pct"]:>+9.2f}% {m["total_pnl"]/10000:>+10,.0f} '
                  f'{m["profit_factor"]:>7.3f} {m["avg_hold_days"]:>6.1f} {m["sl_pct"]:>5.1f}%')

print('\n' + sep)

# JSON 저장
all_results['연도별'] = {}
for lb in labels:
    all_results['연도별'][lb] = {str(yr): m for yr, m in yearly_metrics.get(lb, {}).items()}

out_path = RESULTS_DIR / "tp_compare_ath090_volsurge_sharpe_full.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f'\n  결과 저장 → {out_path}')

# settings 복원
_settings.TP_START_PCT = 0.10
_settings.TP_STEP_PCT = 0.05
_settings.TP_CLOSE_RATIO = 0.50
