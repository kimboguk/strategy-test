#!/usr/bin/env python3
"""
돌파 전략 연도별 Breakdown — 전일 고가ATH 기준, TP 5|0|100, BS504/Vol504
복수 임계값 비교: 1.01, 1.02, 1.03
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

ALLOC_PER_STOCK = 1_000_000
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# TP 5|0|100 (5% 익절 전량)
_settings.TP_START_PCT = 0.05
_settings.TP_STEP_PCT = 0.00
_settings.TP_CLOSE_RATIO = 1.00

PERIOD = (date(2013, 8, 1), date(2026, 3, 31))

# 복수 조합 비교
COMBOS = [
    ("ATH1.01_V2x_T1", 1.01, 2.0, 1),
    ("ATH1.01_V3x_T1", 1.01, 3.0, 1),
    ("ATH1.01_V4x_T1", 1.01, 4.0, 1),
    ("ATH1.02_V2x_T1", 1.02, 2.0, 1),
    ("ATH1.02_V3x_T1", 1.02, 3.0, 1),
    ("ATH1.02_V4x_T1", 1.02, 4.0, 1),
    ("ATH1.03_V2x_T1", 1.03, 2.0, 1),
    ("ATH1.03_V2x_T3", 1.03, 2.0, 3),
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


def precompute(sd, ed):
    """OHLCV + 고가ATH 근접도 + 거래량 + BS + vol504"""
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

    ohlcv_lookup = {}
    signal_by_ticker = {}
    volume_by_ticker = {}

    for pid, grp in prices.groupby('product_id'):
        tk = ticker_map.get(pid)
        if not tk:
            continue
        grp = grp.sort_values('trade_date').reset_index(drop=True)
        tk_lookup = {}
        tk_sig = {}
        tk_vol = {}
        highs = grp['high'].values
        closes = grp['close'].values
        volumes = grp['volume'].values
        dates_arr = grp['trade_date'].values
        n = len(grp)

        opens = grp['open'].values
        lows = grp['low'].values
        running_max = 0.0  # 고가 ATH (전일까지)
        for i in range(n):
            d = pd.Timestamp(dates_arr[i]).date()
            h = highs[i] if not np.isnan(highs[i]) else 0
            c = closes[i] if not np.isnan(closes[i]) else 0
            v = volumes[i] if not np.isnan(volumes[i]) else 0
            o = opens[i] if not np.isnan(opens[i]) else 0
            lo = lows[i] if not np.isnan(lows[i]) else 0
            # OHLCV bar 저장
            tk_lookup[d] = {'open': o, 'high': h, 'low': lo, 'close': c, 'volume': v}
            # 전일까지의 ATH 기준 (돌파 시 > 1.0)
            tk_sig[d] = (c / running_max) if running_max > 0 and c > 0 else 0.0
            running_max = max(running_max, h)

            if i >= 1:
                prev_v = volumes[i - 1] if not np.isnan(volumes[i - 1]) else 0
                tk_vol[d] = (v, prev_v)
            else:
                tk_vol[d] = (v, 0)

        ohlcv_lookup[tk] = tk_lookup
        signal_by_ticker[tk] = tk_sig
        volume_by_ticker[tk] = tk_vol

    print('    BS 기대수익률 로드...')
    bs_by_date = {}
    with engine.connect() as conn:
        bs = pd.read_sql(f"""
            SELECT snapshot_date, product_id, annual_expected_return::float AS bs_return
            FROM expected_returns_snapshot
            WHERE estimation_method = 'bayes_stein' AND lookback_days = 504
              AND product_id IN ({ids_str})
              AND snapshot_date BETWEEN '{sd}' AND '{ed}'
        """, conn, parse_dates=['snapshot_date'])
    for _, row in bs.iterrows():
        d = row['snapshot_date'].date() if hasattr(row['snapshot_date'], 'date') else row['snapshot_date']
        tk = ticker_map.get(row['product_id'])
        if tk is None or np.isnan(row['bs_return']):
            continue
        bs_by_date.setdefault(d, {})[tk] = row['bs_return']

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
        vol_by_date.setdefault(d, {})[tk] = row['vol']

    all_dates = set()
    for d in ohlcv_lookup.values():
        all_dates.update(d.keys())
    calendar = sorted(d for d in all_dates if sd <= d <= ed)

    return ohlcv_lookup, signal_by_ticker, volume_by_ticker, bs_by_date, vol_by_date, calendar


def run_sim(ohlcv_lookup, signal_by_ticker, volume_by_ticker,
            bs_by_date, vol_by_date, calendar,
            threshold, vol_ratio, top_n):
    positions = {}
    all_trades = []
    pending_entry = []
    max_weighted_exposure = 0.0
    daily_exposures = []

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

        daily_exposure = 0.0
        for tk, pos in positions.items():
            if pos.is_closed:
                continue
            bar = ohlcv_lookup.get(tk, {}).get(today)
            if bar:
                daily_exposure += pos.shares * bar['close']
        max_weighted_exposure = max(max_weighted_exposure, daily_exposure)
        daily_exposures.append(daily_exposure)

        day_bs = bs_by_date.get(today, {})
        day_vol = vol_by_date.get(today, {})
        candidates = []
        for tk, sig_data in signal_by_ticker.items():
            if tk in positions:
                continue
            prox = sig_data.get(today, 0)
            if prox < threshold:
                continue
            vol_info = volume_by_ticker.get(tk, {}).get(today)
            if vol_info is None:
                continue
            today_v, prev_v = vol_info
            if prev_v <= 0 or today_v < prev_v * vol_ratio:
                continue
            if tk not in day_bs or tk not in day_vol:
                continue
            expected_sharpe = day_bs[tk] / day_vol[tk]
            candidates.append((tk, expected_sharpe))

        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_entry = [tk for tk, _ in candidates[:top_n]]

    last_day = calendar[-1]
    for tk, pos in list(positions.items()):
        bar = ohlcv_lookup.get(tk, {}).get(last_day)
        if bar:
            _, trade = pos.force_close(last_day, bar['close'])
            if trade:
                all_trades.append(trade)

    avg_exp = np.mean(daily_exposures) if daily_exposures else 0
    return all_trades, max_weighted_exposure, avg_exp


def calc_metrics(trades, max_weighted_exposure=0.0, avg_exposure=0.0):
    if not trades:
        return None
    pnls = np.array([t.pnl for t in trades])
    weights = np.array([t.shares * t.entry_price for t in trades])
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    total_weight = weights.sum()
    wavg = (pnl_pcts * weights).sum() / total_weight if total_weight > 0 else 0

    n = len(trades)
    entries = len({(t.ticker, t.entry_date) for t in trades})
    wins = (pnls > 0).sum()

    win_mask = pnls > 0
    weighted_win_rate = (weights[win_mask].sum() / total_weight * 100) if total_weight > 0 else 0

    avg_hold = np.array([t.holding_days for t in trades]).mean()
    sl_count = sum(1 for t in trades if t.exit_reason == 'SL')
    sl_pct = sl_count / n * 100
    fc_count = sum(1 for t in trades if t.exit_reason == 'FORCE_CLOSE')
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum()) if (pnls <= 0).any() else 1e-10
    pf = gp / gl
    total_pnl = pnls.sum()
    avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_loss = pnls[pnls <= 0].mean() if (pnls <= 0).any() else 0

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        'entries': entries,
        'trades': n,
        'weighted_win_rate': round(weighted_win_rate, 1),
        'wavg_pnl_pct': round(wavg, 2),
        'total_pnl': round(float(total_pnl)),
        'profit_factor': round(pf, 3),
        'avg_hold_days': round(avg_hold, 1),
        'sl_pct': round(sl_pct, 1),
        'sl_count': sl_count,
        'fc_count': fc_count,
        'avg_win': round(float(avg_win)),
        'avg_loss': round(float(avg_loss)),
        'max_exposure': round(max_weighted_exposure),
        'avg_exposure': round(avg_exposure),
        'exit_reasons': exit_reasons,
    }


# === MAIN ===
sep = '=' * 110
print(sep)
print(f'  돌파 전략 연도별 Breakdown — 전일고가ATH | TP 5|0|100 | SL -20% | BS504/Vol504')
print(f'  기간: {PERIOD[0]} ~ {PERIOD[1]}')
print(sep)

sd, ed = PERIOD
print(f'\n  데이터 로드...')
ohlcv, signals, vol_raw, bs_data, vol_data, cal = precompute(sd, ed)
print(f'  거래일: {len(cal)}일, 종목: {len(ohlcv)}')

all_out = {}

for label, threshold, vol_ratio, top_n in COMBOS:
    _settings.TP_START_PCT = 0.05
    _settings.TP_STEP_PCT = 0.00
    _settings.TP_CLOSE_RATIO = 1.00

    print(f'\n  [{label}] 시뮬레이션 중...')
    trades, max_exp, avg_exp = run_sim(
        ohlcv, signals, vol_raw, bs_data, vol_data, cal,
        threshold=threshold, vol_ratio=vol_ratio, top_n=top_n)
    m = calc_metrics(trades, max_exp, avg_exp)

    if m:
        years = len(cal) / 252
        annual_ret = (m['total_pnl'] / years) / avg_exp * 100 if avg_exp > 0 else 0
        print(f'    진입: {m["entries"]:,} | PF: {m["profit_factor"]:.3f} | '
              f'승률: {m["weighted_win_rate"]}% | 보유: {m["avg_hold_days"]}일 | '
              f'SL: {m["sl_pct"]}% | 연수익: {annual_ret:+.1f}%')

    # 연도별
    by_year = defaultdict(list)
    for t in trades:
        by_year[t.entry_date.year].append(t)
    all_years = sorted(by_year.keys())

    print(f'\n  {"연도":>6} {"진입":>5} {"승률%":>6} {"가중PnL%":>9} {"PnL(만)":>9} '
          f'{"PF":>7} {"보유일":>6} {"SL%":>5} {"FC":>3}')
    print('  ' + '-' * 68)

    yearly_results = {}
    for yr in all_years:
        ym = calc_metrics(by_year[yr])
        if ym:
            yearly_results[yr] = ym
            print(f'  {yr:>6} {ym["entries"]:>5} {ym["weighted_win_rate"]:>5.1f}% '
                  f'{ym["wavg_pnl_pct"]:>+8.2f}% {ym["total_pnl"]/10000:>+8,.0f} '
                  f'{ym["profit_factor"]:>7.3f} {ym["avg_hold_days"]:>5.1f} '
                  f'{ym["sl_pct"]:>4.1f}% {ym["fc_count"]:>3}')

    win_years = [yr for yr in all_years if yearly_results.get(yr, {}).get('profit_factor', 0) > 1]
    lose_years = [yr for yr in all_years if yearly_results.get(yr, {}).get('profit_factor', 0) <= 1]
    win_pnl = sum(yearly_results[yr]['total_pnl'] for yr in win_years)
    lose_pnl = sum(yearly_results[yr]['total_pnl'] for yr in lose_years)
    print(f'  승리 {len(win_years)}년({win_pnl/10000:+,.0f}만) | '
          f'패배 {len(lose_years)}년({lose_pnl/10000:+,.0f}만) | '
          f'순 {(win_pnl+lose_pnl)/10000:+,.0f}만')

    all_out[label] = {'전체': m, '연도별': {str(yr): ym for yr, ym in yearly_results.items()}}

print(f'\n{sep}')

out_path = RESULTS_DIR / "breakout_yearly_breakdown.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_out, f, indent=2, ensure_ascii=False, default=str)
print(f'  결과 저장 -> {out_path}')

_settings.TP_START_PCT = 0.10
_settings.TP_STEP_PCT = 0.05
_settings.TP_CLOSE_RATIO = 0.50
