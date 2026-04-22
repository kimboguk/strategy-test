#!/usr/bin/env python3
"""
파라미터 스윕 — TP 5|0|100 (5% 익절 전량), SL -20%
  근접도: 전일 고가ATH 기준 (돌파 시 > 1.0)
  BS 504 / Vol 504 기대샤프
  임계값: 0.98~1.03 (근접+돌파)
  거래량 급등: 200%~500%
  Top N: 1, 3
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

# TP 5|0|100 (5% 익절 전량 청산)
_settings.TP_START_PCT = 0.05
_settings.TP_STEP_PCT = 0.00
_settings.TP_CLOSE_RATIO = 1.00

PERIOD = (date(2013, 8, 1), date(2026, 3, 31))

# 스윕 파라미터
THRESHOLDS = [0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]
VOL_RATIOS = [2.0, 3.0, 4.0, 5.0]
TOP_NS = [1, 3]

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
    """OHLCV + ATH 근접도 + 거래량(raw) + BS + vol504"""

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
    # 거래량 원시 데이터 저장 (비율은 run_sim에서 동적 계산)
    volume_by_ticker = {}   # {ticker: {date: (today_vol, prev_vol)}}

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

        running_max = 0.0   # 고가 기준 ATH (전일까지)
        for i in range(n):
            d = pd.Timestamp(dates_arr[i]).date()
            h = highs[i] if not np.isnan(highs[i]) else 0
            c = closes[i] if not np.isnan(closes[i]) else 0
            v = volumes[i] if not np.isnan(volumes[i]) else 0
            # 근접도: 당일 종가 / 전일까지의 고가 ATH (돌파 시 > 1.0)
            tk_sig[d] = (c / running_max) if running_max > 0 and c > 0 else 0.0
            running_max = max(running_max, h)  # ATH 갱신은 근접도 계산 후
            tk_lookup[d] = {'open': grp.iloc[i]['open'], 'high': h,
                            'low': grp.iloc[i]['low'], 'close': c}

            if i >= 1:
                prev_v = volumes[i - 1] if not np.isnan(volumes[i - 1]) else 0
                tk_vol[d] = (v, prev_v)
            else:
                tk_vol[d] = (v, 0)

        ohlcv_lookup[tk] = tk_lookup
        signal_by_ticker[tk] = tk_sig
        volume_by_ticker[tk] = tk_vol

    # BS 기대수익률
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
        if d not in bs_by_date:
            bs_by_date[d] = {}
        bs_by_date[d][tk] = row['bs_return']

    # 변동성 504
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

    return ohlcv_lookup, signal_by_ticker, volume_by_ticker, bs_by_date, vol_by_date, calendar


def run_sim(ohlcv_lookup, signal_by_ticker, volume_by_ticker,
            bs_by_date, vol_by_date, calendar,
            threshold, vol_ratio, top_n):
    """파라미터화된 시뮬레이션"""
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

        # 신호 생성 (파라미터화)
        day_bs = bs_by_date.get(today, {})
        day_vol = vol_by_date.get(today, {})
        candidates = []
        for tk, sig_data in signal_by_ticker.items():
            if tk in positions:
                continue
            prox = sig_data.get(today, 0)
            if prox < threshold:
                continue
            # 거래량 급등 필터
            vol_info = volume_by_ticker.get(tk, {}).get(today)
            if vol_info is None:
                continue
            today_v, prev_v = vol_info
            if prev_v <= 0 or today_v < prev_v * vol_ratio:
                continue
            # 기대샤프
            if tk not in day_bs or tk not in day_vol:
                continue
            expected_sharpe = day_bs[tk] / day_vol[tk]
            candidates.append((tk, expected_sharpe))

        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_entry = [tk for tk, _ in candidates[:top_n]]

    # 잔여 포지션 강제 청산
    last_day = calendar[-1]
    for tk, pos in list(positions.items()):
        bar = ohlcv_lookup.get(tk, {}).get(last_day)
        if bar:
            _, trade = pos.force_close(last_day, bar['close'])
            if trade:
                all_trades.append(trade)

    avg_exposure = np.mean(daily_exposures) if daily_exposures else 0
    return all_trades, max_weighted_exposure, avg_exposure


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
    win_rate = wins / n * 100

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

    return {
        'entries': entries,
        'trades': n,
        'weighted_win_rate': round(weighted_win_rate, 1),
        'wavg_pnl_pct': round(wavg, 2),
        'total_pnl': round(float(total_pnl)),
        'profit_factor': round(pf, 3),
        'avg_hold_days': round(avg_hold, 1),
        'sl_pct': round(sl_pct, 1),
        'fc_count': fc_count,
        'max_exposure': round(max_weighted_exposure),
        'avg_exposure': round(avg_exposure),
    }


# === MAIN ===
sep = '=' * 120
print(sep)
print('  파라미터 스윕 — TP 5|0|100, SL -20% | 전일고가ATH(돌파가능) | BS504/Vol504')
print(f'  근접: {THRESHOLDS} | 거래량: {VOL_RATIOS} | TopN: {TOP_NS}')
print(f'  기간: {PERIOD[0]} ~ {PERIOD[1]}')
print(sep)

sd, ed = PERIOD
print(f'\n  데이터 로드...')
ohlcv, signals, vol_raw, bs_data, vol_data, cal = precompute(sd, ed)
print(f'  거래일: {len(cal)}일, 종목: {len(ohlcv)}')

results = []

total_combos = len(THRESHOLDS) * len(VOL_RATIOS) * len(TOP_NS)
combo_idx = 0

for threshold in THRESHOLDS:
    for vol_ratio in VOL_RATIOS:
        for top_n in TOP_NS:
            combo_idx += 1
            label = f"ATH{threshold:.2f}_V{vol_ratio:.0f}x_T{top_n}"

            # TP 설정 매번 초기화
            _settings.TP_START_PCT = 0.05
            _settings.TP_STEP_PCT = 0.00
            _settings.TP_CLOSE_RATIO = 1.00

            trades, max_exp, avg_exp = run_sim(
                ohlcv, signals, vol_raw, bs_data, vol_data, cal,
                threshold=threshold, vol_ratio=vol_ratio, top_n=top_n)
            m = calc_metrics(trades, max_exp, avg_exp)

            if m:
                m['label'] = label
                m['threshold'] = threshold
                m['vol_ratio'] = vol_ratio
                m['top_n'] = top_n
                # 연환산 수익률 (평균 노출 자본 대비)
                years = len(cal) / 252
                if avg_exp > 0:
                    annual_return = (m['total_pnl'] / years) / avg_exp * 100
                else:
                    annual_return = 0
                m['annual_return_pct'] = round(annual_return, 2)
                results.append(m)

            print(f'  [{combo_idx:>2}/{total_combos}] {label}: '
                  f'진입 {m["entries"] if m else 0:>5} | '
                  f'PF {m["profit_factor"] if m else 0:.3f} | '
                  f'연수익 {m["annual_return_pct"] if m else 0:>+6.1f}% | '
                  f'가중PnL {m["wavg_pnl_pct"] if m else 0:>+6.2f}%'
                  if m else f'  [{combo_idx:>2}/{total_combos}] {label}: 거래 없음')

# 비교표 출력
print(f'\n\n{sep}')
print('  파라미터 스윕 결과 (PF 순 정렬)')
print(sep)

results.sort(key=lambda x: x['profit_factor'], reverse=True)

print(f'\n  {"조합":<24} {"진입":>5} {"거래":>6} {"가중승률":>7} {"가중PnL%":>9} '
      f'{"총PnL(만)":>10} {"PF":>7} {"보유일":>6} {"SL%":>5} {"FC":>4} '
      f'{"평균노출(만)":>11} {"연수익%":>7}')
print('  ' + '-' * 118)

for m in results:
    print(f'  {m["label"]:<24} {m["entries"]:>5} {m["trades"]:>6} '
          f'{m["weighted_win_rate"]:>6.1f}% {m["wavg_pnl_pct"]:>+8.2f}% '
          f'{m["total_pnl"]/10000:>+9,.0f} {m["profit_factor"]:>7.3f} '
          f'{m["avg_hold_days"]:>5.1f} {m["sl_pct"]:>4.1f}% {m["fc_count"]:>4} '
          f'{m["avg_exposure"]/10000:>10,.0f} {m["annual_return_pct"]:>+6.1f}%')

# Top N별 비교
for top_n in TOP_NS:
    print(f'\n\n  === Top {top_n} 전용 (임계값 × 거래량 매트릭스) ===')
    sub = [r for r in results if r['top_n'] == top_n]

    print(f'\n  {"":>8}', end='')
    for vr in VOL_RATIOS:
        print(f'  {"V" + str(int(vr)) + "x PF":>10} {"연수익":>7}', end='')
    print()
    print('  ' + '-' * (8 + len(VOL_RATIOS) * 19))

    for th in THRESHOLDS:
        print(f'  ATH{th:.2f}', end='')
        for vr in VOL_RATIOS:
            match = [r for r in sub if r['threshold'] == th and r['vol_ratio'] == vr]
            if match:
                r = match[0]
                print(f'  {r["profit_factor"]:>10.3f} {r["annual_return_pct"]:>+6.1f}%', end='')
            else:
                print(f'  {"N/A":>10} {"N/A":>7}', end='')
        print()

print(f'\n{sep}')

# JSON 저장
out_path = RESULTS_DIR / "param_sweep_tp5_breakout_bs504.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f'  결과 저장 -> {out_path}')

# settings 복원
_settings.TP_START_PCT = 0.10
_settings.TP_STEP_PCT = 0.05
_settings.TP_CLOSE_RATIO = 0.50
