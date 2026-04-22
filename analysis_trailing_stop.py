#!/usr/bin/env python3
"""
Trailing Stop 비교 — TP 20|10|50 고정, 4가지 SL 변형
  1) 원본: 고정 SL -20%
  2) 고정 trail 15%: max(초기SL, 고점×0.85)
  3) 고정 trail 12%: max(초기SL, 고점×0.88)
  4) 브레이크이븐 + trail 10%: +10% 도달 시 SL→진입가, 이후 고점×0.90
"""

import json
import numpy as np
from collections import defaultdict
from datetime import date
from pathlib import Path

from strategy_backtest.config import settings as _settings
from strategy_backtest.engine.position import Position
from strategy_backtest.analysis_tp_compare_vol import (
    engine, THRESHOLD, VOL_SURGE_RATIO, TOP_N, ALLOC_PER_STOCK,
    RESULTS_DIR, ticker_map, ids_str, precompute, calc_metrics,
)

# TP 20|10|50 고정
_settings.TP_START_PCT = 0.20
_settings.TP_STEP_PCT = 0.10
_settings.TP_CLOSE_RATIO = 0.50

PERIOD = ("전체", date(2013, 8, 1), date(2026, 3, 31))

# Trailing Stop 변형 정의
# (label, trail_type, trail_pct, breakeven_threshold)
#   trail_type: 'fixed' = 고정 trail, 'breakeven' = BE후 trail, None = 원본
TRAIL_STRATEGIES = [
    ("원본(SL-20%)",    None,        0.0,  0.0,  False),
    ("SL없음",          None,        0.0,  0.0,  True),
]


def run_sim_trail(ohlcv_lookup, signal_by_ticker, vol_surge_by_ticker,
                  bs_by_date, vol_by_date, calendar,
                  trail_type=None, trail_pct=0.0, be_threshold=0.0,
                  no_sl=False):
    """Trailing stop 변형 시뮬레이션"""
    positions = {}
    high_watermarks = {}   # {ticker: 진입 후 최고가}
    all_trades = []
    pending_entry = []
    max_weighted_exposure = 0.0

    for today in calendar:
        # 진입
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
                if no_sl:
                    positions[tk].sl_price = 0.0  # SL 무력화
                high_watermarks[tk] = op
            pending_entry = []

        # SL/TP 체크 (trailing stop 적용)
        closed = []
        for tk, pos in list(positions.items()):
            bar = ohlcv_lookup.get(tk, {}).get(today)
            if bar is None:
                continue

            # 고점 갱신 (당일 고가 기준)
            if tk in high_watermarks:
                high_watermarks[tk] = max(high_watermarks[tk], bar['high'])

            # trailing stop 적용: sl_price를 올림
            if trail_type is not None and tk in high_watermarks:
                hw = high_watermarks[tk]
                entry_p = pos.entry_price

                if trail_type == 'fixed':
                    # 고점 × (1 - trail_pct), 초기 SL보다 높으면 갱신
                    trail_sl = hw * (1 - trail_pct)
                    if trail_sl > pos.sl_price:
                        pos.sl_price = trail_sl

                elif trail_type == 'breakeven':
                    # 브레이크이븐 조건: 고점이 진입가 × (1 + be_threshold) 이상
                    if hw >= entry_p * (1 + be_threshold):
                        # SL을 최소 진입가로, 이후 trail
                        trail_sl = max(entry_p, hw * (1 - trail_pct))
                        if trail_sl > pos.sl_price:
                            pos.sl_price = trail_sl

            _, trades = pos.check_and_execute(today, bar['high'], bar['low'], bar['close'])
            all_trades.extend(trades)
            if pos.is_closed:
                closed.append(tk)
        for tk in closed:
            del positions[tk]
            high_watermarks.pop(tk, None)

        # 가중 보유량
        daily_exposure = 0.0
        for tk, pos in positions.items():
            if pos.is_closed:
                continue
            bar = ohlcv_lookup.get(tk, {}).get(today)
            if bar:
                daily_exposure += pos.shares * bar['close']
        max_weighted_exposure = max(max_weighted_exposure, daily_exposure)

        # 신호 생성
        day_bs = bs_by_date.get(today, {})
        day_vol = vol_by_date.get(today, {})
        candidates = []
        for tk, sig_data in signal_by_ticker.items():
            if tk in positions:
                continue
            prox = sig_data.get(today, 0)
            if prox < THRESHOLD:
                continue
            if not vol_surge_by_ticker.get(tk, {}).get(today, False):
                continue
            if tk not in day_bs or tk not in day_vol:
                continue
            expected_sharpe = day_bs[tk] / day_vol[tk]
            candidates.append((tk, expected_sharpe))
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

    return all_trades, max_weighted_exposure


# === MAIN ===
sep = '=' * 100
print(sep)
print('  Trailing Stop 비교 — TP 20|10|50 고정')
print('  ATH>=0.90 + 거래량300%급등 + 기대샤프 Top 1 | 2013.08~2026.03')
print(sep)

period_label, sd, ed = PERIOD
print(f'\n  데이터 로드: {period_label} ({sd} ~ {ed})...')
ohlcv, signals, vol_surge, bs_data, vol_data, cal = precompute(sd, ed)
print(f'  거래일: {len(cal)}일, 종목: {len(ohlcv)}')

all_results = {}
all_trades_by_strategy = {}

for label, trail_type, trail_pct, be_threshold, no_sl in TRAIL_STRATEGIES:
    # TP 설정 매번 초기화 (Position 생성 시 읽으므로)
    _settings.TP_START_PCT = 0.20
    _settings.TP_STEP_PCT = 0.10
    _settings.TP_CLOSE_RATIO = 0.50

    trades, max_exp = run_sim_trail(
        ohlcv, signals, vol_surge, bs_data, vol_data, cal,
        trail_type=trail_type, trail_pct=trail_pct, be_threshold=be_threshold,
        no_sl=no_sl)
    m = calc_metrics(trades, max_exp)
    all_results[label] = m
    all_trades_by_strategy[label] = trades

    if m:
        print(f'\n  [{label}]')
        print(f'    진입: {m["entries"]:,} | 거래: {m["trades"]:,} | '
              f'가중승률: {m["weighted_win_rate"]}% | 가중PnL: {m["wavg_pnl_pct"]:+.2f}%')
        print(f'    총PnL: {m["total_pnl"]:+,}원 | PF: {m["profit_factor"]:.3f} | '
              f'보유일: {m["avg_hold_days"]}일 | SL: {m["sl_pct"]}%')

# 비교표
print(f'\n\n{sep}')
print('  비교 요약 (TP 20|10|50)')
print(sep)

labels = [lb for lb, *_ in TRAIL_STRATEGIES]
print(f'\n  {"SL전략":<18} {"진입":>6} {"거래":>6} {"가중승률":>8} {"가중PnL%":>10} '
      f'{"총PnL(만원)":>12} {"PF":>7} {"보유일":>7} {"SL%":>6}')
print('  ' + '-' * 88)

for lb in labels:
    m = all_results.get(lb)
    if m is None:
        print(f'  {lb:<18} {"N/A":>6}')
        continue
    print(f'  {lb:<18} {m["entries"]:>6,} {m["trades"]:>6,} {m["weighted_win_rate"]:>7.1f}% '
          f'{m["wavg_pnl_pct"]:>+9.2f}% {m["total_pnl"]/10000:>+11,.0f} '
          f'{m["profit_factor"]:>7.3f} {m["avg_hold_days"]:>6.1f} {m["sl_pct"]:>5.1f}%')

# 연도별 breakdown
print(f'\n\n{sep}')
print('  연도별 Breakdown')
print(sep)

all_years = set()
for lb, trades in all_trades_by_strategy.items():
    for t in trades:
        all_years.add(t.entry_date.year)
all_years = sorted(all_years)

yearly_metrics = {}
for lb, trades in all_trades_by_strategy.items():
    by_year = defaultdict(list)
    for t in trades:
        by_year[t.entry_date.year].append(t)
    yearly_metrics[lb] = {}
    for yr in sorted(by_year):
        m = calc_metrics(by_year[yr])
        if m:
            yearly_metrics[lb][yr] = m

for lb in labels:
    print(f'\n  {lb}')
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

out_path = RESULTS_DIR / "trailing_stop_compare_20_10_50.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f'\n  결과 저장 → {out_path}')

# settings 복원
_settings.TP_START_PCT = 0.10
_settings.TP_STEP_PCT = 0.05
_settings.TP_CLOSE_RATIO = 0.50
