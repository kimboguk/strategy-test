#!/usr/bin/env python3
"""
복리 투자 시뮬레이션 — 초기 자본 1,000만원, 진입 시 보유현금의 1/10 투자
TP 20|10|50 고정, SL -20%
최적 파라미터 조합 + 비교 대상 포함
"""

import json
import numpy as np
from collections import defaultdict
from datetime import date
from pathlib import Path

from strategy_backtest.config import settings as _settings
from strategy_backtest.engine.position import Position
from strategy_backtest.analysis_param_sweep import (
    engine, precompute, ticker_map, ids_str,
    RESULTS_DIR,
)

# TP 20|10|50 고정
_settings.TP_START_PCT = 0.20
_settings.TP_STEP_PCT = 0.10
_settings.TP_CLOSE_RATIO = 0.50

INITIAL_CAPITAL = 10_000_000   # 1,000만원
FRACTION = 10                  # 보유현금의 1/N 투자

PERIOD = (date(2013, 8, 1), date(2026, 3, 31))

# 테스트 조합: (label, threshold, vol_ratio, top_n)
COMBOS = [
    ("ATH0.93_V3x_T1", 0.93, 3.0, 1),   # 파라미터 스윕 1위
    ("ATH0.95_V4x_T3", 0.95, 4.0, 3),   # Top3 1위
    ("ATH0.90_V2x_T1", 0.90, 2.0, 1),   # 느슨한 필터
    ("ATH0.95_V4x_T1", 0.95, 4.0, 1),   # 높은 임계값+높은 거래량
    ("ATH0.90_V3x_T1", 0.90, 3.0, 1),   # 기존 기본 설정
]


def run_sim_compounding(ohlcv_lookup, signal_by_ticker, volume_by_ticker,
                        bs_by_date, vol_by_date, calendar,
                        threshold, vol_ratio, top_n):
    """복리 투자 시뮬레이션: 진입 시 보유현금의 1/FRACTION 투자"""
    cash = float(INITIAL_CAPITAL)
    positions = {}          # {ticker: Position}
    entry_costs = {}        # {ticker: 진입 시 지불한 현금}
    all_trades = []
    pending_entry = []

    # 일별 포트폴리오 가치 추적
    daily_values = []       # [(date, cash, position_value, total)]
    max_total = 0.0
    max_drawdown = 0.0

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

                # 보유현금의 1/FRACTION 배분
                alloc = cash / FRACTION
                if alloc < 10000:   # 최소 1만원
                    continue

                shares = int(alloc / (op * (1 + _settings.BUY_COMMISSION)))
                if shares <= 0:
                    continue
                entry_cost = shares * op * (1 + _settings.BUY_COMMISSION)
                if entry_cost > cash:
                    # 현금 부족 시 가능한 만큼
                    shares = int(cash / (op * (1 + _settings.BUY_COMMISSION)))
                    if shares <= 0:
                        continue
                    entry_cost = shares * op * (1 + _settings.BUY_COMMISSION)

                cash -= entry_cost
                positions[tk] = Position(
                    ticker=tk, entry_date=today, entry_price=op,
                    shares=shares, entry_cost=entry_cost)
                entry_costs[tk] = entry_cost
            pending_entry = []

        # SL/TP 체크
        closed = []
        for tk, pos in list(positions.items()):
            bar = ohlcv_lookup.get(tk, {}).get(today)
            if bar is None:
                continue
            cash_received, trades = pos.check_and_execute(
                today, bar['high'], bar['low'], bar['close'])
            cash += cash_received
            all_trades.extend(trades)
            if pos.is_closed:
                closed.append(tk)
        for tk in closed:
            del positions[tk]
            entry_costs.pop(tk, None)

        # 일별 포트폴리오 가치
        pos_value = 0.0
        for tk, pos in positions.items():
            if pos.is_closed:
                continue
            bar = ohlcv_lookup.get(tk, {}).get(today)
            if bar:
                pos_value += pos.shares * bar['close']
        total_value = cash + pos_value
        daily_values.append((today, cash, pos_value, total_value))

        # MDD 추적
        if total_value > max_total:
            max_total = total_value
        dd = (max_total - total_value) / max_total * 100 if max_total > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        # 신호 생성
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

    # 잔여 포지션 강제 청산
    last_day = calendar[-1]
    for tk, pos in list(positions.items()):
        bar = ohlcv_lookup.get(tk, {}).get(last_day)
        if bar:
            net, trade = pos.force_close(last_day, bar['close'])
            cash += net
            if trade:
                all_trades.append(trade)

    final_value = cash  # 모든 포지션 청산 후
    return all_trades, daily_values, final_value, max_drawdown


def calc_summary(daily_values, final_value, all_trades, max_drawdown):
    """복리 시뮬레이션 요약 통계"""
    if not daily_values:
        return None

    entries = len({(t.ticker, t.entry_date) for t in all_trades})
    n_trades = len(all_trades)
    pnls = np.array([t.pnl for t in all_trades])
    wins = (pnls > 0).sum()

    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    years = len(daily_values) / 252
    cagr = ((final_value / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 연도별 수익률
    yearly = defaultdict(list)
    for d, c, pv, tv in daily_values:
        yearly[d.year].append(tv)
    yearly_returns = {}
    prev_end = INITIAL_CAPITAL
    for yr in sorted(yearly):
        yr_end = yearly[yr][-1]
        yr_ret = (yr_end / prev_end - 1) * 100
        yearly_returns[yr] = round(yr_ret, 2)
        prev_end = yr_end

    # FORCE_CLOSE 비중
    fc_count = sum(1 for t in all_trades if t.exit_reason == 'FORCE_CLOSE')

    return {
        'initial_capital': INITIAL_CAPITAL,
        'final_value': round(final_value),
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'entries': entries,
        'trades': n_trades,
        'win_rate': round(wins / n_trades * 100, 1) if n_trades > 0 else 0,
        'fc_count': fc_count,
        'years': round(years, 1),
        'yearly_returns': yearly_returns,
    }


# === MAIN ===
sep = '=' * 100
print(sep)
print(f'  복리 투자 시뮬레이션 — 초기 {INITIAL_CAPITAL/10000:.0f}만원, 진입 시 보유현금의 1/{FRACTION}')
print(f'  TP 20|10|50 | SL -20% | 기간: {PERIOD[0]} ~ {PERIOD[1]}')
print(sep)

sd, ed = PERIOD
print(f'\n  데이터 로드...')
ohlcv, signals, vol_raw, bs_data, vol_data, cal = precompute(sd, ed)
print(f'  거래일: {len(cal)}일, 종목: {len(ohlcv)}')

all_summaries = {}

for label, threshold, vol_ratio, top_n in COMBOS:
    # TP 설정 초기화
    _settings.TP_START_PCT = 0.20
    _settings.TP_STEP_PCT = 0.10
    _settings.TP_CLOSE_RATIO = 0.50

    print(f'\n  [{label}] 시뮬레이션 중...')
    trades, dv, final_val, mdd = run_sim_compounding(
        ohlcv, signals, vol_raw, bs_data, vol_data, cal,
        threshold=threshold, vol_ratio=vol_ratio, top_n=top_n)

    s = calc_summary(dv, final_val, trades, mdd)
    if s:
        s['label'] = label
        s['threshold'] = threshold
        s['vol_ratio'] = vol_ratio
        s['top_n'] = top_n
        all_summaries[label] = s

        print(f'    최종자산: {final_val/10000:,.0f}만원 | 총수익: {s["total_return_pct"]:+.1f}% | '
              f'CAGR: {s["cagr_pct"]:+.1f}% | MDD: -{s["max_drawdown_pct"]:.1f}%')

# 비교표
print(f'\n\n{sep}')
print(f'  복리 비교표 (초기 {INITIAL_CAPITAL/10000:.0f}만원 → 최종 자산)')
print(sep)

print(f'\n  {"조합":<24} {"최종(만원)":>10} {"총수익%":>8} {"CAGR%":>7} {"MDD%":>7} '
      f'{"진입":>5} {"거래":>6} {"승률%":>6} {"FC":>4}')
print('  ' + '-' * 90)

for label, _, _, _ in COMBOS:
    s = all_summaries.get(label)
    if not s:
        continue
    print(f'  {label:<24} {s["final_value"]/10000:>9,.0f} '
          f'{s["total_return_pct"]:>+7.1f}% {s["cagr_pct"]:>+6.1f}% '
          f'{s["max_drawdown_pct"]:>6.1f}% {s["entries"]:>5} {s["trades"]:>6} '
          f'{s["win_rate"]:>5.1f}% {s["fc_count"]:>4}')

# 연도별 수익률 비교
print(f'\n\n{sep}')
print('  연도별 수익률 비교 (%)')
print(sep)

all_years = set()
for s in all_summaries.values():
    all_years.update(s['yearly_returns'].keys())
all_years = sorted(all_years)

print(f'\n  {"연도":>6}', end='')
for label, _, _, _ in COMBOS:
    print(f'  {label:>20}', end='')
print()
print('  ' + '-' * (6 + len(COMBOS) * 22))

for yr in all_years:
    print(f'  {yr:>6}', end='')
    for label, _, _, _ in COMBOS:
        s = all_summaries.get(label)
        if s and yr in s['yearly_returns']:
            print(f'  {s["yearly_returns"][yr]:>+19.1f}%', end='')
        else:
            print(f'  {"N/A":>20}', end='')
    print()

print(f'\n{sep}')

# JSON 저장
out_path = RESULTS_DIR / "compounding_sim.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_summaries, f, indent=2, ensure_ascii=False, default=str)
print(f'  결과 저장 -> {out_path}')

# settings 복원
_settings.TP_START_PCT = 0.10
_settings.TP_STEP_PCT = 0.05
_settings.TP_CLOSE_RATIO = 0.50
