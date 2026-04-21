"""
백테스트 성과 지표 계산
- 수익률, 리스크, 리스크 조정(SR/Sortino/Calmar)
- 트레이드 통계(승률, PF, Expectancy)
- 보유 기간, 동시 보유 종목 수
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from strategy_backtest.engine.portfolio import DailySnapshot
from strategy_backtest.engine.position import TradeRecord
from strategy_backtest.config.settings import (
    TRADING_DAYS_PER_YEAR, RISK_FREE_RATE,
)


def compute_metrics(
    snapshots: List[DailySnapshot],
    trades: List[TradeRecord],
    initial_capital: float,
    total_commission: float,
) -> Dict[str, Any]:
    """전체 성과 지표 계산"""
    if not snapshots:
        return {}

    # --- Equity curve ---
    dates = [s.date for s in snapshots]
    equity = np.array([s.total_equity for s in snapshots])
    daily_returns = np.array([s.daily_return for s in snapshots])
    n_positions = np.array([s.n_positions for s in snapshots])

    # NaN 제거 (첫째 날 등)
    valid_returns = daily_returns[~np.isnan(daily_returns)]

    n_days = len(snapshots)
    n_years = n_days / TRADING_DAYS_PER_YEAR

    # --- 수익률 ---
    final_equity = equity[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    ann_return = ((final_equity / initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # --- 리스크 ---
    ann_vol = np.std(valid_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    # MDD
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    mdd = np.min(drawdowns) * 100
    mdd_idx = np.argmin(drawdowns)
    peak_idx = np.argmax(equity[:mdd_idx + 1]) if mdd_idx > 0 else 0
    # 회복일
    recovery_idx = None
    for j in range(mdd_idx, len(equity)):
        if equity[j] >= running_max[mdd_idx]:
            recovery_idx = j
            break
    mdd_peak_date = dates[peak_idx]
    mdd_trough_date = dates[mdd_idx]
    mdd_recovery_date = dates[recovery_idx] if recovery_idx else None
    mdd_recovery_days = (recovery_idx - mdd_idx) if recovery_idx else None

    # --- 리스크 조정 ---
    rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = valid_returns - rf_daily
    vol = np.std(valid_returns, ddof=1)
    sharpe = (np.mean(excess_returns) / vol
              * np.sqrt(TRADING_DAYS_PER_YEAR)) if vol > 0 else 0

    # Sortino
    downside = valid_returns[valid_returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = (np.mean(excess_returns) / downside_std
               * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std > 0 else 0

    # Calmar
    calmar = (ann_return / 100) / abs(mdd / 100) if mdd != 0 else 0

    # --- 트레이드 통계 ---
    metrics_trades = {}
    if trades:
        pnls = np.array([t.pnl for t in trades])
        pnl_pcts = np.array([t.pnl_pct for t in trades])
        holding_days = np.array([t.holding_days for t in trades])

        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100 if total_trades else 0
        avg_win = np.mean(wins) if len(wins) else 0
        avg_loss = np.mean(losses) if len(losses) else 0

        gross_profit = np.sum(wins) if len(wins) else 0
        gross_loss = abs(np.sum(losses)) if len(losses) else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # RRR (Risk-Reward Ratio)
        rrr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Expectancy (기대값)
        expectancy = np.mean(pnls) if total_trades else 0

        # 청산 사유별 분류
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = 0
            exit_reasons[reason] += 1

        metrics_trades = {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate_pct': round(win_rate, 2),
            'avg_win': round(avg_win, 0),
            'avg_loss': round(avg_loss, 0),
            'avg_pnl': round(np.mean(pnls), 0),
            'avg_pnl_pct': round(np.mean(pnl_pcts), 2),
            'gross_profit': round(gross_profit, 0),
            'gross_loss': round(-abs(np.sum(losses)) if len(losses) else 0, 0),
            'profit_factor': round(profit_factor, 3),
            'rrr': round(rrr, 3),
            'expectancy': round(expectancy, 0),
            'max_pnl': round(np.max(pnls), 0),
            'min_pnl': round(np.min(pnls), 0),
            'avg_holding_days': round(np.mean(holding_days), 1),
            'median_holding_days': round(np.median(holding_days), 1),
            'max_holding_days': int(np.max(holding_days)),
            'exit_reasons': exit_reasons,
        }

    # --- 포트폴리오 통계 ---
    avg_positions = np.mean(n_positions)
    max_positions = int(np.max(n_positions))

    return {
        'period': {
            'start': str(dates[0]),
            'end': str(dates[-1]),
            'trading_days': n_days,
            'years': round(n_years, 2),
        },
        'returns': {
            'total_return_pct': round(total_return, 2),
            'ann_return_pct': round(ann_return, 2),
            'initial_capital': initial_capital,
            'final_equity': round(final_equity, 0),
        },
        'risk': {
            'ann_volatility_pct': round(ann_vol, 2),
            'mdd_pct': round(mdd, 2),
            'mdd_peak_date': str(mdd_peak_date),
            'mdd_trough_date': str(mdd_trough_date),
            'mdd_recovery_date': str(mdd_recovery_date) if mdd_recovery_date else 'N/A',
            'mdd_recovery_days': mdd_recovery_days,
        },
        'risk_adjusted': {
            'sharpe_ratio': round(sharpe, 4),
            'sortino_ratio': round(sortino, 4),
            'calmar_ratio': round(calmar, 4),
        },
        'trades': metrics_trades,
        'portfolio': {
            'avg_positions': round(avg_positions, 1),
            'max_positions': max_positions,
            'total_commission': round(total_commission, 0),
        },
    }


def build_equity_df(snapshots: List[DailySnapshot]) -> pd.DataFrame:
    """스냅샷 → DataFrame 변환 (CSV 내보내기용)"""
    records = []
    for s in snapshots:
        records.append({
            'date': s.date,
            'cash': round(s.cash, 0),
            'positions_value': round(s.positions_value, 0),
            'total_equity': round(s.total_equity, 0),
            'daily_return': round(s.daily_return, 6),
            'n_positions': s.n_positions,
            'n_entries': s.n_entries,
            'n_exits': s.n_exits,
        })
    return pd.DataFrame(records)


def build_trades_df(trades: List[TradeRecord]) -> pd.DataFrame:
    """트레이드 로그 → DataFrame 변환"""
    records = []
    for t in trades:
        records.append({
            'ticker': t.ticker,
            'entry_date': t.entry_date,
            'entry_price': round(t.entry_price, 0),
            'exit_date': t.exit_date,
            'exit_price': round(t.exit_price, 0),
            'shares': t.shares,
            'pnl': round(t.pnl, 0),
            'pnl_pct': round(t.pnl_pct, 2),
            'exit_reason': t.exit_reason,
            'commission': round(t.commission, 0),
            'holding_days': t.holding_days,
        })
    return pd.DataFrame(records)
