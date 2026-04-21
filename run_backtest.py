#!/usr/bin/env python3
"""
KRX ATH/52주 신고가 돌파 전략 백테스트 — CLI 진입점

사용법:
  APP_ENV=dev conda run -n financial_modeling python -m strategy_backtest.run_backtest --signal ath
  APP_ENV=dev conda run -n financial_modeling python -m strategy_backtest.run_backtest --signal 52w
  APP_ENV=dev conda run -n financial_modeling python -m strategy_backtest.run_backtest --signal both
"""

import argparse
import sys
import time
from datetime import date, datetime

from strategy_backtest.config import settings
from strategy_backtest.data.data_loader import DataLoader
from strategy_backtest.signals.breakout_detector import BreakoutDetector, SignalType
from strategy_backtest.engine.portfolio import PortfolioSimulator
from strategy_backtest.metrics.performance import (
    compute_metrics, build_equity_df, build_trades_df,
)
from strategy_backtest.reports.console_report import print_report
from strategy_backtest.reports.csv_export import export_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="KRX ATH/52주 신고가 돌파 전략 백테스트"
    )
    parser.add_argument(
        '--signal', type=str, default='both',
        choices=['ath', '52w', 'both'],
        help="신호 유형: ath(역사적 신고가), 52w(52주 신고가), both(양쪽 비교)"
    )
    parser.add_argument('--stop-loss', type=float, default=None,
                        help=f"손절 비율 (기본: {settings.STOP_LOSS_PCT})")
    parser.add_argument('--tp-start', type=float, default=None,
                        help=f"익절 시작 비율 (기본: {settings.TP_START_PCT})")
    parser.add_argument('--tp-step', type=float, default=None,
                        help=f"익절 간격 비율 (기본: {settings.TP_STEP_PCT})")
    parser.add_argument('--capital', type=float, default=None,
                        help=f"초기 자본 (기본: {settings.INITIAL_CAPITAL:,.0f})")
    parser.add_argument('--start-date', type=str, default=None,
                        help="시작일 (YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, default=None,
                        help="종료일 (YYYY-MM-DD)")
    parser.add_argument('--suffix', type=str, default='',
                        help="결과 파일명 접미사 (예: _tp20_20)")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="상세 출력")
    return parser.parse_args()


def apply_overrides(args):
    """CLI 파라미터로 settings 오버라이드"""
    if args.stop_loss is not None:
        settings.STOP_LOSS_PCT = args.stop_loss
    if args.tp_start is not None:
        settings.TP_START_PCT = args.tp_start
    if args.tp_step is not None:
        settings.TP_STEP_PCT = args.tp_step
    if args.capital is not None:
        settings.INITIAL_CAPITAL = args.capital


def run_single(
    signal_type: SignalType,
    ticker_data, ohlcv_lookup, trading_calendar,
    start_date, end_date, verbose,
    suffix: str = "",
):
    """단일 신호 유형 백테스트 실행"""
    label = signal_type.value
    print(f"\n{'=' * 70}")
    print(f"  신호 유형: {label.upper()}")
    print(f"{'=' * 70}")

    # 신호 사전 계산
    print("\n[신호 계산]")
    detector = BreakoutDetector(signal_type=signal_type)
    signals_by_date = detector.precompute_all(ticker_data)
    signal_stats = detector.count_signals(signals_by_date)
    print(f"  총 신호: {signal_stats['total_signals']:,}, "
          f"신호 발생일: {signal_stats['signal_days']:,}, "
          f"고유 종목: {signal_stats['unique_tickers']:,}")

    # 시뮬레이션
    print("\n[시뮬레이션]")
    sim = PortfolioSimulator(
        initial_capital=settings.INITIAL_CAPITAL,
        verbose=verbose,
    )
    sim.run(
        trading_calendar=trading_calendar,
        signals_by_date=signals_by_date,
        ohlcv_lookup=ohlcv_lookup,
        start_date=start_date,
        end_date=end_date,
    )

    # 지표 계산
    metrics = compute_metrics(
        snapshots=sim.snapshots,
        trades=sim.all_trades,
        initial_capital=settings.INITIAL_CAPITAL,
        total_commission=sim.total_commission,
    )

    # 리포트
    print_report(metrics, label, signal_stats)

    # 파일 내보내기
    print("\n[결과 내보내기]")
    equity_df = build_equity_df(sim.snapshots)
    trades_df = build_trades_df(sim.all_trades)
    export_results(equity_df, trades_df, metrics, label, suffix=suffix)

    return metrics


def main():
    args = parse_args()
    apply_overrides(args)

    print("=" * 70)
    print("  KRX ATH/52주 신고가 돌파 전략 백테스트")
    print(f"  DB: {settings.DATABASE_NAME} | ENV: {settings.APP_ENV}")
    print(f"  SL: {settings.STOP_LOSS_PCT*100:.1f}% | "
          f"TP: {settings.TP_START_PCT*100:.1f}%+{settings.TP_STEP_PCT*100:.1f}% | "
          f"자본: {settings.INITIAL_CAPITAL:,.0f}원")
    print("=" * 70)

    start_date = (datetime.strptime(args.start_date, '%Y-%m-%d').date()
                  if args.start_date else None)
    end_date = (datetime.strptime(args.end_date, '%Y-%m-%d').date()
                if args.end_date else None)

    # 데이터 로드 (공통)
    t0 = time.time()
    loader = DataLoader(verbose=True)
    ticker_data, ohlcv_lookup, trading_calendar, tickers_df = loader.load_all(
        start_date=start_date, end_date=end_date,
    )
    print(f"\n데이터 로드 완료: {time.time() - t0:.1f}초\n")

    # 전략 실행
    signal_types = []
    if args.signal in ('ath', 'both'):
        signal_types.append(SignalType.ATH)
    if args.signal in ('52w', 'both'):
        signal_types.append(SignalType.HIGH_52W)

    results = {}
    for st in signal_types:
        metrics = run_single(
            signal_type=st,
            ticker_data=ticker_data,
            ohlcv_lookup=ohlcv_lookup,
            trading_calendar=trading_calendar,
            start_date=start_date,
            end_date=end_date,
            verbose=args.verbose,
            suffix=args.suffix,
        )
        results[st.value] = metrics

    # both 모드일 때 비교 요약
    if len(results) == 2:
        print("\n" + "=" * 70)
        print("  ATH vs 52W 비교 요약")
        print("=" * 70)
        header = f"  {'지표':<20} {'ATH':>15} {'52W':>15}"
        print(header)
        print("  " + "─" * 52)

        for label, key_path in [
            ('총 수익률 (%)', ('returns', 'total_return_pct')),
            ('연환산 수익률 (%)', ('returns', 'ann_return_pct')),
            ('연환산 변동성 (%)', ('risk', 'ann_volatility_pct')),
            ('MDD (%)', ('risk', 'mdd_pct')),
            ('Sharpe', ('risk_adjusted', 'sharpe_ratio')),
            ('Sortino', ('risk_adjusted', 'sortino_ratio')),
            ('Calmar', ('risk_adjusted', 'calmar_ratio')),
        ]:
            k1, k2 = key_path
            v_ath = results['ath'].get(k1, {}).get(k2, 'N/A')
            v_52w = results['52w'].get(k1, {}).get(k2, 'N/A')
            print(f"  {label:<20} {v_ath:>15} {v_52w:>15}")

        for label, key_path in [
            ('총 거래 수', ('trades', 'total_trades')),
            ('승률 (%)', ('trades', 'win_rate_pct')),
            ('Profit Factor', ('trades', 'profit_factor')),
            ('Expectancy (원)', ('trades', 'expectancy')),
        ]:
            k1, k2 = key_path
            v_ath = results['ath'].get(k1, {}).get(k2, 'N/A')
            v_52w = results['52w'].get(k1, {}).get(k2, 'N/A')
            print(f"  {label:<20} {v_ath:>15} {v_52w:>15}")

        print("=" * 70)

    print("\n완료.")


if __name__ == '__main__':
    main()
