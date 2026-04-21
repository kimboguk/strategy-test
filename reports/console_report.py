"""
콘솔 리포트 — 구조화된 텍스트 출력
"""

from typing import Dict, Any


def print_report(metrics: Dict[str, Any], signal_type: str, signal_stats: Dict[str, int]):
    """백테스트 결과 콘솔 출력"""
    period = metrics['period']
    returns = metrics['returns']
    risk = metrics['risk']
    ra = metrics['risk_adjusted']
    trades = metrics.get('trades', {})
    portfolio = metrics['portfolio']

    print("\n" + "=" * 70)
    print(f"  KRX 신고가 돌파 전략 백테스트 결과 — {signal_type.upper()}")
    print("=" * 70)

    print(f"\n{'─' * 40}")
    print(f"  기간")
    print(f"{'─' * 40}")
    print(f"  시작일:        {period['start']}")
    print(f"  종료일:        {period['end']}")
    print(f"  거래일:        {period['trading_days']:,} days ({period['years']:.1f} years)")

    print(f"\n{'─' * 40}")
    print(f"  신호 통계")
    print(f"{'─' * 40}")
    print(f"  총 신호:       {signal_stats.get('total_signals', 0):,}")
    print(f"  신호 발생일:   {signal_stats.get('signal_days', 0):,}")
    print(f"  고유 종목:     {signal_stats.get('unique_tickers', 0):,}")
    print(f"  일 평균 신호:  {signal_stats.get('avg_signals_per_day', 0):.1f}")

    print(f"\n{'─' * 40}")
    print(f"  수익률")
    print(f"{'─' * 40}")
    print(f"  초기 자본:     {returns['initial_capital']:>20,.0f}원")
    print(f"  최종 자산:     {returns['final_equity']:>20,.0f}원")
    print(f"  총 수익률:     {returns['total_return_pct']:>+10.2f}%")
    print(f"  연환산 수익률: {returns['ann_return_pct']:>+10.2f}%")

    print(f"\n{'─' * 40}")
    print(f"  리스크")
    print(f"{'─' * 40}")
    print(f"  연환산 변동성: {risk['ann_volatility_pct']:>10.2f}%")
    print(f"  MDD:           {risk['mdd_pct']:>10.2f}%")
    print(f"  MDD 구간:      {risk['mdd_peak_date']} → {risk['mdd_trough_date']}")
    print(f"  MDD 회복:      {risk['mdd_recovery_date']}"
          f" ({risk['mdd_recovery_days']}일)" if risk['mdd_recovery_days'] else
          f"  MDD 회복:      미회복")

    print(f"\n{'─' * 40}")
    print(f"  리스크 조정 지표")
    print(f"{'─' * 40}")
    print(f"  Sharpe Ratio:  {ra['sharpe_ratio']:>10.4f}")
    print(f"  Sortino Ratio: {ra['sortino_ratio']:>10.4f}")
    print(f"  Calmar Ratio:  {ra['calmar_ratio']:>10.4f}")

    if trades:
        print(f"\n{'─' * 40}")
        print(f"  트레이드 통계")
        print(f"{'─' * 40}")
        print(f"  총 거래:       {trades['total_trades']:>10,}")
        print(f"  승리:          {trades['winning_trades']:>10,}")
        print(f"  패배:          {trades['losing_trades']:>10,}")
        print(f"  승률:          {trades['win_rate_pct']:>10.2f}%")
        print(f"  평균 수익:     {trades['avg_win']:>+10,.0f}원")
        print(f"  평균 손실:     {trades['avg_loss']:>+10,.0f}원")
        print(f"  평균 PnL:      {trades['avg_pnl']:>+10,.0f}원")
        print(f"  평균 PnL%:     {trades['avg_pnl_pct']:>+10.2f}%")
        print(f"  Profit Factor: {trades['profit_factor']:>10.3f}")
        print(f"  RRR:           {trades['rrr']:>10.3f}")
        print(f"  Expectancy:    {trades['expectancy']:>+10,.0f}원")
        print(f"  최대 이익:     {trades['max_pnl']:>+10,.0f}원")
        print(f"  최대 손실:     {trades['min_pnl']:>+10,.0f}원")

        print(f"\n  보유 기간:")
        print(f"    평균:        {trades['avg_holding_days']:>6.1f}일")
        print(f"    중위수:      {trades['median_holding_days']:>6.1f}일")
        print(f"    최대:        {trades['max_holding_days']:>6}일")

        print(f"\n  청산 사유:")
        for reason, count in sorted(trades['exit_reasons'].items()):
            print(f"    {reason:<15} {count:>8,}")

    print(f"\n{'─' * 40}")
    print(f"  포트폴리오")
    print(f"{'─' * 40}")
    print(f"  평균 보유 종목: {portfolio['avg_positions']:>10.1f}")
    print(f"  최대 보유 종목: {portfolio['max_positions']:>10}")
    print(f"  총 수수료:      {portfolio['total_commission']:>10,.0f}원")

    print("\n" + "=" * 70)
