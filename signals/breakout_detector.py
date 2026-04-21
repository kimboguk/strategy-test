"""
ATH / 52주 신고가 돌파 신호 사전 계산
- ATH: expanding().max() — 누적 최고가 돌파
- 52W: rolling(252).max() — 252거래일 롤링 최고가 돌파
"""

from enum import Enum
from datetime import date
from typing import Dict, Set, List

import numpy as np
import pandas as pd

from strategy_backtest.config.settings import MIN_HISTORY_DAYS, ROLLING_52W_DAYS


class SignalType(Enum):
    ATH = "ath"
    HIGH_52W = "52w"


class BreakoutDetector:

    def __init__(
        self,
        signal_type: SignalType,
        min_history_days: int = MIN_HISTORY_DAYS,
    ):
        self.signal_type = signal_type
        self.min_history_days = min_history_days

    def detect_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        단일 종목의 돌파 신호 감지
        df: trade_date, high, close 컬럼 필요 (정렬 가정)
        Returns: boolean Series (True = 돌파 신호)
        """
        high = df['high']
        close = df['close']

        if self.signal_type == SignalType.ATH:
            prev_max = high.shift(1).expanding().max()
        else:  # HIGH_52W
            prev_max = high.shift(1).rolling(
                window=ROLLING_52W_DAYS, min_periods=ROLLING_52W_DAYS
            ).max()

        # 당일 종가가 이전까지의 고가 최대값을 돌파
        signal = close > prev_max

        # 최소 히스토리 미만 구간은 신호 제거 (허위 신호 방지)
        signal.iloc[:self.min_history_days] = False

        # NaN 처리
        signal = signal.fillna(False)

        return signal

    def precompute_all(
        self, ticker_data: Dict[str, pd.DataFrame]
    ) -> Dict[date, Set[str]]:
        """
        전 종목 신호를 사전 계산하여 {date: set(tickers)} 반환
        """
        signals_by_date: Dict[date, Set[str]] = {}

        for ticker, df in ticker_data.items():
            if len(df) < self.min_history_days:
                continue

            sig = self.detect_signals(df)
            signal_dates = df.loc[sig, 'trade_date']

            for td in signal_dates:
                d = td.date() if hasattr(td, 'date') else td
                if d not in signals_by_date:
                    signals_by_date[d] = set()
                signals_by_date[d].add(ticker)

        return signals_by_date

    def count_signals(
        self, signals_by_date: Dict[date, Set[str]]
    ) -> Dict[str, int]:
        """신호 통계 요약"""
        total_signals = sum(len(v) for v in signals_by_date.values())
        signal_days = len(signals_by_date)
        unique_tickers = set()
        for tickers in signals_by_date.values():
            unique_tickers.update(tickers)

        return {
            'total_signals': total_signals,
            'signal_days': signal_days,
            'unique_tickers': len(unique_tickers),
            'avg_signals_per_day': total_signals / signal_days if signal_days else 0,
        }
