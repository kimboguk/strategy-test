"""
포트폴리오 시뮬레이션 엔진

매일 루프:
1. 기존 포지션 SL/TP 체크 (당일 OHLCV)
2. 전일 신호 종목 당일 시가 진입 (pending_signals)
3. 당일 종가 기준 신호 감지 → pending_signals 갱신
4. 일별 스냅샷 기록
"""

import time
from datetime import date
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field

from strategy_backtest.config.settings import (
    INITIAL_CAPITAL, BUY_COMMISSION,
)
from strategy_backtest.engine.position import Position, TradeRecord


@dataclass
class DailySnapshot:
    date: date
    cash: float
    positions_value: float  # 종가 기준 포지션 시가총액
    total_equity: float
    daily_return: float
    n_positions: int
    n_entries: int          # 당일 신규 진입 수
    n_exits: int            # 당일 청산 수


class PortfolioSimulator:

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        verbose: bool = False,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # ticker → Position
        self.all_trades: List[TradeRecord] = []
        self.snapshots: List[DailySnapshot] = []
        self.verbose = verbose
        self.total_commission = 0.0

    def run(
        self,
        trading_calendar: List[date],
        signals_by_date: Dict[date, Set[str]],
        ohlcv_lookup: Dict[str, Dict[date, dict]],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        """메인 시뮬레이션 루프"""
        t0 = time.time()

        # 날짜 범위 필터
        calendar = trading_calendar
        if start_date:
            calendar = [d for d in calendar if d >= start_date]
        if end_date:
            calendar = [d for d in calendar if d <= end_date]

        if not calendar:
            print("ERROR: 거래일이 없습니다.")
            return

        pending_signals: Set[str] = set()  # 전일 신호 → 당일 진입 대기
        prev_equity = self.initial_capital
        last_close: Dict[str, float] = {}  # 종목별 최근 종가 캐시 (거래정지 대응)

        total_days = len(calendar)
        report_interval = max(1, total_days // 20)

        for i, today in enumerate(calendar):
            n_entries = 0
            n_exits = 0

            # === Step 1: 기존 포지션 SL/TP 체크 ===
            closed_tickers = []
            for ticker, pos in list(self.positions.items()):
                bar = ohlcv_lookup.get(ticker, {}).get(today)
                if bar is None:
                    continue
                cash_received, trades = pos.check_and_execute(
                    today, bar['high'], bar['low'], bar['close']
                )
                self.cash += cash_received
                for t in trades:
                    self.all_trades.append(t)
                    self.total_commission += t.commission
                    n_exits += 1
                if pos.is_closed:
                    closed_tickers.append(ticker)

            for ticker in closed_tickers:
                del self.positions[ticker]

            # === Step 2: pending_signals → 당일 시가 진입 ===
            if pending_signals:
                # 이미 보유 중인 종목 제외
                valid = [
                    t for t in pending_signals
                    if t not in self.positions
                    and ohlcv_lookup.get(t, {}).get(today) is not None
                    and ohlcv_lookup[t][today]['open'] > 0
                ]

                if valid and self.cash > 0:
                    alloc_per_stock = self.cash / len(valid)
                    for ticker in valid:
                        bar = ohlcv_lookup[ticker][today]
                        open_price = bar['open']
                        # 수수료 고려 후 매수 가능 주식 수
                        max_cost = alloc_per_stock
                        shares = int(max_cost / (open_price * (1 + BUY_COMMISSION)))
                        if shares <= 0:
                            continue
                        entry_cost = shares * open_price * (1 + BUY_COMMISSION)
                        if entry_cost > self.cash:
                            shares = int(self.cash / (open_price * (1 + BUY_COMMISSION)))
                            if shares <= 0:
                                continue
                            entry_cost = shares * open_price * (1 + BUY_COMMISSION)

                        self.cash -= entry_cost
                        self.total_commission += shares * open_price * BUY_COMMISSION

                        pos = Position(
                            ticker=ticker,
                            entry_date=today,
                            entry_price=open_price,
                            shares=shares,
                            entry_cost=entry_cost,
                        )
                        self.positions[ticker] = pos
                        n_entries += 1

                pending_signals = set()

            # === Step 3: 당일 종가 신호 → pending_signals 갱신 ===
            day_signals = signals_by_date.get(today, set())
            if day_signals:
                # 이미 보유 중이면 무시 (다음 날 진입 대기에서도 제외)
                pending_signals = day_signals - set(self.positions.keys())

            # === Step 4: 일별 스냅샷 ===
            positions_value = 0.0
            for ticker, pos in self.positions.items():
                bar = ohlcv_lookup.get(ticker, {}).get(today)
                if bar:
                    close_price = bar['close']
                    last_close[ticker] = close_price
                else:
                    # 거래정지 등 — 최근 종가로 캐리
                    close_price = last_close.get(ticker, pos.entry_price)
                positions_value += pos.shares * close_price

            total_equity = self.cash + positions_value
            daily_return = (total_equity / prev_equity - 1) if prev_equity > 0 else 0.0

            self.snapshots.append(DailySnapshot(
                date=today,
                cash=self.cash,
                positions_value=positions_value,
                total_equity=total_equity,
                daily_return=daily_return,
                n_positions=len(self.positions),
                n_entries=n_entries,
                n_exits=n_exits,
            ))
            prev_equity = total_equity

            if self.verbose and (i + 1) % report_interval == 0:
                pct = (i + 1) / total_days * 100
                print(f"  [{pct:5.1f}%] {today} | "
                      f"Equity: {total_equity:,.0f} | "
                      f"Pos: {len(self.positions)} | "
                      f"Cash: {self.cash:,.0f}")

        # === 백테스트 종료: 잔여 포지션 강제 청산 ===
        last_day = calendar[-1]
        for ticker, pos in list(self.positions.items()):
            bar = ohlcv_lookup.get(ticker, {}).get(last_day)
            if bar:
                cash_received, trade = pos.force_close(last_day, bar['close'])
                self.cash += cash_received
                if trade:
                    self.all_trades.append(trade)
                    self.total_commission += trade.commission

        self.positions.clear()

        elapsed = time.time() - t0
        print(f"\n시뮬레이션 완료: {elapsed:.1f}초, "
              f"총 거래: {len(self.all_trades)}, "
              f"최종 자산: {self.cash:,.0f}원")
