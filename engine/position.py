"""
개별 포지션 관리 — SL/TP 로직
- SL: 진입가 대비 -20% → 전량 청산
- TP: +10% 시작, +5% 간격, 잔여의 50%씩 청산
- KRX 정수 주식 수 처리
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Tuple

from strategy_backtest.config import settings as _settings


@dataclass
class TradeRecord:
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'SL', 'TP_L1', 'TP_L2', ..., 'FORCE_CLOSE'
    commission: float
    holding_days: int


@dataclass
class Position:
    ticker: str
    entry_date: date
    entry_price: float
    shares: int
    entry_cost: float  # 매수 시 지불 총액 (수수료 포함)

    # SL/TP 레벨
    sl_price: float = 0.0
    next_tp_level: int = 1
    next_tp_price: float = 0.0

    # 누적 실현 손익
    realized_trades: List[TradeRecord] = field(default_factory=list)

    def __post_init__(self):
        self.sl_price = self.entry_price * (1 - _settings.STOP_LOSS_PCT)
        self._set_next_tp_price()

    def _set_next_tp_price(self):
        """다음 TP 레벨 가격 계산"""
        tp_pct = _settings.TP_START_PCT + (self.next_tp_level - 1) * _settings.TP_STEP_PCT
        self.next_tp_price = self.entry_price * (1 + tp_pct)

    @property
    def is_closed(self) -> bool:
        return self.shares <= 0

    def market_value(self, price: float) -> float:
        return self.shares * price

    def check_and_execute(
        self, current_date: date, high: float, low: float, close: float
    ) -> Tuple[float, List[TradeRecord]]:
        """
        당일 OHLC로 SL/TP 체크 및 실행
        Returns: (cash_received, new_trades)
        """
        if self.is_closed:
            return 0.0, []

        cash_received = 0.0
        new_trades = []
        holding_days = (current_date - self.entry_date).days

        # 1) SL 판정: 저가가 SL 이하 → 전량 청산
        if low <= self.sl_price:
            exit_price = self.sl_price  # SL 가격에 체결 가정
            sell_shares = self.shares
            gross = sell_shares * exit_price
            commission = gross * (_settings.SELL_COMMISSION + _settings.SELL_TAX)
            net = gross - commission
            pnl = net - (sell_shares / (self.shares + sell_shares) * self.entry_cost
                         if self.shares + sell_shares > 0 else 0)
            # 정확한 PnL: 진입 단가 기준
            cost_basis = sell_shares * self.entry_price * (1 + _settings.BUY_COMMISSION)
            pnl = net - cost_basis
            pnl_pct = (net / cost_basis - 1) * 100  # 수수료/세금 포함 수익률

            trade = TradeRecord(
                ticker=self.ticker,
                entry_date=self.entry_date,
                entry_price=self.entry_price,
                exit_date=current_date,
                exit_price=exit_price,
                shares=sell_shares,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason='SL',
                commission=commission + sell_shares * self.entry_price * _settings.BUY_COMMISSION,
                holding_days=holding_days,
            )
            new_trades.append(trade)
            self.realized_trades.append(trade)
            self.shares = 0
            cash_received = net
            return cash_received, new_trades

        # 2) TP 판정: 연속 레벨 돌파 가능
        while not self.is_closed and high >= self.next_tp_price:
            sell_shares = max(1, int(self.shares * _settings.TP_CLOSE_RATIO))
            if sell_shares >= self.shares:
                sell_shares = self.shares  # 잔여 전부

            exit_price = self.next_tp_price
            gross = sell_shares * exit_price
            commission = gross * (_settings.SELL_COMMISSION + _settings.SELL_TAX)
            net = gross - commission
            cost_basis = sell_shares * self.entry_price * (1 + _settings.BUY_COMMISSION)
            pnl = net - cost_basis
            pnl_pct = (net / cost_basis - 1) * 100  # 수수료/세금 포함 수익률

            trade = TradeRecord(
                ticker=self.ticker,
                entry_date=self.entry_date,
                entry_price=self.entry_price,
                exit_date=current_date,
                exit_price=exit_price,
                shares=sell_shares,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=f'TP_L{self.next_tp_level}',
                commission=commission + sell_shares * self.entry_price * _settings.BUY_COMMISSION,
                holding_days=holding_days,
            )
            new_trades.append(trade)
            self.realized_trades.append(trade)
            cash_received += net
            self.shares -= sell_shares

            if not self.is_closed:
                self.next_tp_level += 1
                self._set_next_tp_price()

        return cash_received, new_trades

    def force_close(
        self, current_date: date, close_price: float
    ) -> Tuple[float, TradeRecord]:
        """백테스트 종료 시 잔여 포지션 강제 청산"""
        if self.is_closed:
            return 0.0, None

        sell_shares = self.shares
        gross = sell_shares * close_price
        commission = gross * (_settings.SELL_COMMISSION + _settings.SELL_TAX)
        net = gross - commission
        cost_basis = sell_shares * self.entry_price * (1 + _settings.BUY_COMMISSION)
        pnl = net - cost_basis
        pnl_pct = (net / cost_basis - 1) * 100  # 수수료/세금 포함 수익률
        holding_days = (current_date - self.entry_date).days

        trade = TradeRecord(
            ticker=self.ticker,
            entry_date=self.entry_date,
            entry_price=self.entry_price,
            exit_date=current_date,
            exit_price=close_price,
            shares=sell_shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason='FORCE_CLOSE',
            commission=commission + sell_shares * self.entry_price * _settings.BUY_COMMISSION,
            holding_days=holding_days,
        )
        self.realized_trades.append(trade)
        self.shares = 0
        return net, trade
