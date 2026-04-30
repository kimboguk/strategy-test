#!/usr/bin/env python3
"""
ATH*1.02 돌파 + 거래량 300% 필터 + Bayes-Stein 랭킹 (한국 종목)

신호:
  - 당일 high >= prev_ATH * 1.02 (전일까지의 누적 최고 high)
  - 당일 volume >= 전일 volume * 3.0
  - 양쪽 동시 충족

진입:
  - 신호 발생 다음날 시가 (open)
  - 일별 Bayes-Stein 연환산 기대수익률 상위 3종목까지
  - 이미 보유 중인 종목 추가 진입 안 함

청산:
  - TP +20% (전량) 또는 SL -5% (전량) — 단일 레벨, 분할 X
  - 백테스트 마지막일 잔여 포지션 강제 청산

가격: market_data.adj_close가 NULL이라 raw close/open/high/low 사용.
주의: split/배당 미보정 → 액면분할 시점 부근에서 신호/PnL이 왜곡될 수 있음.

사용법:
  python run_ath_volume_breakout.py --start 2014-01-01 --end 2026-02-13
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psycopg2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "equity"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

INITIAL_CAPITAL = 100_000_000      # 1억원
TP_PCT = 0.20                       # +20% 익절 (전량)
SL_PCT = 0.05                       # -5% 손절 (전량)
TOP_N_PER_DAY = 3                   # 일별 진입 상한
ATH_BREAKOUT_RATIO = 1.02           # 신고가 돌파 비율 (CLI override 가능)
VOLUME_RATIO = 3.0                  # 거래량 배율 (CLI override 가능)
MIN_TRADING_VALUE = 0.0             # 20일 평균 거래대금 하한 (KRW). 0 = 필터 없음
TV_LOOKBACK_DAYS = 20               # 거래대금 이동평균 기간
SLOT_FRACTION = 0.0                 # 0 = legacy (cash/N 균등). >0 = 총자산 대비 슬롯 비중
MIN_HISTORY_DAYS = 252              # 최소 데이터 일수 (허위신호 방지)
KR_CURRENCY = "KRW"
ESTIMATION_METHOD = "bayes_stein"

BUY_COMMISSION = 0.00015            # 매수 수수료 0.015%
SELL_COMMISSION = 0.00015           # 매도 수수료 0.015%
SELL_TAX = 0.0020                   # 매도 세금 0.20%
                                    # round-trip 합계 0.23%


# ── Data Loaders ───────────────────────────────────────────────

def connect():
    return psycopg2.connect(**DB_CONFIG)


def load_kr_universe(conn) -> pd.DataFrame:
    q = """
        SELECT product_id, ticker, name, market
        FROM products
        WHERE currency = %s
        ORDER BY product_id
    """
    return pd.read_sql(q, conn, params=(KR_CURRENCY,))


def load_ohlcv(conn, product_ids: List[int],
               end_date: Optional[date] = None) -> pd.DataFrame:
    ids = ",".join(str(pid) for pid in product_ids)
    where = [f"product_id IN ({ids})"]
    if end_date:
        where.append(f"trade_date <= '{end_date}'")
    q = f"""
        SELECT product_id, trade_date,
               open::float AS open, high::float AS high,
               low::float AS low, close::float AS close,
               COALESCE(volume, 0)::bigint AS volume
        FROM market_data
        WHERE {' AND '.join(where)}
          AND open IS NOT NULL AND close IS NOT NULL AND close > 0
        ORDER BY product_id, trade_date
    """
    return pd.read_sql(q, conn, parse_dates=["trade_date"])


def load_expected_returns(conn, product_ids: List[int]) -> pd.DataFrame:
    ids = ",".join(str(pid) for pid in product_ids)
    q = f"""
        SELECT product_id, snapshot_date, annual_expected_return::float AS er
        FROM expected_returns_snapshot
        WHERE product_id IN ({ids})
          AND estimation_method = %s
          AND annual_expected_return IS NOT NULL
        ORDER BY product_id, snapshot_date
    """
    return pd.read_sql(q, conn, params=(ESTIMATION_METHOD,),
                       parse_dates=["snapshot_date"])


# ── Signal Detection ───────────────────────────────────────────

def detect_signals_per_ticker(df: pd.DataFrame,
                              ath_ratio: float = ATH_BREAKOUT_RATIO,
                              vol_ratio: float = VOLUME_RATIO,
                              min_tv: float = MIN_TRADING_VALUE,
                              tv_lookback: int = TV_LOOKBACK_DAYS) -> pd.Series:
    """
    df: 단일 종목, trade_date 정렬됨. 컬럼: open, high, low, close, volume

    Filter:
      - 당일 high >= prev_ATH * ath_ratio
      - 당일 volume >= prev_vol * vol_ratio
      - 직전 N일 평균 거래대금 (close*volume) >= min_tv (look-ahead 방지 위해 shift(1))
    """
    if len(df) < MIN_HISTORY_DAYS:
        return pd.Series(False, index=df.index)

    high = df["high"]
    prev_ath = high.shift(1).expanding().max()
    ath_break = high >= prev_ath * ath_ratio

    prev_vol = df["volume"].shift(1)
    vol_break = (prev_vol > 0) & (df["volume"] >= prev_vol * vol_ratio)

    # 거래대금 필터: 신호일 직전 N일 평균 (close * volume)
    if min_tv > 0:
        trading_value = df["close"] * df["volume"]
        avg_tv = trading_value.shift(1).rolling(tv_lookback,
                                                min_periods=tv_lookback).mean()
        tv_ok = avg_tv >= min_tv
    else:
        tv_ok = pd.Series(True, index=df.index)

    sig = ath_break & vol_break & tv_ok
    sig.iloc[:MIN_HISTORY_DAYS] = False
    return sig.fillna(False)


def precompute_signals(ticker_data: Dict[str, pd.DataFrame],
                       ath_ratio: float = ATH_BREAKOUT_RATIO,
                       vol_ratio: float = VOLUME_RATIO,
                       min_tv: float = MIN_TRADING_VALUE,
                       verbose: bool = True) -> Dict[date, Set[str]]:
    out: Dict[date, Set[str]] = defaultdict(set)
    n_signals = 0
    for ticker, df in ticker_data.items():
        sig = detect_signals_per_ticker(df, ath_ratio=ath_ratio,
                                        vol_ratio=vol_ratio, min_tv=min_tv)
        if not sig.any():
            continue
        sig_dates = df.loc[sig, "trade_date"]
        for td in sig_dates:
            d = td.date() if hasattr(td, "date") else td
            out[d].add(ticker)
            n_signals += 1
    if verbose:
        print(f"  signal events: {n_signals:,}, signal days: {len(out):,}")
    return dict(out)


# ── Ranking ────────────────────────────────────────────────────

def build_er_lookup(er_df: pd.DataFrame, ticker_map: Dict[int, str]) -> Dict[str, pd.Series]:
    """ticker → date-indexed expected_return Series (정렬됨)"""
    out: Dict[str, pd.Series] = {}
    for pid, grp in er_df.groupby("product_id"):
        ticker = ticker_map.get(pid)
        if not ticker:
            continue
        s = grp.set_index("snapshot_date")["er"].sort_index()
        out[ticker] = s
    return out


def lookup_er_asof(er_lookup: Dict[str, pd.Series], ticker: str, d: date) -> Optional[float]:
    """ticker의 d 시점 직전(또는 동일) 가장 최근 expected_return"""
    s = er_lookup.get(ticker)
    if s is None or len(s) == 0:
        return None
    target = pd.Timestamp(d)
    # 신호일 직전 스냅샷만 사용 (당일 포함하지 않음 — 미래 정보 차단)
    valid = s.loc[s.index < target]
    if len(valid) == 0:
        return None
    return float(valid.iloc[-1])


def rank_signals(signals_by_date: Dict[date, Set[str]],
                 er_lookup: Dict[str, pd.Series],
                 top_n: int) -> Dict[date, List[str]]:
    """일별 신호를 Bayes-Stein 기대수익률 내림차순으로 정렬, 상위 N개.

    재현성 보장: set 순회 → sorted 사용. ER 동률 시 ticker 이름 알파벳 순 tie-break.
    """
    out: Dict[date, List[str]] = {}
    for d, tickers in signals_by_date.items():
        ranked = []
        for t in sorted(tickers):                # deterministic 순회
            er = lookup_er_asof(er_lookup, t, d)
            if er is not None:
                ranked.append((er, t))
        if not ranked:
            continue
        # ER 내림차순, 동률 시 ticker 알파벳 오름차순
        ranked.sort(key=lambda x: (-x[0], x[1]))
        out[d] = [t for _, t in ranked[:top_n]]
    return out


# ── Position / Trade ────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'TP', 'SL', 'FORCE_CLOSE'
    holding_days: int


@dataclass
class Position:
    ticker: str
    entry_date: date
    entry_price: float
    shares: int
    entry_cost: float                # 수수료 포함 매수 총액
    tp_pct: float = TP_PCT
    sl_pct: float = SL_PCT
    tp_price: float = 0.0
    sl_price: float = 0.0
    no_sl: bool = False              # True면 SL 비활성, 본전 회복 시 청산
    has_been_underwater: bool = False  # entry 아래로 한 번이라도 내려갔는지

    def __post_init__(self):
        self.tp_price = self.entry_price * (1 + self.tp_pct)
        self.sl_price = self.entry_price * (1 - self.sl_pct)

    def check_exit(self, today: date, high: float, low: float
                   ) -> Optional[Tuple[float, Trade]]:
        """SL이 TP보다 우선 (보수적). 하나라도 맞으면 전량 청산.

        no_sl=True 모드: SL 무시, 본전(entry_price) 회복 시 청산.
        - has_been_underwater state는 전일까지 기준 → 같은 바 내 down→up 모호함 회피
        """
        if self.no_sl:
            # 1) TP 최우선
            if high >= self.tp_price:
                return self._close(today, self.tp_price, "TP")
            # 2) 이전 바까지 underwater 였고 오늘 entry 회복
            if self.has_been_underwater and high >= self.entry_price:
                return self._close(today, self.entry_price, "BE")
            # 3) 오늘 underwater 진입 → 미래 바를 위해 state 갱신
            if low < self.entry_price:
                self.has_been_underwater = True
            return None

        # 기본 모드: SL/TP
        if low <= self.sl_price:
            return self._close(today, self.sl_price, "SL")
        if high >= self.tp_price:
            return self._close(today, self.tp_price, "TP")
        return None

    def force_close(self, today: date, close_price: float) -> Tuple[float, Trade]:
        return self._close(today, close_price, "FORCE_CLOSE")

    def _close(self, today: date, price: float, reason: str) -> Tuple[float, Trade]:
        gross = self.shares * price
        sell_cost = gross * (SELL_COMMISSION + SELL_TAX)
        net = gross - sell_cost
        pnl = net - self.entry_cost
        pnl_pct = (net / self.entry_cost - 1) * 100
        trade = Trade(
            ticker=self.ticker,
            entry_date=self.entry_date,
            exit_date=today,
            entry_price=self.entry_price,
            exit_price=price,
            shares=self.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            holding_days=(today - self.entry_date).days,
        )
        self.shares = 0
        return net, trade


# ── Simulator ──────────────────────────────────────────────────

@dataclass
class DailySnap:
    date: date
    cash: float
    positions_value: float
    total_equity: float
    n_positions: int
    n_entries: int
    n_exits: int


class Simulator:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL, verbose: bool = False,
                 tp_pct: float = TP_PCT, sl_pct: float = SL_PCT,
                 no_sl: bool = False, slot_fraction: float = SLOT_FRACTION):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.snaps: List[DailySnap] = []
        self.verbose = verbose
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.no_sl = no_sl
        self.slot_fraction = slot_fraction

    def run(self, calendar: List[date],
            ranked_signals: Dict[date, List[str]],
            bar_lookup: Dict[str, Dict[date, dict]]):
        pending: List[str] = []   # 전일 신호 → 당일 시가 진입
        last_close: Dict[str, float] = {}

        n_days = len(calendar)
        report_step = max(1, n_days // 20)

        for i, today in enumerate(calendar):
            n_entries = n_exits = 0

            # 1) 보유 포지션 SL/TP 체크 (당일 high/low 기반, intra-bar)
            closed = []
            for tkr, pos in list(self.positions.items()):
                bar = bar_lookup.get(tkr, {}).get(today)
                if bar is None:
                    continue
                result = pos.check_exit(today, bar["high"], bar["low"])
                if result:
                    cash_in, trade = result
                    self.cash += cash_in
                    self.trades.append(trade)
                    n_exits += 1
                    closed.append(tkr)
            for t in closed:
                del self.positions[t]

            # 2) 전일 신호 → 당일 시가 진입 (top N from ranking)
            if pending:
                # 보유 중인 종목 제외, 시가 데이터 없는 종목 제외
                candidates = [
                    t for t in pending
                    if t not in self.positions
                    and bar_lookup.get(t, {}).get(today) is not None
                    and bar_lookup[t][today]["open"] > 0
                ]
                if candidates and self.cash > 0:
                    if self.slot_fraction > 0:
                        # Fixed slot: 각 신규 진입 = 총자산 × slot_fraction
                        # 총자산 = cash + 보유 포지션 mark-to-market (전일 종가)
                        positions_value = sum(
                            p.shares * last_close.get(t, p.entry_price)
                            for t, p in self.positions.items()
                        )
                        total_equity = self.cash + positions_value
                        slot_size = total_equity * self.slot_fraction
                    else:
                        # Legacy: cash/n_candidates 균등 분배
                        slot_size = self.cash / len(candidates)

                    for tkr in candidates:
                        if self.cash <= 0:
                            break
                        alloc = min(slot_size, self.cash)
                        bar = bar_lookup[tkr][today]
                        op = bar["open"]
                        shares = int(alloc / (op * (1 + BUY_COMMISSION)))
                        if shares <= 0:
                            continue
                        cost = shares * op * (1 + BUY_COMMISSION)
                        # 안전망: 잔여 cash 초과 방지
                        if cost > self.cash:
                            shares = int(self.cash / (op * (1 + BUY_COMMISSION)))
                            if shares <= 0:
                                continue
                            cost = shares * op * (1 + BUY_COMMISSION)
                        self.cash -= cost
                        self.positions[tkr] = Position(
                            ticker=tkr, entry_date=today,
                            entry_price=op, shares=shares, entry_cost=cost,
                            tp_pct=self.tp_pct, sl_pct=self.sl_pct,
                            no_sl=self.no_sl,
                        )
                        n_entries += 1
                pending = []

            # 3) 당일 종가 기준 신호 → 익일 진입 대기 (이미 보유 중인 종목 제외)
            today_signals = ranked_signals.get(today, [])
            if today_signals:
                pending = [t for t in today_signals if t not in self.positions]

            # 4) 일별 스냅샷 (adj_close 기준 평가)
            pos_value = 0.0
            for tkr, pos in self.positions.items():
                bar = bar_lookup.get(tkr, {}).get(today)
                if bar:
                    cp = bar["close"]
                    last_close[tkr] = cp
                else:
                    cp = last_close.get(tkr, pos.entry_price)
                pos_value += pos.shares * cp
            equity = self.cash + pos_value
            self.snaps.append(DailySnap(
                date=today, cash=self.cash, positions_value=pos_value,
                total_equity=equity, n_positions=len(self.positions),
                n_entries=n_entries, n_exits=n_exits,
            ))

            if self.verbose and (i + 1) % report_step == 0:
                pct = (i + 1) / n_days * 100
                print(f"  [{pct:5.1f}%] {today} | equity={equity:,.0f} "
                      f"pos={len(self.positions)} cash={self.cash:,.0f}")

        # 5) 마지막일 잔여 포지션 강제 청산
        last_day = calendar[-1]
        for tkr, pos in list(self.positions.items()):
            bar = bar_lookup.get(tkr, {}).get(last_day)
            if bar:
                cash_in, trade = pos.force_close(last_day, bar["close"])
                self.cash += cash_in
                self.trades.append(trade)
        self.positions.clear()


# ── Stats ──────────────────────────────────────────────────────

def compute_stats(sim: Simulator) -> dict:
    trades = sim.trades
    n = len(trades)
    if n == 0:
        return {"trades": 0}

    df = pd.DataFrame([t.__dict__ for t in trades])
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    win_rate = len(wins) / n * 100
    avg_win = wins["pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    rrr = abs(avg_win / avg_loss) if avg_loss < 0 else float("inf")

    gp = wins["pnl"].sum() if len(wins) else 0.0
    gl = abs(losses["pnl"].sum()) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")

    total_pnl = df["pnl"].sum()
    final_equity = sim.snaps[-1].total_equity if sim.snaps else sim.initial_capital
    total_return = (final_equity / sim.initial_capital - 1) * 100

    eq = np.array([s.total_equity for s in sim.snaps])
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    mdd = float(dd.min() * 100) if len(dd) else 0.0

    n_days = len(sim.snaps)
    n_years = n_days / 252 if n_days else 1
    ann_return = ((final_equity / sim.initial_capital) ** (1 / n_years) - 1) * 100 \
                 if n_years > 0 else 0.0

    by_reason = df["exit_reason"].value_counts().to_dict()

    return {
        "trades": n,
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "rrr": rrr,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_pnl": float(total_pnl),
        "final_equity": float(final_equity),
        "total_return_pct": total_return,
        "ann_return_pct": ann_return,
        "mdd_pct": mdd,
        "exit_reasons": by_reason,
        "avg_holding_days": float(df["holding_days"].mean()),
    }


def print_yearly_breakdown(sim: Simulator):
    """청산기준 연도별 trade 통계 + 연말 자산 스냅샷"""
    if not sim.trades:
        print("  (no trades)")
        return

    # Trade-based stats (청산 연도 기준)
    df = pd.DataFrame([t.__dict__ for t in sim.trades])
    df["year"] = pd.to_datetime(df["exit_date"]).dt.year

    # Equity snapshot per year-end + 연내 진짜 MDD 계산
    snap_df = pd.DataFrame([
        {"date": s.date, "equity": s.total_equity} for s in sim.snaps
    ])
    snap_df["year"] = pd.to_datetime(snap_df["date"]).dt.year
    year_end_eq = snap_df.groupby("year")["equity"].last()

    # 진짜 within-year MDD: 연내 running peak 기준 trough까지의 최대 drop
    # (peak가 trough보다 먼저 발생해야 진짜 drawdown)
    yr_mdd: Dict[int, float] = {}
    for yr, sub in snap_df.groupby("year"):
        eq = sub["equity"].values
        if len(eq) == 0:
            yr_mdd[int(yr)] = 0.0
            continue
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / running_max
        yr_mdd[int(yr)] = float(dd.min() * 100)

    print(f"\n  === Yearly Breakdown ===")
    print(f"  {'Year':>4} {'Trades':>7} {'WR%':>6} {'PF':>6} {'RRR':>6} "
          f"{'PnL(M)':>10} {'EoY자산(M)':>11} {'Yr Ret%':>8} {'Yr MDD%':>8}")
    print(f"  {'-'*84}")

    prev_eoy = sim.initial_capital
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        n = len(sub)
        wins = sub[sub["pnl"] > 0]
        losses = sub[sub["pnl"] <= 0]
        wr = len(wins) / n * 100 if n else 0
        gw = wins["pnl"].sum() if len(wins) else 0
        gl = abs(losses["pnl"].sum()) if len(losses) else 0
        pf = gw / gl if gl > 0 else float("inf")
        avg_w = wins["pnl"].mean() if len(wins) else 0
        avg_l = losses["pnl"].mean() if len(losses) else 0
        rrr = abs(avg_w / avg_l) if avg_l < 0 else float("inf")
        pnl_m = sub["pnl"].sum() / 1e6
        eoy = year_end_eq.get(yr, prev_eoy)
        yr_ret = (eoy / prev_eoy - 1) * 100 if prev_eoy > 0 else 0
        yr_mdd_val = yr_mdd.get(int(yr), 0.0)
        print(f"  {yr:>4} {n:>7} {wr:>5.1f}% {pf:>6.2f} {rrr:>6.2f} "
              f"{pnl_m:>+10.1f} {eoy/1e6:>11,.1f} {yr_ret:>+7.2f}% {yr_mdd_val:>+7.2f}%")
        prev_eoy = eoy

    print(f"  {'-'*84}")


def print_stats(stats: dict, label: str = ""):
    print("\n" + "=" * 70)
    print(f"  백테스트 결과 {label}")
    print("=" * 70)
    if stats.get("trades", 0) == 0:
        print("  거래 없음")
        return
    print(f"  총 진입수            : {stats['trades']:,}")
    print(f"  승률                 : {stats['win_rate_pct']:.2f}%")
    print(f"  Profit Factor        : {stats['profit_factor']:.2f}")
    print(f"  RRR (avg_win/avg_loss): {stats['rrr']:.2f}")
    print(f"  평균 익절            : {stats['avg_win']:,.0f}")
    print(f"  평균 손절            : {stats['avg_loss']:,.0f}")
    print(f"  순손익               : {stats['total_pnl']:,.0f}")
    print(f"  최종자산             : {stats['final_equity']:,.0f}")
    print(f"  총수익률             : {stats['total_return_pct']:+.2f}%")
    print(f"  연환산수익률         : {stats['ann_return_pct']:+.2f}%")
    print(f"  MDD                  : {stats['mdd_pct']:.2f}%")
    print(f"  평균 보유일수        : {stats['avg_holding_days']:.1f}")
    print(f"  청산사유 분포        : {stats['exit_reasons']}")
    print("=" * 70)


# ── Reusable: data loading + simulation ───────────────────────

def load_all_data(end_d: Optional[date] = None, verbose: bool = True) -> dict:
    """DB에서 universe/OHLCV/expected_returns 로드 → ticker_data, bar_lookup, er_lookup, calendar"""
    if verbose:
        print("\n[데이터 로딩]")
    with connect() as conn:
        if verbose: print("  - KR universe...")
        universe = load_kr_universe(conn)
        if verbose: print(f"    KR 종목 (currency=KRW): {len(universe):,}")
        ticker_map = dict(zip(universe["product_id"], universe["ticker"]))
        pids = universe["product_id"].tolist()

        if verbose: print("  - OHLCV 벌크...")
        ohlcv = load_ohlcv(conn, pids, end_date=end_d)
        if verbose: print(f"    {len(ohlcv):,} bars")

        if verbose: print("  - Bayes-Stein 기대수익률...")
        er_df = load_expected_returns(conn, pids)
        if verbose: print(f"    {len(er_df):,} rows")

    if verbose: print("  - 종목별 분할 + bar lookup...")
    ticker_data: Dict[str, pd.DataFrame] = {}
    bar_lookup: Dict[str, Dict[date, dict]] = {}
    for pid, grp in ohlcv.groupby("product_id"):
        tkr = ticker_map.get(pid)
        if not tkr:
            continue
        df = grp.sort_values("trade_date").reset_index(drop=True)
        ticker_data[tkr] = df
        d_lookup = {}
        for row in df.itertuples(index=False):
            d = row.trade_date.date() if hasattr(row.trade_date, "date") else row.trade_date
            d_lookup[d] = {
                "open": float(row.open),
                "high": float(row.high),
                "low":  float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
            }
        bar_lookup[tkr] = d_lookup
    if verbose: print(f"    종목 수: {len(ticker_data):,}")

    full_dates = sorted({d for lkp in bar_lookup.values() for d in lkp.keys()})
    er_lookup = build_er_lookup(er_df, ticker_map)

    return {
        "ticker_data": ticker_data,
        "bar_lookup": bar_lookup,
        "er_lookup": er_lookup,
        "full_dates": full_dates,
    }


def run_with_params(loaded: dict, *,
                    ath_ratio: float = ATH_BREAKOUT_RATIO,
                    vol_ratio: float = VOLUME_RATIO,
                    min_tv: float = MIN_TRADING_VALUE,
                    top_n: int = TOP_N_PER_DAY,
                    tp_pct: float = TP_PCT,
                    sl_pct: float = SL_PCT,
                    no_sl: bool = False,
                    slot_fraction: float = SLOT_FRACTION,
                    initial_capital: float = INITIAL_CAPITAL,
                    start_d: Optional[date] = None,
                    end_d: Optional[date] = None,
                    verbose: bool = False,
                    cached_signals: Optional[dict] = None
                    ) -> Tuple[Simulator, dict]:
    """주어진 파라미터로 신호 계산 + 시뮬레이션 실행

    cached_signals: precomputed signals_by_date (재사용 시 신호 재계산 생략)
    """
    bar_lookup = loaded["bar_lookup"]
    er_lookup = loaded["er_lookup"]

    if cached_signals is not None:
        signals_by_date = cached_signals
    else:
        signals_by_date = precompute_signals(
            loaded["ticker_data"], ath_ratio=ath_ratio, vol_ratio=vol_ratio,
            min_tv=min_tv, verbose=verbose,
        )
    ranked = rank_signals(signals_by_date, er_lookup, top_n)

    cal = loaded["full_dates"]
    if start_d:
        cal = [d for d in cal if d >= start_d]
    if end_d:
        cal = [d for d in cal if d <= end_d]

    sim = Simulator(initial_capital=initial_capital, verbose=verbose,
                    tp_pct=tp_pct, sl_pct=sl_pct, no_sl=no_sl,
                    slot_fraction=slot_fraction)
    sim.run(cal, ranked, bar_lookup)
    stats = compute_stats(sim)
    return sim, stats


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--ath-ratio", type=float, default=ATH_BREAKOUT_RATIO,
                        help=f"ATH 돌파 배수 (기본 {ATH_BREAKOUT_RATIO})")
    parser.add_argument("--vol-ratio", type=float, default=VOLUME_RATIO,
                        help=f"거래량 배수 (기본 {VOLUME_RATIO})")
    parser.add_argument("--min-tv", type=float, default=MIN_TRADING_VALUE,
                        help="20일 평균 거래대금 하한 (KRW). 0=미적용")
    parser.add_argument("--top-n", type=int, default=TOP_N_PER_DAY,
                        help=f"일별 진입 상한 (기본 {TOP_N_PER_DAY})")
    parser.add_argument("--tp-pct", type=float, default=TP_PCT,
                        help=f"익절률 (기본 {TP_PCT})")
    parser.add_argument("--sl-pct", type=float, default=SL_PCT,
                        help=f"손절률 (기본 {SL_PCT})")
    parser.add_argument("--no-sl", action="store_true",
                        help="SL 비활성, 본전 회복 시 청산 (BE 모드)")
    parser.add_argument("--slot-fraction", type=float, default=SLOT_FRACTION,
                        help="슬롯 크기 (총자산 대비 비율). 0=legacy cash/N. 권장 0.10")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--yearly", action="store_true", help="연도별 breakdown 출력")
    args = parser.parse_args()

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    print("=" * 70)
    print("  ATH 돌파 + 거래량 필터 + Bayes-Stein TopN — 한국 주식")
    sl_label = "BE(no SL)" if args.no_sl else f"SL=-{args.sl_pct*100:.0f}%"
    sizing_label = (f"slot={args.slot_fraction*100:.0f}% of equity"
                    if args.slot_fraction > 0 else "cash/N (legacy)")
    print(f"  TP=+{args.tp_pct*100:.0f}% / {sl_label} | "
          f"ATH×{args.ath_ratio} | Vol×{args.vol_ratio} | "
          f"min_TV={args.min_tv:,.0f} | Top {args.top_n}")
    print(f"  Sizing: {sizing_label} | 자본: {INITIAL_CAPITAL:,.0f}원")
    print("=" * 70)

    t0 = time.time()
    loaded = load_all_data(end_d=end_d, verbose=True)

    print("\n[시뮬레이션]")
    sim, stats = run_with_params(
        loaded,
        ath_ratio=args.ath_ratio, vol_ratio=args.vol_ratio,
        min_tv=args.min_tv, top_n=args.top_n,
        tp_pct=args.tp_pct, sl_pct=args.sl_pct,
        no_sl=args.no_sl,
        slot_fraction=args.slot_fraction,
        initial_capital=INITIAL_CAPITAL,
        start_d=start_d, end_d=end_d,
        verbose=args.verbose,
    )
    print(f"  완료: 총 {len(sim.trades):,}건, "
          f"최종자산 {sim.snaps[-1].total_equity:,.0f}원")

    cal = loaded["full_dates"]
    if start_d: cal = [d for d in cal if d >= start_d]
    if end_d: cal = [d for d in cal if d <= end_d]
    print_stats(stats, label=f"({cal[0]} ~ {cal[-1]})")

    if args.yearly:
        print_yearly_breakdown(sim)

    print(f"\n총 소요시간: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
