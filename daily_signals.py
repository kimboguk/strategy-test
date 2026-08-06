#!/usr/bin/env python3
"""
일별 신호 산출 + forward-test 가상 포지션 트래킹.

매일 sync 이후 실행:
  $env:DB_NAME="equity"
  $env:PGPASSWORD="postgres"
  python sync_equity_db.py
  python daily_signals.py [--as-of YYYY-MM-DD]

흐름 (오늘 = today):
  1) 가상 자본 초기화 (forward_capital이 비어있으면 FORWARD_INITIAL_CAPITAL로 시작)
  2) Pending 포지션 처리 — D+1 entry price 확정 (avg_close_open: yesterday close + today open / 2)
  3) Open 포지션 TP/SL 체크 — 오늘 high/low 기반
  4) 오늘 종가 기준 신호 검출 → top-N 산출 (baseline 파라미터)
  5) 신호를 forward_signals에 기록 + forward_positions에 pending으로 INSERT
  6) 일별 자본 스냅샷 갱신
  7) 콘솔 리포트
"""
import argparse
import os
import sys
import warnings
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# pandas read_sql warning suppression (psycopg2 connection 사용 시 발생)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# baseline 파라미터/유틸 재사용
from run_ath_volume_breakout import (
    DB_CONFIG,
    TP_PCT, SL_PCT,
    SLOT_FRACTION,
    ATH_BREAKOUT_RATIO, VOLUME_RATIO,
    MIN_TRADING_VALUE, TV_LOOKBACK_DAYS,
    MIN_HISTORY_DAYS, KR_CURRENCY,
    RANKING_METHOD, LOOKBACK_DAYS, RISK_FREE_RATE,
    ENTRY_TIMING, TOP_N_PER_DAY,
    BUY_COMMISSION, SELL_COMMISSION, SELL_TAX,
    load_kr_universe, load_ohlcv, load_expected_returns,
    detect_signals_per_ticker,
    build_er_lookup, lookup_er_asof,
)

# forward 모의 장부 초기자본 — 백테스트 공유 상수(INITIAL_CAPITAL=1억)와 분리.
# 실거래 기준선이므로 실제 투입 예정 금액(1,000만원)으로 추적.
FORWARD_INITIAL_CAPITAL = 10_000_000


# ── DB connection ──────────────────────────────────────────────

def connect():
    return psycopg2.connect(**DB_CONFIG)


# ── 1) 가상 자본 초기화/조회 ─────────────────────────────────────

def get_or_init_capital(cur, today: date) -> Tuple[float, float, float]:
    """전일 자본 상태 조회. 없으면 초기화.
    Returns: (cash, positions_value, total_equity)
    """
    cur.execute("""
        SELECT cash, positions_value, total_equity
        FROM forward_capital
        WHERE snapshot_date < %s
        ORDER BY snapshot_date DESC LIMIT 1
    """, (today,))
    row = cur.fetchone()
    if row:
        return float(row[0]), float(row[1]), float(row[2])
    # 첫 실행
    return float(FORWARD_INITIAL_CAPITAL), 0.0, float(FORWARD_INITIAL_CAPITAL)


# ── 2) Pending → Open: entry_price 확정 ────────────────────────

def finalize_pending_entries(cur, today: date,
                             cash_available: float,
                             total_equity: float) -> Tuple[float, int, list]:
    """status='pending' 포지션을 오늘 데이터로 entry_price 확정.

    avg_close_open: signal_date의 close + today open 의 평균 = entry_price.
    """
    # avg_close_open / next_open / next_close 모두 D+1 데이터 필수
    # → signal_date < today 조건 필수
    cur.execute("""
        SELECT id, signal_date, ticker, product_id, entry_timing,
               tp_pct, sl_pct, slot_fraction, rank_at_signal
        FROM forward_positions
        WHERE status = 'pending' AND signal_date < %s
        ORDER BY signal_date, rank_at_signal
    """, (today,))
    pending_rows = cur.fetchall()
    if not pending_rows:
        return cash_available, 0, []

    n_entries = 0
    entries_log = []
    for row in pending_rows:
        pos_id, sig_date, ticker, pid, e_timing, tp, sl, slot_frac, rank_at = row

        # 가격 데이터 조회: signal_date close + today open
        cur.execute("""
            SELECT trade_date, open, close
            FROM market_data
            WHERE product_id = %s
              AND trade_date IN (%s, %s)
            ORDER BY trade_date
        """, (pid, sig_date, today))
        bars = cur.fetchall()
        bars_dict = {r[0]: (float(r[1]) if r[1] else 0.0,
                             float(r[2]) if r[2] else 0.0)
                     for r in bars}

        sig_bar = bars_dict.get(sig_date)
        today_bar = bars_dict.get(today)

        # avg_close_open: 신호일 close + 오늘 open
        if e_timing == "avg_close_open":
            if sig_bar is None or today_bar is None:
                # 데이터 누락 — skip
                cur.execute("""
                    UPDATE forward_positions
                    SET status='skipped', exit_reason='SKIP',
                        notes='no price data for entry'
                    WHERE id=%s
                """, (pos_id,))
                continue
            entry_price = (float(sig_bar[1]) + float(today_bar[0])) / 2  # close + open / 2
        elif e_timing == "next_open":
            if today_bar is None:
                cur.execute("""
                    UPDATE forward_positions
                    SET status='skipped', exit_reason='SKIP',
                        notes='no open price'
                    WHERE id=%s
                """, (pos_id,))
                continue
            entry_price = float(today_bar[0])
        elif e_timing == "next_close":
            if today_bar is None:
                cur.execute("""
                    UPDATE forward_positions
                    SET status='skipped', exit_reason='SKIP',
                        notes='no close price'
                    WHERE id=%s
                """, (pos_id,))
                continue
            entry_price = float(today_bar[1])
        else:
            # same_close 는 pending 단계 없이 바로 처리 → 여기 안 옴
            continue

        if entry_price <= 0:
            cur.execute("""
                UPDATE forward_positions
                SET status='skipped', exit_reason='SKIP',
                    notes='zero/negative price'
                WHERE id=%s
            """, (pos_id,))
            continue

        # 슬롯 사이징
        slot_size = total_equity * float(slot_frac)
        alloc = min(slot_size, cash_available)
        shares = int(alloc / (entry_price * (1 + BUY_COMMISSION)))
        if shares <= 0:
            cur.execute("""
                UPDATE forward_positions
                SET status='skipped', exit_reason='SKIP',
                    notes='cash insufficient'
                WHERE id=%s
            """, (pos_id,))
            continue

        cost = shares * entry_price * (1 + BUY_COMMISSION)
        if cost > cash_available:
            shares = int(cash_available / (entry_price * (1 + BUY_COMMISSION)))
            if shares <= 0:
                cur.execute("""
                    UPDATE forward_positions
                    SET status='skipped', exit_reason='SKIP',
                        notes='cash insufficient (recalc)'
                    WHERE id=%s
                """, (pos_id,))
                continue
            cost = shares * entry_price * (1 + BUY_COMMISSION)

        tp_price = entry_price * (1 + float(tp))
        sl_price = entry_price * (1 - float(sl))

        cur.execute("""
            UPDATE forward_positions
            SET status='open',
                entry_date=%s,
                entry_price=%s,
                shares=%s,
                cost_basis=%s,
                capital_at_entry=%s,
                tp_price=%s,
                sl_price=%s
            WHERE id=%s
        """, (today, entry_price, shares, cost, total_equity,
              tp_price, sl_price, pos_id))

        cash_available -= cost
        n_entries += 1
        entries_log.append((ticker, entry_price, shares, cost))

    return cash_available, n_entries, entries_log


# ── 3) Open 포지션 TP/SL 체크 ──────────────────────────────────

def check_exits(cur, today: date) -> Tuple[float, int, list]:
    """status='open' 포지션 → 오늘 high/low로 TP/SL 체크."""
    cur.execute("""
        SELECT id, ticker, product_id, entry_date, entry_price,
               shares, cost_basis, tp_price, sl_price
        FROM forward_positions
        WHERE status='open'
    """)
    open_rows = cur.fetchall()

    cash_in = 0.0
    n_exits = 0
    exits_log = []
    for row in open_rows:
        pos_id, ticker, pid, e_date, e_price, shares, cost_basis, tp_p, sl_p = row

        cur.execute("""
            SELECT high, low FROM market_data
            WHERE product_id=%s AND trade_date=%s
        """, (pid, today))
        bar = cur.fetchone()
        if bar is None:
            continue
        high, low = float(bar[0]), float(bar[1])

        exit_reason = None
        exit_price = None
        # SL 우선
        if low <= float(sl_p):
            exit_reason = "SL"
            exit_price = float(sl_p)
        elif high >= float(tp_p):
            exit_reason = "TP"
            exit_price = float(tp_p)

        if exit_reason is None:
            continue

        gross = shares * exit_price
        sell_cost = gross * (SELL_COMMISSION + SELL_TAX)
        net = gross - sell_cost
        pnl_krw = net - float(cost_basis)
        pnl_pct = (net / float(cost_basis) - 1) * 100
        holding_days = (today - e_date).days

        cur.execute("""
            UPDATE forward_positions
            SET status='closed',
                exit_date=%s, exit_price=%s, exit_reason=%s,
                pnl_krw=%s, pnl_pct=%s, holding_days=%s
            WHERE id=%s
        """, (today, exit_price, exit_reason, pnl_krw, pnl_pct,
              holding_days, pos_id))

        cash_in += net
        n_exits += 1
        exits_log.append((ticker, exit_reason, exit_price, pnl_krw, pnl_pct))

    return cash_in, n_exits, exits_log


# ── 4) 오늘 신호 검출 + ranking ────────────────────────────────

def detect_today_signals(today: date) -> Tuple[List[dict], dict]:
    """오늘 종가 기준 신호 검출. baseline 파라미터.

    Returns: (top_n_picks, signal_meta)
      top_n_picks: 상위 N개 [{ticker, product_id, metric_value, ...}, ...]
    """
    with connect() as conn:
        # universe (필터 적용)
        universe = load_kr_universe(conn, apply_quality_filter=True)
        ticker_map = dict(zip(universe["product_id"], universe["ticker"]))
        pids = universe["product_id"].tolist()

        # OHLCV 전 기간 (ATH expanding max)
        ohlcv = load_ohlcv(conn, pids, end_date=today)

        # ER lookup
        er_df = load_expected_returns(conn, pids,
                                      method=RANKING_METHOD,
                                      lookback=LOOKBACK_DAYS,
                                      source="rt")
        er_lookup = build_er_lookup(er_df, ticker_map)

    # ticker별 signal 검출 (오늘만 관심)
    picks = []
    for pid, grp in ohlcv.groupby("product_id"):
        ticker = ticker_map.get(pid)
        if not ticker:
            continue
        df = grp.sort_values("trade_date").reset_index(drop=True)
        if len(df) < MIN_HISTORY_DAYS:
            continue

        # 오늘 행 찾기
        df["d"] = df["trade_date"].dt.date
        today_mask = df["d"] == today
        if not today_mask.any():
            continue
        i = df.index[today_mask][0]
        if i < MIN_HISTORY_DAYS:
            continue

        # 오늘 신호 조건
        high = df.iloc[i]["high"]
        prev_high_max = df["high"].iloc[:i].max()
        if pd.isna(prev_high_max) or high < prev_high_max * ATH_BREAKOUT_RATIO:
            continue

        prev_vol = df.iloc[i - 1]["volume"]
        vol = df.iloc[i]["volume"]
        if not (prev_vol > 0 and vol >= prev_vol * VOLUME_RATIO):
            continue

        # 거래대금 필터
        if MIN_TRADING_VALUE > 0:
            if i < TV_LOOKBACK_DAYS:
                continue
            recent = df.iloc[i - TV_LOOKBACK_DAYS:i]
            avg_tv = (recent["close"] * recent["volume"]).mean()
            if avg_tv < MIN_TRADING_VALUE:
                continue
        else:
            avg_tv = None

        # ranking metric (오늘 직전 스냅샷)
        er = lookup_er_asof(er_lookup, ticker, today)
        if er is None:
            continue

        picks.append({
            "ticker": ticker,
            "product_id": int(pid),
            "metric_value": float(er),
            "high_at_signal": float(high),
            "prev_ath": float(prev_high_max),
            "close_at_signal": float(df.iloc[i]["close"]),
            "volume_at_signal": int(vol),
            "prev_volume": int(prev_vol),
            "avg_trading_value": float(avg_tv) if avg_tv is not None else None,
        })

    # ranking
    picks.sort(key=lambda x: (-x["metric_value"], x["ticker"]))
    top = picks[:TOP_N_PER_DAY]
    for r, p in enumerate(top, 1):
        p["rank"] = r

    return top, {
        "ranking_method": RANKING_METHOD,
        "lookback_days": LOOKBACK_DAYS,
        "total_candidates": len(picks),
    }


# ── 5) 신호/포지션 INSERT ──────────────────────────────────────

def record_signals(cur, today: date, picks: List[dict], meta: dict,
                   already_open: set):
    for p in picks:
        if p["ticker"] in already_open:
            continue   # 보유 중인 종목은 추가 진입 안 함

        # forward_signals (UPSERT — 같은 날 재실행 가능)
        cur.execute("""
            INSERT INTO forward_signals (
                signal_date, rank, ticker, product_id,
                metric_value, ranking_method, lookback_days,
                high_at_signal, prev_ath, close_at_signal,
                volume_at_signal, prev_volume, avg_trading_value
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (signal_date, ticker)
            DO UPDATE SET
                rank=EXCLUDED.rank, metric_value=EXCLUDED.metric_value
        """, (today, p["rank"], p["ticker"], p["product_id"],
              p["metric_value"], meta["ranking_method"], meta["lookback_days"],
              p["high_at_signal"], p["prev_ath"], p["close_at_signal"],
              p["volume_at_signal"], p["prev_volume"], p["avg_trading_value"]))

        # forward_positions (pending) — 같은 날 동일 ticker 신호 재발생 시 무시
        cur.execute("""
            INSERT INTO forward_positions (
                signal_date, rank_at_signal, ticker, product_id,
                entry_timing, status,
                tp_pct, sl_pct, slot_fraction
            ) VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s, %s)
            ON CONFLICT (signal_date, ticker) DO NOTHING
        """, (today, p["rank"], p["ticker"], p["product_id"],
              ENTRY_TIMING, TP_PCT, SL_PCT, SLOT_FRACTION))


# ── 6) 일별 자본 스냅샷 ────────────────────────────────────────

def update_capital_snapshot(cur, today: date, cash: float,
                            n_entries: int, n_exits: int):
    # ── 상태 기반(멱등) 재계산 ──
    # 전달받은 cash/n_entries/n_exits(증분값)는 무시하고 today 시점 원장에서 재계산.
    # → 같은 날 사이클을 여러 번 재실행해도 실현손익이 소실되지 않음.
    #
    # today 시점 open = entry_date<=today AND (exit_date IS NULL OR exit_date>today)
    cur.execute("""
        SELECT fp.product_id, fp.shares, fp.cost_basis
        FROM forward_positions fp
        WHERE fp.entry_date IS NOT NULL AND fp.entry_date <= %s
          AND (fp.exit_date IS NULL OR fp.exit_date > %s)
    """, (today, today))
    pos_rows = cur.fetchall()

    positions_value = 0.0
    open_cost = 0.0
    for pid, shares, cost in pos_rows:
        open_cost += float(cost) if cost is not None else 0.0
        cur.execute("""
            SELECT close FROM market_data
            WHERE product_id=%s AND trade_date<=%s
            ORDER BY trade_date DESC LIMIT 1
        """, (pid, today))
        r = cur.fetchone()
        if r and r[0]:
            positions_value += float(shares) * float(r[0])

    # 실현손익 누적 (exit_date <= today) + 당일 진입/청산 건수 — 모두 상태 기반
    cur.execute("""SELECT COALESCE(SUM(pnl_krw), 0) FROM forward_positions
                   WHERE exit_date IS NOT NULL AND exit_date <= %s""", (today,))
    realized = float(cur.fetchone()[0] or 0.0)
    cur.execute("SELECT count(*) FROM forward_positions WHERE entry_date=%s", (today,))
    n_entries = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM forward_positions WHERE exit_date=%s", (today,))
    n_exits = cur.fetchone()[0]

    # 현금 = 초기자본 + 실현손익(청산분) − open 포지션 투입원가
    cash = float(FORWARD_INITIAL_CAPITAL) + realized - open_cost
    total_equity = cash + positions_value

    # 전일 equity 조회 → 오늘 daily_pnl
    cur.execute("""
        SELECT total_equity FROM forward_capital
        WHERE snapshot_date<%s ORDER BY snapshot_date DESC LIMIT 1
    """, (today,))
    prev = cur.fetchone()
    prev_eq = float(prev[0]) if prev else float(FORWARD_INITIAL_CAPITAL)
    daily_pnl = total_equity - prev_eq
    daily_ret = (total_equity / prev_eq - 1) * 100 if prev_eq > 0 else 0.0
    cum_ret = (total_equity / float(FORWARD_INITIAL_CAPITAL) - 1) * 100

    cur.execute("""
        INSERT INTO forward_capital (
            snapshot_date, cash, positions_value, total_equity,
            n_open_positions, n_entries_today, n_exits_today,
            daily_pnl, daily_return_pct, cum_return_pct
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (snapshot_date) DO UPDATE SET
            cash=EXCLUDED.cash,
            positions_value=EXCLUDED.positions_value,
            total_equity=EXCLUDED.total_equity,
            n_open_positions=EXCLUDED.n_open_positions,
            n_entries_today=EXCLUDED.n_entries_today,
            n_exits_today=EXCLUDED.n_exits_today,
            daily_pnl=EXCLUDED.daily_pnl,
            daily_return_pct=EXCLUDED.daily_return_pct,
            cum_return_pct=EXCLUDED.cum_return_pct
    """, (today, cash, positions_value, total_equity,
          len(pos_rows), n_entries, n_exits,
          daily_pnl, daily_ret, cum_ret))

    return total_equity, positions_value, len(pos_rows), daily_ret, cum_ret


# ── 7) Reporter ────────────────────────────────────────────────

def print_report(today, picks, meta, entries, exits,
                 cash, total_equity, n_open, daily_ret, cum_ret):
    print("\n" + "=" * 72)
    print(f"  Daily Report — {today}")
    print("=" * 72)
    print(f"\n[자본]")
    print(f"  cash             : {cash:>20,.0f} KRW")
    print(f"  positions value  : {total_equity-cash:>20,.0f} KRW")
    print(f"  total equity     : {total_equity:>20,.0f} KRW")
    print(f"  open positions   : {n_open:>20}")
    print(f"  daily return     : {daily_ret:>+19.2f}%")
    print(f"  cumulative return: {cum_ret:>+19.2f}%")

    if exits:
        print(f"\n[Exits ({len(exits)})]")
        for tkr, reason, px, pnl, pct in exits:
            print(f"  {tkr:8s}  {reason:5s}  exit={px:>10,.2f}  "
                  f"PnL={pnl:>+12,.0f} ({pct:+.2f}%)")

    if entries:
        print(f"\n[Entries ({len(entries)})]")
        for tkr, px, sh, cost in entries:
            print(f"  {tkr:8s}  entry={px:>10,.2f}  "
                  f"shares={sh:>6}  cost={cost:>14,.0f}")

    print(f"\n[Today's Signals — {meta['ranking_method']}, "
          f"lookback={meta['lookback_days']}일]")
    print(f"  candidates passing all filters: {meta['total_candidates']}")
    if picks:
        print(f"  Top {len(picks)} (ranked):")
        for p in picks:
            print(f"   #{p['rank']} {p['ticker']:8s}  metric={p['metric_value']:+.4f}  "
                  f"close={p['close_at_signal']:>10,.0f}  "
                  f"high={p['high_at_signal']:>10,.0f}  "
                  f"vol_x={p['volume_at_signal']/p['prev_volume']:.1f}")
    else:
        print("  (no signals today)")
    print("\n" + "=" * 72)


# ── Main ────────────────────────────────────────────────────────

def get_pending_trading_days(cur, upto: date) -> List[date]:
    """forward_capital 마지막 스냅샷 다음 거래일 ~ upto 까지의 실제 거래일 목록.

    스케줄러가 하루 걸러도 다음 실행에서 순서대로 메워지도록(갭 방지) 사용.
    forward_capital 이 비어있으면(리셋 직후) 당일만 처리.
    """
    cur.execute("SELECT MAX(snapshot_date) FROM forward_capital")
    last = cur.fetchone()[0]
    if last is None:
        return [upto]
    cur.execute("""
        SELECT DISTINCT trade_date FROM market_data
        WHERE trade_date > %s AND trade_date <= %s
        ORDER BY trade_date
    """, (last, upto))
    return [r[0] for r in cur.fetchall()]


def run_cycle(today: date):
    """하루치 사이클 (단일 트랜잭션). 반환: 리포트 인자 묶음."""
    with connect() as conn:
        cur = conn.cursor()

        # 1) 자본 상태
        cash, _, total_equity = get_or_init_capital(cur, today)

        # 2) Pending → Open
        cash, n_entries, entries_log = finalize_pending_entries(
            cur, today, cash, total_equity)

        # 3) Open 포지션 exits 체크
        cash_in, n_exits, exits_log = check_exits(cur, today)
        cash += cash_in

        # 4) 오늘 신호 검출
        picks, meta = detect_today_signals(today)

        # 보유 중인 ticker (중복 진입 방지)
        cur.execute("SELECT ticker FROM forward_positions WHERE status='open'")
        already_open = {r[0] for r in cur.fetchall()}

        # 5) 신호 기록
        record_signals(cur, today, picks, meta, already_open)

        # 6) 자본 스냅샷 (상태 기반 재계산 — 멱등)
        total_equity, pos_value, n_open, daily_ret, cum_ret = update_capital_snapshot(
            cur, today, cash, n_entries, n_exits)
        cash = total_equity - pos_value   # 스냅샷 정합 현금 (스레드값 대체)

        conn.commit()

    return (picks, meta, entries_log, exits_log,
            cash, total_equity, n_open, daily_ret, cum_ret)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=str, default=None,
                        help="처리 기준일 (YYYY-MM-DD). 기본: 로컬 DB의 max(trade_date)")
    parser.add_argument("--no-catchup", action="store_true",
                        help="누락 거래일 백필 없이 당일만 처리")
    args = parser.parse_args()

    # 처리할 거래일 결정 (기본: 누락분 catch-up)
    with connect() as conn:
        cur = conn.cursor()
        if args.as_of:
            days = [datetime.strptime(args.as_of, "%Y-%m-%d").date()]
        else:
            cur.execute("SELECT MAX(trade_date) FROM market_data")
            upto = cur.fetchone()[0]
            if upto is None:
                print("market_data 비어있음. sync 먼저 실행.")
                sys.exit(1)
            if args.no_catchup:
                days = [upto]
            else:
                days = get_pending_trading_days(cur, upto)
                if not days:
                    print(f"이미 최신 ({upto}) — 처리할 거래일 없음.")
                    return

    print(f"\n=== Daily Signal Pipeline — {len(days)}개 거래일 "
          f"({days[0]} ~ {days[-1]}) ===")
    print(f"  baseline: ATH×{ATH_BREAKOUT_RATIO} Vol×{VOLUME_RATIO} "
          f"min_TV={MIN_TRADING_VALUE/1e9:.0f}B Top{TOP_N_PER_DAY}")
    print(f"  TP=+{TP_PCT*100:.0f}% SL=-{SL_PCT*100:.0f}% slot={SLOT_FRACTION*100:.0f}% "
          f"entry={ENTRY_TIMING}")
    print(f"  ranking={RANKING_METHOD}({LOOKBACK_DAYS}일) "
          f"filter=is_selected=TRUE")
    if len(days) > 10:
        print(f"  ⚠ 백필 {len(days)}일 — 장기 미실행 구간을 메웁니다.")

    # 거래일 순차 처리 (갭 없이 진행돼야 자본 회계가 정합)
    for i, day in enumerate(days):
        result = run_cycle(day)
        if i < len(days) - 1:
            _, _, _, _, _, eq, n_open, _, _ = result
            print(f"  [백필 {i+1}/{len(days)}] {day} "
                  f"equity={eq:,.0f} open={n_open}")

    # 마지막 날 리포트
    print_report(days[-1], *result)


if __name__ == "__main__":
    main()
