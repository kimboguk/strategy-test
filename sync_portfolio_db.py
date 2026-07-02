#!/usr/bin/env python3
"""
portfolio_db (GPU 서버 운영 DB) → 로컬 equity 전면 재취득.

리팩터링된 신스키마 대응:
- ER/지표 정본이 bt_*(백테스트 historical) / rt_*(실시간) 로 분리됨 → 로컬도 분리 미러링.
- bt_*/rt_* 는 원격에서 ticker 키 → COPY 시 products JOIN 으로 product_id 부여(엔진은 product_id 기반).
- 전면 재적재(full reload): 수정주가/리팩터링으로 과거 값이 바뀌므로 TRUNCATE 후 COPY.
- 공분산/클러스터링은 취득 대상 아님.

전제: ~/.ssh/config 의 'portfolio-ops' alias, 원격 peer 인증(psql), 로컬 equity peer/trust.

실행:
  python sync_portfolio_db.py             # 전체
  python sync_portfolio_db.py --only market_data bt_expected_returns
  python sync_portfolio_db.py --list
"""
import argparse
import os
import subprocess
import sys
import time

import psycopg2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────
LOCAL_DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "equity"),
    "user": os.getenv("DB_USER", "postgres"),
}
_pw = os.getenv("PGPASSWORD", "").strip()
if _pw:
    LOCAL_DB["password"] = _pw            # 빈값이면 생략 (peer/trust)

REMOTE_HOST = "portfolio-ops"
REMOTE_DB = "portfolio_db"

ER_METHOD = "bayes_stein"                 # 랭킹에 쓰는 유일 method

# ── 테이블 스펙 ──────────────────────────────────────────────────
# name        : 로컬 테이블명
# ddl         : 로컬 생성 DDL (없으면 기존 테이블 가정)
# cols        : COPY 컬럼 리스트 (원격 SELECT 순서 == 로컬 INSERT 순서)
# remote_sql  : 원격 COPY (...) TO STDOUT 의 SELECT 본문
# mode        : 'truncate' (TRUNCATE 후 COPY) | 'upsert' (FK 보호: ON CONFLICT)
# conflict    : upsert 시 충돌 키
SPECS = [
    {
        "name": "products",
        "ddl": None,
        "cols": ["product_id", "ticker", "name", "product_type", "market",
                 "currency", "sector", "industry", "status", "isin",
                 "listing_date", "delisting_date", "delisting_reason", "data_source"],
        "remote_sql": ("SELECT product_id, ticker, name, product_type, market, "
                       "currency, sector, industry, status, isin, listing_date, "
                       "delisting_date, delisting_reason, data_source FROM products"),
        "mode": "upsert",            # 다른 테이블이 product_id FK → TRUNCATE 불가
        "conflict": ["product_id"],
    },
    {
        "name": "asset_quality",
        "ddl": None,
        "cols": ["product_id", "ticker", "trading_days", "mean_return", "volatility",
                 "skewness", "kurtosis", "avg_volume", "volume_coverage_pct",
                 "is_selected", "filter_reason", "recent_trading_days", "recent_coverage_pct"],
        "remote_sql": ("SELECT product_id, ticker, trading_days, mean_return, volatility, "
                       "skewness, kurtosis, avg_volume, volume_coverage_pct, is_selected, "
                       "filter_reason, recent_trading_days, recent_coverage_pct FROM asset_quality"),
        "mode": "upsert",
        "conflict": ["product_id"],
    },
    {
        "name": "market_data",
        "ddl": None,                 # 기존 로컬 스키마 재사용 (adj_close 포함)
        "cols": ["product_id", "trade_date", "open", "high", "low", "close",
                 "adj_close", "volume", "trading_halt", "is_estimated"],
        "remote_sql": ("SELECT product_id, trade_date, open, high, low, close, "
                       "adj_close, COALESCE(volume,0), trading_halt, is_estimated "
                       "FROM market_data"),
        "mode": "truncate",          # market_data 는 child → TRUNCATE 가능
    },
    {
        "name": "bt_expected_returns",
        "ddl": ("CREATE TABLE IF NOT EXISTS bt_expected_returns ("
                "product_id BIGINT, snapshot_date DATE, lookback_days INT, "
                "annual_expected_return DOUBLE PRECISION)"),
        "cols": ["product_id", "snapshot_date", "lookback_days", "annual_expected_return"],
        "remote_sql": (f"SELECT p.product_id, b.snapshot_date, b.lookback_days, "
                       f"b.annual_expected_return FROM bt_expected_returns b "
                       f"JOIN products p ON p.ticker=b.ticker "
                       f"WHERE b.estimation_method='{ER_METHOD}' "
                       f"AND b.annual_expected_return IS NOT NULL"),
        "mode": "truncate",
    },
    {
        "name": "bt_asset_metrics",
        "ddl": ("CREATE TABLE IF NOT EXISTS bt_asset_metrics ("
                "product_id BIGINT, snapshot_date DATE, lookback_days INT, "
                "annual_mean_return DOUBLE PRECISION, annual_volatility DOUBLE PRECISION)"),
        "cols": ["product_id", "snapshot_date", "lookback_days",
                 "annual_mean_return", "annual_volatility"],
        "remote_sql": ("SELECT DISTINCT p.product_id, b.snapshot_date, b.lookback_days, "
                       "b.annual_mean_return, b.annual_volatility FROM bt_asset_metrics b "
                       "JOIN products p ON p.ticker=b.ticker"),
        "mode": "truncate",
    },
    {
        "name": "rt_expected_returns",
        "ddl": ("CREATE TABLE IF NOT EXISTS rt_expected_returns ("
                "product_id BIGINT, snapshot_date DATE, lookback_days INT, "
                "annual_expected_return DOUBLE PRECISION)"),
        "cols": ["product_id", "snapshot_date", "lookback_days", "annual_expected_return"],
        "remote_sql": (f"SELECT p.product_id, b.snapshot_date, b.lookback_days, "
                       f"b.annual_expected_return FROM rt_expected_returns b "
                       f"JOIN products p ON p.ticker=b.ticker "
                       f"WHERE b.estimation_method='{ER_METHOD}' "
                       f"AND b.annual_expected_return IS NOT NULL"),
        "mode": "truncate",
    },
    {
        "name": "rt_asset_metrics",
        "ddl": ("CREATE TABLE IF NOT EXISTS rt_asset_metrics ("
                "product_id BIGINT, snapshot_date DATE, lookback_days INT, "
                "annual_mean_return DOUBLE PRECISION, annual_volatility DOUBLE PRECISION)"),
        "cols": ["product_id", "snapshot_date", "lookback_days",
                 "annual_mean_return", "annual_volatility"],
        "remote_sql": ("SELECT DISTINCT p.product_id, b.snapshot_date, b.lookback_days, "
                       "b.annual_mean_return, b.annual_volatility FROM rt_asset_metrics b "
                       "JOIN products p ON p.ticker=b.ticker"),
        "mode": "truncate",
    },
]


# ── Helpers ────────────────────────────────────────────────────
def local_conn():
    return psycopg2.connect(**LOCAL_DB)


def remote_copy(remote_sql: str) -> subprocess.Popen:
    """ssh portfolio-ops 'psql -d portfolio_db -c "COPY (...) TO STDOUT"'"""
    inner = f"COPY ({remote_sql}) TO STDOUT"
    cmd_str = f'psql -d {REMOTE_DB} -c "{inner}"'
    return subprocess.Popen(
        ["ssh", REMOTE_HOST, cmd_str],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def sync_table(spec: dict) -> int:
    name = spec["name"]
    cols = spec["cols"]
    col_list = ", ".join(cols)
    print(f"\n[{name}] {spec['mode']} ...", flush=True)
    t0 = time.time()

    # DDL (신규 테이블)
    if spec.get("ddl"):
        with local_conn() as conn:
            cur = conn.cursor()
            cur.execute(spec["ddl"])
            conn.commit()

    proc = remote_copy(spec["remote_sql"])

    with local_conn() as conn:
        cur = conn.cursor()
        if spec["mode"] == "truncate":
            cur.execute(f"TRUNCATE {name}")
            try:
                cur.copy_expert(f"COPY {name} ({col_list}) FROM STDIN", proc.stdout)
            except Exception:
                err = proc.stderr.read().decode("utf-8", "replace")
                raise RuntimeError(f"COPY 실패. 원격 stderr:\n{err}")
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read().decode("utf-8", "replace")
                raise RuntimeError(f"원격 psql 실패 (rc={proc.returncode}):\n{err}")
            cur.execute(f"SELECT count(*) FROM {name}")
            n = cur.fetchone()[0]
            conn.commit()
        else:  # upsert (FK 보호)
            stage = f"_stage_{name}"
            cur.execute(f"DROP TABLE IF EXISTS {stage}; "
                        f"CREATE UNLOGGED TABLE {stage} AS SELECT * FROM {name} WHERE FALSE")
            try:
                cur.copy_expert(f"COPY {stage} ({col_list}) FROM STDIN", proc.stdout)
            except Exception:
                err = proc.stderr.read().decode("utf-8", "replace")
                raise RuntimeError(f"COPY 실패. 원격 stderr:\n{err}")
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read().decode("utf-8", "replace")
                raise RuntimeError(f"원격 psql 실패 (rc={proc.returncode}):\n{err}")
            upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in spec["conflict"])
            cur.execute(
                f"INSERT INTO {name} ({col_list}) SELECT {col_list} FROM {stage} "
                f"ON CONFLICT ({', '.join(spec['conflict'])}) DO UPDATE SET {upd}"
            )
            cur.execute(f"DROP TABLE {stage}")
            cur.execute(f"SELECT count(*) FROM {name}")
            n = cur.fetchone()[0]
            conn.commit()

    print(f"  → {n:,} rows ({time.time()-t0:.1f}s)", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="특정 테이블만")
    ap.add_argument("--list", action="store_true", help="테이블 목록 출력")
    args = ap.parse_args()

    if args.list:
        for s in SPECS:
            print(s["name"])
        return

    print("=" * 70)
    print(f"  portfolio_db → local equity 전면 재취득 ({REMOTE_HOST}:{REMOTE_DB})")
    print("=" * 70)
    only = set(args.only) if args.only else None
    t0 = time.time()
    for spec in SPECS:
        if only and spec["name"] not in only:
            continue
        sync_table(spec)
    print(f"\n{'='*70}\n  총 소요: {time.time()-t0:.1f}초\n{'='*70}")


if __name__ == "__main__":
    main()
