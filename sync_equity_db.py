#!/usr/bin/env python3
"""
운영 플랫폼 (portfolio_db) → 로컬 (equity) 일별 증분 sync.

전제:
- SSH 키 인증 + ~/.ssh/config 의 'portfolio-ops' alias 설정됨
- 원격: peer auth로 PostgreSQL 접속 (kimboguk 계정)
- 로컬: postgres / postgres@localhost:5432
- COPY pipe 통한 효율적 전송 (SSH stdout → psql stdin)

실행:
  $env:PGPASSWORD="postgres"
  python sync_equity_db.py            # incremental 만 (market_data, returns, ER)
  python sync_equity_db.py --full     # + products, asset_quality 전체 reload
  python sync_equity_db.py --only market_data
"""
import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

import psycopg2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────

LOCAL_DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "equity",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", "postgres"),
}
REMOTE_HOST = "portfolio-ops"
REMOTE_DB = "portfolio_db"

# 증분 sync (date_col 기준 max 이후만)
INCREMENTAL_TABLES = [
    {
        "name": "market_data",
        "date_col": "trade_date",
        "columns": [
            "product_id", "trade_date",
            "open", "high", "low", "close", "adj_close", "volume",
            "trading_halt", "is_estimated",
        ],
        "conflict": ["product_id", "trade_date"],
    },
    {
        "name": "daily_returns_snapshot",
        "date_col": "snapshot_date",
        "columns": [
            "snapshot_date", "product_id", "ticker", "lookback_days",
            "mean_return_daily", "annual_mean_return",
            "volatility_daily", "annual_volatility",
            "skewness", "kurtosis",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        ],
        "conflict": ["snapshot_date", "product_id", "lookback_days"],
    },
    {
        "name": "expected_returns_snapshot",
        "date_col": "snapshot_date",
        "columns": [
            "snapshot_date", "product_id", "ticker", "lookback_days",
            "expected_return", "unconditional_return", "ppa_adjustment",
            "annual_expected_return", "estimation_method",
            "shrinkage_intensity", "ct_constraints_applied",
        ],
        "conflict": ["snapshot_date", "product_id",
                     "estimation_method", "lookback_days"],
    },
]

# 전체 reload (작은 테이블, UPSERT 방식 — products는 FK 때문에 TRUNCATE 불가)
FULL_TABLES = [
    {
        "name": "products",
        "columns": [
            "product_id", "ticker", "name", "product_type", "market",
            "currency", "sector", "industry", "status", "isin",
            "listing_date", "delisting_date", "delisting_reason",
            "data_source",
        ],
        "conflict": ["product_id"],
    },
    {
        "name": "asset_quality",
        "columns": [
            "product_id", "ticker", "trading_days",
            "mean_return", "volatility", "skewness", "kurtosis",
            "avg_volume", "volume_coverage_pct",
            "is_selected", "filter_reason",
            "recent_trading_days", "recent_coverage_pct",
        ],
        "conflict": ["product_id"],
    },
]


# ── Helpers ────────────────────────────────────────────────────

def local_conn():
    return psycopg2.connect(**LOCAL_DB)


def get_local_state(cur, table: str, date_col: Optional[str]):
    """로컬 max(date_col) + count"""
    if date_col:
        cur.execute(f"SELECT MAX({date_col})::text, COUNT(*) FROM {table}")
    else:
        cur.execute(f"SELECT NULL, COUNT(*) FROM {table}")
    return cur.fetchone()


def remote_copy(remote_sql: str) -> subprocess.Popen:
    """ssh portfolio-ops 'psql -d portfolio_db -c "COPY (...) TO STDOUT"' 실행"""
    cmd_str = f'psql -d {REMOTE_DB} -c "{remote_sql}"'
    return subprocess.Popen(
        ["ssh", REMOTE_HOST, cmd_str],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def stage_table_sql(name: str) -> str:
    """target과 동일 컬럼 + 제약 없는 UNLOGGED 스테이징 테이블"""
    stage = f"_stage_{name}"
    return (
        f"DROP TABLE IF EXISTS {stage}; "
        f"CREATE UNLOGGED TABLE {stage} AS SELECT * FROM {name} WHERE FALSE;"
    )


# ── Sync 본체 ───────────────────────────────────────────────────

def sync_incremental(spec: dict) -> int:
    name = spec["name"]
    date_col = spec["date_col"]
    cols = spec["columns"]
    conflict = spec["conflict"]
    col_list = ", ".join(cols)
    conflict_clause = ", ".join(conflict)

    print(f"\n[{name}] incremental sync ({date_col} > local max)")

    with local_conn() as conn:
        cur = conn.cursor()
        max_date, prev_count = get_local_state(cur, name, date_col)

    print(f"  로컬 max = {max_date}, count = {prev_count:,}")

    where = f"{date_col} > '{max_date}'" if max_date else "TRUE"
    remote_sql = f"COPY (SELECT {col_list} FROM {name} WHERE {where}) TO STDOUT"

    proc = remote_copy(remote_sql)

    t0 = time.time()
    stage = f"_stage_{name}"
    with local_conn() as conn:
        cur = conn.cursor()
        cur.execute(stage_table_sql(name))
        try:
            cur.copy_expert(f"COPY {stage} ({col_list}) FROM STDIN", proc.stdout)
        except Exception:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"COPY 실패. 원격 stderr:\n{err}")
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"원격 psql 실패 (rc={proc.returncode}):\n{err}")

        cur.execute(
            f"INSERT INTO {name} ({col_list}) "
            f"SELECT {col_list} FROM {stage} "
            f"ON CONFLICT ({conflict_clause}) DO NOTHING"
        )
        added = cur.rowcount
        cur.execute(f"DROP TABLE {stage}")
        conn.commit()

    elapsed = time.time() - t0
    print(f"  fetched + inserted: {added:,} new rows ({elapsed:.1f}s)")
    return added


def sync_full_upsert(spec: dict) -> int:
    """전체 reload — UPSERT (FK 보호 위해 TRUNCATE 금지)"""
    name = spec["name"]
    cols = spec["columns"]
    conflict = spec["conflict"]
    col_list = ", ".join(cols)
    conflict_clause = ", ".join(conflict)
    update_clause = ", ".join(
        f"{c}=EXCLUDED.{c}" for c in cols if c not in conflict
    )

    print(f"\n[{name}] full UPSERT")

    with local_conn() as conn:
        cur = conn.cursor()
        _, prev_count = get_local_state(cur, name, None)
    print(f"  로컬 기존 count = {prev_count:,}")

    remote_sql = f"COPY (SELECT {col_list} FROM {name}) TO STDOUT"
    proc = remote_copy(remote_sql)

    t0 = time.time()
    stage = f"_stage_{name}"
    with local_conn() as conn:
        cur = conn.cursor()
        cur.execute(stage_table_sql(name))
        try:
            cur.copy_expert(f"COPY {stage} ({col_list}) FROM STDIN", proc.stdout)
        except Exception:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"COPY 실패. 원격 stderr:\n{err}")
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"원격 psql 실패 (rc={proc.returncode}):\n{err}")

        if update_clause:
            cur.execute(
                f"INSERT INTO {name} ({col_list}) "
                f"SELECT {col_list} FROM {stage} "
                f"ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}"
            )
        else:
            cur.execute(
                f"INSERT INTO {name} ({col_list}) "
                f"SELECT {col_list} FROM {stage} "
                f"ON CONFLICT ({conflict_clause}) DO NOTHING"
            )
        affected = cur.rowcount
        cur.execute(f"DROP TABLE {stage}")
        conn.commit()

        cur.execute(f"SELECT COUNT(*) FROM {name}")
        new_count = cur.fetchone()[0]

    elapsed = time.time() - t0
    print(f"  upserted: {affected:,} rows | total = {new_count:,} "
          f"(was {prev_count:,}) ({elapsed:.1f}s)")
    return affected


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="products, asset_quality도 sync (전체 reload)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="특정 테이블만 sync (이름)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Equity DB Sync — {REMOTE_HOST}:{REMOTE_DB} → local equity")
    print("=" * 70)

    t0 = time.time()
    only = set(args.only) if args.only else None

    # 증분 우선
    for spec in INCREMENTAL_TABLES:
        if only and spec["name"] not in only:
            continue
        sync_incremental(spec)

    # 전체 (옵션)
    if args.full or only:
        for spec in FULL_TABLES:
            if only and spec["name"] not in only:
                continue
            if not args.full and not only:
                continue
            sync_full_upsert(spec)
    else:
        print("\n[products / asset_quality] 생략 — --full 옵션 시 sync")

    print(f"\n{'='*70}")
    print(f"  총 소요: {time.time()-t0:.1f}초")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
