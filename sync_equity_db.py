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
  # 수정주가 basis 교체(전면 재적재) — dev 소스에서 통째로 갈아끼움:
  python sync_equity_db.py --source-db portfolio_db_dev \
      --full-reload market_data rt_expected_returns rt_asset_metrics
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
# 정본 소스 = portfolio_db_dev (수정주가 재계산본, 가격 스케줄러가 매일 갱신).
# 구 portfolio_db 는 market_data 가 07-03 에서 정체된 레거시. --source-db 로 override 가능.
REMOTE_DB = "portfolio_db_dev"

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
    # ── rt_* (실시간 ER/지표) — 엔진 라이브 경로(source='rt')가 사용 ──
    # 원격은 ticker+estimation_method 키 → products JOIN + bayes_stein 필터로 product_id 부여
    {
        "name": "rt_expected_returns",
        "date_col": "snapshot_date",
        "columns": ["product_id", "snapshot_date", "lookback_days",
                    "annual_expected_return"],
        "no_conflict": True,   # >local max 만 취득 → 기존행 충돌 없음(plain insert)
        "remote_select": (
            "SELECT p.product_id, b.snapshot_date, b.lookback_days, b.annual_expected_return "
            "FROM rt_expected_returns b JOIN products p ON p.ticker=b.ticker "
            "WHERE b.estimation_method='bayes_stein' "
            "AND b.annual_expected_return IS NOT NULL AND {where}"),
    },
    {
        "name": "rt_asset_metrics",
        "date_col": "snapshot_date",
        "columns": ["product_id", "snapshot_date", "lookback_days",
                    "annual_mean_return", "annual_volatility"],
        "no_conflict": True,
        # 원격은 (date,ticker,lookback)당 여러 행 → DISTINCT ON 으로 키당 1행
        "remote_select": (
            "SELECT DISTINCT ON (p.product_id, b.snapshot_date, b.lookback_days) "
            "p.product_id, b.snapshot_date, b.lookback_days, "
            "b.annual_mean_return, b.annual_volatility "
            "FROM rt_asset_metrics b JOIN products p ON p.ticker=b.ticker "
            "WHERE {where} "
            "ORDER BY p.product_id, b.snapshot_date, b.lookback_days, b.id"),
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
    conflict = spec.get("conflict") or []
    col_list = ", ".join(cols)
    conflict_clause = ", ".join(conflict)

    print(f"\n[{name}] incremental sync ({date_col} > local max)")

    with local_conn() as conn:
        cur = conn.cursor()
        max_date, prev_count = get_local_state(cur, name, date_col)

    print(f"  로컬 max = {max_date}, count = {prev_count:,}")

    where = f"{date_col} > '{max_date}'" if max_date else "TRUE"
    if spec.get("remote_select"):
        inner = spec["remote_select"].format(where=where)
    else:
        inner = f"SELECT {col_list} FROM {name} WHERE {where}"
    remote_sql = f"COPY ({inner}) TO STDOUT"

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

        if spec.get("no_conflict"):
            cur.execute(
                f"INSERT INTO {name} ({col_list}) "
                f"SELECT {col_list} FROM {stage}"
            )
        else:
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


def sync_full_reload(spec: dict) -> int:
    """증분 스펙 테이블을 전면 재적재 (수정주가 basis 교체용).

    스테이징에 원격 전량 COPY 완료 후, **단일 트랜잭션**에서 TRUNCATE + INSERT 스왑.
    COPY 실패 시 TRUNCATE 이전에 raise → 타깃 테이블 무손상(원자적).
    """
    name = spec["name"]
    cols = spec["columns"]
    col_list = ", ".join(cols)

    print(f"\n[{name}] FULL RELOAD (전면 재적재)")

    with local_conn() as conn:
        cur = conn.cursor()
        _, prev_count = get_local_state(cur, name, None)
    print(f"  로컬 기존 count = {prev_count:,}")

    if spec.get("remote_select"):
        inner = spec["remote_select"].format(where="TRUE")
    else:
        inner = f"SELECT {col_list} FROM {name}"
    remote_sql = f"COPY ({inner}) TO STDOUT"
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

        # 스왑 — 여기서부터 타깃 교체 (COPY 성공 후에만 도달)
        cur.execute(f"TRUNCATE {name}")
        cur.execute(
            f"INSERT INTO {name} ({col_list}) SELECT {col_list} FROM {stage}"
        )
        added = cur.rowcount
        cur.execute(f"DROP TABLE {stage}")
        conn.commit()

    elapsed = time.time() - t0
    print(f"  reloaded: {added:,} rows (was {prev_count:,}) ({elapsed:.1f}s)")
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
    global REMOTE_DB

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="products, asset_quality도 sync (전체 reload)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="특정 테이블만 sync (이름)")
    parser.add_argument("--full-reload", nargs="+", default=None,
                        dest="full_reload",
                        help="지정 증분테이블을 전면 재적재 (수정주가 basis 교체). "
                             "지정 시 해당 테이블만 처리")
    parser.add_argument("--source-db", default=None,
                        help=f"원격 DB override (기본 {REMOTE_DB})")
    args = parser.parse_args()

    if args.source_db:
        REMOTE_DB = args.source_db

    print("=" * 70)
    print(f"  Equity DB Sync — {REMOTE_HOST}:{REMOTE_DB} → local equity")
    print("=" * 70)

    t0 = time.time()
    only = set(args.only) if args.only else None
    full_reload = set(args.full_reload) if args.full_reload else None

    # 전면 재적재 모드 — 지정 증분테이블만 통째로 교체하고 종료
    if full_reload:
        matched = set()
        for spec in INCREMENTAL_TABLES:
            if spec["name"] in full_reload:
                sync_full_reload(spec)
                matched.add(spec["name"])
        unknown = full_reload - matched
        if unknown:
            print(f"\n[경고] 알 수 없는 --full-reload 테이블: {', '.join(sorted(unknown))}")
        print(f"\n{'='*70}\n  총 소요: {time.time()-t0:.1f}초\n{'='*70}")
        return

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
