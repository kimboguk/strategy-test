# 수정주가 재적재 계획 (로컬 `equity` ← 원격 portfolio_db_dev)

> 배경: 원격(우분투) 시장 백필이 **수정주가(adjusted price) 방식으로 전면 재계산**됨.
> 과거 행의 *값*이 바뀌었고 날짜는 그대로이므로 [sync_equity_db.py](sync_equity_db.py)의
> 증분(`date_col > local max`) 로직으로는 변경분을 잡지 못함 → **전면 재적재 필요**.
> 원격 portfolio_db_dev가 인덱스 재생성 중이라 I/O 경합 우려 → **취득은 준비만, 실행은 인덱스 완료 후**.

## 기준선 — 현재 로컬 `equity` 상태 (postgres, peer 인증)

| 테이블 | 현재 보유 | 비고 |
|---|---|---|
| `products` | 7,315행 (KRW 6,769) | 마켓 KOSPI/KOSDAQ/KONEX/KRX |
| `asset_quality` | 3,288행 / is_selected=TRUE 1,354 (KRW 1,088) | **최근일 기준 단일 스냅샷** — 과거 연도에 오적용 |
| `market_data` | 11,113,330행 / 종목 3,298 / 1980-01-02 ~ 2026-04-14 | 전면 교체 대상 |
| `daily_returns_snapshot` (샘플수익률) | 10,150,687행 / lookback **252·504만** / ~2026-02-13 | **126 없음** |
| `expected_returns_snapshot` (기대수익률) | 30,458,331행 / lookback **252·504만** / method bayes_stein·ppa_cluster·ppa_pca / ~2026-02-13 | **126 없음** |

## 재취득 항목

### ① market_data — 전체 유니버스 · 전체 히스토리 (전면 재적재)
- 범위: 필터 없이 전 종목 · 상장~현재 전 기간 (ATH 누적 최고가 정합성 위해 전 기간 필수)
- 컬럼: `product_id, trade_date, open, high, low, close, adj_close, volume, trading_halt, is_estimated`
- 방식: truncate → full reload (수정주가 반영값으로 전면 교체)

### ② daily_returns_snapshot (샘플수익률) — lookback 126·252·504
- **126 = 신규**(로컬 없음, 원격 신규 생성분) / 252·504 = 전면 교체
- 컬럼: `snapshot_date, product_id, ticker, lookback_days, mean_return_daily, annual_mean_return, volatility_daily, annual_volatility, skewness, kurtosis, sharpe_ratio, sortino_ratio, calmar_ratio`
- 유니버스: 시점별 필터된 종목만 (④ 참조)

### ③ expected_returns_snapshot (기대수익률) — lookback 126·252·504 · `estimation_method='bayes_stein'`만
- bayes_stein 단일 (ppa_cluster·ppa_pca 제외 → 용량 약 1/3). repo가 실제 쓰는 유일 method.
- **126 = 신규** / 252·504 = 전면 교체
- 컬럼: `snapshot_date, product_id, ticker, lookback_days, expected_return, unconditional_return, ppa_adjustment, annual_expected_return, estimation_method, shrinkage_intensity, ct_constraints_applied`
- 유니버스: 시점별 필터된 종목만 (④ 참조)

### ④ 시점별(per-date) 종목 선택 정보 — **출처 확정 보류**
- ②③의 "필터된 종목"은 매일 달라짐. 현재 로컬 `asset_quality`는 최근일 단일 스냅샷이라 과거 연도에 잘못 적용됨.
- **유력 후보**: 원격 returns/ER 스냅샷 행이 *그 시점 선택된 종목에 대해서만* 산출 → 스냅샷을 그대로 전부 받으면 시점별 필터가 자동 반영(별도 테이블 불필요).
- **대안(fallback)**: 취득한 market_data에서 recent N일 거래일수/coverage 기준으로 시점별 선택을 로컬에서 직접 재산출.
- → 원격 서버 추가 확인 후 결정.

## 실행 시 유의 (인덱스 완료 후)
- 증분이 아닌 **full reload**: 기존 행 truncate/replace 필요. 현 [sync_equity_db.py](sync_equity_db.py)는 증분 전용이므로 full-reload 경로(`--full-reload` 등) 보강 필요.
- 접속: 로컬 peer 인증(`psql -U postgres -d equity`), 원격은 `ssh portfolio-ops 'psql -d portfolio_db_dev ... COPY TO STDOUT'`.
- 원격 DB명이 `portfolio_db` → **`portfolio_db_dev`** 로 바뀐 점 반영 ([sync_equity_db.py:39](sync_equity_db.py#L39) `REMOTE_DB` 수정).
