-- forward-test 트래킹 schema
-- daily_signals.py가 매일 실행되며 갱신

-- ── forward_signals: 매일 산출된 top-N 신호 로그 ───────────────

CREATE TABLE IF NOT EXISTS forward_signals (
    signal_date         DATE         NOT NULL,
    rank                INT          NOT NULL,         -- 1, 2, 3 (top-N)
    ticker              VARCHAR(20)  NOT NULL,
    product_id          BIGINT,

    -- 신호 발생 시 컨텍스트 (재현/디버깅용)
    metric_value        NUMERIC,                       -- BS ER 또는 Sharpe
    ranking_method      VARCHAR(20)  NOT NULL,
    lookback_days       INT          NOT NULL,

    -- 신호 검증값
    high_at_signal      NUMERIC,
    prev_ath            NUMERIC,
    close_at_signal     NUMERIC,
    volume_at_signal    BIGINT,
    prev_volume         BIGINT,
    avg_trading_value   NUMERIC,

    created_at          TIMESTAMPTZ  DEFAULT NOW(),
    PRIMARY KEY (signal_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_fwd_signals_date
    ON forward_signals (signal_date);


-- ── forward_positions: 가상 포지션 트래킹 ──────────────────────

CREATE TABLE IF NOT EXISTS forward_positions (
    id                  SERIAL       PRIMARY KEY,

    -- 신호 정보
    signal_date         DATE         NOT NULL,        -- 신호 발생일 (D+0)
    rank_at_signal      INT,                          -- 신호 시점 순위
    ticker              VARCHAR(20)  NOT NULL,
    product_id          BIGINT,

    -- 진입 정보
    entry_timing        VARCHAR(20)  NOT NULL,        -- avg_close_open / next_open ...
    entry_date          DATE,                          -- 실제 entry price 확정된 날
    entry_price         NUMERIC,
    shares              INT,
    cost_basis          NUMERIC,                       -- shares * entry_price * (1 + buy_commission)
    capital_at_entry    NUMERIC,                       -- 진입 시점 가상 자본
    slot_fraction       NUMERIC,                       -- 적용된 슬롯 비율

    -- TP/SL 레벨
    tp_pct              NUMERIC,
    sl_pct              NUMERIC,
    tp_price            NUMERIC,
    sl_price            NUMERIC,

    -- 상태
    status              VARCHAR(15)  NOT NULL DEFAULT 'pending',
    -- pending  : 신호 발생, entry_price 미확정 (D+1 시가 대기)
    -- open     : entry 완료, 진행 중
    -- closed   : TP/SL/FORCE 등으로 청산
    -- skipped  : 진입 불가 (자금 부족 / 데이터 없음)

    -- 청산 정보
    exit_date           DATE,
    exit_price          NUMERIC,
    exit_reason         VARCHAR(20),                   -- TP / SL / FORCE / BE / SKIP
    holding_days        INT,
    pnl_krw             NUMERIC,
    pnl_pct             NUMERIC,

    notes               TEXT,
    created_at          TIMESTAMPTZ  DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  DEFAULT NOW(),

    UNIQUE (signal_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_fwd_pos_status
    ON forward_positions (status);
CREATE INDEX IF NOT EXISTS idx_fwd_pos_signal_date
    ON forward_positions (signal_date);
CREATE INDEX IF NOT EXISTS idx_fwd_pos_entry_date
    ON forward_positions (entry_date);
CREATE INDEX IF NOT EXISTS idx_fwd_pos_ticker
    ON forward_positions (ticker);


-- ── forward_capital: 일별 가상 자본 트래킹 ─────────────────────

CREATE TABLE IF NOT EXISTS forward_capital (
    snapshot_date       DATE         PRIMARY KEY,
    cash                NUMERIC      NOT NULL,
    positions_value     NUMERIC      NOT NULL,        -- mark-to-market
    total_equity        NUMERIC      NOT NULL,
    n_open_positions    INT          NOT NULL,
    n_entries_today     INT          NOT NULL DEFAULT 0,
    n_exits_today       INT          NOT NULL DEFAULT 0,
    daily_pnl           NUMERIC,
    daily_return_pct    NUMERIC,
    cum_return_pct      NUMERIC,
    notes               TEXT,
    created_at          TIMESTAMPTZ  DEFAULT NOW()
);


-- ── update trigger ────────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_forward_positions_modtime ON forward_positions;
CREATE TRIGGER update_forward_positions_modtime
    BEFORE UPDATE ON forward_positions
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();
