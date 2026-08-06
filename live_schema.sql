-- 라이브 트레이딩 환경 schema (자동매매 설정 + 브로커 주문 로그)
-- backend main.py lifespan 의 ensure_live_tables() 가 실행

-- ── live_settings: 자동매매 정본 상태 (단일행) ────────────────────
CREATE TABLE IF NOT EXISTS live_settings (
    id            INT          PRIMARY KEY DEFAULT 1,
    auto_trade    BOOLEAN      NOT NULL DEFAULT FALSE,
    kiwoom_env    VARCHAR(10)  NOT NULL DEFAULT 'mock',   -- mock | real
    updated_at    TIMESTAMPTZ  DEFAULT NOW(),
    updated_by    VARCHAR(50),
    CONSTRAINT live_settings_singleton CHECK (id = 1)
);
INSERT INTO live_settings (id, auto_trade) VALUES (1, FALSE)
    ON CONFLICT (id) DO NOTHING;

-- ── order_log: 모든 브로커 주문 제출 감사 (재조정·안전 백본) ───────
CREATE TABLE IF NOT EXISTS order_log (
    id                  SERIAL       PRIMARY KEY,
    created_at          TIMESTAMPTZ  DEFAULT NOW(),
    cycle_as_of         DATE,
    market              VARCHAR(10),
    side                VARCHAR(4),                       -- BUY | SELL
    ticker              VARCHAR(20),
    product_id          BIGINT,
    qty                 INT,
    price               NUMERIC,
    order_type          VARCHAR(20),
    broker_env          VARCHAR(10),                      -- mock | real
    adapter             VARCHAR(20),                      -- manual | kiwoom
    intent              VARCHAR(10),                      -- auto | manual
    broker_order_id     VARCHAR(50),
    status              VARCHAR(20),                      -- submitted|filled|rejected|failed
    raw_request         JSONB,
    raw_response        JSONB,
    error               TEXT,
    forward_position_id INT
);
CREATE INDEX IF NOT EXISTS idx_order_log_asof   ON order_log (cycle_as_of);
CREATE INDEX IF NOT EXISTS idx_order_log_status ON order_log (status);

-- 멱등성: 자동 사이클 재실행 시 동일 (as_of, side, ticker) 이중 제출 방지 (Phase 3)
CREATE UNIQUE INDEX IF NOT EXISTS uq_order_log_auto
    ON order_log (cycle_as_of, side, ticker)
    WHERE intent = 'auto';
