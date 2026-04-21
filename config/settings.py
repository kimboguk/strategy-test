"""
KRX ATH/52주 신고가 돌파 전략 백테스트 설정
"""

import os

# ============================================================================
# ENVIRONMENT
# ============================================================================
APP_ENV = os.getenv('APP_ENV', 'prod')

# ============================================================================
# DATABASE CONFIGURATION (financial_modeling 패턴 재사용)
# ============================================================================
DATABASE_HOST = os.getenv("DATABASE_HOST", "").strip()
DATABASE_PORT_STR = os.getenv("DATABASE_PORT", "").strip()
DATABASE_PORT = int(DATABASE_PORT_STR) if DATABASE_PORT_STR else None
_db_default = 'portfolio_db_dev' if APP_ENV == 'dev' else 'portfolio_db'
DATABASE_NAME = os.getenv("DATABASE_NAME", _db_default)
DATABASE_USER = os.getenv("DATABASE_USER", "kimboguk")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "").strip()

DB_CONFIG = {
    'host': DATABASE_HOST,
    'port': DATABASE_PORT,
    'database': DATABASE_NAME,
    'user': DATABASE_USER,
    'password': DATABASE_PASSWORD,
}

if DATABASE_PASSWORD and DATABASE_HOST:
    DATABASE_URL = (
        f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}"
        f"@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    )
else:
    DATABASE_URL = f"postgresql://{DATABASE_USER}@/{DATABASE_NAME}"

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

INITIAL_CAPITAL = 100_000_000  # 1억원

# 손절/익절
STOP_LOSS_PCT = 0.20           # 진입가 대비 -20%
TP_START_PCT = 0.10            # 익절 시작: +10%
TP_STEP_PCT = 0.05             # 익절 간격: +5%
TP_CLOSE_RATIO = 0.50          # 각 TP 레벨에서 잔여의 50% 청산

# 거래 비용
BUY_COMMISSION = 0.00015       # 매수 수수료 0.015%
SELL_COMMISSION = 0.00015      # 매도 수수료 0.015%
SELL_TAX = 0.0018              # 매도 세금 0.18%

# 신호 필터
MIN_HISTORY_DAYS = 252         # 최소 데이터 일수 (허위 신호 방지)

# 시장 필터
KRX_MARKETS = ('KRX', 'KOSPI', 'KOSDAQ')

# 연간 거래일
TRADING_DAYS_PER_YEAR = 252
ROLLING_52W_DAYS = 252

# 무위험 수익률
RISK_FREE_RATE = 0.035         # 3.5% (한국 국고채 기준)
