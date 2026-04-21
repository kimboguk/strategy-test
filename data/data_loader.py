"""
OHLCV 벌크 데이터 로더
- asset_quality.is_selected=TRUE & KRX 마켓 필터
- 전체 OHLCV 한 번에 로드 → 종목별 딕셔너리 분할
"""

import time
import pandas as pd
from sqlalchemy import create_engine
from datetime import date
from typing import Dict, List, Optional, Tuple

from strategy_backtest.config.settings import DATABASE_URL, KRX_MARKETS


class DataLoader:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._engine = create_engine(DATABASE_URL)

    def _connect(self):
        return self._engine.connect()

    def load_filtered_tickers(self) -> pd.DataFrame:
        """asset_quality.is_selected=TRUE & KRX 마켓 종목 로드"""
        markets_placeholder = ','.join(f"'{m}'" for m in KRX_MARKETS)
        query = f"""
            SELECT p.product_id, p.ticker, p.name, p.market, p.status
            FROM asset_quality aq
            JOIN products p ON aq.product_id = p.product_id
            WHERE aq.is_selected = TRUE
              AND p.market IN ({markets_placeholder})
            ORDER BY p.product_id
        """
        with self._connect() as conn:
            df = pd.read_sql(query, conn)
        if self.verbose:
            print(f"  필터링 종목 수: {len(df)} (KRX markets)")
        return df

    def load_ohlcv_bulk(
        self,
        product_ids: List[int],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """전체 OHLCV 한 번에 벌크 로드"""
        t0 = time.time()

        ids_str = ','.join(str(pid) for pid in product_ids)
        where_clauses = [f"product_id IN ({ids_str})"]
        if start_date:
            where_clauses.append(f"trade_date >= '{start_date}'")
        if end_date:
            where_clauses.append(f"trade_date <= '{end_date}'")
        where = ' AND '.join(where_clauses)

        query = f"""
            SELECT product_id, trade_date,
                   open::float, high::float, low::float, close::float,
                   volume
            FROM market_data
            WHERE {where}
            ORDER BY product_id, trade_date
        """
        with self._connect() as conn:
            df = pd.read_sql(query, conn, parse_dates=['trade_date'])
        elapsed = time.time() - t0
        if self.verbose:
            print(f"  OHLCV 로드: {len(df):,} rows, {elapsed:.1f}초")
        return df

    def split_by_ticker(
        self, ohlcv: pd.DataFrame, ticker_map: Dict[int, str]
    ) -> Dict[str, pd.DataFrame]:
        """product_id 기준으로 종목별 DataFrame 딕셔너리 분할"""
        result = {}
        for pid, group in ohlcv.groupby('product_id'):
            ticker = ticker_map.get(pid)
            if ticker:
                df = group.sort_values('trade_date').reset_index(drop=True)
                result[ticker] = df
        if self.verbose:
            print(f"  종목별 분할: {len(result)} tickers")
        return result

    def build_ohlcv_lookup(
        self, ticker_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[date, dict]]:
        """
        {ticker: {date: {open, high, low, close, volume}}} 룩업 테이블
        O(1) 접근용
        """
        lookup = {}
        for ticker, df in ticker_data.items():
            ticker_lookup = {}
            for row in df.itertuples(index=False):
                d = row.trade_date.date() if hasattr(row.trade_date, 'date') else row.trade_date
                ticker_lookup[d] = {
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume,
                }
            lookup[ticker] = ticker_lookup
        return lookup

    def load_trading_calendar(
        self, ohlcv: pd.DataFrame, threshold: float = 0.5
    ) -> List[date]:
        """
        KRX 거래일 캘린더 생성
        threshold 비율 이상 종목이 거래한 날만 거래일로 인정
        """
        n_tickers = ohlcv['product_id'].nunique()
        min_count = max(1, int(n_tickers * threshold))

        daily_counts = ohlcv.groupby('trade_date')['product_id'].nunique()
        trading_days = daily_counts[daily_counts >= min_count].index
        calendar = sorted([
            d.date() if hasattr(d, 'date') else d for d in trading_days
        ])
        if self.verbose:
            print(f"  거래일 캘린더: {len(calendar)} days "
                  f"({calendar[0]} ~ {calendar[-1]})")
        return calendar

    def load_all(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict], List[date], pd.DataFrame]:
        """
        전체 데이터 로드 통합 메서드
        - OHLCV는 전 기간 로드 (ATH 누적 최고가 계산을 위해 히스토리 전부 필요)
        - 거래일 캘린더만 start_date ~ end_date 범위로 제한
        Returns: (ticker_data, ohlcv_lookup, trading_calendar, tickers_df)
        """
        print("[1/4] 필터링 종목 로드...")
        tickers_df = self.load_filtered_tickers()
        product_ids = tickers_df['product_id'].tolist()
        ticker_map = dict(zip(tickers_df['product_id'], tickers_df['ticker']))

        # OHLCV는 전 기간 로드 — 신호 계산에 히스토리 전부 필요
        # end_date만 적용 (미래 데이터 차단)
        print("[2/4] OHLCV 벌크 로드 (전 기간)...")
        ohlcv = self.load_ohlcv_bulk(product_ids, start_date=None, end_date=end_date)

        print("[3/4] 종목별 분할 & 룩업 테이블 빌드...")
        ticker_data = self.split_by_ticker(ohlcv, ticker_map)
        ohlcv_lookup = self.build_ohlcv_lookup(ticker_data)

        print("[4/4] 거래일 캘린더 생성...")
        full_calendar = self.load_trading_calendar(ohlcv)
        # start_date 필터는 시뮬레이터 측 run()에서도 처리하지만 여기서도 안내
        if start_date:
            n_before = len([d for d in full_calendar if d < start_date])
            if self.verbose:
                print(f"  (시뮬레이션 범위 외 히스토리 거래일: {n_before} days)")

        return ticker_data, ohlcv_lookup, full_calendar, tickers_df
