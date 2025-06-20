import pyupbit
import pandas as pd
import time
import streamlit as st
import logging

secs = {
    "minute1": 60,
    "minute3": 180,
    "minute5": 300,
    "minute10": 600,
    "minute15": 900,
    "minute30": 1800,
    "minute60": 3600,
    "day": 86400,
}

logger = logging.getLogger(__name__)


def stream_candles(
    ticker: str,
    interval: str,
    q=None,
    max_retry: int = 5,
    retry_wait: int = 3,
    stop_event=None,
):
    import streamlit as st

    def standardize_ohlcv(df):
        if df is None or df.empty:
            raise ValueError(f"OHLCV 데이터 수집 실패: {ticker}, {interval}")
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        if "value" in df.columns:
            df = df.drop(columns=["value"])
        df.index = pd.to_datetime(df.index)
        return df.dropna().sort_index()

    # 최초 데이터 수집: 재시도 로직 적용
    retry_cnt = 0
    df = None
    while retry_cnt < max_retry:
        # 종료 신호 체크
        if stop_event.is_set():
            msg = f"stream_candles stop - while retry_cnt < max_retry:)"
            logger.warning(msg)
            return
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=200)
        if df is not None and not df.empty:
            break
        retry_cnt += 1
        msg = f"pyupbit.get_ohlcv() 실패: ticker={ticker}, interval={interval} (재시도 {retry_cnt}/{max_retry})"
        logger.warning(msg)
        if q is not None:
            q.put(("WARNING", msg))
        time.sleep(retry_wait)
    else:
        msg = f"pyupbit.get_ohlcv() 최종 실패: ticker={ticker}, interval={interval}"
        logger.error(msg)
        if q is not None:
            q.put(("ERROR", msg))
        return  # 종료 신호가 아니어도, 실패 시 함수 종료

    df = standardize_ohlcv(df)
    yield df

    last_candle_time = df.index[-1]
    while True:
        # 종료 신호 체크
        if stop_event.is_set():
            msg = f"stream_candles stop - while True:)"
            logger.warning(msg)
            return
        time.sleep(secs[interval] // 3)
        retry_cnt = 0
        new = None
        while retry_cnt < max_retry:
            # 종료 신호 체크 (재시도 루프 내에서도)
            if stop_event.is_set():
                msg = f"stream_candles stop - while True: while retry_cnt < max_retry:)"
                logger.warning(msg)
                return
            new = pyupbit.get_ohlcv(ticker, interval=interval, count=1)
            if new is not None and not new.empty:
                break
            retry_cnt += 1
            msg = f"pyupbit.get_ohlcv() (실시간) 실패: ticker={ticker}, interval={interval} (재시도 {retry_cnt}/{max_retry})"
            logger.warning(msg)
            if q is not None:
                q.put(("WARNING", msg))
            time.sleep(retry_wait)
        else:
            msg = f"pyupbit.get_ohlcv() (실시간) 최종 실패: ticker={ticker}, interval={interval}"
            logger.error(msg)
            if q is not None:
                q.put(("ERROR", msg))
            return  # 실패 시 함수 종료

        new = standardize_ohlcv(new)
        new_candle_time = new.index[-1]
        if new_candle_time == last_candle_time:
            continue  # 새로운 캔들이 아직 생성되지 않음
        last_candle_time = new_candle_time
        df = pd.concat([df, new]).iloc[-500:]
        yield df
