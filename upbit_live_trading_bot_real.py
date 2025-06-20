"""
Upbit Live Trading Bot v1 — Optimised Edition
============================================
Author  : 30‑year Automated‑Trading Veteran (Streamlit‑ready)
Updated : 2025‑06‑11

Key upgrades
------------
1. **Pydantic 2** — uses `@field_validator`, no deprecation warnings.
2. **Thread‑safe UI** — worker thread attaches Streamlit context; communication via `queue.Queue`.
3. **Robust session/state** — single source of truth in `st.session_state`; clean reset logic.
4. **Graceful shutdown** — `_stop_live()` + signal handler (CLI only; ignored when running as Streamlit page).
5. **Rich logging** — TSV‑friendly file + stderr stream.

Run standalone:
```
streamlit run upbit_live_trading_bot.py
```
Or place in *pages* directory (`pages/app_live_test.py`).
Or place in *pages* directory (`pages/app_live_real.py`).
"""

from __future__ import annotations

import streamlit as st
import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional


st.set_page_config(
    page_title="Upbit Live Trading Bot v1 (Optimised)", page_icon="🤖", layout="wide"
)


import pandas as pd
from pydantic import BaseModel, Field, field_validator

from streamlit_extras.metric_cards import style_metric_cards  # type: ignore

# ───────────────────────── External business logic ───────────────────────────
from backtesting import Backtest  # type: ignore
from data_feed import stream_candles  # type: ignore
from strategy_v2 import MACDStrategy  # type: ignore
from trader import UpbitTrader  # type: ignore

from streamlit_autorefresh import st_autorefresh

# 예시: 5초(5000ms)마다 자동 새로고침
st_autorefresh(interval=5000, key="logrefresh")

# ───────────────────────────── Logging setup ─────────────────────────────────
LOG_PATH = "upbit_live.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf‑8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("upbit.live")

# ─────────────────────────────── Constants ───────────────────────────────────
INTERVAL_OPTIONS: dict[str, str] = {
    "1분봉": "minute1",
    "3분봉": "minute3",
    "5분봉": "minute5",
    "10분봉": "minute10",
    "15분봉": "minute15",
    "30분봉": "minute30",
    "60분봉": "minute60",
    "일봉": "day",
}


# ───────────────────────────── Pydantic model ────────────────────────────────
class LiveParams(BaseModel):
    ticker: str = Field(..., description="KRW‑BTC 형식 혹은 BTC")
    interval: str = Field(..., description="Upbit candle interval id")
    days: int = Field(90, ge=1, le=365)

    fast_period: int = Field(12, ge=5, le=50)
    slow_period: int = Field(26, ge=20, le=100)
    signal_period: int = Field(7, ge=5, le=20)

    macd_threshold: float = 0.0
    take_profit: float = Field(0.05, gt=0)
    stop_loss: float = Field(0.01, gt=0)

    cash: int = Field(1_000_000, ge=100_000)
    commission: float = Field(0.0005, ge=0)

    min_holding_period: int = 1
    macd_crossover_threshold: float = 0.0

    @field_validator("ticker")
    def _validate_ticker(cls, v: str) -> str:  # noqa: N805
        v = v.upper().strip()
        if "-" in v:
            base, quote = v.split("-", 1)
            if base != "KRW" or not quote.isalpha():
                raise ValueError("Format must be KRW-XXX or simply XXX")
            return v
        if not v.isalpha():
            raise ValueError("Ticker must be alphabetic, e.g. BTC, ETH")
        return v

    @property
    def upbit_ticker(self) -> str:
        return self.ticker if "-" in self.ticker else f"KRW-{self.ticker}"


# ─────────────────────────────── UI helpers ──────────────────────────────────


def validate_ticker(ticker: str) -> Optional[str]:
    if not ticker or not ticker.isalnum():
        st.error("❌ 거래 종목은 영문/숫자만 입력하세요.")
        return None
    # 실제 업비트 지원 종목 리스트와 비교 (권장)
    # ex) if ticker.upper() not in upbit_tickers: ...
    return ticker.upper()


def make_sidebar() -> Optional[LiveParams]:
    """Render sidebar form and return validated params (or None)."""
    with st.sidebar:
        st.header("⚙️ 파라미터 설정")
        with st.form("input_form"):
            ticker = st.text_input("거래 종목", value="DOGE")
            interval_name = st.selectbox(
                "차트 단위", list(INTERVAL_OPTIONS.keys()), index=5
            )
            days = st.number_input("데이터 기간(일)", 1, 365, 90)

            fast = st.number_input("단기 EMA", 5, 50, 12)
            slow = st.number_input("장기 EMA", 20, 100, 26)
            signal = st.number_input("신호선 기간", 5, 20, 7)
            macd_threshold = st.number_input("MACD 기준값", -10.0, 10.0, 0.0, 0.01)

            tp = st.number_input("Take Profit(%)", 0.1, 50.0, 5.0, 0.1) / 100
            sl = st.number_input("Stop Loss(%)", 0.1, 50.0, 1.0, 0.1) / 100
            cash = st.number_input(
                "초기 자본(원)", 100_000, 100_000_000_000, 1_000_000, 10_000
            )

            str_submitted = "🚀 Just Do It !!!"
            if st.session_state.test_mode:
                str_submitted = "🧪 Just Do It !!!"
            submitted = st.form_submit_button(str_submitted)

        if not submitted:
            return None

        try:
            return LiveParams(
                ticker=ticker,
                interval=INTERVAL_OPTIONS[interval_name],
                days=int(days),
                fast_period=int(fast),
                slow_period=int(slow),
                signal_period=int(signal),
                macd_threshold=macd_threshold,
                take_profit=tp,
                stop_loss=sl,
                cash=int(cash),
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"❌ 파라미터 오류: {exc}")
            return None


# ────────────────────────── Live engine (thread) ─────────────────────────────


def run_live_loop(
    params: LiveParams, q: queue.Queue, trader: UpbitTrader, stop_event: threading.Event
) -> None:
    """Worker: stream candles, backtest, send events to queue."""
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    import streamlit as st

    add_script_run_ctx(threading.current_thread())

    try:
        while not stop_event.is_set():
            for df in stream_candles(
                params.upbit_ticker, params.interval, q, stop_event=stop_event
            ):
                if stop_event.is_set():
                    break

                # dynamic strategy subclass
                class LiveStrategy(MACDStrategy):
                    fast_period = params.fast_period
                    slow_period = params.slow_period
                    signal_period = params.signal_period
                    take_profit = params.take_profit
                    stop_loss = params.stop_loss
                    macd_threshold = params.macd_threshold
                    min_holding_period = params.min_holding_period
                    macd_crossover_threshold = params.macd_crossover_threshold

                bt = Backtest(
                    df,
                    LiveStrategy,
                    cash=params.cash,
                    commission=params.commission,
                    exclusive_orders=True,
                )
                strategy: LiveStrategy = bt._strategy
                res = bt.run()

                signal_events = getattr(strategy, "signal_events", [])
                last_bar = len(df) - 1
                sig = None
                # 마지막 Candle에서 신호 발생 여부만 체크
                for bar_index, signal_type in reversed(signal_events):
                    if bar_index == last_bar:
                        sig = 1 if signal_type == "BUY" else -1
                        break

                price = df.Close.iloc[-1]
                have_coin = trader._coin_balance(params.upbit_ticker) > 0

                if sig == 1 and not have_coin:
                    trader.buy_market(
                        price, ticker=params.upbit_ticker, ts=df.index[-1]
                    )
                    q.put((df.index[-1], "BUY", price))
                elif sig == -1 and have_coin:
                    qty = trader._coin_balance(params.upbit_ticker)
                    trader.sell_market(
                        qty, ticker=params.upbit_ticker, price=price, ts=df.index[-1]
                    )
                    q.put((df.index[-1], "SELL", price))

                q.put((df.index[-1], "STATUS", price, sig))
                q.put(
                    (
                        df.index[-1],
                        "LOG",
                        f"{df.index[-1]} | price={price} | sig={sig}",
                    )
                )
    except Exception:  # pragma: no cover
        logger.exception("Live loop error")
        q.put(("EXCEPTION", *sys.exc_info()))


# ────────────────────────────── Helper funcs ─────────────────────────────────


def render_live_table(log_placeholder) -> None:
    q: queue.Queue = st.session_state.queue
    orders_box = st.empty()
    status_box = st.empty()

    tail = st.session_state["log_tail"]

    while not q.empty():
        event = q.get()
        if event[0] == "EXCEPTION":
            _, exc_type, exc_val, exc_tb = event
            import traceback as _tb

            st.session_state.live_error = str(exc_val)
            st.session_state.live_traceback = "".join(
                _tb.format_exception(exc_type, exc_val, exc_tb)
            )
            stop_live()
            return
        elif event[1] in ("BUY", "SELL"):
            st.session_state.orders.append(event[:3])
        elif event[1] == "STATUS":
            ts, _, price, sig = event
            status_box.info(f"{ts} | Close = {price} | Sig = {sig}")
        elif event[1] == "LOG":
            st.session_state.log_tail.append(event[2])
            st.session_state.log_tail = st.session_state.log_tail[-100:]
            log_placeholder.code("\n".join(st.session_state.log_tail))
        elif event[0] == "WARNING":
            # st.warning(event[1])
            tail.append(event[1])
        elif event[0] == "ERROR":
            # st.error(event[1])
            tail.append(event[1])
            st.session_state.live = False

    if st.session_state.orders:
        df_orders = pd.DataFrame(
            st.session_state.orders[-20:],  # 최근 20건만
            columns=["Time", "Side", "Price"],
        )
        orders_box.dataframe(df_orders, use_container_width=True)

    # 로그 새로고침 버튼 (expander/empty를 새로 만들지 않음)
    if st.button("🔄 로그 새로고침"):
        log_placeholder.code("\n".join(st.session_state.log_tail))


def stop_live() -> None:
    if "stop_event" in st.session_state:
        st.session_state.stop_event.set()
    st.session_state.live = False
    st.session_state.params = None
    st.session_state.trader = None
    # 모든 worker 종료 대기
    worker = st.session_state.get("worker", None)
    if worker is not None:
        worker.join(timeout=5)
        if worker.is_alive():
            st.warning(
                "⚠️ 백그라운드 스레드가 아직 종료되지 않았습니다. 잠시 후 다시 시도하세요."
            )
            st.session_state.worker = None
            st.session_state.queue = None
    st.session_state.orders = []
    st.session_state.log_tail = []
    logger.info("Live trading loop stopped.")


def clear_state(full: bool = False) -> None:
    st.session_state.pop("orders", None)
    st.session_state.pop("queue", None)
    if full:
        st.session_state.pop("live_error", None)
        st.session_state.pop("live_traceback", None)


# ────────────────────────────────── Main UI ──────────────────────────────────


def main() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarNavItems"],
        [data-testid="stSidebarNavSeparator"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ─────────── Session-state 초기화 ───────────
    if "live" not in st.session_state:
        st.session_state.live = False
    if "orders" not in st.session_state:
        st.session_state.orders: list[tuple] = []  # type: ignore
    if "stop_event" not in st.session_state or not isinstance(
        st.session_state.stop_event, threading.Event
    ):
        st.session_state.stop_event = threading.Event()
    if "test_mode" not in st.session_state:
        st.session_state.test_mode = True
    if "log_tail" not in st.session_state:
        st.session_state.log_tail = []
    if "log_placeholder" not in st.session_state:
        st.session_state.log_placeholder = None

    with st.sidebar:
        st.subheader("🔧 거래 모드 설정")
        trade_mode = st.radio(
            "거래 모드",
            ["🧪 테스트 모드", "🚀 실전 모드"],
            index=0,
            horizontal=True,
            help="실전 모드는 실제 자산에 영향을 미치니 신중히 선택하세요.",
        )
        test_mode = trade_mode == "🧪 테스트 모드"
        if not test_mode:
            st.warning(
                "실전 거래 모드입니다. 모든 작업이 실제 자산에 영향을 미칩니다.",
                icon="⚠️",
            )
            # 추가 보안: 비밀번호 등 2차 확인 절차
            password = st.text_input("실전 모드 비밀번호 입력", type="password")
            if password != st.secrets.get("UPBIT_PASSWORD", ""):
                st.stop()
            st.success("실전 모드 인증 완료!", icon="✅")
        st.session_state["test_mode"] = test_mode

    st.title("🤖 Upbit Live Trading Bot v1 (Optimised)")

    params = make_sidebar()

    # ─────────── 라이브 루프 시작 ───────────
    if params and not st.session_state.live:
        st.session_state.params = params

        # 이전 스레드 종료 보장
        prev_worker = st.session_state.get("worker", None)
        if prev_worker is not None and prev_worker.is_alive():
            st.session_state.stop_event.set()
            prev_worker.join(timeout=5)
            st.session_state.worker = None
            st.session_state.queue = None

        # 새로운 stop_event 생성 (매번 새로 생성)
        st.session_state.stop_event = threading.Event()

        st.session_state.live = True
        st.session_state.orders.clear()

        q: queue.Queue = queue.Queue()
        trader: UpbitTrader = UpbitTrader(test_mode=test_mode)
        st.session_state.trader = trader
        stop_event = st.session_state.stop_event
        worker = threading.Thread(
            target=run_live_loop,
            args=(params, q, trader, stop_event),
            daemon=True,
        )
        worker.start()
        st.session_state.worker = worker
        st.session_state.queue = q
        st.success("✅ Upbit Live Trading Bot이 실행되었습니다.")

    params = st.session_state.get("params", None)
    trader = st.session_state.get("trader", None)
    if params is None:
        st.info("사이드바에서 거래 파라미터 설정 후 진행하세요.")
        # 또는 return 등으로 함수 종료
    if trader is None:
        str_submitted = "🚀 Just Do It !!!"
        if st.session_state.test_mode:
            str_submitted = "🧪 Just Do It !!!"
        st.info(f"거래를 시작해 주세요. [{str_submitted}]")
        # return
    if st.session_state.live and params is not None and trader is not None:
        # 안전하게 params 사용
        render_account_metrics(trader, params.upbit_ticker, params.cash)

    # ─────────── 진행 중 UI ───────────
    if st.session_state.live:
        if test_mode:
            st.subheader("🟢 Trading Bot 실행 중… (TEST)")
        else:
            st.subheader("🟢 Trading Bot 실행 중…")

        if st.button("🛑 Tading Bot 실행 중단"):
            stop_live()
            st.success("✅ Upbit Live Trading Bot이 중단되었습니다.")
            worker = st.session_state.get("worker", None)
            if worker is not None:
                worker.join(timeout=5)  # 최대 5초 대기 (필요시 조정)
                if worker.is_alive():
                    st.warning(
                        "⚠️ 백그라운드 스레드가 아직 종료되지 않았습니다. 잠시 후 다시 시도하세요."
                    )
                st.session_state.worker = None  # worker 객체 해제
                st.session_state.queue = None
            st.rerun()

        log_expander = st.expander("📜 실시간 로그", expanded=True)
        with log_expander:
            log_placeholder = st.empty()  # 세션 상태에 저장하지 않음
            # 최초 실행 시 로그 출력
            log_placeholder.code("\n".join(st.session_state.log_tail))

        render_live_table(log_placeholder)

    # ─────────── 오류 UI ───────────
    if st.session_state.get("live_error"):
        st.error(f"❌ 라이브 루프 오류: {st.session_state.live_error}")
        with st.expander("자세한 오류 정보 보기"):
            st.text(st.session_state.live_traceback)
        if st.button("초기화면으로 돌아가기"):
            clear_state(full=True)
            st.rerun()


def render_account_metrics(trader: UpbitTrader, market: str, base_cash: int):
    krw = trader._krw_balance()
    holding = trader._coin_balance(market)
    # pnl = trader.realised_pnl()

    col1, col2, col3 = st.columns(3)
    col1.metric("💵 KRW 잔고", f"{krw:,.0f} 원")
    col2.metric("📦 코인 수량", f"{holding:,.4f}")
    # col3.metric("📈 누적 PnL", f"{pnl:,.0f} 원", delta=f"{(pnl/base_cash)*100:.2f}%")
    style_metric_cards()


def style_metric_cards():
    st.markdown(
        """
        <style>
        /* metric 카드 배경/글자색 다크모드/라이트모드 대응 */
        [data-testid="stMetric"] {
            background-color: var(--background-color);
            border-radius: 0.5em;
            padding: 1em;
            margin: 0.5em 0;
            color: var(--text-color);
            border: 1px solid #44444422;
        }
        /* 라이트모드 */
        @media (prefers-color-scheme: light) {
          [data-testid="stMetric"] {
            background-color: #f7f7f7;
            color: #222;
          }
        }
        /* 다크모드 */
        @media (prefers-color-scheme: dark) {
          [data-testid="stMetric"] {
            background-color: #22272b;
            color: #f7f7f7;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def update_log():
    if st.session_state.log_placeholder:
        st.session_state.log_placeholder.code("\n".join(st.session_state.log_tail))


# ──────────────────────────── CLI Signal Hook ───────────────────────────────
# Streamlit 페이지가 아닌, 터미널에서 단독 실행 시에만 SIGINT 핸들러 등록
try:
    signal.signal(signal.SIGINT, lambda *_: stop_live())
except ValueError:
    # Streamlit 내부(서브 스레드)에서는 signal 사용 불가 → 무시
    pass


if __name__ == "__main__":
    main()
