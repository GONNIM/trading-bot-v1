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
Or place in *pages* directory (`pages/app_live.py`).
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
    page_title="Upbit Live Trading Bot v1", page_icon="🤖", layout="wide"
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

st_autorefresh(interval=5000, key="logrefresh")


# ───────────────────────────── Logging setup ─────────────────────────────────
LOG_PATH = "upbit_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf‑8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("upbit.test")

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

CASH_OPTIONS = {
    "10-percent": {
        "button": "10%",
        "ratio": 0.1,
    },
    "25-percent": {
        "button": "25%",
        "ratio": 0.25,
    },
    "50-percent": {
        "button": "50%",
        "ratio": 0.5,
    },
    "100-percent": {
        "button": "100%",
        "ratio": 1,
    },
}


from config import MIN_CASH, MIN_FEE_RATIO


params = st.query_params
virtual_krw = int(params.get("virtual_krw", 0))
user_id = params.get("user_id", "")
# st.write(f"가상 보유자산: {virtual_krw:,} KRW")
# st.write(f"User ID: {user_id}")
if virtual_krw < MIN_CASH:
    st.switch_page("pages/select_test.py")

if "virtual_amount" not in st.session_state:
    st.session_state.virtual_amount = virtual_krw
if "order_ratio" not in st.session_state:
    st.session_state.order_ratio = 1
if "order_amount" not in st.session_state:
    st.session_state.order_amount = virtual_krw


# ───────────────────────────── Pydantic model ────────────────────────────────
class LiveParams(BaseModel):
    ticker: str = Field(..., description="KRW‑BTC 형식 혹은 BTC")
    interval: str = Field(..., description="Upbit candle interval id")

    fast_period: int = Field(12, ge=1, le=50)
    slow_period: int = Field(26, ge=1, le=100)
    signal_period: int = Field(7, ge=1, le=20)

    macd_threshold: float = 0.0
    take_profit: float = Field(0.05, gt=0)
    stop_loss: float = Field(0.01, gt=0)

    cash: int = Field(MIN_CASH, ge=MIN_CASH)
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

            fast = st.number_input("단기 EMA", 5, 50, 12)
            slow = st.number_input("장기 EMA", 20, 100, 26)
            signal = st.number_input("신호선 기간", 5, 20, 7)
            macd_threshold = st.number_input("MACD 기준값", -10.0, 10.0, 0.0, 0.01)

            tp = st.number_input("Take Profit (%)", 0.1, 50.0, 5.0, 0.1) / 100
            sl = st.number_input("Stop Loss (%)", 0.1, 50.0, 1.0, 0.1) / 100

            # cash = st.number_input(
            #     "주문총액 (KRW)", 0, 100_000_000_000, int(virtual_krw), 10_000
            # )
            st.write("주문총액 (KRW)")
            st.info(f"{st.session_state.order_amount:,.0f}")
            cash = st.session_state.order_amount

            st.subheader("테스트 모드 실행")
            str_submitted = "🧪 Just Do It !!!"
            submitted = st.form_submit_button(str_submitted, use_container_width=True)

        st.write("")

        columns = st.columns(4)
        for i, (name, info) in enumerate(CASH_OPTIONS.items()):
            if columns[i].button(info["button"], key=name, use_container_width=True):
                st.session_state.order_ratio = info["ratio"]
                st.session_state.order_amount = (
                    st.session_state.virtual_amount * st.session_state.order_ratio
                )
                st.rerun()

        st.subheader("가상 보유자산")
        st.info(f"{st.session_state.virtual_amount:,.0f} KRW")

        if not submitted:
            return None

    if st.session_state.test_mode:
        try:
            return LiveParams(
                ticker=ticker,
                interval=INTERVAL_OPTIONS[interval_name],
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
    else:
        if MIN_CASH > virtual_krw:
            st.error(f"가상 보유자산이 최소주문가능금액({MIN_CASH} KRW)보다 작습니다.")
            st.stop()
            return None

        try:
            return LiveParams(
                ticker=ticker,
                interval=INTERVAL_OPTIONS[interval_name],
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
    params: LiveParams,
    q: queue.Queue,
    trader: UpbitTrader,
    stop_event: threading.Event,
    test_mode: bool,
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
                # res = bt.run()
                # strategy = bt._strategy
                bt.run()
                signal_events = MACDStrategy.signal_events

                # signal_events = getattr(strategy, "signal_events", [])
                last_bar = len(df) - 1

                for event in signal_events:
                    if event[1] == "LOG":
                        bar_index = event[0]
                        position = event[2]
                        macd = event[3]
                        signal = event[4]
                        price = event[5]
                        q.put(
                            (
                                df.index[bar_index],
                                "LOG",
                                f"{df.index[bar_index]} | price={price} | position={position} | macd={macd} | signal={signal} | bar={bar_index}",
                            )
                        )

                sig = None
                position = None
                macd = None
                signal = None
                for event in reversed(signal_events):
                    if event[0] == last_bar:
                        if event[1] == "BUY":
                            sig = 1
                        elif event[1] == "SELL":
                            sig = -1
                        else:
                            continue
                        if len(event) >= 5:
                            position = event[2]
                            macd = event[3]
                            signal = event[4]
                        else:
                            position = macd = signal = None
                        break

                price = df.Close.iloc[-1]
                # have_coin = trader._coin_balance(params.upbit_ticker) > 0
                have_coin = st.session_state.get("holding", 0) > 0

                if sig == 1 and not have_coin:
                    result_buy = trader.buy_market(
                        price, ticker=params.upbit_ticker, ts=df.index[-1]
                    )
                    q.put(
                        (
                            df.index[-1],
                            "BUY",
                            result_buy["qty"],
                            result_buy["price"],
                            position,
                            macd,
                            signal,
                        )
                    )
                elif sig == -1 and have_coin:
                    # qty = trader._coin_balance(params.upbit_ticker)
                    qty = st.session_state.holding
                    result_sell = trader.sell_market(
                        qty, ticker=params.upbit_ticker, price=price, ts=df.index[-1]
                    )
                    q.put(
                        (
                            df.index[-1],
                            "SELL",
                            result_sell["qty"],
                            result_sell["price"],
                            position,
                            macd,
                            signal,
                        )
                    )
    except Exception:  # pragma: no cover
        logger.exception("Live loop error")
        q.put(("EXCEPTION", *sys.exc_info()))


# ────────────────────────────── Helper funcs ─────────────────────────────────


def render_live_table(log_placeholder) -> None:
    q: queue.Queue = st.session_state.queue
    orders_box = st.empty()

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
            # st.session_state.orders.append(event[:3])
            if len(event) >= 6:
                ts, event_type, qty, price, position, macd, signal = event[:6]
                # macd, signal 값을 테이블, 로그 등에 활용
            else:
                ts, event_type, qty, price = event[:3]
                position = macd = signal = None

            if event[1] == "BUY":
                buy_amount = qty * price
                st.session_state.holding += qty
                st.session_state.virtual_amount -= buy_amount
            elif event[1] == "SELL":
                sell_amount = qty * price
                sell_fee = sell_amount * MIN_FEE_RATIO
                sell_total_amount = sell_amount - sell_fee
                st.session_state.holding -= qty
                st.session_state.virtual_amount += sell_total_amount
            st.session_state.order_amount = (
                st.session_state.virtual_amount * st.session_state.order_ratio
            )
            st.session_state.orders.append(
                [ts, event_type, qty, price, position, macd, signal]
            )
        elif event[1] == "LOG":
            st.session_state.log_tail.append(event[2])
            st.session_state.log_tail = st.session_state.log_tail[-10:]
            log_placeholder.code("\n".join(st.session_state.log_tail))
        elif event[0] == "WARNING":
            st.warning(event[1])
            st.session_state.log_tail.append(event[1])
            st.session_state.log_tail = st.session_state.log_tail[-10:]
            log_placeholder.code("\n".join(st.session_state.log_tail))
        elif event[0] == "ERROR":
            if len(event) == 4:
                _, exc_type, exc_val, exc_tb = event
                import traceback as _tb

                st.session_state.live_error = str(exc_val)
                st.session_state.live_traceback = "".join(
                    _tb.format_exception(exc_type, exc_val, exc_tb)
                )
            elif len(event) == 2:
                _, message = event
                st.session_state.live_error = str(message)
                st.session_state.live_traceback = ""
            else:
                st.session_state.live_error = f"알 수 없는 ERROR 이벤트 형식: {event}"
                st.session_state.live_traceback = ""
            stop_live()
            return

    # if st.session_state.orders:
    #     df_orders = pd.DataFrame(
    #         st.session_state.orders[-20:],  # 최근 20건만
    #         # columns=["Time", "Side", "Price"],
    #         columns=["Time", "Event", "Qty", "Price", "Position", "MACD", "Signal"],
    #     )
    #     orders_box.dataframe(df_orders, use_container_width=True)
    recent_orders = st.session_state.orders[-20:]
    df_orders = pd.DataFrame(
        recent_orders,
        columns=["Time", "Event", "Qty", "Price", "Position", "MACD", "Signal"],
    )
    orders_box.dataframe(df_orders, use_container_width=True)

    # 로그 새로고침 버튼 (expander/empty를 새로 만들지 않음)
    if st.button("🔄 로그 새로고침", use_container_width=True):
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
    logger.info("Live trading loop stopped.")


def clear_state(full: bool = False) -> None:
    st.session_state.pop("orders", None)
    st.session_state.pop("queue", None)
    st.session_state.orders = []
    st.session_state.log_tail = []
    if full:
        st.session_state.pop("live_error", None)
        st.session_state.pop("live_traceback", None)


# ────────────────────────────────── Main UI ──────────────────────────────────


def main() -> None:
    st.markdown(
        """
        <style>
        /* 헤더와 본문 사이 간격 제거 */
        div.block-container {
            padding-top: 1rem;  /* 기본값은 3rem */
        }

        /* 제목 상단 마진 제거 */
        h1 {
            margin-top: 0 !important;
        }

        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarNavItems"],
        [data-testid="stSidebarNavSeparator"] { display: none !important; }
        div.stButton > button, div.stForm > form > button {
            height: 60px !important;
            font-size: 30px !important;
            font-weight: 900 !important;
        }
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

    if "holding" not in st.session_state:
        st.session_state.holding = 0

    test_mode = True
    st.session_state["test_mode"] = test_mode

    st.title(f"🤖 Upbit Live Trading Bot v1 (TEST) - {user_id}")

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
            args=(params, q, trader, stop_event, test_mode),
            daemon=True,
        )
        worker.start()
        st.session_state.worker = worker
        st.session_state.queue = q
        st.success("✅ Upbit Live Trading Bot이 실행되었습니다.")

    if not st.session_state.get("live_error"):
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

        if st.button("🛑 Tading Bot 실행 중단", use_container_width=True):
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

        st.write("")

        log_expander = st.expander("📜 실시간 로그", expanded=True)
        with log_expander:
            log_placeholder = st.empty()  # 세션 상태에 저장하지 않음
            # 최초 실행 시 로그 출력
            log_placeholder.code("\n".join(st.session_state.log_tail))

        render_live_table(log_placeholder)

    # ─────────── 오류 UI ───────────
    if st.session_state.get("live_error"):
        st.error(f"❌ Trading Bot Error: {st.session_state.live_error}")
        with st.expander("자세한 오류 정보 보기"):
            st.text(st.session_state.live_traceback)
        if st.button("초기화면으로 돌아가기", use_container_width=True):
            clear_state(full=True)
            st.rerun()


def render_account_metrics(trader: UpbitTrader, market: str, base_cash: int):
    # krw = trader._krw_balance()
    # holding = trader._coin_balance(market)
    # pnl = trader.realised_pnl()
    krw = st.session_state.get("virtual_amount", 0)
    holding = st.session_state.get("holding", 0)
    pnl = krw - virtual_krw

    col1, col2, col3 = st.columns(3)
    col1.metric("💵 KRW 잔고", f"{krw:,.0f} KRW")
    col2.metric("📦 코인 수량", f"{holding:,.4f}")
    col3.metric("📈 누적 PnL", f"{pnl:,.0f} KRW", delta=f"{(pnl/base_cash)*100:.2f}%")
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
