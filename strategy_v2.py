from backtesting import Strategy
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MACDStrategy(Strategy):
    fast_period = 12
    slow_period = 26
    signal_period = 7
    take_profit = 0.05
    stop_loss = 0.01
    macd_threshold = 0.0
    min_holding_period = 1
    macd_crossover_threshold = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"MACDStrategy __init__ id={id(self)}")

    def init(self):
        logger.info("전략 초기화 시작")
        close = self.data.Close
        self.macd_line = self.I(
            self._calculate_macd, close, self.fast_period, self.slow_period
        )
        self.signal_line = self.I(
            self._calculate_signal, self.macd_line, self.signal_period
        )
        self.entry_price = None
        self.entry_bar = None
        self.last_signal_bar = None
        # self.signal_events = []  # (bar_index, 'BUY'/'SELL')
        # logger.info(f"init: self.signal_events id={id(self.signal_events)}")
        MACDStrategy.signal_events = []
        logger.info(
            f"init: MACDStrategy.signal_events id={id(MACDStrategy.signal_events)}"
        )

    def _calculate_macd(self, series, fast, slow):
        series = pd.Series(series)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.values

    def _calculate_signal(self, macd, period):
        macd = pd.Series(macd)
        signal = macd.ewm(span=period, adjust=False).mean()
        return signal.values

    def next(self):
        current_bar = len(self.data) - 1
        current_price = self.data.Close[-1]
        macd_val = float(self.macd_line[-1])
        signal_val = float(self.signal_line[-1])
        position_val = "Gold" if self.position else "Dead"

        # 같은 봉에서 신호 중복 방지
        if self.last_signal_bar == current_bar:
            return

        # 상태 로그 남기기 (매번)
        MACDStrategy.signal_events.append(
            (current_bar, "LOG", position_val, macd_val, signal_val, current_price)
        )

        if self.position:
            bars_since_entry = current_bar - self.entry_bar
            # 익절/손절
            tp_price = self.entry_price * (1 + self.take_profit)
            sl_price = self.entry_price * (1 - self.stop_loss)
            if current_price >= tp_price or current_price <= sl_price:
                self.position.close()
                MACDStrategy.signal_events.append(
                    (
                        current_bar,
                        "SELL",
                        position_val,
                        macd_val,
                        signal_val,
                    )
                )
                self.entry_price = None
                self.entry_bar = None
                self.last_signal_bar = current_bar
                return

            # 최소 보유 기간
            if bars_since_entry < self.min_holding_period:
                return

            # 매도 신호
            macd_diff = self.macd_line[-1] - self.signal_line[-1]
            if (
                macd_diff < -self.macd_crossover_threshold
                and self.macd_line[-2] >= self.signal_line[-2]
                and self.macd_line[-1] >= self.macd_threshold
            ):
                self.position.close()
                MACDStrategy.signal_events.append(
                    (
                        current_bar,
                        "SELL",
                        position_val,
                        macd_val,
                        signal_val,
                    )
                )
                self.entry_price = None
                self.entry_bar = None
                self.last_signal_bar = current_bar

        if not self.position:
            # 매수 신호
            macd_diff = self.macd_line[-1] - self.signal_line[-1]
            if (
                macd_diff > self.macd_crossover_threshold
                and self.macd_line[-2] <= self.signal_line[-2]
                and self.macd_line[-1] >= self.macd_threshold
            ):
                self.buy()
                MACDStrategy.signal_events.append(
                    (
                        current_bar,
                        "BUY",
                        position_val,
                        macd_val,
                        signal_val,
                    )
                )
                self.entry_price = current_price
                self.entry_bar = current_bar
                self.last_signal_bar = current_bar
