import pyupbit, logging, math, time, traceback, pandas as pd
from config import ACCESS, SECRET

logger = logging.getLogger(__name__)


class UpbitTrader:
    def __init__(self, risk_pct=0.1, test_mode=True):
        self.test_mode, self.risk_pct = test_mode, risk_pct
        self.upbit = None if test_mode else pyupbit.Upbit(ACCESS, SECRET)

    # 잔고 헬퍼
    def _krw_balance(self):
        if self.test_mode:
            return 1_000_000
        try:
            # 공식 메서드 사용
            balance = self.upbit.get_balance(ticker="KRW")
            return float(balance) if balance else 0.0
        except Exception as e:
            print(f"[CRITICAL] 잔고 조회 실패: {e}")
            return 0.0

    def _coin_balance(self, ticker):
        cur = ticker.split("-")[1]
        if self.test_mode:
            return 0.0
        for b in self.upbit.get_balances():
            if b["currency"] == cur:
                return float(b["balance"])
        return 0.0

    # 주문 API
    def _buy(self, ticker, qty, price, ts):
        if self.test_mode:
            logger.info(f"[TEST BUY] {ticker} {qty}@{price}")
            return dict(time=ts, action="BUY", qty=qty, price=price)
        return self.upbit.buy_market_order(ticker, qty)

    def _sell(self, ticker, qty, price, ts):
        if qty <= 0:
            return None
        if self.test_mode:
            logger.info(f"[TEST SELL] {ticker} {qty}@{price}")
            return dict(time=ts, action="SELL", qty=qty, price=price)
        return self.upbit.sell_market_order(ticker, qty)

    def execute_signals(self, df: pd.DataFrame, ticker: str):
        orders = []
        for _, row in df.iterrows():
            sig = int(row["signal"])
            price = row.get("EntryPrice") if sig == 1 else row.get("ExitPrice")
            ts = row.get("EntryTime") or row.get("ExitTime")
            if pd.isna(price):
                continue
            try:
                if sig == 1:
                    krw = self._krw_balance() * self.risk_pct
                    qty = round(krw / price, 8)
                    orders.append(self._buy(ticker, qty, price, ts))
                else:
                    qty = self._coin_balance(ticker)
                    orders.append(self._sell(ticker, qty, price, ts))
                time.sleep(0.2)
            except Exception:
                logger.error(traceback.format_exc())
        return orders

    # ------------------------------------------------------------
    # 새로 추가!  buy_market  /  sell_market
    # ------------------------------------------------------------
    def buy_market(self, price: float, ticker: str, ts=None):
        krw = self._krw_balance() * self.risk_pct
        qty = round(krw / price, 8)
        if self.test_mode:
            logger.info(f"[TEST BUY] {ticker}  {qty}@{price}")
            return {"time": ts, "side": "BUY", "qty": qty, "price": price}
        return self.upbit.buy_market_order(ticker, qty)

    def sell_market(self, qty: float, ticker: str, price: float, ts=None):
        if qty <= 0:
            return None
        if self.test_mode:
            logger.info(f"[TEST SELL] {ticker}  {qty}@{price}")
            return {"time": ts, "side": "SELL", "qty": qty, "price": price}
        return self.upbit.sell_market_order(ticker, qty)
