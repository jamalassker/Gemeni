import pandas as pd
import numpy as np
import asyncio, logging, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt 

# ================= CONFIG =================
TIMEFRAME = "5m"
MAX_POSITIONS = 1
MIN_PROB = 0.55        
TAKER_FEE = 0.001

ATR_MULTIPLIER_TP = 0.6
ATR_MULTIPLIER_SL = 0.9
MAX_TRADE_SECONDS = 180
MIN_ATR_PCT = 0.004   # volatility filter

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

SYMBOLS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","AVAX/USDT","DOGE/USDT",
    "PEPE/USDT","WIF/USDT","BONK/USDT","FLOKI/USDT","INJ/USDT"
]

class ProSniperV7:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 100
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.report_id = None
        self.session = None

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def send_tg(self, text, edit=False):
        await self.init_session()
        url = f"https://api.telegram.org/bot{TG_TOKEN}/"
        method = "editMessageText" if edit and self.report_id else "sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
        if edit: payload["message_id"] = self.report_id
        async with self.session.post(url + method, json=payload):
            pass

    async def get_data(self, sym, limit=150):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        df['tr'] = np.maximum(df['h'] - df['l'], 
            np.maximum(abs(df['h'] - df['c'].shift()), abs(df['l'] - df['c'].shift())))
        df['atr'] = df['tr'].rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().clip(lower=0).rolling(14).mean() /
                                       df['c'].diff().abs().rolling(14).mean().replace(0,0.001))))
        return df.dropna()

    async def training_cycle(self):
        while True:
            new_models = {}
            for sym in SYMBOLS:
                try:
                    df = await self.get_data(sym)
                    df['target'] = (df['c'].shift(-3) > df['c'] * 1.003).astype(int)
                    X = df[['rsi','atr']]
                    model = Pipeline([
                        ('s', StandardScaler()),
                        ('rf', RandomForestClassifier(n_estimators=50))
                    ])
                    model.fit(X, df['target'])
                    new_models[sym] = model
                except: pass
            self.models.update(new_models)
            await asyncio.sleep(3600)

    async def risk_manager(self):
        while True:
            for sym, p in list(self.positions.items()):
                ticker = await self.exchange.fetch_ticker(sym)
                price = ticker['last']

                # TIME EXIT
                if time.time() - p['time'] > MAX_TRADE_SECONDS:
                    price = ticker['last']
                    val = p['size'] * price
                    self.cash += val * (1 - TAKER_FEE)
                    self.realized_pnl += val - (p['size'] * p['entry'])
                    del self.positions[sym]
                    await self.send_tg(f"⏱ FORCE EXIT {sym}")
                    continue

                if price >= p['tp'] or price <= p['sl']:
                    val = p['size'] * price
                    self.cash += val * (1 - TAKER_FEE)
                    self.realized_pnl += val - (p['size'] * p['entry'])
                    del self.positions[sym]
                    await self.send_tg(f"🏁 EXIT {sym}")
            await asyncio.sleep(1)

    async def trade_loop(self):
        asyncio.create_task(self.training_cycle())
        asyncio.create_task(self.risk_manager())

        while True:
            if not self.models or self.positions:
                await asyncio.sleep(3)
                continue

            candidates = []

            for sym in SYMBOLS:
                try:
                    df = await self.get_data(sym, 30)
                    atr_pct = df['atr'].iloc[-1] / df['c'].iloc[-1]
                    if atr_pct < MIN_ATR_PCT:
                        continue

                    prob = self.models[sym].predict_proba(df[['rsi','atr']].iloc[-1:])[0][1]
                    candidates.append((prob, atr_pct, sym, df))
                except: pass

            if candidates:
                prob, _, sym, df = max(candidates)
                if prob >= MIN_PROB:
                    entry = df['c'].iloc[-1]
                    atr = df['atr'].iloc[-1]

                    size = (self.cash * 0.1) / entry
                    self.cash -= self.cash * 0.1

                    self.positions[sym] = {
                        'entry': entry,
                        'size': size,
                        'tp': entry + atr * ATR_MULTIPLIER_TP,
                        'sl': entry - atr * ATR_MULTIPLIER_SL,
                        'time': time.time()
                    }

                    await self.send_tg(f"🚀 BUY {sym} | Prob {prob:.2f}")

            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(ProSniperV7().trade_loop())





