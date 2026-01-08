import pandas as pd
import numpy as np
import asyncio, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt

TIMEFRAME = "1m"
MAX_POSITIONS = 3
MIN_PROB = 0.52
TAKER_FEE = 0.001
ATR_TP = 0.7
ATR_SL = 0.9
MAX_HOLD = 180
TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"
SYMBOLS = ["PEPE/USDT","WIF/USDT","SOL/USDT","SUI/USDT","BONK/USDT","FLOKI/USDT","SEI/USDT","AVAX/USDT","NEAR/USDT","OP/USDT"]

class Sniper:
    def __init__(self):
        self.ex = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.pos = {}
        self.cash = 100.0
        self.pnl = 0.0
        self.msg_id = None
        self.sess = None

    async def tg(self, txt, edit=False):
        if not self.sess: self.sess = aiohttp.ClientSession()
        url = f"https://api.telegram.org/bot{TG_TOKEN}/" + ("editMessageText" if edit and self.msg_id else "sendMessage")
        data = {"chat_id": TG_CHAT, "text": txt, "parse_mode": "HTML"}
        if edit and self.msg_id: data["message_id"] = self.msg_id
        try:
            async with self.sess.post(url, json=data, timeout=5) as r:
                res = await r.json()
                if not edit and res.get("ok"): self.msg_id = res["result"]["message_id"]
        except: pass

    async def fetch(self, sym, n=60):
        ohlcv = await self.ex.fetch_ohlcv(sym, TIMEFRAME, limit=n)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        df['atr'] = (df['h'] - df['l']).rolling(10).mean()
        delta = df['c'].diff()
        gain = delta.clip(lower=0).rolling(10).mean()
        loss = (-delta.clip(upper=0)).rolling(10).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
        df['ema_fast'] = df['c'].ewm(span=8).mean()
        df['ema_slow'] = df['c'].ewm(span=21).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['mom'] = df['c'].pct_change(5)
        df['vol_spike'] = df['v'] / df['v'].rolling(20).mean()
        df['atr_pct'] = df['atr'] / df['c']
        return df.dropna()

    async def train(self):
        await self.tg("⚡ Training models...")
        for sym in SYMBOLS:
            try:
                df = await self.fetch(sym, 100)
                df['target'] = (df['c'].shift(-2) > df['c'] * 1.0015).astype(int)
                feats = ['rsi','macd','mom','vol_spike','atr_pct']
                X, y = df[feats].fillna(0), df['target']
                pipe = Pipeline([('sc', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=25, max_depth=5))])
                pipe.fit(X, y)
                self.models[sym] = (pipe, feats)
            except: pass
        await self.tg(f"✅ Trained {len(self.models)} models")

    async def retrain_loop(self):
        while True:
            await asyncio.sleep(600)
            await self.train()

    async def manage(self):
        while True:
            for sym in list(self.pos):
                try:
                    p = self.pos[sym]
                    tick = await self.ex.fetch_ticker(sym)
                    price = tick['last']
                    held = time.time() - p['t']
                    reason = None
                    if price >= p['tp']: reason = "✅TP"
                    elif price <= p['sl']: reason = "❌SL"
                    elif held > MAX_HOLD: reason = "⏱TIME"
                    elif held > 60 and price > p['entry'] * 1.001: reason = "🔒LOCK"
                    if reason:
                        val = p['sz'] * price
                        self.cash += val * (1 - TAKER_FEE)
                        gain = val - p['cost']
                        self.pnl += gain
                        del self.pos[sym]
                        await self.tg(f"🏁 {reason} {sym} | +${gain:.3f} | Net: ${self.pnl:+.2f}")
                except: pass
            slots = ", ".join(self.pos.keys()) or "—"
            await self.tg(f"💎 <b>SNIPER LIVE</b>\n💰 Cash: ${self.cash:.2f}\n📈 PnL: ${self.pnl:+.2f}\n🎯 Pos: {slots}", edit=True)
            await asyncio.sleep(2)

    async def scan(self):
        while True:
            if len(self.pos) >= MAX_POSITIONS:
                await asyncio.sleep(1); continue
            for sym in SYMBOLS:
                if sym in self.pos or sym not in self.models: continue
                try:
                    df = await self.fetch(sym, 30)
                    row = df.iloc[-1]
                    pipe, feats = self.models[sym]
                    X = df[feats].iloc[-1:].fillna(0)
                    prob = pipe.predict_proba(X)[0][1]
                    trend_ok = row['ema_fast'] > row['ema_slow']
                    rsi_ok = 30 < row['rsi'] < 70
                    if prob >= MIN_PROB and trend_ok and rsi_ok:
                        price = row['c']
                        atr = row['atr']
                        alloc = (self.cash / (MAX_POSITIONS - len(self.pos))) * 0.9
                        cost = alloc * (1 + TAKER_FEE)
                        if cost > self.cash: continue
                        self.cash -= cost
                        self.pos[sym] = {
                            'entry': price, 'sz': alloc / price, 'cost': cost, 't': time.time(),
                            'tp': price + atr * ATR_TP, 'sl': price - atr * ATR_SL
                        }
                        await self.tg(f"🚀 <b>BUY {sym}</b> @ {price:.6f} | P={prob:.2f}")
                except: pass
            await asyncio.sleep(0.8)

    async def run(self):
        await self.train()
        await asyncio.gather(self.retrain_loop(), self.manage(), self.scan())

if __name__ == "__main__":
    asyncio.run(Sniper().run())
