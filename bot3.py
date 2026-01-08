import pandas as pd
import numpy as np
import asyncio, time, aiohttp
import ccxt.async_support as ccxt

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
TIMEFRAME = "1m"
MAX_HOLD = 180
TAKER_FEE = 0.001
TG_TOKEN = "8287625785:AAH5CzpIgBiDYWO3WGikKYSaTwgz0rgc2y0"
TG_CHAT = "8287625785"

class ZScalper:
    def __init__(self):
        self.ex = ccxt.binance({'enableRateLimit': True})
        self.pos = None
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

    def calc(self, df):
        # Z-Score: how far price deviates from mean
        df['sma20'] = df['c'].rolling(20).mean()
        df['std20'] = df['c'].rolling(20).std()
        df['zscore'] = (df['c'] - df['sma20']) / df['std20'].replace(0, 1e-9)
        # RSI
        delta = df['c'].diff()
        gain = delta.clip(lower=0).rolling(10).mean()
        loss = (-delta.clip(upper=0)).rolling(10).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
        # Momentum & Volume
        df['mom'] = df['c'].pct_change(5) * 100
        df['vol_z'] = (df['v'] - df['v'].rolling(20).mean()) / df['v'].rolling(20).std().replace(0, 1e-9)
        df['atr'] = (df['h'] - df['l']).rolling(10).mean()
        return df.dropna()

    async def fetch(self, sym):
        ohlcv = await self.ex.fetch_ohlcv(sym, TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        return self.calc(df)

    async def run(self):
        await self.tg("🚀 <b>Z-SCORE SCALPER ONLINE</b>")
        while True:
            # EXIT LOGIC
            if self.pos:
                try:
                    tick = await self.ex.fetch_ticker(self.pos['sym'])
                    price = tick['last']
                    held = time.time() - self.pos['t']
                    reason = None
                    if price >= self.pos['tp']: reason = "✅TP"
                    elif price <= self.pos['sl']: reason = "❌SL"
                    elif held > MAX_HOLD: reason = "⏱TIME"
                    elif held > 45 and price > self.pos['entry'] * 1.001: reason = "🔒LOCK"
                    if reason:
                        val = self.pos['sz'] * price * (1 - TAKER_FEE)
                        gain = val - self.pos['cost']
                        self.pnl += gain
                        self.cash = val
                        await self.tg(f"🏁 {reason} {self.pos['sym']} | ${gain:+.3f} | Net: ${self.pnl:+.2f}")
                        self.pos = None
                except: pass

            # ENTRY LOGIC
            if not self.pos:
                best, best_score = None, 0
                for sym in SYMBOLS:
                    try:
                        df = await self.fetch(sym)
                        r = df.iloc[-1]
                        # BUY SIGNAL: Z-score dip + RSI oversold + positive momentum + volume spike
                        # Z < -1.5 = price below normal, RSI < 40 = oversold, mom > 0 = recovering
                        score = 0
                        if r['zscore'] < -1.2: score += 2
                        if r['rsi'] < 40: score += 2
                        if r['mom'] > 0.05: score += 1
                        if r['vol_z'] > 1: score += 1
                        if score > best_score:
                            best_score, best = score, (sym, r)
                    except: pass
                
                if best and best_score >= 4:
                    sym, r = best
                    price = r['c']
                    atr = r['atr']
                    cost = self.cash * (1 + TAKER_FEE)
                    self.pos = {
                        'sym': sym, 'entry': price, 'sz': self.cash / price,
                        'cost': cost, 't': time.time(),
                        'tp': price + atr * 0.8, 'sl': price - atr * 1.0
                    }
                    self.cash = 0
                    await self.tg(f"⚡ <b>BUY {sym}</b> @ {price:.4f} | Z:{r['zscore']:.1f} RSI:{r['rsi']:.0f}")

            # DASHBOARD
            val = self.cash if not self.pos else self.pos['sz'] * (await self.ex.fetch_ticker(self.pos['sym']))['last']
            st = f"🎯 {self.pos['sym']}" if self.pos else "🔍 Scanning"
            await self.tg(f"💎 <b>Z-SCALPER</b>\n💰 ${val:.2f} | PnL: ${self.pnl:+.2f}\n{st}", edit=True)
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(ZScalper().run())
