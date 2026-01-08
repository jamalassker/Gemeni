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
ATR_MULTIPLIER_TP = 0.6    # Tight TP for fast exit
ATR_MULTIPLIER_SL = 0.9    # Tight SL
MAX_TRADE_SECONDS = 180    # 3 Minutes max hold
MIN_ATR_PCT = 0.004        # Only trade if volatility > 0.4%

# CREDENTIALS
TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "DOGE/USDT",
    "PEPE/USDT", "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "INJ/USDT",
    "NEAR/USDT", "SUI/USDT", "APT/USDT", "FET/USDT", "RENDER/USDT"
]

class ProSniperV7:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 100  # Updated to reflect your current balance
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.report_id = None
        self.session = None

    async def init_session(self):
        if not self.session: self.session = aiohttp.ClientSession()

    async def send_tg(self, text, edit=False):
        await self.init_session()
        url = f"https://api.telegram.org/bot{TG_TOKEN}/"
        method = "editMessageText" if edit and self.report_id else "sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
        if edit: payload["message_id"] = self.report_id
        try:
            async with self.session.post(url + method, json=payload, timeout=8) as resp:
                data = await resp.json()
                if not edit and data.get("ok"): self.report_id = data["result"]["message_id"]
        except: pass

    async def get_data(self, sym, limit=150):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        # Technicals
        df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().where(df['c'].diff() > 0, 0).rolling(14).mean() / df['c'].diff().where(df['c'].diff() < 0, 0).abs().rolling(14).mean().replace(0,0.001))))
        df['vol_change'] = df['v'].pct_change()
        return df.dropna()

    async def training_cycle(self):
        while True:
            await self.send_tg("⚙️ <b>Scalp Brain Training...</b>")
            new_models = {}
            for sym in SYMBOLS:
                try:
                    df = await self.get_data(sym)
                    # Target: 0.3% move within 3 bars
                    df['target'] = (df['c'].shift(-3) > df['c'] * 1.003).astype(int)
                    X = df[['rsi', 'atr', 'vol_change']].fillna(0)
                    model = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50))])
                    model.fit(X, df['target'])
                    new_models[sym] = model
                except: continue
            self.models.update(new_models)
            await self.send_tg("✅ <b>Brains Ready.</b>")
            await asyncio.sleep(3600)

    async def risk_manager(self):
        while True:
            for sym, p in list(self.positions.items()):
                try:
                    ticker = await self.exchange.fetch_ticker(sym)
                    curr_price = ticker['last']
                    
                    # 1. TIME EXIT
                    if time.time() - p['time'] > MAX_TRADE_SECONDS:
                        reason = "⏱ TIME LIMIT"
                    # 2. TP / SL EXIT
                    elif curr_price >= p['tp']: reason = "✅ TP HIT"
                    elif curr_price <= p['sl']: reason = "❌ SL HIT"
                    else: continue

                    # Execute Closure
                    val = p['size'] * curr_price
                    fee = val * TAKER_FEE
                    self.cash += (val - fee)
                    self.realized_pnl += (val - (p['size'] * p['entry'])) - fee
                    self.total_fees += fee
                    del self.positions[sym]
                    await self.send_tg(f"🏁 <b>EXIT {sym}:</b> {reason}")
                except Exception as e: print(f"Risk Error: {e}")
            
            # Dashboard Update
            net = self.realized_pnl
            pos_text = "<i>Waiting for signals...</i>" if not self.positions else f"▫️ {list(self.positions.keys())[0]}"
            report = f"💎 <b>PRO DASHBOARD</b>\nCash: {self.cash:.2f}\nNet: {net:+.2f}\n\n{pos_text}"
            await self.send_tg(report, edit=True)
            await asyncio.sleep(2)

    async def trade_loop(self):
        asyncio.create_task(self.training_cycle())
        asyncio.create_task(self.risk_manager())
        while True:
            if not self.models or self.positions:
                await asyncio.sleep(5); continue
            
            candidates = []
            for sym in SYMBOLS:
                try:
                    df = await self.get_data(sym, limit=30)
                    curr = df.iloc[-1]
                    atr_pct = curr['atr'] / curr['c']
                    
                    if atr_pct < MIN_ATR_PCT: continue # Skip if too boring

                    prob = self.models[sym].predict_proba(df[['rsi', 'atr', 'vol_change']].iloc[-1:])[0][1]
                    candidates.append((prob, atr_pct, sym, curr))
                except: continue

            if candidates:
                # Pick the BEST candidate (Highest Probability)
                prob, atr_pct, sym, curr = max(candidates, key=lambda x: x[0])
                
                if prob >= MIN_PROB:
                    entry = curr['c']
                    trade_val = self.cash * 0.95 # All-in for small $10 balance
                    size = trade_val / entry
                    
                    self.cash -= trade_val # Reserved
                    self.positions[sym] = {
                        'entry': entry, 'size': size, 'time': time.time(),
                        'tp': entry + (curr['atr'] * ATR_MULTIPLIER_TP),
                        'sl': entry - (curr['atr'] * ATR_MULTIPLIER_SL)
                    }
                    await self.send_tg(f"🚀 <b>BUY:</b> {sym} (Prob: {prob:.2f} | Vol: {atr_pct:.2%})")

            await asyncio.sleep(1)

if __name__ == "__main__":
    print("Initializing Engine...")
    asyncio.run(ProSniperV7().trade_loop())






