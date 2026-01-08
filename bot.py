import pandas as pd
import numpy as np
import asyncio, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt

# ================= CONFIG =================
TIMEFRAME = "1m"             # Dropped to 1m for faster entry/exit
MAX_POSITIONS = 1
MIN_PROB = 0.52              # Slightly lower to increase trade frequency
TAKER_FEE = 0.001

# Exit Parameters
ATR_MULTIPLIER_TP = 0.7      # Target 0.7x the average move
ATR_MULTIPLIER_SL = 1.0      # Stop at 1.0x the average move
MAX_TRADE_SECONDS = 300      # 5 minutes max hold time
MIN_ATR_PCT = 0.005          # Only trade if vol is > 0.5% (High Volatility)

# CREDENTIALS
TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

# High Volatility Symbols Only
SYMBOLS = [
    "PEPE/USDT", "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "SOL/USDT",
    "SUI/USDT", "FET/USDT", "RENDER/USDT", "APT/USDT", "AVAX/USDT",
    "INJ/USDT", "NEAR/USDT", "OP/USDT", "ARB/USDT", "TIA/USDT"
]

class ProSniperV7:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 10.0  # Starting balance
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
            async with self.session.post(url + method, json=payload, timeout=8) as r:
                data = await r.json()
                if not edit and data.get("ok"): self.report_id = data["result"]["message_id"]
        except: pass

    async def get_data(self, sym, limit=100):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        # ATR Calculation
        df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift()), abs(df['l'] - df['c'].shift())))
        df['atr'] = df['tr'].rolling(14).mean()
        # RSI Calculation
        diff = df['c'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = diff.abs().rolling(14).mean().replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        # Volatility & Volume
        df['vol_change'] = df['v'].pct_change()
        df['atr_pct'] = df['atr'] / df['c']
        return df.dropna()

    async def training_cycle(self):
        while True:
            await self.send_tg("⚙️ <b>Brain Training (Scalp Mode)...</b>")
            new_models = {}
            for sym in SYMBOLS:
                try:
                    df = await self.get_data(sym)
                    # Target: Price is 0.4% higher within 3 minutes
                    df['target'] = (df['c'].shift(-3) > df['c'] * 1.004).astype(int)
                    X = df[['rsi','atr','vol_change']].fillna(0)
                    model = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50))])
                    model.fit(X, df['target'])
                    new_models[sym] = model
                except: continue
            self.models = new_models
            await self.send_tg("✅ <b>Brains Ready. Watching for Spikes.</b>")
            await asyncio.sleep(1800) # Retrain every 30 mins for freshness

    async def risk_manager(self):
        while True:
            for sym, p in list(self.positions.items()):
                try:
                    ticker = await self.exchange.fetch_ticker(sym)
                    price = ticker['last']
                    exit_reason = None

                    # 1. TIME EXIT (The Scalper's best friend)
                    if time.time() - p['time'] > MAX_TRADE_SECONDS:
                        exit_reason = "⏱ TIME"
                    # 2. PROFIT / LOSS
                    elif price >= p['tp']: exit_reason = "✅ TP"
                    elif price <= p['sl']: exit_reason = "❌ SL"

                    if exit_reason:
                        val = p['size'] * price
                        fee = val * TAKER_FEE
                        self.cash += (val - fee)
                        self.realized_pnl += (val - (p['size'] * p['entry'])) - fee
                        self.total_fees += fee
                        del self.positions[sym]
                        await self.send_tg(f"🏁 <b>EXIT {sym}:</b> {exit_reason} | PnL: {(val - (p['size'] * p['entry'])):.2f}")
                except: continue

            # Update Dashboard
            report = (f"💎 <b>PRO DASHBOARD</b>\nCash: {self.cash:.2f}\nNet: {self.realized_pnl:+.2f}\n\n"
                      f"{'<i>Waiting for Volatility...</i>' if not self.positions else '🔥 TRADING: ' + list(self.positions.keys())[0]}")
            await self.send_tg(report, edit=True)
            await asyncio.sleep(1)

    async def trade_loop(self):
        asyncio.create_task(self.training_cycle())
        asyncio.create_task(self.risk_manager())
        while True:
            if not self.models or self.positions:
                await asyncio.sleep(2); continue

            candidates = []
            for sym in SYMBOLS:
                try:
                    df = await self.get_data(sym, 30)
                    curr = df.iloc[-1]
                    # Filter for volatility first
                    if curr['atr_pct'] < MIN_ATR_PCT: continue
                    
                    # Probability check
                    prob = self.models[sym].predict_proba(df[['rsi','atr','vol_change']].iloc[-1:])[0][1]
                    candidates.append((prob, sym, curr))
                except: continue

            if candidates:
                # Pick the coin with the highest probability + volatility combo
                prob, sym, curr = max(candidates, key=lambda x: x[0])
                if prob >= MIN_PROB:
                    entry = curr['c']
                    trade_val = self.cash * 0.90 # Use 90% of cash ($9.00 approx)
                    fee = trade_val * TAKER_FEE
                    
                    self.cash -= (trade_val + fee)
                    self.total_fees += fee
                    self.positions[sym] = {
                        'entry': entry, 'size': trade_val / entry, 'time': time.time(),
                        'tp': entry + (curr['atr'] * ATR_MULTIPLIER_TP),
                        'sl': entry - (curr['atr'] * ATR_MULTIPLIER_SL)
                    }
                    await self.send_tg(f"🚀 <b>BUY {sym}</b>\nProb: {prob:.2f} | Vol: {curr['atr_pct']:.2%}")

            await asyncio.sleep(0.5)

if __name__ == "__main__":
    print("Scalp Engine Online...")
    asyncio.run(ProSniperV7().trade_loop())







