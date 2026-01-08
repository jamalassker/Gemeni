import pandas as pd
import numpy as np
import asyncio, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt

# ================= PRO-SCALPER CONFIG =================
TIMEFRAME = "1m"
MAX_POSITIONS = 3            # Use 3 slots of $33 for better frequency
MIN_PROB = 0.51              # Aggressive entry for high frequency
TAKER_FEE = 0.001

# Fast Exit Logic
ATR_MULTIPLIER_TP = 0.8      # Close at 0.8x ATR move
ATR_MULTIPLIER_SL = 1.0      # Stop at 1.0x ATR
MAX_TRADE_SECONDS = 240      # 4 minutes max hold time
MIN_ATR_PCT = 0.002          # Lowered to 0.2% for 1m timeframe

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

SYMBOLS = [
    "PEPE/USDT", "WIF/USDT", "SOL/USDT", "SUI/USDT", "FET/USDT",
    "RENDER/USDT", "BONK/USDT", "FLOKI/USDT", "AR/USDT", "TIA/USDT",
    "TAO/USDT", "SEI/USDT", "AVAX/USDT", "NEAR/USDT", "OP/USDT"
]

class SpeedSniperV8:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 100.0
        self.realized_pnl = 0.0
        self.report_id = None
        self.session = None

    async def init_session(self):
        if not self.session: self.session = aiohttp.ClientSession()

    async def send_tg(self, text, edit=False):
        await self.init_session()
        url = f"https://api.telegram.org/bot{TG_TOKEN}/" + ("editMessageText" if edit and self.report_id else "sendMessage")
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
        if edit: payload["message_id"] = self.report_id
        try:
            async with self.session.post(url, json=payload, timeout=5) as r:
                data = await r.json()
                if not edit and data.get("ok"): self.report_id = data["result"]["message_id"]
        except: pass

    async def get_data(self, sym, limit=100):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        
        # Indicators: ATR, RSI, MACD, VWAP
        df['atr'] = df['h'].rolling(14).max() - df['l'].rolling(14).min()
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().clip(lower=0).rolling(14).mean() / df['c'].diff().abs().rolling(14).mean().replace(0, 0.001))))
        
        # Simple MACD
        df['ema12'] = df['c'].ewm(span=12).mean()
        df['ema26'] = df['c'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        
        # Simple VWAP
        df['vwap'] = (df['v'] * (df['h'] + df['l'] + df['c']) / 3).cumsum() / df['v'].cumsum()
        
        df['vol_change'] = df['v'].pct_change()
        df['atr_pct'] = df['atr'] / df['c']
        return df.dropna()

    async def training_cycle(self):
        while True:
            new_models = {}
            for sym in SYMBOLS:
                try:
                    df = await self.get_data(sym)
                    df['target'] = (df['c'].shift(-3) > df['c'] * 1.002).astype(int)
                    X = df[['rsi', 'macd', 'vol_change', 'atr_pct']].fillna(0)
                    model = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=30))])
                    model.fit(X, df['target'])
                    new_models[sym] = model
                except: continue
            self.models = new_models
            await self.send_tg("⚡ <b>Brains Updated. Watching 15 charts.</b>")
            await asyncio.sleep(1200)

    async def risk_manager(self):
        while True:
            for sym, p in list(self.positions.items()):
                try:
                    ticker = await self.exchange.fetch_ticker(sym)
                    curr = ticker['last']
                    
                    # FETCH LATEST RSI FOR EMERGENCY EXIT
                    df = await self.get_data(sym, 20)
                    last_rsi = df['rsi'].iloc[-1]

                    exit_reason = None
                    if time.time() - p['time'] > MAX_TRADE_SECONDS: exit_reason = "⏱ TIME"
                    elif curr >= p['tp']: exit_reason = "✅ TP"
                    elif curr <= p['sl']: exit_reason = "❌ SL"
                    elif last_rsi > 85: exit_reason = "🔥 RSI OVERHEAT" # Exit on overbought spike

                    if exit_reason:
                        val = p['size'] * curr
                        fee = val * TAKER_FEE
                        self.cash += (val - fee)
                        self.realized_pnl += (val - (p['size'] * p['entry'])) - fee
                        del self.positions[sym]
                        await self.send_tg(f"🏁 <b>EXIT {sym}:</b> {exit_reason} | PnL: {self.realized_pnl:+.2f}")
                except: continue
            
            # Dashboard
            pos_list = ", ".join(self.positions.keys()) if self.positions else "None"
            await self.send_tg(f"💎 <b>LIVE SNIPER</b>\nCash: {self.cash:.2f}\nNet: {self.realized_pnl:+.2f}\nPositions: {pos_list}", edit=True)
            await asyncio.sleep(1)

    async def trade_loop(self):
        asyncio.create_task(self.training_cycle())
        asyncio.create_task(self.risk_manager())
        while True:
            if not self.models or len(self.positions) >= MAX_POSITIONS:
                await asyncio.sleep(1); continue

            for sym in SYMBOLS:
                if sym in self.positions: continue
                try:
                    df = await self.get_data(sym, 30)
                    curr = df.iloc[-1]
                    
                    # VWAP Filter: Only BUY if price is above VWAP (Bullish bias)
                    if curr['c'] < curr['vwap'] or curr['atr_pct'] < MIN_ATR_PCT: continue

                    prob = self.models[sym].predict_proba(df[['rsi', 'macd', 'vol_change', 'atr_pct']].iloc[-1:])[0][1]
                    
                    if prob >= MIN_PROB:
                        entry = curr['c']
                        trade_val = (self.cash / (MAX_POSITIONS - len(self.positions))) * 0.95
                        fee = trade_val * TAKER_FEE
                        
                        self.cash -= (trade_val + fee)
                        self.positions[sym] = {
                            'entry': entry, 'size': trade_val / entry, 'time': time.time(),
                            'tp': entry + (curr['atr'] * ATR_MULTIPLIER_TP),
                            'sl': entry - (curr['atr'] * ATR_MULTIPLIER_SL)
                        }
                        await self.send_tg(f"🚀 <b>SNIPE {sym}</b> | Prob: {prob:.2f}")
                except: continue
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(SpeedSniperV8().trade_loop())





