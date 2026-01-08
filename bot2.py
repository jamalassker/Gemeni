import pandas as pd
import numpy as np
import asyncio, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt 

# ================= ADVANCED CONFIG =================
TIMEFRAME = "1m"             # 1-minute for high-speed accuracy
MAX_POSITIONS = 3            # Allows diversification for your $100
MIN_PROB = 0.58              # Higher precision threshold
TAKER_FEE = 0.001            # Binance standard
ATR_MULTIPLIER_TP = 0.8      # Target profit
ATR_MULTIPLIER_SL = 1.0      # initial stop
MAX_HOLD_SECONDS = 300       # 5-minute hard exit (Scalp rule)

# Credentials
TG_TOKEN = "8488789199:AAHhViKmhXlvE7WpgZGVDS4WjCjUuBVtqzQ"
TG_CHAT = "5665906172"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT", "WIF/USDT", 
    "BONK/USDT", "SUI/USDT", "FET/USDT", "RENDER/USDT", "AVAX/USDT"
]

class PersistentProSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 100.0
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.report_id = None
        self.session = None
        self.market_safe = True # Global trend flag

    async def init_session(self):
        if not self.session: self.session = aiohttp.ClientSession()

    async def send_tg(self, text, edit=False):
        await self.init_session()
        url = f"https://api.telegram.org/bot{TG_TOKEN}/" + ("editMessageText" if edit and self.report_id else "sendMessage")
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
        if edit: payload["message_id"] = self.report_id
        try:
            async with self.session.post(url, json=payload, timeout=8) as resp:
                data = await resp.json()
                if not edit and data.get("ok"): self.report_id = data["result"]["message_id"]
        except: pass

    async def get_processed_data(self, sym, limit=100):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        # Indicators
        df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().clip(lower=0).rolling(14).mean() / df['c'].diff().abs().rolling(14).mean().replace(0,0.001))))
        df['vol_change'] = df['v'].pct_change()
        return df.dropna()

    async def check_global_trend(self):
        """Trick #1: BTC Trend Filter"""
        while True:
            try:
                df = await self.get_processed_data("BTC/USDT", limit=10)
                # If BTC drops > 0.3% in last 10 mins, mark market as unsafe
                change = (df['c'].iloc[-1] - df['c'].iloc[0]) / df['c'].iloc[0]
                self.market_safe = change > -0.003
            except: pass
            await asyncio.sleep(30)

    async def train_background(self):
        while True:
            await self.send_tg("🧠 <b>Brain Training: Advanced Scalp Mode...</b>")
            new_models = {}
            for sym in SYMBOLS:
                try:
                    df = await self.get_processed_data(sym)
                    # Label: 0.4% move within 3 minutes
                    df['target'] = (df['c'].shift(-3) > df['c'] * 1.004).astype(int)
                    X = df[['rsi', 'atr', 'vol_change']].fillna(0)
                    new_models[sym] = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50))]).fit(X, df['target'])
                except: continue
            self.models.update(new_models)
            await self.send_tg("✅ <b>Brains Optimized.</b> Ready for volatility.")
            await asyncio.sleep(3600)

    async def risk_manager(self):
        """Trick #2: Dynamic Trailing Stop Loss"""
        while True:
            for sym, p in list(self.positions.items()):
                try:
                    ticker = await self.exchange.fetch_ticker(sym)
                    curr = ticker['last']
                    
                    # Trailing Logic: If price hits 50% of TP target, move SL to Break Even
                    if curr >= (p['entry'] + (p['tp'] - p['entry']) * 0.5) and not p.get('breakeven'):
                        p['sl'] = p['entry'] + (p['entry'] * 0.001) # Entry + small fee buffer
                        p['breakeven'] = True
                        await self.send_tg(f"🛡️ <b>BE ACTIVE:</b> {sym} protected.")

                    # Exit Conditions
                    reason = None
                    if time.time() - p['time'] > MAX_HOLD_SECONDS: reason = "⏱ TIME"
                    elif curr >= p['tp']: reason = "✅ TP"
                    elif curr <= p['sl']: reason = "❌ SL"

                    if reason:
                        val = p['size'] * curr
                        fee = val * TAKER_FEE
                        self.cash += (val - fee)
                        self.realized_pnl += (val - (p['size'] * p['entry'])) - fee
                        self.total_fees += fee
                        del self.positions[sym]
                        await self.send_tg(f"🏁 <b>EXIT {sym}:</b> {reason} | PnL: {(val - (p['size'] * p['entry'])):.2f}")
                except: continue

            net = self.realized_pnl
            report = f"💎 <b>PRO DASHBOARD ($100)</b>\nCash: {self.cash:.2f}\nNet: {net:+.2f}\nSafe: {'✅' if self.market_safe else '⚠️ WAIT'}"
            await self.send_tg(report, edit=True)
            await asyncio.sleep(1)

    async def trade_loop(self):
        asyncio.create_task(self.check_global_trend())
        asyncio.create_task(self.train_background())
        asyncio.create_task(self.risk_manager())
        
        while True:
            if not self.models or len(self.positions) >= MAX_POSITIONS or not self.market_safe:
                await asyncio.sleep(2); continue
                
            for sym in SYMBOLS:
                if sym in self.positions: continue
                try:
                    df = await self.get_processed_data(sym, limit=20)
                    curr = df.iloc[-1]
                    # Trick #3: Volatility Filter (ATR must be > 0.4% of price)
                    if (curr['atr'] / curr['c']) < 0.004: continue

                    prob = self.models[sym].predict_proba(df[['rsi', 'atr', 'vol_change']].iloc[-1:])[0][1]
                    
                    if prob >= MIN_PROB:
                        entry = curr['c']
                        # Trick #4: Adaptive TP/SL based on ATR
                        tp = entry + (curr['atr'] * ATR_MULTIPLIER_TP)
                        sl = entry - (curr['atr'] * ATR_MULTIPLIER_SL)
                        
                        trade_val = self.cash * 0.30 # Stake 30% per position
                        size = trade_val / entry
                        self.cash -= (trade_val * (1 + TAKER_FEE))
                        self.total_fees += (trade_val * TAKER_FEE)
                        
                        self.positions[sym] = {'entry': entry, 'size': size, 'tp': tp, 'sl': sl, 'time': time.time(), 'breakeven': False}
                        await self.send_tg(f"🚀 <b>SCALP BUY:</b> {sym} (Prob: {prob:.2f})")
                except: continue
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(PersistentProSniper().trade_loop())




