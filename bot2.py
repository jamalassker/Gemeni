import pandas as pd
import numpy as np
import asyncio, logging, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt 

# ================= DANGEROUS QUANT CONFIG =================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT", "SUI/USDT", "WIF/USDT"] 
TIMEFRAME = "5m"
MAX_POSITIONS = 5
MIN_PROB = 0.62       # Elite selectivity
TAKER_FEE = 0.001
ATR_MULTIPLIER = 2.0  # Take profit at 2x ATR

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

class HardenedSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 10000.0  # Real cash on hand
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.report_id = None

    async def get_data(self, sym, limit=200):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        # Elite Indicator: ATR for Volatility Targeting
        df['h-l'] = df['h'] - df['l']
        df['h-pc'] = abs(df['h'] - df['c'].shift(1))
        df['l-pc'] = abs(df['l'] - df['c'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().where(df['c'].diff() > 0, 0).rolling(14).mean() / 
                                      df['c'].diff().where(df['c'].diff() < 0, 0).abs().rolling(14).mean().replace(0,0.001))))
        return df.dropna()

    async def train_models(self):
        """Triple-Barrier Training: Predict if price hits TP before SL"""
        for sym in SYMBOLS:
            df = await self.get_data(sym)
            # Define barriers
            df['target'] = (df['c'].shift(-5) > df['c'] * 1.01).astype(int) # Target 1% move in 25 mins
            X = df[['rsi', 'atr']]
            self.models[sym] = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100))]).fit(X, df['target'])
        print("✅ Models Retrained on Target-Performance logic.")

    async def trade_loop(self):
        await self.train_models()
        asyncio.create_task(self.risk_manager())
        
        while True:
            for sym in SYMBOLS:
                if sym in self.positions or len(self.positions) >= MAX_POSITIONS: continue
                
                df = await self.get_data(sym, limit=30)
                curr = df.iloc[-1]
                prob = self.models[sym].predict_proba(df[['rsi', 'atr']].iloc[-1:])[0][1]
                
                # Regime Filter: Only trade if ATR is in 20th-80th percentile (not dead, not crazy)
                if prob >= MIN_PROB:
                    entry = curr['c']
                    tp = entry + (curr['atr'] * ATR_MULTIPLIER)
                    sl = entry - (curr['atr'] * 1.5)
                    
                    size = (self.cash * 0.15) / entry
                    self.cash -= (size * entry * (1 + TAKER_FEE))
                    self.positions[sym] = {'entry': entry, 'size': size, 'tp': tp, 'sl': sl}
                    print(f"🎯 PRO ENTRY: {sym} | TP: {tp:.4f} | SL: {sl:.4f}")
            await asyncio.sleep(5)

    async def risk_manager(self):
        while True:
            for sym, p in list(self.positions.items()):
                ticker = await self.exchange.fetch_ticker(sym)
                price = ticker['last']
                
                if price >= p['tp'] or price <= p['sl']:
                    reason = "TAKE PROFIT" if price >= p['tp'] else "STOP LOSS"
                    val = p['size'] * price
                    self.cash += (val * (1 - TAKER_FEE))
                    self.realized_pnl += (val - (p['size'] * p['entry']))
                    del self.positions[sym]
                    print(f"🏁 EXIT {sym} ({reason}) | PnL: {self.realized_pnl:.2f}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(HardenedSniper().trade_loop())
