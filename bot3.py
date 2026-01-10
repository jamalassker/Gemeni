import os
import time
import logging
import pandas as pd
import numpy as np
import requests
import ccxt
import warnings
from datetime import datetime
from typing import Dict, Optional

warnings.filterwarnings('ignore')

# ============================================================================
# RELAXED CONFIGURATION FOR MAXIMUM ACTIVITY
# ============================================================================

class UltimateConfig:
    SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
        'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'TRX/USDT',
        'LINK/USDT', 'MATIC/USDT', 'NEAR/USDT', 'LTC/USDT', 'BCH/USDT',
        'SHIB/USDT', 'UNI/USDT', 'STX/USDT', 'FIL/USDT', 'ARB/USDT'
    ]
    
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.10      # Increased to 10% for faster growth
    MAX_OPEN_TRADES = 10       # More simultaneous trades
    
    # Fast Scalping Targets
    BASE_TAKE_PROFIT = 0.0070   # 0.7%
    BASE_STOP_LOSS = 0.0050     # 0.5%
    BREAK_EVEN_TRIGGER = 0.0020 # Secure profit very early
    
    TAKER_FEE = 0.0010
    CHECK_INTERVAL = 2          # Ultra-fast scanning (2 seconds)
    
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ============================================================================
# TELEGRAM ALERTS
# ============================================================================

def send_telegram(msg: str):
    if not UltimateConfig.TELEGRAM_TOKEN: return
    try:
        url = f"https://api.telegram.org/bot{UltimateConfig.TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": UltimateConfig.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=5)
    except: pass

# ============================================================================
# ENGINE & LOGIC
# ============================================================================

class PortfolioManager:
    def __init__(self, initial):
        self.balance = initial
        self.open_trades = {}

    def update_and_report(self, prices):
        closed = []
        for tid, t in self.open_trades.items():
            price = prices.get(t['symbol'])
            if not price: continue
            
            # P&L calculation
            pnl = ((price - t['entry']) / t['entry']) if t['side'] == 'BUY' else ((t['entry'] - price) / t['entry'])
            pnl -= (UltimateConfig.TAKER_FEE * 2)

            # Break-Even Protection
            if pnl >= UltimateConfig.BREAK_EVEN_TRIGGER and not t.get('be'):
                t['sl'] = -0.0001
                t['be'] = True
                send_telegram(f"🛡️ <b>SAFE:</b> {t['symbol']} at Break-Even")

            # Exit Logic
            if pnl >= UltimateConfig.BASE_TAKE_PROFIT: closed.append((tid, "✅ TP", pnl))
            elif pnl <= -t['sl']: closed.append((tid, "❌ SL", pnl))

        for tid, reason, final_pnl in closed:
            trade = self.open_trades.pop(tid)
            profit = final_pnl * trade['size']
            self.balance += (trade['size'] + profit)
            send_telegram(f"🏁 <b>CLOSED: {trade['symbol']}</b>\nNet: ${profit:+.2f}\nBalance: ${self.balance:.2f}")

class FastBot:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.portfolio = PortfolioManager(UltimateConfig.INITIAL_CAPITAL)
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    def get_signal(self, df):
        # Relaxed logic: Just check EMA slope and RSI
        ema8 = df['close'].ewm(span=8).mean().iloc[-1]
        ema21 = df['close'].ewm(span=21).mean().iloc[-1]
        rsi = self.rsi(df['close']).iloc[-1]
        
        if ema8 > ema21 and rsi < 70: return 'BUY'
        if ema8 < ema21 and rsi > 30: return 'SELL'
        return None

    def rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        return 100 - (100 / (1 + (gain / loss)))

    def run(self):
        print("🚀 SCALPER STARTING...") # This will show in Railway immediately
        send_telegram("🚀 <b>BOT STARTED</b>\nAggressive Scalping Active.")
        
        while True:
            prices = {}
            for sym in UltimateConfig.SYMBOLS:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(sym, '1m', limit=30)
                    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                    prices[sym] = df['c'].iloc[-1]
                    
                    if len(self.portfolio.open_trades) < UltimateConfig.MAX_OPEN_TRADES:
                        if sym not in [t['symbol'] for t in self.portfolio.open_trades.values()]:
                            sig = self.get_signal(df)
                            if sig:
                                size = self.portfolio.balance * UltimateConfig.RISK_PER_TRADE
                                tid = f"{sym}_{int(time.time())}"
                                self.portfolio.open_trades[tid] = {'symbol': sym, 'entry': prices[sym], 'size': size, 'side': sig, 'sl': UltimateConfig.BASE_STOP_LOSS, 'be': False}
                                logging.info(f"OPENED: {sig} {sym} at {prices[sym]}")
                                send_telegram(f"💰 <b>OPEN: {sig} {sym}</b>\nPrice: {prices[sym]}")
                    time.sleep(0.05)
                except: continue
            
            self.portfolio.update_and_report(prices)
            time.sleep(UltimateConfig.CHECK_INTERVAL)

if __name__ == "__main__":
    FastBot().run()

