import os
import time
import asyncio
import logging
import pandas as pd
import numpy as np
import aiohttp
import ccxt.async_support as ccxt
import warnings
from datetime import datetime
from typing import Dict, Optional

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'TRX/USDT',
    'LINK/USDT', 'MATIC/USDT', 'NEAR/USDT', 'LTC/USDT', 'BCH/USDT',
    'SHIB/USDT', 'UNI/USDT', 'STX/USDT', 'FIL/USDT', 'ARB/USDT'
]

class ZScalper:
    def __init__(self):
        self.ex = ccxt.binance({'enableRateLimit': True})
        self.cash = 100.0
        self.pnl = 0.0
        self.pos = None
        self.last_update = 0
        self.sess = None
        self.tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.tg_chat = os.getenv('TELEGRAM_CHAT_ID')
        self.last_msg_id = None

    async def tg(self, msg: str, edit: bool = False):
        if not self.tg_token: return
        if not self.sess: self.sess = aiohttp.ClientSession()
        
        url = f"https://api.telegram.org/bot{self.tg_token}/"
        try:
            if edit and self.last_msg_id:
                method = "editMessageText"
                payload = {"chat_id": self.tg_chat, "message_id": self.last_msg_id, "text": msg, "parse_mode": "HTML"}
            else:
                method = "sendMessage"
                payload = {"chat_id": self.tg_chat, "text": msg, "parse_mode": "HTML"}
            
            async with self.sess.post(url + method, json=payload) as resp:
                data = await resp.json()
                if not edit and data.get('ok'):
                    self.last_msg_id = data['result']['message_id']
        except Exception as e:
            print(f"Telegram error: {e}")

    async def fetch(self, sym: str):
        ohlcv = await self.ex.fetch_ohlcv(sym, '1m', limit=50)
        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        # Technicals
        df['rsi'] = self.rsi(df['c'])
        df['ma'] = df['c'].rolling(20).mean()
        df['std'] = df['c'].rolling(20).std()
        df['zscore'] = (df['c'] - df['ma']) / df['std']
        df['mom'] = df['c'].pct_change(3)
        return df

    def rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        return 100 - (100 / (1 + (gain / loss)))

    async def check_entry(self):
        for sym in SYMBOLS:
            try:
                df = await self.fetch(sym)
                r = df.iloc[-1]
                price = r['c']
                
                # Signal: Oversold + Momentum turning
                score = 0
                if r['zscore'] < -1.5: score += 4
                if r['rsi'] < 35: score += 3
                if r['mom'] > 0: score += 3

                if score >= 7:
                    sz = (self.cash * 0.1) / price
                    self.pos = {
                        'sym': sym, 'entry': price, 'sz': sz, 
                        't': time.time(), 'tp': price * 1.008, 
                        'sl': price * 0.995
                    }
                    self.cash -= (self.pos['sz'] * price)
                    await self.tg(f"⚡ <b>BUY {sym}</b>\nPrice: ${price:.4f}\nScore: {score}/10\nZ: {r['zscore']:.2f}")
                    return True
            except: continue
        return False

    async def check_exit(self):
        if not self.pos: return False
        try:
            tick = await self.ex.fetch_ticker(self.pos['sym'])
            curr = tick['last']
            pnl_pct = (curr / self.pos['entry'] - 1)
            
            exit_signal = False
            reason = ""

            if curr >= self.pos['tp']:
                exit_signal, reason = True, "✅ TAKE PROFIT"
            elif curr <= self.pos['sl']:
                exit_signal, reason = True, "❌ STOP LOSS"

            if exit_signal:
                val = self.pos['sz'] * curr
                profit = val - (self.pos['sz'] * self.pos['entry'])
                self.cash += val
                self.pnl += profit
                await self.tg(f"🏁 <b>{reason}</b>\nProfit: ${profit:+.2f}\nSymbol: {self.pos['sym']}")
                self.pos = None
                return True
        except: pass
        return False

    async def update_dashboard(self):
        if time.time() - self.last_update < 5: return
        try:
            header = f"💎 <b>Z-SCALPER v2.0</b>\n💰 Cash: ${self.cash:.2f} | PnL: ${self.pnl:+.2f}\n"
            if self.pos:
                tick = await self.ex.fetch_ticker(self.pos['sym'])
                curr = tick['last']
                perf = (curr / self.pos['entry'] - 1) * 100
                status = f"📊 <b>HOLDING {self.pos['sym']}</b>\nProfit: {perf:+.2f}%"
            else:
                status = "🔍 <b>SCANNING MARKET...</b>"
            
            await self.tg(header + status, edit=True)
            self.last_update = time.time()
        except: pass

    async def run(self):
        await self.tg("🚀 <b>Z-SCORE SCALPER ONLINE</b>")
        while True:
            try:
                if not self.pos: await self.check_entry()
                else: await self.check_exit()
                await self.update_dashboard()
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Loop error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = ZScalper()
    asyncio.run(bot.run())

