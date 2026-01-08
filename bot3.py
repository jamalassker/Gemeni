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
        self.last_update = 0

    async def tg(self, txt, edit=False):
        """Send or update Telegram message"""
        if not self.sess:
            self.sess = aiohttp.ClientSession()
        
        url = f"https://api.telegram.org/bot{TG_TOKEN}/"
        url += "editMessageText" if edit and self.msg_id else "sendMessage"
        
        data = {
            "chat_id": TG_CHAT,
            "text": txt,
            "parse_mode": "HTML",
            "disable_notification": edit  # Only notify on new messages, not updates
        }
        
        if edit and self.msg_id:
            data["message_id"] = self.msg_id
            
        try:
            async with self.sess.post(url, json=data, timeout=5) as r:
                res = await r.json()
                if not edit and res.get("ok"):
                    self.msg_id = res["result"]["message_id"]
        except Exception as e:
            print(f"Telegram error: {e}")

    def calc(self, df):
        """Calculate indicators"""
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
        
        # Entry/Exit specific signals
        df['buy_signal'] = (df['zscore'] < -1.2) & (df['rsi'] < 40) & (df['mom'] > 0.05)
        df['sell_signal'] = (df['zscore'] > 1.2) | (df['rsi'] > 70)
        
        return df.dropna()

    async def fetch(self, sym):
        """Fetch OHLCV data"""
        ohlcv = await self.ex.fetch_ohlcv(sym, TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        return self.calc(df)

    async def check_exit(self):
        """Check if we should exit current position"""
        if not self.pos:
            return False
            
        try:
            tick = await self.ex.fetch_ticker(self.pos['sym'])
            price = tick['last']
            held = time.time() - self.pos['t']
            
            # Quick exit conditions
            profit_pct = (price / self.pos['entry'] - 1) * 100
            fee_adjusted = profit_pct - (TAKER_FEE * 2 * 100)  # Account for entry+exit fees
            
            # Faster exit conditions
            if fee_adjusted >= 0.2:  # Exit at 0.2% profit after fees
                reason = "✅ QUICK PROFIT"
            elif price >= self.pos['tp']:
                reason = "✅ TP HIT"
            elif price <= self.pos['sl']:
                reason = "❌ SL HIT"
            elif held > MAX_HOLD:
                reason = "⏱ TIME OUT"
            elif held > 30 and profit_pct > 0.05:  # Faster lock-in (30s instead of 45s)
                reason = "🔒 LOCK PROFIT"
            else:
                return False
                
            # Execute exit
            val = self.pos['sz'] * price * (1 - TAKER_FEE)
            gain = val - self.pos['cost']
            self.pnl += gain
            self.cash = val
            profit_loss = "PROFIT" if gain > 0 else "LOSS"
            
            await self.tg(f"🏁 EXIT: {self.pos['sym']}\n"
                         f"Reason: {reason}\n"
                         f"Entry: ${self.pos['entry']:.4f} | Exit: ${price:.4f}\n"
                         f"P/L: ${gain:+.3f} ({profit_pct:+.2f}%)\n"
                         f"Net Total: ${self.pnl:+.2f}")
            
            self.pos = None
            return True
            
        except Exception as e:
            print(f"Exit check error: {e}")
            return False

    async def check_entry(self):
        """Check for new entry opportunities"""
        if self.pos:
            return False
            
        best_score = 0
        best_entry = None
        
        for sym in SYMBOLS:
            try:
                df = await self.fetch(sym)
                r = df.iloc[-1]
                
                # Enhanced scoring system
                score = 0
                
                # Strong reversal signals
                if r['zscore'] < -1.5:
                    score += 3  # Strongly oversold
                elif r['zscore'] < -1.2:
                    score += 2
                    
                if r['rsi'] < 35:
                    score += 3  # Deep oversold
                elif r['rsi'] < 40:
                    score += 2
                    
                if r['mom'] > 0.1:  # Strong momentum
                    score += 2
                elif r['mom'] > 0.05:
                    score += 1
                    
                if r['vol_z'] > 1.5:  # High volume spike
                    score += 2
                elif r['vol_z'] > 1:
                    score += 1
                    
                # Check for actual buy signal from dataframe
                if r['buy_signal']:
                    score += 2
                    
                if score > best_score and score >= 5:  # Lower threshold for faster entries
                    best_score = score
                    best_entry = (sym, r, score)
                    
            except Exception as e:
                print(f"Entry check error for {sym}: {e}")
        
        if best_entry:
            sym, r, score = best_entry
            price = r['c']
            atr = r['atr']
            
            # More aggressive position sizing for faster trades
            position_size = min(0.8, 0.5 + (score - 5) * 0.1)  # 50-80% of capital
            cost = self.cash * position_size * (1 + TAKER_FEE)
            sz = (self.cash * position_size) / price
            
            self.pos = {
                'sym': sym,
                'entry': price,
                'sz': sz,
                'cost': cost,
                't': time.time(),
                'tp': price + atr * 0.5,  # Tighter TP for faster exits
                'sl': price - atr * 0.8,  # Tighter SL for faster exits
                'score': score
            }
            self.cash -= (self.cash * position_size)
            
            await self.tg(f"⚡ <b>BUY {sym}</b>\n"
                         f"Price: ${price:.4f}\n"
                         f"Score: {score}/10\n"
                         f"Z: {r['zscore']:.2f} | RSI: {r['rsi']:.0f}\n"
                         f"TP: ${self.pos['tp']:.4f} | SL: ${self.pos['sl']:.4f}")
            return True
            
        return False

    async def update_dashboard(self):
        """Update status dashboard"""
        if time.time() - self.last_update < 5:  # Update every 5 seconds
            return
            
        try:
            if self.pos:
                tick = await self.ex.fetch_ticker(self.pos['sym'])
                current_price = tick['last']
                profit = (current_price / self.pos['entry'] - 1) * 100
                held = int(time.time() - self.pos['t'])
                
                status = (f"📊 <b>HOLDING {self.pos['sym']}</b>\n"
                         f"Entry: ${self.pos['entry']:.4f} | Now: ${current_price:.4f}\n"
                         f"P/L: {profit:+.2f}% | Held: {held}s\n"
                         f"TP: ${self.pos['tp']:.4f} | SL: ${self.pos['sl']:.4f}")
            else:
                # Show scanning status with top 3 candidates
                status = "🔍 <b>SCANNING FOR ENTRIES</b>\n"
                
                # Quick scan of all symbols
                scans = []
                for sym in SYMBOLS[:3]:  # Limit to first 3 for speed
                    try:
                        df = await self.fetch(sym)
                        r = df.iloc[-1]
                        score = 0
                        if r['zscore'] < -1.2: score += 2
                        if r['rsi'] < 40: score += 2
                        if r['mom'] > 0.05: score += 1
                        scans.append(f"{sym.split('/')[0]}: Z{r['zscore']:.1f} RSI{r['rsi']:.0f}")
                    except:
                        pass
                
                status += " | ".join(scans) if scans else "No signals"
            
            total_value = self.cash
            if self.pos:
                tick = await self.ex.fetch_ticker(self.pos['sym'])
                total_value += self.pos['sz'] * tick['last']
                
            header = f"💎 <b>Z-SCALPER v2.0</b>\n💰 ${total_value:.2f} | PnL: ${self.pnl:+.2f}\n"
            
            await self.tg(header + status, edit=True)
            self.last_update = time.time()
            
        except Exception as e:
            print(f"Dashboard error: {e}")

    async def run(self):
        """Main trading loop"""
        await self.tg("🚀 <b>Z-SCORE SCALPER v2.0 ONLINE</b>\n"
                     "Faster trades | Improved exits | Fixed alerts")
        
        print("Bot started. Press Ctrl+C to stop.")
        
        try:
            while True:
                # Check exit first (priority)
                if await self.check_exit():
                    await asyncio.sleep(1)  # Brief pause after exit
                    
                # Check for new entry
                if not self.pos:
                    await self.check_entry()
                
                # Update dashboard
                await self.update_dashboard()
                
                # Faster loop for quicker reactions
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            await self.tg(f"❌ <b>BOT CRASHED</b>\nError: {str(e)}")
            print(f"Critical error: {e}")
        finally:
            if self.sess:
                await self.sess.close()
            await self.ex.close()

if __name__ == "__main__":
    bot = ZScalper()
    asyncio.run(bot.run())
