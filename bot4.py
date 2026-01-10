import os, time, requests, ccxt, logging, pandas as pd, numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TECHNICAL CONFIGURATION - OPTIMIZED FOR 1M SCALPING
# ============================================================================
class ScalpConfig:
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'AVAX/USDT']
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.15      # 15% position size to move the $100 faster
    MAX_TRADES = 3             # Focus on the 3 best setups
    
    # Precision Targets (Tight for $100 growth)
    TP_PCT = 0.0090            # 0.90% Target
    SL_PCT = 0.0045            # 0.45% Hard Stop
    BE_PCT = 0.0035            # Move to Break-Even at +0.35%
    
    # Cost Protection
    MAX_SPREAD_PCT = 0.0005    # Don't enter if spread is > 0.05%
    ESTIMATED_FEE = 0.001      # Binance Taker Fee (0.1%)
    
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ============================================================================
# TECHNICAL INDICATORS (PURE MATH, NO AI)
# ============================================================================
class TechIndicators:
    @staticmethod
    def get_indicators(df):
        # 1. EMA (Trend)
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        # 2. RSI (Momentum Filter)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        
        # 3. Stochastic (Entry Timing)
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['d'] = df['k'].rolling(3).mean()
        
        return df.dropna()

# ============================================================================
# TRADING ENGINE
# ============================================================================
class AccurateScalper:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.balance = ScalpConfig.INITIAL_CAPITAL
        self.trades = {}
        logging.basicConfig(level=logging.INFO)

    def notify(self, msg):
        if not ScalpConfig.TELEGRAM_TOKEN: return
        try:
            url = f"https://api.telegram.org/bot{ScalpConfig.TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": ScalpConfig.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=5)
        except: pass

    def check_spread(self, symbol):
        orderbook = self.exchange.fetch_order_book(symbol)
        bid = orderbook['bids'][0][0]
        ask = orderbook['asks'][0][0]
        spread = (ask - bid) / bid
        return spread <= ScalpConfig.MAX_SPREAD_PCT, ask

    def run(self):
        self.notify("⚡ <b>SCALPER ACTIVE</b>\nMode: Pure Technical (No AI)")
        while True:
            for symbol in ScalpConfig.SYMBOLS:
                try:
                    # 1. Fetch Data
                    bars = self.exchange.fetch_ohlcv(symbol, '1m', limit=50)
                    df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
                    df = TechIndicators.get_indicators(df)
                    last = df.iloc[-1]
                    prev = df.iloc[-2]
                    
                    # 2. Check Costs & Spread
                    is_cheap, price = self.check_spread(symbol)
                    if not is_cheap: continue

                    # 3. Entry Logic (EMA Cross + RSI Filter + Stochastic Hook)
                    # Long Entry: EMA9 > EMA21 AND RSI < 70 AND Stochastic K crosses D above 20
                    if (last['ema9'] > last['ema21'] and last['rsi'] < 70 and 
                        prev['k'] < prev['d'] and last['k'] > last['d'] and last['k'] < 30):
                        
                        if len(self.trades) < ScalpConfig.MAX_TRADES and symbol not in [t['sym'] for t in self.trades.values()]:
                            tid = f"{symbol}_{int(time.time())}"
                            size = self.balance * ScalpConfig.RISK_PER_TRADE
                            self.trades[tid] = {'sym': symbol, 'entry': price, 'size': size, 'be': False}
                            self.notify(f"🚀 <b>ENTRY: {symbol}</b>\nPrice: {price}\nIndicator: EMA/Stoch Confirm")

                except Exception as e: continue
                time.sleep(0.1)

            # 4. Manage Exits
            self.manage_exits()
            time.sleep(1)

    def manage_exits(self):
        closed = []
        for tid, t in self.trades.items():
            curr_price = self.exchange.fetch_ticker(t['sym'])['last']
            pnl = (curr_price - t['entry']) / t['entry']
            
            # Fee Deduction (Entry + Exit)
            net_pnl = pnl - (ScalpConfig.ESTIMATED_FEE * 2)

            # Break-Even Trigger
            if net_pnl >= ScalpConfig.BE_PCT and not t['be']:
                t['be'] = True
                self.notify(f"🛡️ <b>BE: {t['sym']}</b>\nProfit secured.")

            # Hard Exit
            if net_pnl >= ScalpConfig.TP_PCT: closed.append((tid, "✅ TP", net_pnl))
            elif net_pnl <= -ScalpConfig.SL_PCT: closed.append((tid, "❌ SL", net_pnl))

        for tid, reason, final_pnl in closed:
            trade = self.trades.pop(tid)
            profit = final_pnl * trade['size']
            self.balance += (trade['size'] + profit)
            self.notify(f"🏁 <b>{reason} {trade['sym']}</b>\nGain: ${profit:+.2f}\nNew Wallet: ${self.balance:.2f}")

if __name__ == "__main__":
    AccurateScalper().run()
