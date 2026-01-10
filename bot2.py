import os, time, requests, ccxt, logging, pandas as pd, sys
import warnings
warnings.filterwarnings('ignore')

# Force logs to show up in Railway immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class ScalpConfig:
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'AVAX/USDT']
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.20      # 20% to grow $100 aggressively
    MAX_TRADES = 3             
    TP_PCT = 0.0085            # 0.85% (Covers fees + profit)
    SL_PCT = 0.0040            # 0.40% (Tight protection)
    BE_PCT = 0.0030            # Move to BE at +0.3%
    CHECK_INTERVAL = 5         
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_tg(msg):
    if not ScalpConfig.TELEGRAM_TOKEN: return
    try:
        url = f"https://api.telegram.org/bot{ScalpConfig.TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": ScalpConfig.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=5)
    except: pass

class FastBot:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.balance = ScalpConfig.INITIAL_CAPITAL
        self.open_trades = {}

    def get_signals(self, df):
        # Technicals: EMA + Stochastic
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        # Stochastic
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['d'] = df['k'].rolling(3).mean()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # BUY: EMA Trend is UP + Stochastic Crosses UP in Oversold (<30)
        if last['ema9'] > last['ema21'] and prev['k'] < prev['d'] and last['k'] > last['d'] and last['k'] < 30:
            return 'BUY'
        return None

    def run(self):
        logging.info("⭐ BOT PROCESS INITIALIZED")
        send_tg("🚀 <b>BOT STARTED</b>\nMonitoring indicators...")
        
        while True:
            prices = {}
            for sym in ScalpConfig.SYMBOLS:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(sym, '1m', limit=50)
                    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                    prices[sym] = df['c'].iloc[-1]
                    
                    if len(self.open_trades) < ScalpConfig.MAX_TRADES:
                        if sym not in [t['sym'] for t in self.open_trades.values()]:
                            signal = self.get_signals(df)
                            if signal == 'BUY':
                                tid = f"{sym}_{int(time.time())}"
                                size = self.balance * ScalpConfig.RISK_PER_TRADE
                                self.open_trades[tid] = {'sym': sym, 'entry': prices[sym], 'size': size, 'be': False}
                                logging.info(f"TRADE OPENED: {sym}")
                                send_tg(f"💰 <b>BUY {sym}</b>\nPrice: {prices[sym]}")
                except Exception as e:
                    continue
                time.sleep(0.1)

            # Exit Management
            closed = []
            for tid, t in self.open_trades.items():
                price = prices.get(t['sym'])
                if not price: continue
                pnl = (price - t['entry']) / t['entry']
                
                if pnl >= ScalpConfig.BE_PCT and not t['be']:
                    t['be'] = True
                    send_tg(f"🛡️ <b>BE: {t['sym']}</b>")

                if pnl >= ScalpConfig.TP_PCT: closed.append((tid, "✅ TP", pnl))
                elif pnl <= -ScalpConfig.SL_PCT: closed.append((tid, "❌ SL", pnl))

            for tid, res, f_pnl in closed:
                trade = self.open_trades.pop(tid)
                profit = (f_pnl - 0.002) * trade['size'] # -0.2% for fees
                self.balance += (trade['size'] + profit)
                send_tg(f"🏁 <b>{res} {trade['sym']}</b>\nP&L: ${profit:+.2f}\nWallet: ${self.balance:.2f}")
            
            time.sleep(ScalpConfig.CHECK_INTERVAL)

if __name__ == "__main__":
    FastBot().run()



