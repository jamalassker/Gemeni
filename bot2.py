import os, time, requests, ccxt, logging, pandas as pd, sys
import warnings
warnings.filterwarnings('ignore')

# FORCE logs to show in Railway immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class AggressiveConfig:
    # Aggressive list of 10 highly liquid symbols
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 
               'AVAX/USDT', 'ADA/USDT', 'DOGE/USDT', 'LINK/USDT', 'NEAR/USDT']
    
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.25      # 25% of wallet ($25) per trade to grow fast
    MAX_TRADES = 4             # Up to 4 trades at once
    
    # Aggressive Targets
    TP_PCT = 0.0095            # 0.95% Target
    SL_PCT = 0.0045            # 0.45% Stop Loss
    BE_PCT = 0.0035            # Set to Break-Even at +0.35%
    
    CHECK_INTERVAL = 3         # Scan every 3 seconds
    
    # Telegram
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_tg(msg):
    if not AggressiveConfig.TOKEN: return
    try:
        url = f"https://api.telegram.org/bot{AggressiveConfig.TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": AggressiveConfig.CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=5)
    except: pass

class FastAggressor:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.balance = AggressiveConfig.INITIAL_CAPITAL
        self.open_trades = {}

    def get_signal(self, df):
        # EMA Trend
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        # Stochastic
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['d'] = df['k'].rolling(3).mean()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # AGGRESSIVE BUY: Trend is UP and Stoch just crossed UP below 40
        if last['ema9'] > last['ema21'] and prev['k'] < prev['d'] and last['k'] > last['d'] and last['k'] < 40:
            return 'BUY'
        return None

    def run(self):
        logging.info("⚡ AGGRESSIVE BOT STARTING...")
        send_tg("⚡ <b>AGGRESSIVE BOT ONLINE</b>\nScanning for quick profits.")
        
        while True:
            prices = {}
            for sym in AggressiveConfig.SYMBOLS:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(sym, '1m', limit=50)
                    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                    prices[sym] = df['c'].iloc[-1]
                    
                    # Entry logic
                    if len(self.open_trades) < AggressiveConfig.MAX_TRADES:
                        if sym not in [t['sym'] for t in self.open_trades.values()]:
                            sig = self.get_signal(df)
                            if sig:
                                tid = f"{sym}_{int(time.time())}"
                                pos_size = self.balance * AggressiveConfig.RISK_PER_TRADE
                                self.open_trades[tid] = {'sym': sym, 'entry': prices[sym], 'size': pos_size, 'be': False}
                                logging.info(f"🔥 OPENED {sym} at {prices[sym]}")
                                send_tg(f"🚀 <b>BUY: {sym}</b>\nPrice: {prices[sym]}")
                except: continue
                time.sleep(0.05)

            # Exit logic
            closed = []
            for tid, t in self.open_trades.items():
                p = prices.get(t['sym'])
                if not p: continue
                pnl = (p - t['entry']) / t['entry']
                
                # Move to Break-Even early
                if pnl >= AggressiveConfig.BE_PCT and not t['be']:
                    t['be'] = True
                    send_tg(f"🛡️ <b>SECURED: {t['sym']}</b> (BE Set)")

                if pnl >= AggressiveConfig.TP_PCT: closed.append((tid, "🟢 PROFIT", pnl))
                elif pnl <= -AggressiveConfig.SL_PCT: closed.append((tid, "🔴 STOP", pnl))

            for tid, res, f_pnl in closed:
                trade = self.open_trades.pop(tid)
                net_pnl = f_pnl - 0.002 # Deduct 0.2% for spread/fees
                gain = net_pnl * trade['size']
                self.balance += (trade['size'] + gain)
                send_tg(f"🏁 <b>{res} {trade['sym']}</b>\nP&L: ${gain:+.2f}\nWallet: ${self.balance:.2f}")

            time.sleep(AggressiveConfig.CHECK_INTERVAL)

if __name__ == "__main__":
    FastAggressor().run()




