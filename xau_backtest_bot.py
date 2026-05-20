#!/usr/bin/env python3
"""
XAU/USD Backtest Bot - Telegram integrated
Stores params via /phiên, /vốn, /timeframe, /time_start, /time_end, /x%
Backtests on higher timeframe, validates exit at minute level
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import telebot
import os
import json
import threading
from typing import Dict, Optional, Tuple, List

# Initialize Telegram bot
TELEGRAM_API_KEY = os.getenv('TELEGRAM_BT_VangDo_bot_API')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not TELEGRAM_API_KEY or not TELEGRAM_CHAT_ID:
    print("Error: Missing TELEGRAM_BT_VangDo_bot_API or TELEGRAM_CHAT_ID")
    exit(1)

bot = telebot.TeleBot(TELEGRAM_API_KEY)

# Global user parameters (stored in memory, can add JSON persistence)
USER_PARAMS = {
    'n_periods': 20,
    'initial_capital': 500.0,
    'timeframe': None,
    'start_date': '01/01/2023',
    'end_date': None,
    'trailing_stop_pct': 0.0
}


class XAUUSDBacktester:
    def __init__(self, timeframe: str, n_periods: int = 20, trailing_stop_pct: float = 0.0,
                 initial_capital: float = 500.0, start_date: str = None, end_date: str = None):
        """Initialize backtest parameters"""
        self.timeframe = timeframe.lower()
        self.n_periods = n_periods
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        
        # Parse dates
        self.start_date = self._parse_date(start_date) if start_date else '2023-01-01'
        self.end_date = self._parse_date(end_date) if end_date else datetime.now().strftime('%Y-%m-%d')
        
        # Map timeframe
        self.interval_map = {'d': '1d', 'w': '1wk', 'm': '1mo'}
        self.interval = self.interval_map.get(self.timeframe, '1d')
        
        # Trading state
        self.position = None
        self.entry_price = None
        self.entry_date = None
        self.balance = initial_capital
        self.closed_trades = []
        self.blowup_date = None
        
    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse dd/mm/yy to yyyy-mm-dd"""
        try:
            date_obj = datetime.strptime(date_str, '%d/%m/%y')
            return date_obj.strftime('%Y-%m-%d')
        except:
            return None
    
    def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch XAU/USD data"""
        try:
            df = yf.download('GC=F', start=self.start_date, end=self.end_date,
                           interval=self.interval, progress=False)
            if df.empty:
                return None
            df = df[['Close', 'High', 'Low', 'Volume']].copy()
            df.index.name = 'Date'
            return df
        except:
            return None
    
    def fetch_minute_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """Fetch 1-minute data for a specific date to find exact exit price"""
        try:
            # Parse date
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            start = date_obj.strftime('%Y-%m-%d')
            end = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Fetch minute data
            df = yf.download('GC=F', start=start, end=end, interval='1m', progress=False)
            if df.empty:
                return None
            return df[['Close', 'High', 'Low']]
        except:
            return None
    
    def calculate_rsi_wilder(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using Wilder's smoothing (TradingView method)"""
        # Use numpy for cleaner calculation
        delta = prices.diff().values
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = np.zeros_like(gain, dtype=float)
        avg_loss = np.zeros_like(loss, dtype=float)
        
        # Initialize first average with simple MA
        if period < len(gain):
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder's smoothing
        for i in range(period + 1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        
        # Calculate RSI safely
        rs = np.divide(avg_gain, avg_loss, where=avg_loss!=0, out=np.zeros_like(avg_gain))
        rsi = 100 - (100 / (1 + rs))
        
        return pd.Series(rsi, index=prices.index)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""
        df['RSI'] = self.calculate_rsi_wilder(df['Close'], period=14)
        df['SMA_RSI'] = df['RSI'].rolling(window=14).mean()
        df['Price_Change_Pct'] = df['Close'].pct_change().abs() * 100
        df['Avg_Price_Change'] = df['Price_Change_Pct'].rolling(window=self.n_periods).mean()
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover signals - RSI crossing SMA(RSI)"""
        df['Signal'] = 0
        
        for i in range(1, len(df)):
            # Skip if NaN values
            if (pd.isna(df['RSI'].iloc[i]) or pd.isna(df['SMA_RSI'].iloc[i]) or
                pd.isna(df['RSI'].iloc[i-1]) or pd.isna(df['SMA_RSI'].iloc[i-1])):
                continue
            
            rsi_prev = df['RSI'].iloc[i-1]
            sma_prev = df['SMA_RSI'].iloc[i-1]
            rsi_curr = df['RSI'].iloc[i]
            sma_curr = df['SMA_RSI'].iloc[i]
            
            # Buy: RSI crosses ABOVE SMA(RSI) - strict crossover
            # Previous: RSI <= SMA, Current: RSI > SMA
            if rsi_prev <= sma_prev and rsi_curr > sma_curr:
                df.loc[df.index[i], 'Signal'] = 1
            
            # Sell: RSI crosses BELOW SMA(RSI) - strict crossover
            # Previous: RSI >= SMA, Current: RSI < SMA
            elif rsi_prev >= sma_prev and rsi_curr < sma_curr:
                df.loc[df.index[i], 'Signal'] = -1
        
        return df
    
    def get_exact_exit_price(self, entry_date, exit_date_str, trailing_stop_level):
        """
        Get exact exit price at minute level when stop loss is hit.
        Returns the close price of the minute when price first touched stop level.
        """
        try:
            minute_df = self.fetch_minute_data(exit_date_str)
            if minute_df is None or minute_df.empty:
                # Fallback to daily close
                return None
            
            # Find first minute where Low <= stop_level
            for idx, row in minute_df.iterrows():
                if row['Low'] <= trailing_stop_level:
                    return row['Close']
            
            return None
        except:
            return None
    
    def run_backtest(self, df: pd.DataFrame) -> Tuple[Dict, List]:
        """Run backtest with proper trailing stop logic"""
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return_pct': 0,
            'final_balance': self.balance,
            'max_drawdown': 0,
            'blowup': False,
            'blowup_date': None
        }
        
        max_balance = self.balance
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            current_date = df.index[i]
            signal = df['Signal'].iloc[i]
            avg_change = df['Avg_Price_Change'].iloc[i]
            
            # In position: check sell signal OR trailing stop
            if self.position == 'long':
                # Calculate current trailing stop level
                # Trailing stop = current price - (avg_change + x%) * current_price
                stop_distance_pct = (avg_change + self.trailing_stop_pct * 100) / 100
                trailing_stop_level = current_price * (1 - stop_distance_pct)
                
                # Exit if:
                # 1. Sell signal (RSI crosses below SMA)
                # 2. Price hits trailing stop (Low touches or goes below stop)
                
                exit_price = None
                exit_type = None
                
                # Check sell signal first
                if signal == -1:
                    exit_price = current_price
                    exit_type = 'sell_signal'
                
                # Check trailing stop
                elif df['Low'].iloc[i] <= trailing_stop_level:
                    # Try to get minute-level exit price
                    exit_date_str = str(current_date.date())
                    minute_exit = self.get_exact_exit_price(self.entry_date, exit_date_str, trailing_stop_level)
                    exit_price = minute_exit if minute_exit else df['Low'].iloc[i]
                    exit_type = 'trailing_stop'
                
                # Execute exit if triggered
                if exit_price is not None:
                    pnl = (exit_price - self.entry_price) / self.entry_price
                    self.balance += self.initial_capital * pnl
                    
                    self.closed_trades.append({
                        'entry_date': str(self.entry_date.date()),
                        'entry_price': round(self.entry_price, 2),
                        'exit_date': exit_date_str if exit_type == 'trailing_stop' else str(current_date.date()),
                        'exit_price': round(exit_price, 2),
                        'pnl_pct': round(pnl * 100, 2),
                        'type': exit_type
                    })
                    
                    self.position = None
                    results['total_trades'] += 1
                    if pnl > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
            
            # Entry signal (only if not already in position)
            elif signal == 1 and self.position is None:
                self.position = 'long'
                self.entry_price = current_price
                self.entry_date = current_date
            
            # Check blowup
            if self.balance <= 0:
                self.blowup_date = str(current_date.date())
                results['blowup'] = True
                results['blowup_date'] = self.blowup_date
                results['final_balance'] = self.balance
                return results, self.closed_trades
            
            # Track drawdown
            max_balance = max(max_balance, self.balance)
            drawdown = (max_balance - self.balance) / max_balance * 100 if max_balance > 0 else 0
            results['max_drawdown'] = max(results['max_drawdown'], drawdown)
        
        results['total_return_pct'] = (self.balance - self.initial_capital) / self.initial_capital * 100
        results['final_balance'] = self.balance
        
        return results, self.closed_trades


def format_results(results: Dict, trades: List) -> str:
    """Format backtest results for Telegram"""
    msg = f"<b>📊 XAU/USD Backtest Results</b>\n"
    msg += f"{'─' * 40}\n\n"
    
    msg += f"<b>📌 Parameters:</b>\n"
    msg += f"Timeframe: <code>{USER_PARAMS['timeframe'].upper()}</code>\n"
    msg += f"Period: <code>{USER_PARAMS['start_date']}</code> → <code>{USER_PARAMS['end_date'] or 'today'}</code>\n"
    msg += f"Capital: <code>${USER_PARAMS['initial_capital']:.2f}</code>\n"
    msg += f"N Periods: <code>{USER_PARAMS['n_periods']}</code>\n"
    msg += f"Trailing Stop: <code>Avg Change + {USER_PARAMS['trailing_stop_pct']:.1f}%</code>\n\n"
    
    msg += f"<b>📈 Results:</b>\n"
    msg += f"Total Trades: <code>{results['total_trades']}</code>\n"
    
    if results['total_trades'] > 0:
        win_rate = (results['winning_trades'] / results['total_trades']) * 100
        msg += f"Win/Loss: <code>{results['winning_trades']}/{results['losing_trades']}</code> ({win_rate:.1f}%)\n"
    else:
        msg += f"Win/Loss: <code>0/0</code> (N/A)\n"
    
    msg += f"Final Balance: <code>${results['final_balance']:.2f}</code>\n"
    msg += f"Return: <code>{results['total_return_pct']:+.2f}%</code>\n"
    msg += f"Max Drawdown: <code>{results['max_drawdown']:.2f}%</code>\n\n"
    
    if results['blowup']:
        msg += f"⚠️ <b>BLOWUP!</b> Date: <code>{results['blowup_date']}</code>\n\n"
    
    # Add trade details
    if trades:
        msg += f"<b>📋 All {len(trades)} Trades:</b>\n"
        for trade in trades:
            sign = "✅" if trade['pnl_pct'] > 0 else "❌"
            exit_type = "📍TS" if trade['type'] == 'trailing_stop' else "🔴SL"
            msg += f"{sign} {exit_type} {trade['entry_date']}→{trade['exit_date']}: "
            msg += f"<code>{trade['entry_price']:.2f}</code>→<code>{trade['exit_price']:.2f}</code> "
            msg += f"<code>{trade['pnl_pct']:+.2f}%</code>\n"
    else:
        msg += f"<b>ℹ️ No trades detected in this period</b>\n"
        msg += f"(No RSI(14) crossover signals found)\n"
    
    return msg


# ============== TELEGRAM COMMAND HANDLERS ==============

@bot.message_handler(commands=['start'])
def start(message):
    """Start command"""
    msg = """
<b>🤖 XAU/USD Backtest Bot</b>

<b>Commands:</b>
/phiên - Set N periods (default: 20)
/vốn - Set initial capital (default: $500)
/timeframe - Set d/w/m (required)
/time_start - Set start date dd/mm/yy (default: 01/01/2023)
/time_end - Set end date dd/mm/yy (default: today)
/x% - Set trailing stop % (default: 0%)
/status - Show current parameters
/run - Execute backtest
/help - Detailed help
    """
    bot.send_message(message.chat.id, msg, parse_mode='HTML')

@bot.message_handler(commands=['help'])
def help_cmd(message):
    """Help command"""
    msg = """
<b>📖 Parameter Guide</b>

<b>🔹 /phiên &lt;number&gt;</b>
N periods for avg price change
Default: 20, Range: 5-100
Example: /phiên 15

<b>🔹 /vốn &lt;amount&gt;</b>
Initial capital in USD
Default: $500, Min: $10
Example: /vốn 1000

<b>🔹 /timeframe &lt;d|w|m&gt;</b>
d = daily, w = weekly, m = monthly
Example: /timeframe d

<b>🔹 /time_start &lt;dd/mm/yy&gt;</b>
Start date (default: 01/01/2023)
Example: /time_start 01/01/2023

<b>🔹 /time_end &lt;dd/mm/yy&gt;</b>
End date (default: today)
Example: /time_end 10/05/2026

<b>🔹 /x% &lt;percent&gt;</b>
Trailing stop % (default: 0%)
Example: /x% 2.5

<b>🔹 /status</b>
Show current parameters

<b>🔹 /run</b>
Execute backtest with current parameters
    """
    bot.send_message(message.chat.id, msg, parse_mode='HTML')

@bot.message_handler(commands=['phiên'])
def set_n_periods(message):
    """Set N periods"""
    try:
        args = message.text.split()
        if len(args) < 2:
            bot.send_message(message.chat.id, "Usage: /phiên <number>\nExample: /phiên 20")
            return
        
        n = int(args[1])
        if n < 5 or n > 100:
            bot.send_message(message.chat.id, "❌ N periods must be 5-100")
            return
        
        USER_PARAMS['n_periods'] = n
        bot.send_message(message.chat.id, f"✅ N periods set to <code>{n}</code>", parse_mode='HTML')
    except:
        bot.send_message(message.chat.id, "❌ Invalid input")

@bot.message_handler(commands=['vốn'])
def set_capital(message):
    """Set initial capital"""
    try:
        args = message.text.split()
        if len(args) < 2:
            bot.send_message(message.chat.id, "Usage: /vốn <amount>\nExample: /vốn 500")
            return
        
        capital = float(args[1])
        if capital < 10:
            bot.send_message(message.chat.id, "❌ Minimum capital: $10")
            return
        
        USER_PARAMS['initial_capital'] = capital
        bot.send_message(message.chat.id, f"✅ Capital set to <code>${capital:.2f}</code>", parse_mode='HTML')
    except:
        bot.send_message(message.chat.id, "❌ Invalid input")

@bot.message_handler(commands=['timeframe'])
def set_timeframe(message):
    """Set timeframe"""
    try:
        args = message.text.split()
        if len(args) < 2:
            bot.send_message(message.chat.id, "Usage: /timeframe <d|w|m>\nExample: /timeframe d")
            return
        
        tf = args[1].lower()
        if tf not in ['d', 'w', 'm']:
            bot.send_message(message.chat.id, "❌ Use d (daily), w (weekly), or m (monthly)")
            return
        
        USER_PARAMS['timeframe'] = tf
        bot.send_message(message.chat.id, f"✅ Timeframe set to <code>{tf.upper()}</code>", parse_mode='HTML')
    except:
        bot.send_message(message.chat.id, "❌ Invalid input")

@bot.message_handler(commands=['time_start'])
def set_start_date(message):
    """Set start date"""
    try:
        args = message.text.split()
        if len(args) < 2:
            bot.send_message(message.chat.id, "Usage: /time_start <dd/mm/yy>\nExample: /time_start 01/01/2023")
            return
        
        date_str = args[1]
        datetime.strptime(date_str, '%d/%m/%y')  # Validate
        USER_PARAMS['start_date'] = date_str
        bot.send_message(message.chat.id, f"✅ Start date set to <code>{date_str}</code>", parse_mode='HTML')
    except:
        bot.send_message(message.chat.id, "❌ Invalid date format (use dd/mm/yy)")

@bot.message_handler(commands=['time_end'])
def set_end_date(message):
    """Set end date"""
    try:
        args = message.text.split()
        if len(args) < 2:
            bot.send_message(message.chat.id, "Usage: /time_end <dd/mm/yy>\nExample: /time_end 10/05/2026")
            return
        
        date_str = args[1]
        datetime.strptime(date_str, '%d/%m/%y')  # Validate
        USER_PARAMS['end_date'] = date_str
        bot.send_message(message.chat.id, f"✅ End date set to <code>{date_str}</code>", parse_mode='HTML')
    except:
        bot.send_message(message.chat.id, "❌ Invalid date format (use dd/mm/yy)")

@bot.message_handler(commands=['x%'])
def set_trailing_stop(message):
    """Set trailing stop %"""
    try:
        args = message.text.split()
        if len(args) < 2:
            bot.send_message(message.chat.id, "Usage: /x% <percent>\nExample: /x% 2.5")
            return
        
        pct = float(args[1])
        if pct < 0 or pct > 100:
            bot.send_message(message.chat.id, "❌ Trailing stop must be 0-100%")
            return
        
        USER_PARAMS['trailing_stop_pct'] = pct
        bot.send_message(message.chat.id, f"✅ Trailing stop set to <code>{pct}%</code>", parse_mode='HTML')
    except:
        bot.send_message(message.chat.id, "❌ Invalid input")

@bot.message_handler(commands=['status'])
def show_status(message):
    """Show current parameters"""
    msg = f"""
<b>📌 Current Parameters</b>

N Periods: <code>{USER_PARAMS['n_periods']}</code>
Capital: <code>${USER_PARAMS['initial_capital']:.2f}</code>
Timeframe: <code>{USER_PARAMS['timeframe'] or 'NOT SET'}</code>
Start Date: <code>{USER_PARAMS['start_date']}</code>
End Date: <code>{USER_PARAMS['end_date'] or 'today'}</code>
Trailing Stop: <code>{USER_PARAMS['trailing_stop_pct']}%</code>

Ready: <code>{'✅ YES' if USER_PARAMS['timeframe'] else '❌ NO (set timeframe)'}</code>
    """
    bot.send_message(message.chat.id, msg, parse_mode='HTML')

@bot.message_handler(commands=['run'])
def run_backtest(message):
    """Execute backtest"""
    chat_id = message.chat.id
    
    # Check required parameters
    if not USER_PARAMS['timeframe']:
        bot.send_message(chat_id, "❌ Set timeframe first: /timeframe d|w|m")
        return
    
    bot.send_message(chat_id, "⏳ Running backtest... (may take 1-2 minutes)")
    
    try:
        # Create backtester
        backtester = XAUUSDBacktester(
            timeframe=USER_PARAMS['timeframe'],
            n_periods=USER_PARAMS['n_periods'],
            trailing_stop_pct=USER_PARAMS['trailing_stop_pct'],
            initial_capital=USER_PARAMS['initial_capital'],
            start_date=USER_PARAMS['start_date'],
            end_date=USER_PARAMS['end_date']
        )
        
        # Fetch and process data
        df = backtester.fetch_data()
        if df is None or df.empty:
            bot.send_message(chat_id, "❌ Cannot fetch data. Check date range.")
            return
        
        df = backtester.calculate_indicators(df)
        df = backtester.generate_signals(df)
        
        # Debug: count signals
        buy_signals = (df['Signal'] == 1).sum()
        sell_signals = (df['Signal'] == -1).sum()
        total_signals = buy_signals + sell_signals
        
        # Show debug info
        debug_msg = f"<b>🔍 Debug Info:</b>\n"
        debug_msg += f"Candles: {len(df)}\n"
        debug_msg += f"Buy signals: {buy_signals}\n"
        debug_msg += f"Sell signals: {sell_signals}\n"
        debug_msg += f"Total signals: {total_signals}\n\n"
        
        if total_signals > 0:
            # Show last few signals
            signal_rows = df[df['Signal'] != 0].tail(5)
            debug_msg += f"<b>Latest signals:</b>\n"
            for idx, row in signal_rows.iterrows():
                sig_type = "🟢 BUY" if row['Signal'] == 1 else "🔴 SELL"
                debug_msg += f"{sig_type} {str(idx.date())}: RSI={row['RSI']:.2f} SMA={row['SMA_RSI']:.2f}\n"
        else:
            # Show RSI vs SMA stats
            valid_df = df.dropna(subset=['RSI', 'SMA_RSI'])
            debug_msg += f"RSI range: {valid_df['RSI'].min():.2f} - {valid_df['RSI'].max():.2f}\n"
            debug_msg += f"SMA(RSI) range: {valid_df['SMA_RSI'].min():.2f} - {valid_df['SMA_RSI'].max():.2f}\n"
            debug_msg += f"<i>No crossovers detected</i>\n"
        
        bot.send_message(chat_id, debug_msg, parse_mode='HTML')
        
        # Run backtest
        results, trades = backtester.run_backtest(df)
        
        # Format and send results
        msg = format_results(results, trades)
        bot.send_message(chat_id, msg, parse_mode='HTML')
    
    except Exception as e:
        bot.send_message(chat_id, f"❌ Error: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle_unknown(message):
    """Handle unknown commands"""
    bot.send_message(message.chat.id, "Unknown command. Type /help for available commands.")


if __name__ == '__main__':
    print("🚀 XAU/USD Backtest Bot is running...")
    bot.infinity_polling(timeout=10, long_polling_timeout=10)