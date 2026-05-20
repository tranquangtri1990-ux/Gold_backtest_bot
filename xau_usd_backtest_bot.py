#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import telebot
import os
import sys
from typing import Dict, List, Tuple, Optional

# Initialize Telegram bot
TELEGRAM_API_KEY = os.getenv('TELEGRAM_BT_VangDo_bot_API')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not TELEGRAM_API_KEY:
    print("Error: TELEGRAM_BT_VangDo_bot_API not found in environment")
    sys.exit(1)

bot = telebot.TeleBot(TELEGRAM_API_KEY)

class XAUUSDBacktester:
    def __init__(self, timeframe: str, n_periods: int = 20, trailing_stop_pct: float = 0.0,
                 initial_capital: float = 500.0, start_date: str = None, end_date: str = None):
        """
        Initialize XAU/USD backtest parameters
        
        Args:
            timeframe: 'd' (daily), 'w' (weekly), 'm' (monthly)
            n_periods: periods for avg price change calculation (default: 20)
            trailing_stop_pct: additional % for trailing stop (default: 0%)
            initial_capital: initial capital in USD (default: 500)
            start_date: start date in dd/mm/yy format (default: 01/01/2023)
            end_date: end date in dd/mm/yy format (default: today)
        """
        self.timeframe = timeframe.lower()
        self.n_periods = n_periods
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        
        # Parse dates
        if start_date is None:
            self.start_date = '2023-01-01'
        else:
            self.start_date = self._parse_date(start_date)
        
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = self._parse_date(end_date)
        
        # Map timeframe
        self.interval_map = {'d': '1d', 'w': '1wk', 'm': '1mo'}
        self.interval = self.interval_map.get(self.timeframe, '1d')
        
        # Trading state
        self.position = None  # 'long' or None
        self.entry_price = None
        self.entry_date = None
        self.entry_index = None
        self.highest_price = None
        self.lowest_price = None
        self.trailing_stop_level = None
        self.balance = initial_capital
        self.trades = []
        self.blowup_date = None
        self.closed_trades = []
        
    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse dd/mm/yy format to yyyy-mm-dd"""
        try:
            date_obj = datetime.strptime(date_str, '%d/%m/%y')
            return date_obj.strftime('%Y-%m-%d')
        except:
            return None
    
    def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch XAU/USD data from yfinance"""
        try:
            df = yf.download('GC=F', start=self.start_date, end=self.end_date,
                           interval=self.interval, progress=False)
            if df.empty:
                return None
            df = df[['Close', 'High', 'Low', 'Volume']].copy()
            df.index.name = 'Date'
            return df
        except Exception as e:
            return None
    
    def calculate_rsi_wilder(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using Wilder's smoothing (TradingView method)"""
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = loss.abs()
        
        # Wilder's smoothing
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Smooth from period onwards
        for i in range(period, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI(14) and SMA(RSI, 14)"""
        # RSI calculation (TradingView Wilder's method)
        df['RSI'] = self.calculate_rsi_wilder(df['Close'], period=14)
        
        # SMA of RSI
        df['SMA_RSI'] = df['RSI'].rolling(window=14).mean()
        
        # Average price change for trailing stop
        df['Price_Change_Pct'] = df['Close'].pct_change().abs() * 100
        df['Avg_Price_Change'] = df['Price_Change_Pct'].rolling(window=self.n_periods).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover signals"""
        df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        for i in range(1, len(df)):
            if pd.isna(df['RSI'].iloc[i]) or pd.isna(df['SMA_RSI'].iloc[i]):
                continue
            
            # Buy signal: RSI crosses above SMA(RSI)
            if (df['RSI'].iloc[i-1] <= df['SMA_RSI'].iloc[i-1] and 
                df['RSI'].iloc[i] > df['SMA_RSI'].iloc[i]):
                df.loc[df.index[i], 'Signal'] = 1
            
            # Sell signal: RSI crosses below SMA(RSI)
            elif (df['RSI'].iloc[i-1] >= df['SMA_RSI'].iloc[i-1] and 
                  df['RSI'].iloc[i] < df['SMA_RSI'].iloc[i]):
                df.loc[df.index[i], 'Signal'] = -1
        
        return df
    
    def run_backtest(self, df: pd.DataFrame) -> Tuple[Dict, List]:
        """Run backtest simulation"""
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
        min_balance = self.balance
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            current_date = df.index[i]
            signal = df['Signal'].iloc[i]
            avg_change = df['Avg_Price_Change'].iloc[i]
            
            # Update trailing stop if in position
            if self.position == 'long':
                # Trailing stop = avg_change + x%
                self.trailing_stop_level = current_price * (1 - (avg_change + self.trailing_stop_pct * 100) / 100)
                
                # Close if hit trailing stop
                if current_price <= self.trailing_stop_level:
                    pnl = (current_price - self.entry_price) / self.entry_price
                    self.balance += self.initial_capital * pnl
                    self.closed_trades.append({
                        'entry_date': self.entry_date,
                        'entry_price': self.entry_price,
                        'exit_date': current_date,
                        'exit_price': current_price,
                        'pnl_pct': pnl * 100,
                        'type': 'trailing_stop'
                    })
                    self.position = None
                    results['total_trades'] += 1
                    if pnl > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
            
            # Entry signal
            if signal == 1 and self.position is None:
                self.position = 'long'
                self.entry_price = current_price
                self.entry_date = current_date
                self.entry_index = i
                self.highest_price = current_price
            
            # Exit signal
            elif signal == -1 and self.position == 'long':
                pnl = (current_price - self.entry_price) / self.entry_price
                self.balance += self.initial_capital * pnl
                self.closed_trades.append({
                    'entry_date': self.entry_date,
                    'entry_price': self.entry_price,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'pnl_pct': pnl * 100,
                    'type': 'sell_signal'
                })
                self.position = None
                results['total_trades'] += 1
                if pnl > 0:
                    results['winning_trades'] += 1
                else:
                    results['losing_trades'] += 1
            
            # Check blowup
            if self.balance <= 0:
                self.blowup_date = current_date
                results['blowup'] = True
                results['blowup_date'] = str(current_date)
                results['final_balance'] = self.balance
                return results, self.closed_trades
            
            # Track max drawdown
            max_balance = max(max_balance, self.balance)
            min_balance = min(min_balance, self.balance)
            drawdown = (max_balance - self.balance) / max_balance * 100 if max_balance > 0 else 0
            results['max_drawdown'] = max(results['max_drawdown'], drawdown)
        
        results['total_return_pct'] = (self.balance - self.initial_capital) / self.initial_capital * 100
        results['final_balance'] = self.balance
        
        return results, self.closed_trades
    
    def format_telegram_message(self, results: Dict) -> str:
        """Format backtest results for Telegram"""
        msg = f"<b>🚀 XAU/USD Backtest Results</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += f"<b>Parameters:</b>\n"
        msg += f"Timeframe: {self.timeframe.upper()}\n"
        msg += f"Period: {self.start_date} → {self.end_date}\n"
        msg += f"Initial Capital: ${self.initial_capital:.2f}\n"
        msg += f"N Periods: {self.n_periods}\n"
        msg += f"Trailing Stop: Avg Change + {self.trailing_stop_pct*100:.1f}%\n\n"
        
        msg += f"<b>Results:</b>\n"
        msg += f"Total Trades: {results['total_trades']}\n"
        msg += f"Win Rate: {results['winning_trades']}/{results['total_trades']} "
        if results['total_trades'] > 0:
            msg += f"({results['winning_trades']/results['total_trades']*100:.1f}%)\n"
        else:
            msg += "(N/A)\n"
        
        msg += f"Final Balance: ${results['final_balance']:.2f}\n"
        msg += f"Return: {results['total_return_pct']:+.2f}%\n"
        msg += f"Max Drawdown: {results['max_drawdown']:.2f}%\n\n"
        
        if results['blowup']:
            msg += f"⚠️ <b>ACCOUNT BLOWUP!</b>\n"
            msg += f"Date: {results['blowup_date']}\n"
        
        return msg


def parse_user_input() -> Dict:
    """Parse command line arguments or interactive input"""
    params = {
        'timeframe': None,
        'n_periods': 20,
        'trailing_stop_pct': 0.0,
        'initial_capital': 500.0,
        'start_date': None,
        'end_date': None
    }
    
    # For Telegram bot: expect arguments passed as command
    # Format: /backtest d 20 0 500 01/01/2023 10/05/2026
    if len(sys.argv) > 1:
        try:
            params['timeframe'] = sys.argv[1]
            if len(sys.argv) > 2:
                params['n_periods'] = int(sys.argv[2])
            if len(sys.argv) > 3:
                params['trailing_stop_pct'] = float(sys.argv[3])
            if len(sys.argv) > 4:
                params['initial_capital'] = float(sys.argv[4])
            if len(sys.argv) > 5:
                params['start_date'] = sys.argv[5]
            if len(sys.argv) > 6:
                params['end_date'] = sys.argv[6]
        except:
            return None
    
    return params if params['timeframe'] else None


def main():
    params = parse_user_input()
    
    if not params:
        print("Usage: python xau_usd_backtest_bot.py <timeframe> [n_periods] [trailing_stop_pct] [initial_capital] [start_date] [end_date]")
        print("Example: python xau_usd_backtest_bot.py d 20 0 500 01/01/2023 10/05/2026")
        sys.exit(1)
    
    # Run backtest
    backtester = XAUUSDBacktester(
        timeframe=params['timeframe'],
        n_periods=params['n_periods'],
        trailing_stop_pct=params['trailing_stop_pct'],
        initial_capital=params['initial_capital'],
        start_date=params['start_date'],
        end_date=params['end_date']
    )
    
    # Fetch data
    df = backtester.fetch_data()
    if df is None or df.empty:
        msg = "❌ Error: Cannot fetch data from yfinance"
        print(msg)
        if TELEGRAM_CHAT_ID:
            bot.send_message(TELEGRAM_CHAT_ID, msg)
        sys.exit(1)
    
    # Calculate indicators
    df = backtester.calculate_indicators(df)
    
    # Generate signals
    df = backtester.generate_signals(df)
    
    # Run backtest
    results, trades = backtester.run_backtest(df)
    
    # Format and send message
    msg = backtester.format_telegram_message(results)
    
    print(msg)
    if TELEGRAM_CHAT_ID:
        try:
            bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
        except Exception as e:
            print(f"Error sending Telegram message: {e}")


if __name__ == '__main__':
    main()
