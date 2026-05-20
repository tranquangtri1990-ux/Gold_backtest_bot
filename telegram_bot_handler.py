#!/usr/bin/env python3
"""
Telegram Bot Handler for XAU/USD Backtest
Receives user input and triggers GitHub Actions workflow
"""

import telebot
import os
import json
import requests
from datetime import datetime

TELEGRAM_API_KEY = os.getenv('TELEGRAM_BT_VangDo_bot_API')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO', 'your_username/your_repo')

bot = telebot.TeleBot(TELEGRAM_API_KEY)

# Store user session for multi-step input
user_sessions = {}

@bot.message_handler(commands=['start'])
def start(message):
    """Start command"""
    chat_id = message.chat.id
    text = """
🤖 <b>XAU/USD Backtest Bot</b>

Available commands:
/backtest - Start new backtest
/help - Show detailed help

Input parameters (with defaults):
• Timeframe: d, w, m
• N Periods: 20
• Trailing Stop %: 0
• Initial Capital: $500
• Date Range: 01/01/2023 to today
    """
    bot.send_message(chat_id, text, parse_mode='HTML')

@bot.message_handler(commands=['help'])
def help_command(message):
    """Help command"""
    chat_id = message.chat.id
    text = """
<b>📋 Parameter Guide:</b>

<b>Timeframe:</b> d (daily), w (weekly), m (monthly)

<b>N Periods:</b> Number of periods for average price change calculation
• Default: 20
• Range: 5-100

<b>Trailing Stop %:</b> Additional percentage on top of avg price change
• Default: 0%
• Example: if avg change is 2% and you input 1%, trailing stop = 3%

<b>Initial Capital:</b> Starting capital in USD
• Default: $500
• Minimum: $10

<b>Date Range:</b> Format dd/mm/yy
• Start date (default: 01/01/2023)
• End date (default: today)

<b>Example:</b>
/backtest d 20 0 500 01/01/2023 10/05/2026

Entry Signals: RSI(14) crossing above/below SMA(RSI,14)
Exit: Sell signal or trailing stop hit
    """
    bot.send_message(chat_id, text, parse_mode='HTML')

@bot.message_handler(commands=['backtest'])
def backtest_command(message):
    """Backtest command - ask for parameters"""
    chat_id = message.chat.id
    user_sessions[chat_id] = {'step': 0, 'params': {}}
    
    # Parse if parameters provided inline
    args = message.text.split()[1:]
    
    if len(args) >= 1:
        # All parameters provided in command
        try:
            params = {
                'timeframe': args[0],
                'n_periods': args[1] if len(args) > 1 else '20',
                'trailing_stop_pct': args[2] if len(args) > 2 else '0',
                'initial_capital': args[3] if len(args) > 3 else '500',
                'start_date': args[4] if len(args) > 4 else '01/01/2023',
                'end_date': args[5] if len(args) > 5 else ''
            }
            run_backtest(chat_id, params)
        except Exception as e:
            bot.send_message(chat_id, f"❌ Error: {str(e)}\n\nUsage: /backtest d 20 0 500 01/01/2023 10/05/2026")
    else:
        # Interactive mode - ask step by step
        msg = bot.send_message(chat_id, "📊 <b>XAU/USD Backtest Setup</b>\n\nStep 1️⃣: Enter timeframe (d/w/m):", parse_mode='HTML')
        bot.register_next_step_handler(msg, step_timeframe)

def step_timeframe(message):
    """Step 1: Get timeframe"""
    chat_id = message.chat.id
    timeframe = message.text.strip().lower()
    
    if timeframe not in ['d', 'w', 'm']:
        msg = bot.send_message(chat_id, "❌ Invalid timeframe. Use: d, w, or m")
        bot.register_next_step_handler(msg, step_timeframe)
        return
    
    user_sessions[chat_id]['params']['timeframe'] = timeframe
    msg = bot.send_message(chat_id, "Step 2️⃣: Enter N periods (default: 20):", parse_mode='HTML')
    bot.register_next_step_handler(msg, step_n_periods)

def step_n_periods(message):
    """Step 2: Get N periods"""
    chat_id = message.chat.id
    text = message.text.strip()
    
    try:
        n_periods = int(text) if text else 20
        if n_periods < 5 or n_periods > 100:
            msg = bot.send_message(chat_id, "❌ N periods must be between 5-100. Try again:")
            bot.register_next_step_handler(msg, step_n_periods)
            return
    except:
        msg = bot.send_message(chat_id, "❌ Invalid number. Try again:")
        bot.register_next_step_handler(msg, step_n_periods)
        return
    
    user_sessions[chat_id]['params']['n_periods'] = str(n_periods)
    msg = bot.send_message(chat_id, "Step 3️⃣: Enter Trailing Stop % (default: 0):", parse_mode='HTML')
    bot.register_next_step_handler(msg, step_trailing_stop)

def step_trailing_stop(message):
    """Step 3: Get trailing stop %"""
    chat_id = message.chat.id
    text = message.text.strip()
    
    try:
        trailing_stop = float(text) if text else 0.0
        if trailing_stop < 0 or trailing_stop > 100:
            msg = bot.send_message(chat_id, "❌ Trailing stop must be 0-100%. Try again:")
            bot.register_next_step_handler(msg, step_trailing_stop)
            return
    except:
        msg = bot.send_message(chat_id, "❌ Invalid number. Try again:")
        bot.register_next_step_handler(msg, step_trailing_stop)
        return
    
    user_sessions[chat_id]['params']['trailing_stop_pct'] = str(trailing_stop)
    msg = bot.send_message(chat_id, "Step 4️⃣: Enter Initial Capital $ (default: 500):", parse_mode='HTML')
    bot.register_next_step_handler(msg, step_initial_capital)

def step_initial_capital(message):
    """Step 4: Get initial capital"""
    chat_id = message.chat.id
    text = message.text.strip()
    
    try:
        capital = float(text) if text else 500.0
        if capital < 10:
            msg = bot.send_message(chat_id, "❌ Capital must be at least $10. Try again:")
            bot.register_next_step_handler(msg, step_initial_capital)
            return
    except:
        msg = bot.send_message(chat_id, "❌ Invalid number. Try again:")
        bot.register_next_step_handler(msg, step_initial_capital)
        return
    
    user_sessions[chat_id]['params']['initial_capital'] = str(capital)
    msg = bot.send_message(chat_id, "Step 5️⃣: Enter Start Date dd/mm/yy (default: 01/01/2023):", parse_mode='HTML')
    bot.register_next_step_handler(msg, step_start_date)

def step_start_date(message):
    """Step 5: Get start date"""
    chat_id = message.chat.id
    text = message.text.strip()
    
    if not text:
        start_date = '01/01/2023'
    else:
        try:
            datetime.strptime(text, '%d/%m/%y')
            start_date = text
        except:
            msg = bot.send_message(chat_id, "❌ Invalid date format. Use dd/mm/yy (e.g., 01/01/23):")
            bot.register_next_step_handler(msg, step_start_date)
            return
    
    user_sessions[chat_id]['params']['start_date'] = start_date
    msg = bot.send_message(chat_id, "Step 6️⃣: Enter End Date dd/mm/yy (default: today):", parse_mode='HTML')
    bot.register_next_step_handler(msg, step_end_date)

def step_end_date(message):
    """Step 6: Get end date and run backtest"""
    chat_id = message.chat.id
    text = message.text.strip()
    
    if not text:
        end_date = ''
    else:
        try:
            datetime.strptime(text, '%d/%m/%y')
            end_date = text
        except:
            msg = bot.send_message(chat_id, "❌ Invalid date format. Use dd/mm/yy or leave blank for today:")
            bot.register_next_step_handler(msg, step_end_date)
            return
    
    user_sessions[chat_id]['params']['end_date'] = end_date
    
    # All parameters collected, run backtest
    params = user_sessions[chat_id]['params']
    run_backtest(chat_id, params)
    
    # Clean up session
    del user_sessions[chat_id]

def run_backtest(chat_id, params):
    """Trigger GitHub Actions workflow"""
    
    bot.send_message(chat_id, "⏳ Starting backtest... this may take a minute")
    
    try:
        # Prepare workflow dispatch payload
        payload = {
            'ref': 'main',  # Change to your default branch
            'inputs': {
                'timeframe': params['timeframe'],
                'n_periods': params['n_periods'],
                'trailing_stop_pct': params['trailing_stop_pct'],
                'initial_capital': params['initial_capital'],
                'start_date': params['start_date'],
                'end_date': params.get('end_date', '')
            }
        }
        
        # Trigger workflow
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json',
            'X-GitHub-Api-Version': '2022-11-28'
        }
        
        url = f'https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/xau-backtest.yml/dispatches'
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 204:
            summary = f"""
✅ Backtest triggered!

<b>Parameters:</b>
Timeframe: {params['timeframe'].upper()}
N Periods: {params['n_periods']}
Trailing Stop: +{params['trailing_stop_pct']}%
Initial Capital: ${params['initial_capital']}
Date Range: {params['start_date']} to {params.get('end_date', 'today')}

Results will be posted shortly...
            """
            bot.send_message(chat_id, summary, parse_mode='HTML')
        else:
            bot.send_message(chat_id, f"❌ Error: {response.text}")
    
    except Exception as e:
        bot.send_message(chat_id, f"❌ Error triggering workflow: {str(e)}")

@bot.message_handler(commands=['cancel'])
def cancel(message):
    """Cancel current operation"""
    chat_id = message.chat.id
    if chat_id in user_sessions:
        del user_sessions[chat_id]
        bot.send_message(chat_id, "❌ Backtest setup cancelled")
    else:
        bot.send_message(chat_id, "No active operation to cancel")

@bot.message_handler(func=lambda message: True)
def handle_all(message):
    """Handle all other messages"""
    chat_id = message.chat.id
    if chat_id in user_sessions:
        # In the middle of backtest setup, ignore
        pass
    else:
        bot.send_message(chat_id, 
                        "🤖 Use /backtest to start a backtest\n/help for details\n/start for info")

if __name__ == '__main__':
    print("🚀 XAU/USD Backtest Bot is running...")
    bot.infinity_polling(timeout=10, long_polling_timeout=10)
