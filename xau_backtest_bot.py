#!/usr/bin/env python3
"""XAU/USD Backtest Bot — RSI(14) crossover SMA(RSI,14), lot-based PnL"""

import os, io, threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import yfinance as yf
import telebot

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY  = os.getenv('TELEGRAM_BT_VangDo_bot_API')
CHAT_ID  = os.getenv('TELEGRAM_CHAT_ID')
if not API_KEY or not CHAT_ID:
    print("Missing TELEGRAM_BT_VangDo_bot_API or TELEGRAM_CHAT_ID"); exit(1)

bot = telebot.TeleBot(API_KEY)

# XAU/USD forex: 1 lot = 100 oz. PnL = Δprice × lot × 100
LOT_SIZE_OZ = 100

P = {   # user params
    'timeframe':    None,
    'start_date':   '01/01/2023',
    'end_date':     None,
    'von':          1000.0,   # vốn ban đầu (USD)
    'lot':          0.01,
    'trailing_pct': 0.0,
    'n_periods':    20,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_date(s: str) -> Optional[str]:
    for fmt in ('%d/%m/%Y', '%d/%m/%y'):
        try: return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError: pass
    return None


# ── Data fetching ─────────────────────────────────────────────────────────────
def _stooq(start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        url = f"https://stooq.com/q/d/l/?s=xauusd&d1={start.replace('-','')}&d2={end.replace('-','')}&i=d"
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or len(r.text) < 50: return None
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().capitalize() for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        if 'Volume' not in df.columns: df['Volume'] = 0
        return df[['Close','High','Low','Volume']].dropna(subset=['Close'])
    except: return None


def _yf(ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df.columns = [c.capitalize() for c in df.columns]
        if 'Volume' not in df.columns: df['Volume'] = 0
        if any(c not in df.columns for c in ['Close','High','Low']): return None
        df = df[['Close','High','Low','Volume']].copy()
        for col in df.columns: df[col] = df[col].squeeze()
        return df.dropna(subset=['Close']).sort_index()
    except: return None


def fetch_data(timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch XAUUSD Spot với 2 năm warmup để RSI converge giống TradingView.
    Nguồn: Stooq (spot forex) → XAUUSD=X → GC=F
    """
    end = end_date or datetime.now().strftime('%Y-%m-%d')
    warmup = (datetime.strptime(start_date,'%Y-%m-%d') - timedelta(days=730)).strftime('%Y-%m-%d')
    iv_map = {'d':'1d','w':'1wk','m':'1mo'}
    interval = iv_map.get(timeframe,'1d')

    # 1. Stooq daily → resample nếu cần
    df = _stooq(warmup, end)
    if df is not None and not df.empty:
        if timeframe == 'w':
            df = df.resample('W').agg({'Close':'last','High':'max','Low':'min','Volume':'sum'}).dropna(subset=['Close'])
        elif timeframe == 'm':
            df = df.resample('ME').agg({'Close':'last','High':'max','Low':'min','Volume':'sum'}).dropna(subset=['Close'])
        df.index.name = 'Date'
        print(f"Stooq OK: {len(df)} bars")
        return df

    # 2-3. yfinance fallback
    for ticker in ('XAUUSD=X','GC=F'):
        df = _yf(ticker, warmup, end, interval)
        if df is not None:
            print(f"yfinance {ticker} OK: {len(df)} bars")
            return df

    return None


def fetch_minute(date_str: str, stop_level: float) -> Optional[float]:
    """Tìm giá exit chính xác tại minute khi trailing stop chạm"""
    try:
        d = datetime.strptime(date_str,'%Y-%m-%d')
        df = _yf('XAUUSD=X', d.strftime('%Y-%m-%d'),
                 (d+timedelta(days=1)).strftime('%Y-%m-%d'), '1m')
        if df is None: return None
        for _, row in df.iterrows():
            if row['Low'] <= stop_level: return float(row['Close'])
    except: pass
    return None


# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder (RMA) — khớp TradingView ta.rsi()"""
    g = close.diff().clip(lower=0).values.astype(float)
    l = (-close.diff()).clip(lower=0).values.astype(float)
    n = len(g)
    ag = np.full(n, np.nan); al = np.full(n, np.nan)
    if n > period:
        ag[period] = np.nanmean(g[1:period+1])
        al[period] = np.nanmean(l[1:period+1])
        for i in range(period+1, n):
            ag[i] = (ag[i-1]*(period-1) + g[i]) / period
            al[i] = (al[i-1]*(period-1) + l[i]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(al==0, np.inf, ag/al)
    rsi = np.where(al==0, 100.0, 100.0 - 100.0/(1.0+rs))
    rsi[:period] = np.nan
    return pd.Series(rsi, index=close.index)


def add_indicators(df: pd.DataFrame, n_periods: int) -> pd.DataFrame:
    df = df.copy()
    close = df['Close'].squeeze()
    df['RSI']     = calc_rsi_wilder(close)
    df['SMA_RSI'] = df['RSI'].rolling(14, min_periods=14).mean()
    df['Avg_Move'] = close.pct_change().abs().rolling(n_periods, min_periods=1).mean() * 100
    return df


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    # Pandas 3.x Copy-on-Write: df['col'].iloc[i]=val là no-op
    # → phải build numpy array rồi assign 1 lần duy nhất
    rsi = df['RSI'].values
    sma = df['SMA_RSI'].values
    sig = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if any(np.isnan(x) for x in [rsi[i-1], sma[i-1], rsi[i], sma[i]]):
            continue
        if   rsi[i-1] <= sma[i-1] and rsi[i] > sma[i]: sig[i] =  1
        elif rsi[i-1] >= sma[i-1] and rsi[i] < sma[i]: sig[i] = -1
    df = df.copy()
    df['Signal'] = sig
    return df


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, start_date: str,
                 von: float, lot: float, trailing_pct: float) -> Tuple[Dict, List]:
    """
    PnL tuyệt đối theo lot forex:
        pnl_usd = (exit - entry) × lot × LOT_SIZE_OZ

    Nhiều lệnh đồng thời: mỗi buy signal mở 1 lệnh mới độc lập.
    Sell signal đóng tất cả lệnh đang mở.
    Trailing stop = highest_high × (1 - stop_pct), chỉ đi lên.
    Cháy tài khoản khi balance <= 0.
    """
    start_ts   = pd.Timestamp(start_date)
    open_trades: List[Dict] = []
    closed:      List[Dict] = []
    balance     = von
    total_pnl   = 0.0
    peak_balance= von
    max_dd      = 0.0
    blowup_date = None

    for i in range(len(df)):
        bar = df.index[i]
        if bar < start_ts: continue

        price   = float(df['Close'].iloc[i])
        high    = float(df['High'].iloc[i])
        low     = float(df['Low'].iloc[i])
        signal  = int(df['Signal'].iloc[i])
        date_s  = str(bar.date())
        # Chỉ dùng trailing_pct thuần — KHÔNG cộng avg_mv
        # avg_mv làm stop quá rộng → trailing stop không bao giờ trigger
        stop_pct = trailing_pct / 100.0

        # Update + check exit for each open trade
        still_open = []
        for t in open_trades:
            t['peak'] = max(t['peak'], high)
            stop_lvl  = t['peak'] * (1 - stop_pct) if stop_pct > 0 else 0
            ep = None; etype = None

            # Trailing stop TRƯỚC signal
            if stop_pct > 0 and low <= stop_lvl:
                ep = fetch_minute(date_s, stop_lvl) or stop_lvl
                etype = 'stop'
            elif signal == -1:
                ep, etype = price, 'signal'

            if ep is not None:
                pnl      = (ep - t['entry']) * lot * LOT_SIZE_OZ
                total_pnl += pnl
                balance   += pnl
                closed.append({
                    'entry_date':  t['date'],
                    'entry_price': round(t['entry'], 2),
                    'exit_date':   date_s,
                    'exit_price':  round(ep, 2),
                    'pnl_usd':     round(pnl, 2),
                    'balance':     round(max(balance, 0), 2),
                    'type':        etype,
                    'stop_lvl':    round(stop_lvl, 2),
                })
                # Cháy tài khoản: ghi nhận ngày, đóng tất cả lệnh còn lại
                if balance <= 0 and blowup_date is None:
                    blowup_date = date_s
                    still_open  = []   # force đóng hết
                    break             # dừng vòng loop trade
            else:
                still_open.append(t)

        open_trades = still_open

        # Dừng giao dịch khi đã cháy
        if blowup_date:
            break

        # Chỉ mở lệnh mới khi còn vốn
        if signal == 1:
            open_trades.append({'entry': price, 'date': date_s, 'peak': high})

        # Track max drawdown theo balance
        peak_balance = max(peak_balance, balance)
        dd = peak_balance - balance
        max_dd = max(max_dd, dd)

    wins   = sum(1 for t in closed if t['pnl_usd'] > 0)
    losses = len(closed) - wins
    return {
        'total_trades':  len(closed),
        'wins':          wins,
        'losses':        losses,
        'total_pnl':     round(total_pnl, 2),
        'final_balance': round(max(balance, 0), 2),
        'max_dd':        round(max_dd, 2),
        'open_count':    len(open_trades),
        'blowup':        blowup_date,
    }, closed


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt_results(res: Dict, trades: List) -> str:
    wr = f"{res['wins']/res['total_trades']*100:.1f}%" if res['total_trades'] else "N/A"
    s  = (f"<b>📊 XAU/USD Backtest</b>\n"
          f"TF: <code>{P['timeframe'].upper()}</code>  "
          f"{P['start_date']} → {P['end_date'] or 'today'}\n"
          f"Vốn: <code>${P['von']:,.2f}</code>  "
          f"Lot: <code>{P['lot']}</code>  Trail: <code>{P['trailing_pct']}%</code>\n\n"
          f"Trades: <code>{res['total_trades']}</code>  "
          f"W/L: <code>{res['wins']}/{res['losses']}</code> ({wr})\n"
          f"Total PnL:     <code>${res['total_pnl']:+.2f}</code>\n"
          f"Final balance: <code>${res['final_balance']:,.2f}</code>\n"
          f"Max Drawdown:  <code>${res['max_dd']:.2f}</code>\n")
    if res['blowup']:
        s += f"💥 <b>CHÁY TÀI KHOẢN</b> ngày <code>{res['blowup']}</code>\n"
    if res['open_count']:
        s += f"⏳ Còn mở: <code>{res['open_count']}</code> lệnh chưa đóng\n"
    if trades:
        s += f"\n<b>Trades:</b>\n"
        for t in trades:
            icon = "✅" if t['pnl_usd'] > 0 else "❌"
            tag  = "🛑" if t['type'] == 'stop' else "📍"
            s += (f"{icon}{tag} {t['entry_date']}→{t['exit_date']}  "
                  f"<code>{t['entry_price']:.2f}→{t['exit_price']:.2f}</code>  "
                  f"<code>${t['pnl_usd']:+.2f}</code>  "
                  f"bal:<code>${t['balance']:,.2f}</code>\n")
    return s


# ── Telegram handlers ─────────────────────────────────────────────────────────
def _arg(msg, n=1):
    parts = msg.text.split()
    return parts[n] if len(parts) > n else None

@bot.message_handler(commands=['start','help'])
def cmd_help(m):
    bot.send_message(m.chat.id, (
        "<b>🤖 XAU/USD Backtest Bot</b>\n\n"
        "/timeframe d|w|m\n"
        "/time_start dd/mm/yyyy\n"
        "/time_end dd/mm/yyyy\n"
        "/vốn 1000  (vốn ban đầu USD)\n"
        "/lot 0.01\n"
        "/x% trailing stop %\n"
        "/phiên N periods (vol avg)\n"
        "/status  xem params\n"
        "/run  chạy backtest"
    ), parse_mode='HTML')

@bot.message_handler(commands=['timeframe'])
def cmd_tf(m):
    v = _arg(m)
    if v and v.lower() in ('d','w','m'):
        P['timeframe'] = v.lower()
        bot.send_message(m.chat.id, f"✅ Timeframe: <code>{v.upper()}</code>", parse_mode='HTML')
    else:
        bot.send_message(m.chat.id, "❌ /timeframe d|w|m")

@bot.message_handler(commands=['time_start'])
def cmd_start(m):
    v = _arg(m)
    if v and parse_date(v):
        P['start_date'] = v
        bot.send_message(m.chat.id, f"✅ Start: <code>{v}</code>", parse_mode='HTML')
    else:
        bot.send_message(m.chat.id, "❌ /time_start dd/mm/yyyy")

@bot.message_handler(commands=['time_end'])
def cmd_end(m):
    v = _arg(m)
    if v and parse_date(v):
        P['end_date'] = v
        bot.send_message(m.chat.id, f"✅ End: <code>{v}</code>", parse_mode='HTML')
    else:
        bot.send_message(m.chat.id, "❌ /time_end dd/mm/yyyy")

@bot.message_handler(commands=['vốn'])
def cmd_von(m):
    try:
        v = float(_arg(m))
        if v > 0:
            P['von'] = v
            bot.send_message(m.chat.id, f"✅ Vốn: <code>${v:,.2f}</code>", parse_mode='HTML')
        else:
            bot.send_message(m.chat.id, "❌ Vốn phải > 0")
    except:
        bot.send_message(m.chat.id, "❌ /vốn 1000")

@bot.message_handler(commands=['lot'])
def cmd_lot(m):
    try:
        v = float(_arg(m))
        if 0.001 <= v <= 100:
            P['lot'] = v
            bot.send_message(m.chat.id, f"✅ Lot: <code>{v}</code>", parse_mode='HTML')
        else:
            bot.send_message(m.chat.id, "❌ Lot: 0.001 – 100")
    except:
        bot.send_message(m.chat.id, "❌ /lot 0.01")

@bot.message_handler(commands=['x%'])
def cmd_trail(m):
    try:
        v = float(_arg(m))
        P['trailing_pct'] = v
        bot.send_message(m.chat.id, f"✅ Trailing: <code>{v}%</code>", parse_mode='HTML')
    except:
        bot.send_message(m.chat.id, "❌ /x% 2.5")

@bot.message_handler(commands=['phiên'])
def cmd_periods(m):
    try:
        v = int(_arg(m))
        if 5 <= v <= 100:
            P['n_periods'] = v
            bot.send_message(m.chat.id, f"✅ N periods: <code>{v}</code>", parse_mode='HTML')
        else:
            bot.send_message(m.chat.id, "❌ 5–100")
    except:
        bot.send_message(m.chat.id, "❌ /phiên 20")

@bot.message_handler(commands=['status'])
def cmd_status(m):
    bot.send_message(m.chat.id, (
        f"<b>📌 Params</b>\n"
        f"Timeframe: <code>{P['timeframe'] or 'NOT SET'}</code>\n"
        f"Start: <code>{P['start_date']}</code>\n"
        f"End: <code>{P['end_date'] or 'today'}</code>\n"
        f"Vốn: <code>${P['von']:,.2f}</code>\n"
        f"Lot: <code>{P['lot']}</code>\n"
        f"Trailing: <code>{P['trailing_pct']}%</code>\n"
        f"N periods: <code>{P['n_periods']}</code>"
    ), parse_mode='HTML')

@bot.message_handler(commands=['run'])
def cmd_run(m):
    if not P['timeframe']:
        bot.send_message(m.chat.id, "❌ Set timeframe first: /timeframe d|w|m"); return

    bot.send_message(m.chat.id, "⏳ Fetching data & running backtest...")
    try:
        start = parse_date(P['start_date'])
        end   = parse_date(P['end_date']) if P['end_date'] else None

        df = fetch_data(P['timeframe'], start, end or datetime.now().strftime('%Y-%m-%d'))
        if df is None or df.empty:
            bot.send_message(m.chat.id, "❌ Cannot fetch data. Check date range."); return

        df = add_indicators(df, P['n_periods'])
        df = add_signals(df)

        buys  = (df['Signal'] == 1).sum()
        sells = (df['Signal'] == -1).sum()
        bot.send_message(m.chat.id,
            f"📶 Signals: 🟢{buys} buy / 🔴{sells} sell", parse_mode='HTML')

        res, trades = run_backtest(df, start, P['von'], P['lot'], P['trailing_pct'])
        bot.send_message(m.chat.id, fmt_results(res, trades), parse_mode='HTML')

    except Exception as e:
        bot.send_message(m.chat.id, f"❌ Error: {e}")

@bot.message_handler(func=lambda m: True)
def cmd_unknown(m):
    bot.send_message(m.chat.id, "Unknown command. /help")


if __name__ == '__main__':
    print("🚀 XAU/USD Backtest Bot running...")
    bot.infinity_polling(timeout=10, long_polling_timeout=10)