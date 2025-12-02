import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import json
import logging
import socket
import random
import requests
import re
from datetime import datetime, timedelta

# ==========================================
# 🎛️ 用户配置区 (USER CONFIGURATION)
# ==========================================
# 1. 填入你的 PushPlus Token (不要留空)
DEFAULT_WECHAT_TOKEN = "4364438ae3014d628e1cae92bbf00cc0" 

# 2. 开启自动巡航 (True = 默认开启，打开网页即自动运行)
DEFAULT_AUTO_PILOT = True  
# ==========================================

# --- 0. 生产环境初始化 ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

try: import psutil
except ImportError: psutil = None

def log_system_status():
    if psutil is None: return "Monitor: Bypass"
    try: return f"✅ Mem: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f}MB"
    except: return "Monitor Fail"

# --- 1. 页面配置 ---
st.set_page_config(page_title="Tod's Studio V10.10 (Golden Master)", page_icon="🎸", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; }
    .advice-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stProgress > div > div > div > div { background-color: #28a745; }
    .stButton button { width: 100%; font-weight: 600; }
    .countdown-box {
        font-family: 'Courier New', monospace; font-size: 1.1em; color: #d63384;
        font-weight: bold; text-align: center; padding: 8px;
        border: 1px dashed #d63384; border-radius: 5px; margin-top: 10px; background-color: #fff0f6;
    }
    .status-updated { color: #198754; font-size: 0.8em; text-align: center; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .no-signal { background-color: #343a40; color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }
    .stChatMessage { background-color: #f0f2f6; border-radius: 10px; padding: 1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# 状态初始化
if 'ticker_status' not in st.session_state: st.session_state['ticker_status'] = {}
if 'last_known_status' not in st.session_state: st.session_state['last_known_status'] = {} 
if 'scan_executed' not in st.session_state: st.session_state['scan_executed'] = False
if 'last_scan_time' not in st.session_state: st.session_state['last_scan_time'] = None
if 'trigger_refresh' not in st.session_state: st.session_state['trigger_refresh'] = False
if 'messages' not in st.session_state: st.session_state['messages'] = []
if 'chat_context_ticker' not in st.session_state: st.session_state['chat_context_ticker'] = ""
if 'pushed_today' not in st.session_state: st.session_state['pushed_today'] = set()

# --- 2. 核心记忆系统 ---
SETTINGS_FILE = 'stock_settings.json'
US_SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SOXX", "AMD": "SOXX", "TSM": "SOXX", "AVGO": "SOXX",
    "TSLA": "XLY", "AMZN": "XLY", "NFLX": "XLY", "GOOG": "XLC", "META": "XLC",
    "JPM": "XLF", "BAC": "XLF", "COIN": "XLF", "HOOD": "XLF", "MSTR": "XLF", "CRCL": "XLF",
    "default": "QQQ"
}

def load_settings():
    default_settings = {
        "portfolio": ["TSLA", "NVDA", "QQQ"], 
        "atr_params": {"TSLA": 3.2, "NVDA": 2.8, "QQQ": 2.0},
        "favorites": ["TSLA", "NVDA", "MSTR", "COIN", "QQQ", "SMH", "AAPL", "AMD", "AMZN", "GOOG", "CRWD", "PLTR", "", "", "", "", "", "", "", ""] 
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                favs = settings.get("favorites", [])
                while len(favs) < 20: favs.append("")
                settings["favorites"] = favs[:20] 
                return settings
        except: return default_settings
    return default_settings

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=2); return True
    except: return False

if 'settings' not in st.session_state: st.session_state['settings'] = load_settings()
if 'current_ticker' not in st.session_state: 
    first_valid = next((x for x in st.session_state['settings']['favorites'] if x), "QQQ")
    st.session_state['current_ticker'] = first_valid

# --- 3. 数据层 (V10.0 Robust) ---
def sanitize_ticker(ticker):
    return re.sub(r'[^A-Za-z0-9\.\^]', '', str(ticker).upper())[:20] if ticker else ""

def fix_china_ticker(ticker):
    t = sanitize_ticker(ticker).strip()
    return f"{t}.HK" if t.isdigit() and len(t) == 5 else t

def get_random_agent():
    return random.choice(['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'])

def validate_stock_data(df, min_days=50):
    if df is None or len(df) < min_days: return False
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols): return False
    return True

@st.cache_data(ttl=600, show_spinner=False)
def fetch_data_safe(ticker, period="2y"):
    ticker = fix_china_ticker(ticker)
    try:
        time.sleep(random.uniform(0.5, 1.2))
        stock = yf.Ticker(ticker) 
        df = stock.history(period=period, interval="1d", auto_adjust=False)
        if validate_stock_data(df, 200 if period=="max" else 50): return df
        
        # Bypass for new stocks/issues
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={period}"
        r = requests.get(url, headers={'User-Agent': get_random_agent()}, timeout=5)
        data = r.json()
        if "chart" in data and data["chart"]["result"]:
            res = data["chart"]["result"][0]
            quote = res["indicators"]["quote"][0]
            df = pd.DataFrame({"Open": quote["open"], "High": quote["high"], "Low": quote["low"], "Close": quote["close"], "Volume": quote["volume"]}, index=pd.to_datetime(res["timestamp"], unit="s"))
            return df.dropna()
    except: pass
    return None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {"PE": info.get('trailingPE', 0), "PB": info.get('priceToBook', 0), "Mkt Cap": info.get('marketCap', 0)}
    except: return None

def get_market_benchmark(ticker):
    ticker = str(ticker).upper()
    if ticker in US_SECTOR_MAP: return US_SECTOR_MAP[ticker]
    return "^HSI" if ticker.endswith(".HK") else "QQQ"

def fetch_pair_data(ticker):
    return fetch_data_safe(ticker, "2y"), fetch_data_safe(get_market_benchmark(ticker), "2y")

# --- 4. 算法与分析 (V10.0 Logic Restored) ---
def optimize_display_data(df, max_points=800):
    return df.tail(max_points).copy() if len(df) > max_points else df

def calculate_rsi_vectorized_fixed(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
    gain[0] = 0; loss[0] = 0
    avg_gain = np.zeros(len(series)); avg_loss = np.zeros(len(series))
    if len(series) > period:
        avg_gain[period] = np.mean(gain[1:period+1]); avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period + 1, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return pd.Series(rsi, index=series.index)

def calculate_percentile_rank_fixed(current_val, history_series, lookback=252):
    if history_series is None or len(history_series) < 20: return 50
    clean = history_series.dropna().tail(lookback)
    return (clean < current_val).mean() * 100 if len(clean) > 0 else 50

def calculate_advanced_metrics(df, bench_df, atr_multiplier=2.5):
    try:
        df = df.copy()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['Bias50'] = (df['Close'] - df['SMA50']) / df['SMA50'] * 100
        
        # ATR Logic
        high_low = df['High'] - df['Low']
        high_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_prev = np.abs(df['Low'] - df['Close'].shift(1))
        tr = np.maximum(high_low, np.maximum(high_prev, low_prev))
        df['ATR'] = tr.rolling(14).mean()
        
        df['Stop_Loss_Long'] = (df['Close'].rolling(20).max() - df['ATR'] * atr_multiplier).clip(lower=df['Close']*0.7)
        
        df['RSI'] = calculate_rsi_vectorized_fixed(df['Close'], 14)
        
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
        df['RVol'] = df['Volume'] / df['Vol_SMA20']
        
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(20).std()
        df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(20).std()

        if bench_df is not None and not bench_df.empty:
            common = df.index.intersection(bench_df.index)
            if len(common) > 20:
                df['RS_Raw'] = (df.loc[common,'Close'].pct_change() - bench_df.loc[common,'Close'].pct_change()).fillna(0)
                df['RS_Trend'] = df['RS_Raw'].rolling(20).mean() * 100
                df['RS_Momentum'] = df['RS_Trend'] - df['RS_Trend'].shift(5)
    except: pass
    return df.dropna()

# [V10.10] 严格复刻 V10.0 的核心评分逻辑 (Uncapped Score)
def calculate_core_score(row, df_hist_slice=None, benchmark_name="Benchmark"):
    score = 50
    reasons = []
    
    rsi_rank = 50
    rvol_rank = 50
    bias_rank = 50
    
    if df_hist_slice is not None and len(df_hist_slice) > 20:
        try:
            rsi_rank = calculate_percentile_rank_fixed(row['RSI'], df_hist_slice['RSI'], 252)
            rvol_rank = calculate_percentile_rank_fixed(row.get('RVol', 1.0), df_hist_slice['RVol'], 252)
            bias_rank = calculate_percentile_rank_fixed(row.get('Bias50', 0), df_hist_slice['Bias50'], 252)
        except: pass

    # 1. 趋势 (Trend) - 35%
    trend_score = 0
    if row['Close'] > row.get('SMA50', 0): trend_score += 10
    if row['Close'] > row.get('SMA200', 0): trend_score += 15
    if row.get('SMA50', 0) > row.get('SMA200', 0): trend_score += 10 
    score += trend_score
    if trend_score >= 25: reasons.append("多头排列")
    
    # 2. 动量 (Momentum) - 30%
    mom_score = 0
    if row.get('MACD', 0) > row.get('MACD_Signal', 0): mom_score += 10
    
    rsi_val = row.get('RSI', 50)
    if 50 < rsi_val <= 75: mom_score += 15 
    elif rsi_val > 75: mom_score += 5 
    if 40 < rsi_rank < 80: mom_score += 5

    score += mom_score
    if mom_score >= 15: reasons.append("动能充沛")
    
    # 3. 相对强度 (RS) - 20%
    rs_score = 0
    if row.get('RS_Trend', 0) > 0: rs_score += 10
    if row.get('RS_Momentum', 0) > 0: rs_score += 10
    score += rs_score
    if rs_score >= 15: reasons.append(f"领跑 {benchmark_name}")
    
    # 4. 量能 (Volume) - 15%
    vol_score = 0
    if rvol_rank > 80: 
        vol_score += 15
        reasons.append("放量攻击")
    elif row.get('RVol', 1.0) > 1.0:
        vol_score += 5
    score += vol_score
    
    # 5. 共振加成
    resonance_bonus = 0
    if trend_score >= 30 and mom_score >= 20 and vol_score >= 10:
        resonance_bonus += 30 
        reasons.append("🔥 主升浪共振")
    score += resonance_bonus

    # 6. 风险扣分
    penalty = 0
    if bias_rank > 95: 
        penalty -= 15
        reasons.append("⚠ 短期乖离过大")
    if rsi_val > 85: 
        penalty -= 10
        reasons.append("⚠ RSI超买")
    score += penalty
    
    # [V10.10] 移除 120 分上限，还原 V10.0 逻辑
    
    return score, reasons

def get_status_emoji(score):
    if score >= 110: return "🟣" 
    if score >= 100: return "🟢" 
    if score >= 90: return "🥎" 
    if score >= 75: return "🟡" 
    if score >= 45: return "🟠" 
    return "🔴"                 

def us_market_advice(curr, atr_mult, benchmark_name, df_hist=None):
    advice = {"status": "", "action": "", "reason": [], "metaphor": "", "score_mod": 0}
    price = curr['Close']
    stop_loss = curr.get('Stop_Loss_Long', price * 0.9)
    
    if price <= stop_loss:
        advice.update({
            "status": "🔴 硬限幅切断 (Hard Clip)",
            "action": "❌ 坚决离场 / 止损",
            "metaphor": f"触发 ATR 风控。为保住本金，必须切断信号。",
            "reason": [f"跌破 ${stop_loss:.2f} 止损线"]
        })
        return advice
        
    score, reasons = calculate_core_score(curr, df_hist, benchmark_name)
    advice["score_mod"] = score
    advice["reason"] = reasons if reasons else ["技术面中性"]
    
    if score >= 110:
        advice.update({"status": "🟣 紫色传说 (Ultra)", "action": "🚀🚀 坚定锁仓 / 享受主升浪", "metaphor": "完美共振，动态范围突破天际！"})
    elif score >= 100:
        advice.update({"status": "🟢 黄金买点 (Golden)", "action": "🚀 积极做多 / 加仓", "metaphor": "信号极强，能量充沛。"})
    elif score >= 90:
        advice.update({"status": "🥎 趋势良好 (Strong)", "action": "✅ 持有 / 适度加仓", "metaphor": "信号清晰，信噪比高。"})
    elif score >= 75:
        advice.update({"status": "🟡 震荡整理 (Linear)", "action": "👀 观望 / 保持仓位", "metaphor": "线性区间，无明显失真。"})
    elif score >= 45:
        advice.update({"status": "🟠 动能减弱 (Weak)", "action": "🛡️ 减仓 / 提高警惕", "metaphor": "高频衰减，声音变闷。"})
    else:
        advice.update({"status": "🔴 风险区域 (Risk)", "action": "❌ 离场 / 避险", "metaphor": "技术面走弱，底噪过大。"})
    
    return advice

# --- 5. 微信推送模块 ---
def send_wechat_msg(token, title, content):
    if not token: return False
    try:
        requests.post('http://www.pushplus.plus/send', json={"token": token, "title": title, "content": content, "template": "html"}, timeout=3)
        return True
    except: return False

# --- 6. 自动扫描 (信号翻转逻辑 + V10.0 Scoring) ---
def perform_auto_scan(push_token=None, force_refresh=False):
    valid_favs = [t for t in st.session_state['settings']['favorites'] if t]
    if not valid_favs: return

    if force_refresh: st.cache_data.clear()
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    if 'last_push_date' not in st.session_state or st.session_state['last_push_date'] != today_str:
        st.session_state['pushed_today'] = set()
        st.session_state['last_push_date'] = today_str

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    bench_cache = {}
    needed_bench = set([get_market_benchmark(t) for t in valid_favs] + ["QQQ"])
    for b in needed_bench:
        try: bench_cache[b] = fetch_data_safe(b, "2y")
        except: pass
        time.sleep(0.5)

    total = len(valid_favs)
    for i, ticker in enumerate(valid_favs):
        status_text.caption(f"📡 Scanning ({i+1}/{total}): {ticker}...")
        try:
            time.sleep(random.uniform(0.5, 1.2))
            df = fetch_data_safe(ticker, "2y")
            if validate_stock_data(df, 200):
                my_bench = get_market_benchmark(ticker)
                
                df = calculate_advanced_metrics(df, bench_cache.get(my_bench))
                if not df.empty:
                    curr = df.iloc[-1]
                    # [V10.10] 使用还原后的 V10.0 评分逻辑
                    score, reasons = calculate_core_score(curr, df, my_bench)
                    new_status = get_status_emoji(score)
                    
                    last_status = st.session_state['last_known_status'].get(ticker)
                    st.session_state['ticker_status'][ticker] = new_status
                    
                    # 信号翻转推送策略 (Signal Flip Strategy)
                    if push_token and last_status and (new_status != last_status):
                        msg = f"<b>🔄 信号翻转: {ticker}</b><br>从 {last_status} 变更为 {new_status}<br>现价: ${curr['Close']:.2f}<br>评分: {score}<br>理由: {', '.join(reasons)}"
                        send_wechat_msg(push_token, f"{new_status} {ticker} 变盘提醒", msg)
                    
                    st.session_state['last_known_status'][ticker] = new_status
            else:
                st.session_state['ticker_status'][ticker] = "⚪"
        except: pass
        progress_bar.progress((i+1)/total)
        
    status_text.empty()
    progress_bar.empty()
    st.session_state['scan_executed'] = True
    st.session_state['last_scan_time'] = datetime.now()

# --- 7. 回测模块 ---
def calculate_max_drawdown(equity_curve):
    s = pd.Series(equity_curve)
    drawdown = (s - s.cummax()) / s.cummax()
    return drawdown.min() * 100 if not s.empty else 0

def calculate_performance_metrics(returns):
    if len(returns) < 2: return {}
    days = len(returns)
    total = (1 + returns).prod() - 1
    cagr = ((1 + total) ** (252/days) - 1) * 100 if days > 0 else 0
    sharpe = np.sqrt(252) * (returns.mean() - 0.02/252) / returns.std() if returns.std() > 0 else 0
    win = (returns > 0).mean() * 100
    return {'CAGR': cagr, 'Sharpe': sharpe, 'WinRate': win}

def run_backtest_dynamic(ticker, years=10, initial_capital=100000, atr_mult=3.0):
    try:
        period = "max" if years > 2 else "5y"
        df = fetch_data_safe(ticker, period)
        bench = fetch_data_safe(get_market_benchmark(ticker), period)
        if df is None or len(df) < 200: return None, 0
        
        df = calculate_advanced_metrics(df, bench, atr_mult)
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        if df.index[0] > cutoff: cutoff = df.index[0]
        df_bt = df[df.index >= cutoff].copy()
        
        cash = initial_capital
        pos = 0
        eq_strat = []
        eq_bh = []
        shares_bh = initial_capital / df_bt['Close'].iloc[0]
        
        in_mkt = False
        stop = 0
        high = 0
        
        for i in range(len(df_bt)):
            curr = df_bt.iloc[i]
            price = curr['Close']
            eq_bh.append(shares_bh * price)
            
            if i < 1: 
                eq_strat.append(cash)
                continue
                
            score, _ = calculate_core_score(curr, df_bt.iloc[:i], "Bench")
            
            if in_mkt:
                if price > high:
                    high = price
                    new_stop = high - curr['ATR'] * atr_mult
                    if new_stop > stop: stop = new_stop
                if price < stop or (score < 45 and curr['MACD'] < curr['MACD_Signal']):
                    cash = pos * price * 0.999 
                    pos = 0
                    in_mkt = False
            else:
                cash *= (1 + 0.03/252) 
                if score >= 80:
                    pos = (cash * 0.999) / price
                    cash = 0
                    in_mkt = True
                    high = price
                    stop = price - curr['ATR'] * atr_mult
            
            eq_strat.append(pos * price if in_mkt else cash)
            
        return pd.DataFrame({'Strategy': eq_strat, 'Buy_Hold': eq_bh}, index=df_bt.index), years
    except: return None, 0

def generate_local_response(prompt, ticker, curr, advice):
    p = prompt.lower()
    if "买" in p or "buy" in p: return f"🤖 {advice['status']} (评分: {advice['score_mod']})"
    if "卖" in p or "sell" in p: return f"🤖 动态止损位: ${curr['Stop_Loss_Long']:.2f}"
    return f"我是 {ticker} 的助理，请问有什么可以帮您？"

# --- UI Render ---
with st.sidebar:
    st.title("🎸 Tod's V10.10")
    st.caption("Golden Master | Cloud")
    
    with st.expander("📡 微信耳返 (Push)", expanded=False):
        wechat_token = st.text_input("PushPlus Token", value=DEFAULT_WECHAT_TOKEN, type="password")
        if wechat_token and st.button("🔔 测试"):
            if send_wechat_msg(wechat_token, "Soundcheck", "System Online"): st.success("OK")
            else: st.error("Fail")

    st.markdown("### 🔄 自动巡航")
    enable_auto_refresh = st.checkbox("沉浸式监控 (60s)", value=DEFAULT_AUTO_PILOT)
    
    countdown_placeholder = st.empty()
    if st.session_state['last_scan_time']:
        st.markdown(f"<div class='status-updated'>✅ 上次刷新: {st.session_state['last_scan_time'].strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

    if st.button("🔄 立即强制刷新"):
        st.cache_data.clear(); st.session_state['scan_executed'] = False; st.session_state['trigger_refresh'] = False; st.rerun()

    if not st.session_state['scan_executed'] or st.session_state.get('trigger_refresh', False):
        token_to_use = wechat_token if enable_auto_refresh else None
        perform_auto_scan(push_token=token_to_use, force_refresh=True)
        st.session_state['trigger_refresh'] = False

    with st.expander("⚙️ 调音台", expanded=False):
        current_ticker = st.session_state['current_ticker']
        current_atr = st.session_state['settings']['atr_params'].get(current_ticker, 2.5)
        new_atr = st.slider(f"{current_ticker} ATR", 1.5, 5.0, float(current_atr), 0.1)
        if new_atr != current_atr:
            st.session_state['settings']['atr_params'][current_ticker] = new_atr
            save_settings(st.session_state['settings'])
            st.success("Saved")

    st.markdown("### 🎹 通道选择")
    favs = st.session_state['settings']['favorites']
    for r in range(5):
        cols = st.columns(4)
        for c in range(4):
            idx = r * 4 + c
            if idx < len(favs):
                t = favs[idx]
                icon = st.session_state['ticker_status'].get(t, "")
                if t:
                    if cols[c].button(f"{icon} {t}", key=f"b_{idx}"):
                        st.session_state['current_ticker'] = t
                        st.rerun()

    with st.expander("🎛️ 通道跳线", expanded=False):
        edited_df = st.data_editor(pd.DataFrame({"Channel": [f"CH {i+1}" for i in range(20)], "Ticker": favs}), hide_index=True)
        if st.button("💾 保存"):
            st.session_state['settings']['favorites'] = [sanitize_ticker(t) for t in edited_df["Ticker"].astype(str).tolist()]
            save_settings(st.session_state['settings'])
            st.rerun()

    st.caption(log_system_status())

ticker = st.session_state['current_ticker']
bench_name = get_market_benchmark(ticker)
st.title(f"{ticker} 频谱深度解析")
st.caption(f"对标: {bench_name}")

try:
    with st.spinner('🎵 调谐中...'):
        df, df_b = fetch_pair_data(ticker)
    
    if df is None or df.empty:
        st.markdown(f"<div class='no-signal'>⚠️ 无信号: {ticker}<br><a href='https://finance.yahoo.com/quote/{ticker}' style='color:#ffc107'>手动验证</a></div>", unsafe_allow_html=True)
    else:
        atr_mult = st.session_state['settings']['atr_params'].get(ticker, 2.5)
        df = calculate_advanced_metrics(df, df_b, atr_mult)
        df_dis = optimize_display_data(df)
        
        if len(df) > 2:
            curr = df.iloc[-1]
            advice = us_market_advice(curr, atr_mult, bench_name, df)
            st.session_state['ticker_status'][ticker] = get_status_emoji(advice['score_mod'])
            
            k1, k2, k3, k4, k5 = st.columns(5)
            
            # [V10.10 Restored] V10.0 的详细 Tooltips 逻辑，这才是你想要的！
            # 1. Price Help
            price_help = f"收盘价: ${curr['Close']:.2f}\n"
            if curr['Close'] > curr['SMA50']: price_help += "✅ >SMA50 (中期强势)\n"
            else: price_help += "⚠️ <SMA50 (中期弱势)\n"
            if curr['Close'] > curr['SMA200']: price_help += "✅ >SMA200 (长期牛市)"
            else: price_help += "⚠️ <SMA200 (长期熊市)"
            
            # 2. RVol Help
            rvol_val = curr['RVol']
            if rvol_val > 1.5: rvol_help = "🔥 爆量: 主力大举介入，波动加剧。"
            elif rvol_val > 1.2: rvol_help = "⚡ 放量: 交易活跃，趋势确认度高。"
            elif rvol_val < 0.8: rvol_help = "🧊 缩量: 市场观望，缺乏方向。"
            else: rvol_help = "温和: 成交量正常。"
            
            # 3. RSI Help
            rsi_val = curr.get('RSI', 50)
            if rsi_val > 75: rsi_help = "🔥 严重超买: 情绪狂热，风险极高。"
            elif rsi_val > 70: rsi_help = "⚠️ 超买区: 短期过热，谨防回调。"
            elif rsi_val < 25: rsi_help = "🧊 严重超卖: 恐慌盘涌出，反弹一触即发。"
            elif rsi_val < 30: rsi_help = "💎 超卖区: 情绪低迷，关注底部机会。"
            elif rsi_val > 50: rsi_help = "🐂 多头区: 买盘占优。"
            else: rsi_help = "🐻 空头区: 卖盘占优。"
            
            # 4. RS Help
            rs_val = curr.get('RS_Momentum', 0)
            if rs_val > 0.1: rs_help = "🚀 显著领跑: 走势大幅强于基准，机构抱团。"
            elif rs_val > 0: rs_help = "✅ 小幅领先: 略强于大盘。"
            else: rs_help = "🐌 跑输大盘: 资金流出或关注度下降。"
            rs_help += f"\n(基准: {bench_name})"
            
            # 5. ATR Help
            stop_p = curr['Stop_Loss_Long']
            dist = (curr['Close'] - stop_p) / curr['Close'] * 100
            atr_help = f"🛡️ 移动止损线: ${stop_p:.2f}\n"
            atr_help += f"当前安全垫: {dist:.1f}%\n"
            atr_help += "原理: 价格跌破此线表示波动率异常放大，趋势可能反转。"

            k1.metric("现价", f"${curr['Close']:.2f}", f"{(curr['Close']-df.iloc[-2]['Close'])/df.iloc[-2]['Close']*100:.2f}%", help=price_help)
            k2.metric("RVol", f"{curr['RVol']:.2f}x", help=rvol_help)
            k3.metric("RSI", f"{curr['RSI']:.1f}", help=rsi_help)
            k4.metric("RS动量", f"{curr.get('RS_Momentum',0):.2f}", help=rs_help)
            k5.metric("ATR止损", f"${curr['Stop_Loss_Long']:.2f}", help=atr_help)
            
            status_color = "#9400D3" if "🟣" in advice['status'] else "#d4edda" if "🟢" in advice['status'] else "#f8d7da"
            text_color = "white" if "🟣" in advice['status'] else "black"
            
            st.markdown(f"""
            <div class="advice-box" style="background-color: {status_color}; border-left: 5px solid #666; color: {text_color};">
                <h3 style="margin:0;">{advice['status']} (评分: {advice['score_mod']})</h3>
                <p style="margin-top:10px;"><b>🔉 声学隐喻：</b> {advice['metaphor']}</p>
                <p><b>👉 操作指令：</b> <strong>{advice['action']}</strong></p>
                <p style="font-size:0.9em; opacity:0.8;"><i>🔍 依据: {', '.join(advice['reason'])}</i></p>
            </div>
            """, unsafe_allow_html=True)
            
            if wechat_token and st.button("📱 手动发送战报"):
                msg = f"<b>{ticker}</b><br>状态: {advice['status']}<br>现价: ${curr['Close']:.2f}"
                if send_wechat_msg(wechat_token, f"{ticker} 战报", msg): st.toast("✅ Sent")
                else: st.error("Fail")
            
            if "❌" not in advice['action']:
                with st.expander("🧮 仓位增益 & 目标锁定", expanded=True):
                    c_g1, c_g2 = st.columns(2)
                    cap = c_g1.number_input("投入", 200000, step=10000)
                    rr = c_g2.number_input("盈亏比", 2.0, step=0.5)
                    risk = st.slider("风控%", 0.5, 5.0, 2.0)
                    
                    risk_share = max(0.01, curr['Close'] - curr['Stop_Loss_Long'])
                    shares = int((cap * risk/100) / max(risk_share, curr['ATR']*2))
                    target = curr['Close'] + risk_share * rr
                    c_g1.info(f"建议仓位: {shares} 股"); c_g2.success(f"止盈目标: ${target:.2f}")

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df_dis.index, open=df_dis['Open'], high=df_dis['High'], low=df_dis['Low'], close=df_dis['Close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_dis.index, y=df_dis['Stop_Loss_Long'], line=dict(color='purple', dash='dash')), row=1, col=1)
            if "❌" not in advice['action']:
                fig.add_hline(y=target, line_dash="dot", line_color="green", row=1, col=1)
            fig.add_trace(go.Bar(x=df_dis.index, y=df_dis['Volume'], marker_color='grey'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_dis.index, y=df_dis['RSI'], line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1); fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
            fig.add_trace(go.Scatter(x=df_dis.index, y=df_dis['MACD']), row=4, col=1)
            fig.add_trace(go.Scatter(x=df_dis.index, y=df_dis['MACD_Signal']), row=4, col=1)
            fig.add_trace(go.Bar(x=df_dis.index, y=df_dis['MACD_Hist']), row=4, col=1)
            fig.update_layout(height=800, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # [V10.10] 找回丢失的 AI 助理
            st.markdown("---")
            with st.expander(f"💬 智能助理 ({ticker} 专属)", expanded=True):
                if st.session_state.get('chat_context_ticker') != ticker:
                    st.session_state['messages'] = []
                    st.session_state['chat_context_ticker'] = ticker
                    st.session_state['messages'].append({"role": "assistant", "content": f"你好 Tod，我是 {ticker} 的专属分析助理。当前评分 **{advice['score_mod']}**。有什么指令？"})
                
                for msg in st.session_state['messages']:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                
                if prompt := st.chat_input(f"关于 {ticker}..."):
                    with st.chat_message("user"): st.markdown(prompt)
                    st.session_state['messages'].append({"role": "user", "content": prompt})
                    with st.chat_message("assistant"):
                        with st.spinner("思考中..."):
                            time.sleep(0.5) 
                            response = generate_local_response(prompt, ticker, curr, advice)
                            st.markdown(response)
                    st.session_state['messages'].append({"role": "assistant", "content": response})

            with st.expander("⏳ 10年时光机 (Backtest Lab)", expanded=False):
                bt_years = st.selectbox("周期", [1, 3, 5, 10], index=2)
                if st.button("🚀 运行仿真"):
                    with st.spinner(f"正在对标 {bench_name} 进行回测..."):
                        res_df, real_years = run_backtest_dynamic(ticker, years=bt_years, atr_mult=atr_mult)
                        if res_df is not None:
                            ret = (res['Strategy'].iloc[-1]-100000)/1000
                            bh = (res['Buy_Hold'].iloc[-1]-100000)/1000
                            dd = calculate_max_drawdown(res['Strategy'])
                            st.write(f"策略收益: {ret:.1f}% (vs {bh:.1f}%) | 最大回撤: {dd:.1f}%")
                            st.line_chart(res)
                        else:
                            st.error("回测数据不足或异常")

except Exception as e:
    st.error(f"Render Error: {e}")

if enable_auto_refresh:
    try:
        with countdown_placeholder.container():
            t_ph = st.empty(); p_ph = st.progress(100)
            for i in range(60, 0, -1):
                t_ph.markdown(f"<div class='countdown-box'>⏳ 下次刷新: {i} s</div>", unsafe_allow_html=True)
                p_ph.progress(i/60); time.sleep(1)
            st.session_state['trigger_refresh'] = True; st.rerun()
    except: pass
