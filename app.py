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

# --- 0. 生产环境初始化 (Production Setup) ---

logging.basicConfig(
    filename='stock_studio.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 软启动：尝试导入 psutil
try:
    import psutil
except ImportError:
    psutil = None

def check_environment():
    """ 开机自检 (Power-On Self-Test) """
    issues = []
    try:
        import yfinance
    except ImportError:
        issues.append("❌ yfinance 未安装")
    return issues

def log_system_status():
    """ 硬件状态监控 (System Monitor) """
    if psutil is None: return "Monitor: Bypass"
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"✅ Mem: {memory_mb:.1f}MB"
    except:
        return "Monitor Fail"

# --- 1. 页面配置 (UI Design) ---
st.set_page_config(
    page_title="Tod's Studio V10.6 (Memory Fixed)",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem; }
        .stMetric { min-height: 80px; }
        h1 { font-size: 1.5rem !important; }
    }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; }
    .advice-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stProgress > div > div > div > div { background-color: #28a745; }
    .stButton button { width: 100%; font-weight: 600; }
    
    /* 倒计时样式 */
    .countdown-box {
        font-family: 'Courier New', monospace; font-size: 1.1em; color: #d63384;
        font-weight: bold; text-align: center; padding: 8px;
        border: 1px dashed #d63384; border-radius: 5px; margin-top: 10px; background-color: #fff0f6;
    }
    .status-updated {
        color: #198754; font-size: 0.8em; text-align: center; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    
    .no-signal { background-color: #343a40; color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 状态初始化
if 'ticker_status' not in st.session_state: st.session_state['ticker_status'] = {}
if 'scan_executed' not in st.session_state: st.session_state['scan_executed'] = False
if 'last_scan_time' not in st.session_state: st.session_state['last_scan_time'] = None
if 'trigger_refresh' not in st.session_state: st.session_state['trigger_refresh'] = False
if 'pushed_today' not in st.session_state: st.session_state['pushed_today'] = set() 

# --- 2. 核心记忆系统 (Settings) ---
SETTINGS_FILE = 'stock_settings.json'
US_SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SOXX", "AMD": "SOXX", "TSM": "SOXX", "AVGO": "SOXX",
    "TSLA": "XLY", "AMZN": "XLY", "NFLX": "XLY", 
    "GOOG": "XLC", "GOOGL": "XLC", "META": "XLC",
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
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except: return False

if 'settings' not in st.session_state: st.session_state['settings'] = load_settings()
if 'current_ticker' not in st.session_state: 
    first_valid = next((x for x in st.session_state['settings']['favorites'] if x), "QQQ")
    st.session_state['current_ticker'] = first_valid

# --- 3. 数据层 (Robust Data) ---
def sanitize_ticker(ticker):
    if not ticker: return ""
    ticker = re.sub(r'[^A-Za-z0-9\.\^]', '', str(ticker).upper())
    return ticker[:20] 

def fix_china_ticker(ticker):
    t = sanitize_ticker(ticker).strip()
    if t.isdigit() and len(t) == 5: return f"{t}.HK"
    return t

def get_random_agent():
    agents = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
    ]
    return random.choice(agents)

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
        
        # Bypass Logic
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={period}"
        headers = {'User-Agent': get_random_agent()}
        r = requests.get(url, headers=headers, timeout=5)
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
    if ticker.endswith(".HK"): return "^HSI"
    return "QQQ"

@st.cache_data(ttl=1800, show_spinner=False)
def get_benchmark_data(target_ticker):
    bench_ticker = get_market_benchmark(target_ticker)
    return fetch_data_safe(bench_ticker, period="2y")

def fetch_pair_data(ticker):
    bench_ticker = get_market_benchmark(ticker)
    df_stock = fetch_data_safe(ticker, "2y")
    time.sleep(random.uniform(0.5, 1.2))
    df_bench = fetch_data_safe(bench_ticker, "2y")
    return df_stock, df_bench

# --- 4. 算法与分析 (DSP) ---
# [修复点] 之前丢失的 optimize_display_data 函数
def optimize_display_data(df, max_points=800):
    if len(df) > max_points:
        return df.tail(max_points).copy()
    return df

def calculate_rsi_vectorized_fixed(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain[0] = 0; loss[0] = 0
    avg_gain = np.zeros(len(series)); avg_loss = np.zeros(len(series))
    if len(series) > period:
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
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

def calculate_advanced_metrics(df, bench_df, atr_mult=2.5):
    try:
        df = df.copy()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['Stop_Loss_Long'] = (df['Close'].rolling(20).max() - df['ATR'] * atr_mult).clip(lower=df['Close']*0.7)
        df['RSI'] = calculate_rsi_vectorized_fixed(df['Close'])
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
        df['RVol'] = df['Volume'] / df['Vol_SMA20']
        
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(20).std()
        df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(20).std()

        if bench_df is not None and not bench_df.empty:
            common = df.index.intersection(bench_df.index)
            if len(common) > 20:
                df['RS_Raw'] = (df.loc[common,'Close'].pct_change() - bench_df.loc[common,'Close'].pct_change()).fillna(0)
                df['RS_Momentum'] = df['RS_Raw'].rolling(20).mean() * 100
    except: pass
    return df.dropna()

def calculate_core_score(row, df_hist, bench_ticker="Benchmark"):
    score = 50
    reasons = []
    
    # 趋势
    trend = 0
    if row['Close'] > row['SMA50']: trend += 10
    if row['Close'] > row['SMA200']: trend += 15
    if row['SMA50'] > row['SMA200']: trend += 10; reasons.append("多头排列")
    score += trend
    
    # 动量
    mom = 0
    if row['MACD'] > row['MACD_Signal']: mom += 10
    rsi = row['RSI']
    if 50 < rsi < 75: mom += 15
    score += mom
    if mom >= 15: reasons.append("动能充沛")
    
    # 量能 & RS
    if row.get('RVol', 0) > 1.2: score += 5; reasons.append("放量")
    if row.get('RS_Momentum', 0) > 0: score += 10; reasons.append(f"领跑{bench_ticker}")
    
    # 风险
    if rsi > 80: score -= 10; reasons.append("⚠ RSI超买")
    
    # 共振
    if trend >= 30 and mom >= 20: score += 30; reasons.append("🔥 主升浪共振")
    
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
    url = 'http://www.pushplus.plus/send'
    data = {"token": token, "title": title, "content": content, "template": "html"}
    try:
        requests.post(url, json=data, timeout=3)
        return True
    except: return False

# --- 6. 自动扫描与UI ---
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
    
    # 预加载基准
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
                    score, reasons = calculate_core_score(curr, df, my_bench)
                    status_icon = get_status_emoji(score)
                    st.session_state['ticker_status'][ticker] = status_icon
                    
                    if push_token and score >= 100 and (ticker not in st.session_state['pushed_today']):
                        msg = f"<b>🚀 {ticker} 信号触发</b><br>状态: {status_icon}<br>现价: ${curr['Close']:.2f}<br>理由: {', '.join(reasons)}"
                        send_wechat_msg(push_token, f"{status_icon} {ticker} 信号", msg)
                        st.session_state['pushed_today'].add(ticker)
            else:
                st.session_state['ticker_status'][ticker] = "⚪"
        except:
            st.session_state['ticker_status'][ticker] = "⚪"
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
                    cash = pos * price * 0.999 # Commission
                    pos = 0
                    in_mkt = False
            else:
                cash *= (1 + 0.03/252) # Interest
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
    st.title("🎸 Tod's V10.6")
    st.caption("Memory Fixed | Auto-Pilot")
    
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
            st.session_state['settings']['favorites'] = [fix_china_ticker(t) if t else "" for t in edited_df["Ticker"].astype(str).tolist()]
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
            k1.metric("现价", f"${curr['Close']:.2f}", f"{(curr['Close']-df.iloc[-2]['Close'])/df.iloc[-2]['Close']*100:.2f}%")
            k2.metric("RVol", f"{curr['RVol']:.2f}x")
            k3.metric("RSI", f"{curr['RSI']:.1f}")
            k4.metric("RS动量", f"{curr.get('RS_Momentum',0):.2f}")
            k5.metric("ATR止损", f"${curr['Stop_Loss_Long']:.2f}")
            
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
                with st.expander("🧮 仓位增益", expanded=True):
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
            
            with st.expander("⏳ 回测", expanded=False):
                if st.button("🚀 运行"):
                    res, _ = run_backtest_dynamic(ticker, 10, 100000, atr_mult)
                    if res is not None:
                        ret = (res['Strategy'].iloc[-1]-100000)/1000
                        bh = (res['Buy_Hold'].iloc[-1]-100000)/1000
                        dd = calculate_max_drawdown(res['Strategy'])
                        st.write(f"策略收益: {ret:.1f}% (vs {bh:.1f}%) | 最大回撤: {dd:.1f}%")
                        st.line_chart(res)

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
