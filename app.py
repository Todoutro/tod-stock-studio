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
    """ 开机自检 (Power-On Self-Test) - V2 (柔和版) """
    issues = []
    try:
        import yfinance
    except ImportError:
        issues.append("❌ yfinance 未安装")
    
    if psutil is None:
        logging.warning("psutil not found, memory monitoring disabled.")
    
    proxy_set = os.environ.get("http_proxy") or os.environ.get("https_proxy")
    if not proxy_set:
        try:
            socket.create_connection(("finance.yahoo.com", 443), timeout=5)
        except:
            try:
                socket.create_connection(("www.google.com", 80), timeout=3)
            except:
                issues.append("⚠️ 网络连接不稳定 (Yahoo/Google 不可达)")
            
    return issues

def log_system_status():
    """ 硬件状态监控 (System Monitor) """
    if psutil is None:
        return "Monitor: Bypass (No Lib)"
        
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        status_msg = f"System Status - Memory: {memory_mb:.1f}MB"
        logging.info(status_msg)
        if memory_mb > 800: return f"⚠️ High Load: {memory_mb:.1f}MB"
        return f"✅ Memory: {memory_mb:.1f}MB"
    except:
        return "Monitor Fail"

# --- 1. 页面配置 (UI Design) ---
st.set_page_config(
    page_title="Tod's Studio V10.5 (Tour Edition)",
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
    h3 { font-family: 'Helvetica Neue', sans-serif; }
    
    /* 优化矩阵按钮样式 */
    .stButton button {
        width: 100%;
        padding: 0.5rem 0.25rem;
        line-height: 1.3;
        font-weight: 600;
        border: 1px solid #dee2e6;
        transition: all 0.2s;
    }
    .stButton button:hover {
        border-color: #0d6efd;
        background-color: #f8f9fa;
    }
    
    /* 聊天气泡样式微调 */
    .stChatMessage {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    
    /* 倒计时样式 */
    .countdown-box {
        font-family: 'Courier New', monospace;
        font-size: 1.1em;
        color: #d63384;
        font-weight: bold;
        text-align: center;
        padding: 8px;
        border: 1px dashed #d63384;
        border-radius: 5px;
        margin-top: 10px;
        background-color: #fff0f6;
    }
    
    /* 状态更新闪烁动画 */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .status-updated {
        color: #198754;
        font-size: 0.8em;
        text-align: center;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

if 'env_checked' not in st.session_state:
    env_issues = check_environment()
    if env_issues: st.error(f"系统自检报告: {', '.join(env_issues)}")
    st.session_state['env_checked'] = True

# 初始化信号状态缓存 (Signal Memory)
if 'ticker_status' not in st.session_state:
    st.session_state['ticker_status'] = {}
# 扫描锁
if 'scan_executed' not in st.session_state:
    st.session_state['scan_executed'] = False
# 上次扫描时间
if 'last_scan_time' not in st.session_state:
    st.session_state['last_scan_time'] = None
# [V9.7] 自动刷新扳机 (Trigger)
if 'trigger_refresh' not in st.session_state:
    st.session_state['trigger_refresh'] = False
# [V10.5] 推送去重
if 'pushed_today' not in st.session_state:
    st.session_state['pushed_today'] = set()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'chat_context_ticker' not in st.session_state:
    st.session_state['chat_context_ticker'] = ""

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
        "portfolio": ["TSLA", "NVDA", "PDD", "QQQ", "AAPL", "COIN", "MSTR", "VOO", "BABA", "AMD", "MSFT", "SMH"], 
        "atr_params": {
            "TSLA": 3.2, "MSTR": 3.5, "COIN": 3.3, 
            "NVDA": 2.8, "AMD": 2.8, "PDD": 2.8,   
            "AAPL": 2.2, "MSFT": 2.0, "GOOG": 2.0, "VOO": 1.8 
        },
        "favorites": ["TSLA", "NVDA", "MSTR", "COIN", "QQQ", "SMH", "AAPL", "AMD", "AMZN", "GOOG", "", "", "", "", "", "", "", "", "", ""] 
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                favs = settings.get("favorites", [])
                while len(favs) < 20:
                    favs.append("")
                settings["favorites"] = favs[:20] 
                return settings
        except:
            return default_settings
    return default_settings

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False

if 'settings' not in st.session_state: st.session_state['settings'] = load_settings()
if 'current_ticker' not in st.session_state: 
    first_valid = next((x for x in st.session_state['settings']['favorites'] if x), "QQQ")
    st.session_state['current_ticker'] = first_valid

# --- 3. 数据层 (Robust Data Layer V5 - Anti-Block) ---
def sanitize_ticker(ticker):
    if not ticker: return ""
    ticker = re.sub(r'[^A-Za-z0-9\.\^]', '', str(ticker).upper())
    return ticker[:20] 

def fix_china_ticker(ticker):
    t = sanitize_ticker(ticker).strip()
    if t.isdigit() and len(t) == 5: return f"{t}.HK"
    return t

def get_random_agent():
    """ [V9.5] 随机 User-Agent 防止风控 """
    agents = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
    ]
    return random.choice(agents)

def validate_stock_data(df, min_days=50):
    if df is None or len(df) < min_days: 
        return False
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols): 
        return False
    
    price_cols = ['Open', 'High', 'Low', 'Close']
    if (df[price_cols] <= 0).any().any(): 
        return False
    
    if df.index.duplicated().any():
        return False
        
    if df['Close'].isnull().sum() > len(df) * 0.2: 
        return False
        
    return True

def fetch_via_direct_api(ticker, period="2y"):
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d"}
        
        if period == "max":
            params["period1"] = 0 
            params["period2"] = int(time.time()) 
        else:
            range_map = {"1y": "1y", "2y": "2y", "5y": "5y"}
            params["range"] = range_map.get(period, "2y")

        headers = { 'User-Agent': get_random_agent() }
        
        http_proxy = os.environ.get("http_proxy")
        https_proxy = os.environ.get("https_proxy")
        proxies = {"https": https_proxy, "http": http_proxy} if http_proxy else None
        
        r = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=10)
        data = r.json()
        
        if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
            res = data["chart"]["result"][0]
            timestamps = res["timestamp"]
            quote = res["indicators"]["quote"][0]
            
            df = pd.DataFrame({
                "Open": quote["open"], "High": quote["high"],
                "Low": quote["low"], "Close": quote["close"],
                "Volume": quote["volume"]
            }, index=pd.to_datetime(timestamps, unit="s"))
            
            df = df.dropna()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
    except Exception as e:
        return None
    return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_data_safe(ticker, period="2y"):
    ticker = fix_china_ticker(ticker)
    
    min_days_required = 200 if period == "max" else 50
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.5, 1.2))
            
            stock = yf.Ticker(ticker) 
            df = stock.history(period=period, interval="1d", auto_adjust=False)
            
            if validate_stock_data(df, min_days=min_days_required): 
                return df
                
            logging.info(f"yfinance incomplete/failed for {ticker}, switching to Direct API...")
            time.sleep(0.5) 
            df_direct = fetch_via_direct_api(ticker, period)
            if validate_stock_data(df_direct, min_days=min_days_required):
                return df_direct
                
        except Exception as e:
            wait_time = 1
            logging.warning(f"Retry {ticker}: {e}. Waiting {wait_time}s")
            time.sleep(wait_time)
            
    return None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    try:
        time.sleep(random.uniform(0.5, 1.0)) 
        t = yf.Ticker(ticker) 
        info = t.info
        return {
            "PE": info.get('trailingPE', 0),
            "Fwd PE": info.get('forwardPE', 0),
            "PB": info.get('priceToBook', 0),
            "Mkt Cap": info.get('marketCap', 0)
        }
    except:
        return None

def get_market_benchmark(ticker):
    ticker = str(ticker).upper()
    if ticker in US_SECTOR_MAP: return US_SECTOR_MAP[ticker]
    if ticker.endswith(".HK") or ticker.isdigit(): return "^HSI"   
    elif ticker.endswith(".SS") or ticker.endswith(".SZ"): return "000300.SS" 
    elif ticker.endswith(".T"): return "^N225" 
    else: return "QQQ" 

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

# --- 4. 算法层 (Master Calibration) ---

# [V9.4.1] 向量化 RSI (Standard Wilder's Smoothing)
def calculate_rsi_vectorized_fixed(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain[0] = 0
    loss[0] = 0
    
    avg_gain = np.zeros(len(series))
    avg_loss = np.zeros(len(series))
    
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

def calculate_advanced_metrics(df, benchmark_df=None, atr_multiplier=2.5):
    try:
        df = df.copy()
        
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['Bias50'] = (df['Close'] - df['SMA50']) / df['SMA50'] * 100
        
        high_low = df['High'] - df['Low']
        high_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_prev = np.abs(df['Low'] - df['Close'].shift(1))
        tr = np.maximum(high_low, np.maximum(high_prev, low_prev))
        df['ATR'] = tr.rolling(14).mean()
        
        rolling_high = df['Close'].rolling(20).max()
        df['Stop_Loss_Long'] = rolling_high - (df['ATR'] * atr_multiplier)
        df['Stop_Loss_Long'] = df['Stop_Loss_Long'].clip(lower=df['Close'] * 0.7, upper=df['Close'] * 0.99)
        
        df['RSI'] = calculate_rsi_vectorized_fixed(df['Close'], 14)
        
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
        df['RVol'] = df['Volume'] / df['Vol_SMA20'] 
        
        if benchmark_df is not None and not benchmark_df.empty and len(benchmark_df) > 20:
            common_idx = df.index.intersection(benchmark_df.index)
            if len(common_idx) > 20:
                try:
                    aligned_stock = df.loc[common_idx, 'Close']
                    aligned_bench = benchmark_df.loc[common_idx, 'Close']
                    stock_ret = aligned_stock.pct_change().fillna(0)
                    bench_ret = aligned_bench.pct_change().fillna(0)
                    rs_raw = stock_ret - bench_ret
                    df['RS_Raw'] = rs_raw.reindex(df.index, fill_value=0)
                    df['RS_Trend'] = df['RS_Raw'].rolling(20, min_periods=1).mean() * 100
                    df['RS_Momentum'] = df['RS_Trend'] - df['RS_Trend'].shift(5)
                except:
                    df['RS_Trend'] = 0; df['RS_Momentum'] = 0
            else:
                df['RS_Trend'] = 0; df['RS_Momentum'] = 0
        else:
            df['RS_Trend'] = 0; df['RS_Momentum'] = 0
            
        return df.dropna()
    except Exception as e:
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def calculate_percentile_rank_fixed(current_val, history_series, lookback=252):
    if history_series is None or len(history_series) < 20: return 50
    clean_series = history_series.dropna()
    if len(clean_series) < 10: return 50
    recent_data = clean_series.tail(lookback)
    return (recent_data < current_val).mean() * 100

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

    # 1. 趋势
    trend_score = 0
    if row['Close'] > row.get('SMA50', 0): trend_score += 10
    if row['Close'] > row.get('SMA200', 0): trend_score += 15
    if row.get('SMA50', 0) > row.get('SMA200', 0): trend_score += 10 
    score += trend_score
    if trend_score >= 25: reasons.append("多头排列")
    
    # 2. 动量
    mom_score = 0
    if row.get('MACD', 0) > row.get('MACD_Signal', 0): mom_score += 10
    
    rsi_val = row.get('RSI', 50)
    if 50 < rsi_val <= 75: mom_score += 15 
    elif rsi_val > 75: mom_score += 5 
    if 40 < rsi_rank < 80: mom_score += 5

    score += mom_score
    if mom_score >= 15: reasons.append("动能充沛")
    
    # 3. RS
    rs_score = 0
    if row.get('RS_Trend', 0) > 0: rs_score += 10
    if row.get('RS_Momentum', 0) > 0: rs_score += 10
    score += rs_score
    if rs_score >= 15: reasons.append(f"领跑 {benchmark_name}")
    
    # 4. 量能
    vol_score = 0
    if rvol_rank > 80: 
        vol_score += 15
        reasons.append("放量攻击")
    elif row.get('RVol', 1.0) > 1.0:
        vol_score += 5
    score += vol_score
    
    # 5. 共振
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
    
    return score, reasons

def get_status_emoji(score):
    if score >= 110: return "🟣" 
    if score >= 100: return "🟢" 
    if score >= 90: return "🥎" 
    if score >= 75: return "🟡" 
    if score >= 45: return "🟠" 
    return "🔴"                 

# --- [V10.5] 微信推送模块 (The Wireless Transmitter) ---
def send_wechat_msg(token, title, content):
    """ 发送微信推送 (PushPlus) """
    if not token: return False
    url = 'http://www.pushplus.plus/send'
    data = {
        "token": token,
        "title": title,
        "content": content,
        "template": "html"
    }
    try:
        requests.post(url, json=data, timeout=3)
        return True
    except:
        return False

# 自动扫描逻辑 (Auto Scan - Precision Optimized + WeChat Push)
def perform_auto_scan(push_token=None, force_refresh=False):
    """ [V10.5] 全精度扫描 + 自动推送 """
    valid_favs = [t for t in st.session_state['settings']['favorites'] if t]
    if not valid_favs: return

    # 清除缓存
    if force_refresh:
        st.cache_data.clear()

    # 重置当日推送记录 (如果是新的一天)
    today_str = datetime.now().strftime('%Y-%m-%d')
    if 'last_push_date' not in st.session_state or st.session_state['last_push_date'] != today_str:
        st.session_state['pushed_today'] = set()
        st.session_state['last_push_date'] = today_str

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    # 第一步：预加载所有需要的 Benchmark (去重)
    needed_benchmarks = set()
    needed_benchmarks.add("QQQ") # 默认时钟
    for t in valid_favs:
        needed_benchmarks.add(get_market_benchmark(t))
    
    bench_data_cache = {}
    
    # 批量预热 Benchmark
    for i, b_ticker in enumerate(needed_benchmarks):
        status_text.caption(f"📡 Syncing Clock: {b_ticker}...")
        try:
            bench_data_cache[b_ticker] = fetch_data_safe(b_ticker, "2y")
        except:
            bench_data_cache[b_ticker] = None
        time.sleep(random.uniform(0.5, 1.0)) # 避免拥堵

    # 第二步：精准扫描
    total = len(valid_favs)
    
    for i, ticker in enumerate(valid_favs):
        status_text.caption(f"📡 Scanning ({i+1}/{total}): {ticker}...")
        
        try:
            time.sleep(random.uniform(0.5, 1.5))
            
            df = fetch_data_safe(ticker, "2y")
            if validate_stock_data(df, min_days=200):
                # 使用该股票专属的 Benchmark 进行计算
                my_bench_ticker = get_market_benchmark(ticker)
                my_bench_df = bench_data_cache.get(my_bench_ticker)
                
                df = calculate_advanced_metrics(df, my_bench_df) 
                if not df.empty:
                    curr = df.iloc[-1]
                    # 计算评分
                    score, reasons = calculate_core_score(curr, df, my_bench_ticker) 
                    status_icon = get_status_emoji(score)
                    st.session_state['ticker_status'][ticker] = status_icon
                    
                    # [V10.5] 自动推送触发逻辑
                    # 条件: 1. 设置了Token 2. 评分>=100 (紫色或绿色) 3. 今天没推过
                    if push_token and score >= 100 and (ticker not in st.session_state['pushed_today']):
                        msg = f"<b>🚀 信号触发: {ticker}</b><br>状态: {status_icon} (评分: {score})<br>现价: ${curr['Close']:.2f}<br>理由: {', '.join(reasons)}"
                        send_wechat_msg(push_token, f"{status_icon} {ticker} 信号提醒", msg)
                        st.session_state['pushed_today'].add(ticker)
            else:
                st.session_state['ticker_status'][ticker] = "⚪"
        except:
            st.session_state['ticker_status'][ticker] = "⚪"
            
        progress_bar.progress((i + 1) / total)

    status_text.empty()
    progress_bar.empty()
    st.session_state['scan_executed'] = True
    st.session_state['last_scan_time'] = datetime.now()

# --- 5. 诊断引擎 ---
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

# --- 6. 回测模块 ---
def calculate_max_drawdown(equity_curve):
    if not equity_curve: return 0.0
    s = pd.Series(equity_curve)
    rolling_max = s.cummax()
    drawdown = (s - rolling_max) / rolling_max
    max_dd = drawdown.min()
    return max_dd * 100 

def calculate_performance_metrics(returns):
    if len(returns) < 2: return {}
    metrics = {}
    total_ret = (1 + returns).prod() - 1
    days = len(returns)
    metrics['CAGR'] = ((1 + total_ret) ** (252/days) - 1) * 100 if days > 0 else 0
    excess_ret = returns - 0.02/252
    metrics['Sharpe'] = np.sqrt(252) * excess_ret.mean() / returns.std() if returns.std() > 0 else 0
    metrics['WinRate'] = (returns > 0).mean() * 100
    return metrics

def run_backtest_dynamic(ticker, years=10, initial_capital=100000, atr_mult=3.0, commission=0.001, slippage=0.0005):
    bench_ticker = get_market_benchmark(ticker)
    try:
        fetch_period = "max" if years > 2 else "5y"
        df = fetch_data_safe(ticker, fetch_period)
        time.sleep(random.uniform(0.8, 1.5))
        df_bench = fetch_data_safe(bench_ticker, fetch_period)
        if df is None or df.empty: return None, None
        if df_bench is None or len(df_bench) < 100: df_bench = pd.DataFrame()

        available_days = len(df)
        warmup_days = 200
        tradable_days = available_days - warmup_days
        if tradable_days < 10: return None, None
        
        required_days = years * 250
        real_years = years
        if tradable_days < required_days:
            real_years = tradable_days / 250
            cutoff = df.index[-tradable_days]
        else:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        if df.index[0] > cutoff: cutoff = df.index[0]
            
        df_metrics = calculate_advanced_metrics(df, df_bench if not df_bench.empty else None, atr_mult)
        df_backtest = df_metrics[df_metrics.index >= cutoff].copy()
        if len(df_backtest) < 10: return None, None 
        
        cash = initial_capital
        position = 0
        equity_strategy = []
        equity_bh = []
        buy_hold_shares = initial_capital / (df_backtest['Close'].iloc[0] * (1 + slippage + commission))
        in_market = False
        stop_loss = 0
        highest_price = 0
        
        for i in range(len(df_backtest)):
            curr = df_backtest.iloc[i]
            current_date = curr.name
            price = curr['Close']
            equity_bh.append(buy_hold_shares * price)
            if i < 1: 
                equity_strategy.append(cash)
                continue
            history_up_to_yesterday = df_metrics.loc[:current_date].iloc[:-1]
            score, _ = calculate_core_score(curr, history_up_to_yesterday)
            
            if in_market:
                if price > highest_price:
                    highest_price = price
                    new_stop = highest_price - (curr['ATR'] * atr_mult)
                    if new_stop > stop_loss: stop_loss = new_stop
                hard_stop = price < stop_loss
                soft_exit = (score < 45) and (curr.get('MACD', 0) < curr.get('MACD_Signal', 0))
                if hard_stop or soft_exit:
                    sell_price = price * (1 - slippage)
                    cash = position * sell_price * (1 - commission)
                    position = 0
                    in_market = False
            else:
                cash = cash * (1 + 0.035/252) 
                if (score >= 80) or (score >= 65 and price > curr.get('SMA50', 0)):
                    buy_price = price * (1 + slippage)
                    cost = cash * (1 - commission)
                    position = cost / buy_price
                    cash = 0
                    in_market = True
                    highest_price = price
                    stop_loss = price - (curr['ATR'] * atr_mult)
            equity_strategy.append(cash if not in_market else position * price)
        res_df = pd.DataFrame({'Strategy': equity_strategy, 'Buy_Hold': equity_bh}, index=df_backtest.index)
        return res_df, real_years
    except Exception as e:
        return None, None

def optimize_display_data(df, max_points=800):
    if len(df) > max_points: return df.tail(max_points).copy()
    return df

def generate_local_response(prompt, ticker, curr_data, advice):
    prompt = prompt.lower()
    if "买" in prompt or "buy" in prompt or "入手" in prompt:
        if "🟢" in advice['status'] or "🟣" in advice['status']:
            return f"🤖 **AI 分析:** {ticker} 目前处于 **{advice['status']}**。系统评分为 {advice['score_mod']}，属于高信噪比区域，建议根据资金管理原则分批介入。"
        elif "🔴" in advice['status']:
            return f"🤖 **AI 分析:** 警告！{ticker} 目前触发 **{advice['status']}**，可能会有进一步下行风险。建议暂时观望，不要接飞刀。"
        else:
            return f"🤖 **AI 分析:** {ticker} 目前处于 **{advice['status']}**，趋势不明显。如果你的策略是趋势跟踪，建议等待信号明确。"
    elif "卖" in prompt or "sell" in prompt or "止损" in prompt:
        return f"🤖 **AI 分析:** 当前的 ATR 动态止损位在 **${curr_data['Stop_Loss_Long']:.2f}**。如果收盘价跌破此位置，系统建议无条件离场以保护本金。"
    elif "rsi" in prompt:
        return f"🤖 **AI 数据:** 当前 RSI 为 **{curr_data['RSI']:.1f}**。"
    elif "atr" in prompt:
        return f"🤖 **AI 数据:** 当前 ATR (波动率) 为 **{curr_data['ATR']:.2f}**。"
    else:
        return f"🤖 **AI 助理:** 我是一个专注于 {ticker} 交易信号的本地 AI。你可以问我关于 **买卖建议**、**止损位**、**RSI** 或 **趋势** 的问题。"

# --- 7. UI 主程序 ---
with st.sidebar:
    st.title("🎸 Tod's V10.5")
    st.caption("Tour Edition | LTS")
    
    # [V10.5] 微信推送配置区
    with st.expander("📡 微信耳返 (Push)", expanded=False):
        wechat_token = st.text_input("PushPlus Token", type="password", help="去 pushplus.plus 获取 Token 填入此处。填入后自动激活报警。")
        if wechat_token and st.button("🔔 测试连通性"):
            if send_wechat_msg(wechat_token, "Tod Studio Soundcheck", "你的耳返系统工作正常！🎤"):
                st.success("Test Signal Sent!")
            else:
                st.error("Connection Failed")

    # [V9.7] 自动巡航 - 核心控制区
    st.markdown("### 🔄 自动巡航 (Auto-Pilot)")
    enable_auto_refresh = st.checkbox("沉浸式监控 (60s刷新)", value=False, help="开启后，系统将自动循环扫描。Emoji 颜色会随股价实时更新。")
    
    countdown_placeholder = st.empty()
    
    if st.session_state['last_scan_time']:
        last_t = st.session_state['last_scan_time'].strftime('%H:%M:%S')
        st.markdown(f"<div class='status-updated'>✅ 上次刷新: {last_t}</div>", unsafe_allow_html=True)

    if st.button("🔄 立即强制刷新 (Manual Reset)"):
        st.cache_data.clear()
        st.session_state['scan_executed'] = False 
        st.session_state['trigger_refresh'] = False
        st.rerun()

    # [V9.7 关键逻辑修复] 检查扳机，如果被扣动，立即扫描
    if not st.session_state['scan_executed'] or st.session_state.get('trigger_refresh', False):
        perform_auto_scan(push_token=wechat_token if enable_auto_refresh else None, force_refresh=True)
        st.session_state['trigger_refresh'] = False # 扫描完，重置扳机

    with st.expander("⚙️ 系统调音台", expanded=False):
        current_ticker = st.session_state['current_ticker']
        current_atr = st.session_state['settings']['atr_params'].get(current_ticker, 2.5)
        new_atr = st.slider(f"{current_ticker} ATR", 1.5, 5.0, float(current_atr), 0.1)
        if new_atr != current_atr:
            st.session_state['settings']['atr_params'][current_ticker] = new_atr
            save_settings(st.session_state['settings'])
            st.success("Saved")
            
    st.markdown("### 🎹 通道选择 (20-Ch Matrix)")
    favs = st.session_state['settings']['favorites']
    
    for r in range(5):
        cols = st.columns(4)
        for c in range(4):
            idx = r * 4 + c
            if idx < len(favs):
                ticker = favs[idx]
                status_icon = st.session_state['ticker_status'].get(ticker, "")
                
                # [V9.7 视觉优化] 按钮 Label 包含 Emoji 信号灯
                if ticker:
                    label = f"{status_icon} {ticker}" if status_icon else ticker
                else:
                    label = "---"
                
                if cols[c].button(label, key=f"btn_{idx}", use_container_width=True, disabled=not ticker):
                    st.session_state['current_ticker'] = ticker
                    st.rerun()

    with st.expander("🎛️ 通道跳线 (Patch Editor)", expanded=False):
        patch_df = pd.DataFrame({"Channel": [f"CH {i+1}" for i in range(20)], "Ticker": favs})
        edited_df = st.data_editor(patch_df, hide_index=True, use_container_width=True, num_rows="fixed")
        if st.button("💾 保存跳线"):
            new_favs = edited_df["Ticker"].fillna("").astype(str).str.strip().str.upper().tolist()
            new_favs = [fix_china_ticker(t) if t else "" for t in new_favs]
            st.session_state['settings']['favorites'] = new_favs
            save_settings(st.session_state['settings'])
            st.session_state['scan_executed'] = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🏠 房间声学 (Fundamentals)")
    fund_data = get_fundamentals(st.session_state['current_ticker'])
    if fund_data:
        c1, c2 = st.columns(2)
        c1.metric("PE (TTM)", f"{fund_data['PE']:.1f}" if fund_data['PE'] else "N/A")
        c2.metric("PB", f"{fund_data['PB']:.1f}" if fund_data['PB'] else "N/A")
        mkt_cap_b = fund_data['Mkt Cap'] / 1e9
        st.caption(f"总市值: ${mkt_cap_b:.1f} B")
    else:
        st.caption("基本面数据暂不可用")

    st.markdown("---")
    with st.expander("🧮 仓位增益", expanded=True):
        account_risk = st.number_input("本笔投入", value=200000, step=10000)
        risk_pct = st.slider("最大风控 %", 0.5, 5.0, 2.0)
    
    st.caption(log_system_status())

# 主显示区
ticker = st.session_state['current_ticker']
bench_name = get_market_benchmark(ticker)

st.title(f"{ticker} 频谱深度解析")
st.caption(f"对标基准 (Benchmark): {bench_name}")

try:
    with st.spinner('🎵 正在调谐信号 (Phase Alignment)...'):
        df, df_bench = fetch_pair_data(ticker)
    
    # [V10.2] Bypass Logic: If signal is lost, show "No Signal" screen instead of crashing
    if df is None or df.empty:
        st.markdown(f"""
        <div class="no-signal">
            <h3>⚠️ 无信号 (No Signal)</h3>
            <p>通道 <b>{ticker}</b> 暂无数据响应。</p>
            <p style="font-size:0.8em; opacity:0.7;">可能原因: 新股数据未收录 / 代码拼写 / 数据源拥堵</p>
            <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" style="color: #ffc107; text-decoration: none;">🔍 手动验证 (Yahoo Finance)</a>
        </div>
        """, unsafe_allow_html=True)
        # 允许程序继续运行，不使用 st.stop() 阻塞其他组件
        
    else:
        atr_mult = st.session_state['settings']['atr_params'].get(ticker, 2.5)
        df = calculate_advanced_metrics(df, df_bench, atr_mult)
        df_display = optimize_display_data(df)

        if len(df) > 2:
            curr = df.iloc[-1]
            change_pct = (curr['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100
            advice = us_market_advice(curr, atr_mult, bench_name, df)
            final_score = advice['score_mod']
            status_emoji = get_status_emoji(final_score)
            st.session_state['ticker_status'][ticker] = status_emoji
            
            # [V9.9] HUD 升级：5路显示，增加 RSI 独立表头
            k1, k2, k3, k4, k5 = st.columns(5)
            
            # [V10.0] Tooltip Logic Injection
            
            # 1. Price Help
            price_help = f"收盘价: ${curr['Close']:.2f}\n"
            if curr['Close'] > curr['SMA50']: price_help += "✅ >SMA50 (中期强势)\n"
            else: price_help += "⚠️ <SMA50 (中期弱势)\n"
            if curr['Close'] > curr['SMA200']: price_help += "✅ >SMA200 (长期牛市)"
            else: price_help += "⚠️ <SMA200 (长期熊市)"
            
            k1.metric("现价", f"${curr['Close']:.2f}", f"{change_pct:+.2f}%", help=price_help)

            # 2. RVol Help
            rvol_val = curr['RVol']
            if rvol_val > 1.5: rvol_help = "🔥 爆量: 主力大举介入，波动加剧。"
            elif rvol_val > 1.2: rvol_help = "⚡ 放量: 交易活跃，趋势确认度高。"
            elif rvol_val < 0.8: rvol_help = "🧊 缩量: 市场观望，缺乏方向。"
            else: rvol_help = "温和: 成交量正常。"
            
            k2.metric("RVol (量能)", f"{curr['RVol']:.2f}x", "放量" if curr['RVol']>1.2 else "缩量", help=rvol_help)
            
            # 3. RSI Help
            rsi_val = curr.get('RSI', 50)
            if rsi_val > 75: rsi_help = "🔥 严重超买: 情绪狂热，风险极高。"
            elif rsi_val > 70: rsi_help = "⚠️ 超买区: 短期过热，谨防回调。"
            elif rsi_val < 25: rsi_help = "🧊 严重超卖: 恐慌盘涌出，反弹一触即发。"
            elif rsi_val < 30: rsi_help = "💎 超卖区: 情绪低迷，关注底部机会。"
            elif rsi_val > 50: rsi_help = "🐂 多头区: 买盘占优。"
            else: rsi_help = "🐻 空头区: 卖盘占优。"
            
            k3.metric("RSI (14)", f"{rsi_val:.1f}", "超买" if rsi_val > 70 else "超卖" if rsi_val < 30 else "中性", help=rsi_help)
            
            # 4. RS Help
            rs_val = curr.get('RS_Momentum', 0)
            if rs_val > 0.1: rs_help = "🚀 显著领跑: 走势大幅强于基准，机构抱团。"
            elif rs_val > 0: rs_help = "✅ 小幅领先: 略强于大盘。"
            else: rs_help = "🐌 跑输大盘: 资金流出或关注度下降。"
            rs_help += f"\n(基准: {bench_name})"
            
            rs_mom = curr.get('RS_Momentum', 0)
            k4.metric(f"RS 动量", f"{rs_mom:.2f}", "🚀" if rs_mom > 0 else "🐌", help=rs_help)
            
            # 5. ATR Help
            stop_p = curr['Stop_Loss_Long']
            dist = (curr['Close'] - stop_p) / curr['Close'] * 100
            atr_help = f"🛡️ 移动止损线: ${stop_p:.2f}\n"
            atr_help += f"当前安全垫: {dist:.1f}%\n"
            atr_help += "原理: 价格跌破此线表示波动率异常放大，趋势可能反转。"
            
            dist_stop = (curr['Close']-curr['Stop_Loss_Long'])/curr['Close']*100
            k5.metric("ATR 止损", f"{dist_stop:.1f}%", f"${curr['Stop_Loss_Long']:.2f}", help=atr_help)

            status_color = "#9400D3" if "🟣" in advice['status'] else "#d4edda" if "🟢" in advice['status'] else "#f8d7da"
            text_color = "white" if "🟣" in advice['status'] else "black"
            
            # [V10.5] 增加手动推送按钮到 Advice Box 下方
            st.markdown(f"""
            <div class="advice-box" style="background-color: {status_color}; border-left: 5px solid #666; color: {text_color};">
                <h3 style="margin:0;">{advice['status']} (评分: {advice['score_mod']})</h3>
                <p style="margin-top:10px;"><b>🔉 声学隐喻：</b> {advice['metaphor']}</p>
                <p><b>👉 操作指令：</b> <strong>{advice['action']}</strong></p>
                <p style="font-size:0.9em; opacity:0.8;"><i>🔍 依据: {', '.join(advice['reason'])}</i></p>
            </div>
            """, unsafe_allow_html=True)
            
            # 手动推送按钮 (仅当设置了 Token 时显示)
            if wechat_token:
                if st.button("📱 手动发送本页战报到微信", use_container_width=True):
                    msg_content = f"<b>{ticker} 实时战报</b><br>现价: ${curr['Close']:.2f}<br>评级: {advice['status']}<br>评分: {final_score}<br>止损位: ${curr['Stop_Loss_Long']:.2f}"
                    if send_wechat_msg(wechat_token, f"{ticker} 分析报告", msg_content):
                        st.toast("✅ 已成功发送到微信！")
                    else:
                        st.error("❌ 发送失败，请检查 Token。")
            
            if "❌" not in advice['action']:
                price_risk = max(0.01, curr['Close'] - curr['Stop_Loss_Long'])
                atr_risk = 2 * curr['ATR']
                risk_unit = max(price_risk, atr_risk)
                vol_factor = 1.0
                if curr['ATR']/curr['Close'] > 0.05: vol_factor = 0.7 
                elif curr['ATR']/curr['Close'] < 0.02: vol_factor = 1.2 
                shares = int((account_risk * (risk_pct/100) * vol_factor) / risk_unit)
                st.info(f"💡 **Gain Staging:** 建议仓位 **{shares}** 股 (波动率系数 {vol_factor:.1f}x)")

            # [V9.9] Chart 升级：4轨道堆叠 (Price, Vol, RSI, MACD)
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03)
            
            # Row 1: K-Line
            fig.add_trace(go.Candlestick(x=df_display.index, open=df_display['Open'], high=df_display['High'], low=df_display['Low'], close=df_display['Close'], name='K线'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['SMA50'], line=dict(color='orange', width=1), name='SMA50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['SMA200'], line=dict(color='royalblue', width=1), name='SMA200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['BB_Upper'], line=dict(color='gray', width=0.5, dash='dot'), name='BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Stop_Loss_Long'], line=dict(color='#9400D3', width=1.5, dash='dash'), name='ATR止损'), row=1, col=1)
            
            # Row 2: Volume
            colors = ['#28a745' if r > 0 else '#dc3545' for r in df_display['Close'].diff()]
            fig.add_trace(go.Bar(x=df_display.index, y=df_display['Volume'], marker_color=colors, name='Vol'), row=2, col=1)
            
            # Row 3: RSI (New!)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['RSI'], line=dict(color='#6f42c1', width=1.5), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
            
            # Row 4: MACD
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MACD'], line=dict(color='#007bff', width=1.5), name='MACD'), row=4, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MACD_Signal'], line=dict(color='#ffc107', width=1.5), name='Signal'), row=4, col=1)
            fig.add_trace(go.Bar(x=df_display.index, y=df_display['MACD_Hist'], marker_color='gray', name='Hist'), row=4, col=1)
            
            fig.update_layout(height=800, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

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
                            strategy_ret = (res_df['Strategy'].iloc[-1] - 100000)/100000*100
                            bh_ret = (res_df['Buy_Hold'].iloc[-1] - 100000)/100000*100
                            strat_dd = calculate_max_drawdown(res_df['Strategy'])
                            perf_metrics = calculate_performance_metrics(res_df['Strategy'].pct_change().dropna())
                            c1, c2, c3 = st.columns(3)
                            c1.metric("策略总收益", f"{strategy_ret:.1f}%", f"vs {bh_ret:.1f}%")
                            c2.metric("最大回撤", f"{strat_dd:.1f}%", f"夏普: {perf_metrics.get('Sharpe', 0):.2f}")
                            c3.metric("胜率", f"{perf_metrics.get('WinRate', 0):.1f}%", f"CAGR: {perf_metrics.get('CAGR', 0):.1f}%")
                            st.line_chart(res_df)
                        else:
                            st.error("回测数据不足或异常")
        
except Exception as e:
    st.error(f"❌ 系统异常: {e}")
    if st.button("🆘 紧急恢复"):
        st.cache_data.clear(); st.session_state.clear(); st.rerun()
    st.stop()

# --- 8. [V9.7] 沉浸式倒计时逻辑 (自动触发) ---
if enable_auto_refresh:
    try:
        with countdown_placeholder.container():
            # 60秒倒计时
            count_text = st.empty()
            progress_bar = st.progress(100)
            for i in range(60, 0, -1):
                count_text.markdown(f"<div class='countdown-box'>⏳ 下次刷新: {i} s</div>", unsafe_allow_html=True)
                progress_bar.progress(i / 60)
                time.sleep(1)
            
            # [V9.7 关键] 扣动扳机，准备重载
            st.session_state['trigger_refresh'] = True
            st.rerun()
            
    except Exception as e:
        st.error(f"Auto-Loop Interrupt: {e}")
