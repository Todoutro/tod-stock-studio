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
    page_title="Tod's Studio V10.4 (Cloud)",
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
    .countdown-box {
        font-family: 'Courier New', monospace; font-size: 1.1em; color: #d63384;
        font-weight: bold; text-align: center; padding: 8px;
        border: 1px dashed #d63384; border-radius: 5px; margin-top: 10px; background-color: #fff0f6;
    }
    .no-signal { background-color: #343a40; color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 状态初始化
if 'ticker_status' not in st.session_state: st.session_state['ticker_status'] = {}
if 'scan_executed' not in st.session_state: st.session_state['scan_executed'] = False
if 'last_scan_time' not in st.session_state: st.session_state['last_scan_time'] = None
if 'trigger_refresh' not in st.session_state: st.session_state['trigger_refresh'] = False
if 'pushed_today' not in st.session_state: st.session_state['pushed_today'] = set() # 防止重复推送

# --- 2. 核心记忆系统 (Settings) ---
SETTINGS_FILE = 'stock_settings.json'
US_SECTOR_MAP = {"AAPL": "XLK", "MSFT": "XLK", "NVDA": "SOXX", "AMD": "SOXX", "TSM": "SOXX", "TSLA": "XLY", "AMZN": "XLY", "GOOG": "XLC", "META": "XLC", "default": "QQQ"}

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
    agents = ['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36']
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
        
        # Bypass Logic (Simple)
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

def fetch_pair_data(ticker):
    bench = get_market_benchmark(ticker)
    return fetch_data_safe(ticker, "2y"), fetch_data_safe(bench, "2y")

# --- 4. 算法与分析 (DSP) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_metrics(df, bench_df, atr_mult=2.5):
    try:
        df = df.copy()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['Stop_Loss_Long'] = (df['Close'].rolling(20).max() - df['ATR'] * atr_mult).clip(lower=df['Close']*0.7)
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
        df['RVol'] = df['Volume'] / df['Vol_SMA20']
        
        if bench_df is not None and not bench_df.empty:
            common = df.index.intersection(bench_df.index)
            if len(common) > 20:
                df['RS_Raw'] = (df.loc[common,'Close'].pct_change() - bench_df.loc[common,'Close'].pct_change()).fillna(0)
                df['RS_Momentum'] = df['RS_Raw'].rolling(20).mean() * 100
        return df.dropna()
    except: return df

def calculate_score(row, df_hist):
    score = 50
    reasons = []
    
    # 趋势
    if row['Close'] > row['SMA50']: score += 10
    if row['Close'] > row['SMA200']: score += 15
    if row['SMA50'] > row['SMA200']: score += 10; reasons.append("多头排列")
    
    # 动量
    if row['MACD'] > row['MACD_Signal']: score += 10
    if 50 < row['RSI'] < 75: score += 15; reasons.append("动能充沛")
    
    # 量能 & RS
    if row.get('RVol', 0) > 1.2: score += 5
    if row.get('RS_Momentum', 0) > 0: score += 10; reasons.append("领跑基准")
    
    # 风险
    if row['RSI'] > 80: score -= 10; reasons.append("⚠ RSI超买")
    
    return score, reasons

def get_emoji(score):
    if score >= 110: return "🟣"
    if score >= 100: return "🟢"
    if score >= 90: return "🥎"
    if score >= 75: return "🟡"
    if score >= 45: return "🟠"
    return "🔴"

# --- 5. 微信推送模块 (The Wireless Transmitter) ---
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

# --- 6. 自动扫描与UI ---
def perform_auto_scan(push_token=None):
    valid_favs = [t for t in st.session_state['settings']['favorites'] if t]
    if not valid_favs: return

    st.sidebar.caption("📡 Scanning Matrix...")
    bench_cache = {} # 简单缓存
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    # 如果日期变了，清空已推送列表
    if 'last_push_date' not in st.session_state or st.session_state['last_push_date'] != today_str:
        st.session_state['pushed_today'] = set()
        st.session_state['last_push_date'] = today_str

    for ticker in valid_favs:
        try:
            df = fetch_data_safe(ticker)
            if df is not None and not df.empty:
                # 简化版计算，只为获取状态
                df = calculate_metrics(df, None) # 自动扫描时暂不加载Benchmark以提速
                if not df.empty:
                    curr = df.iloc[-1]
                    score, reasons = calculate_score(curr, df)
                    status = get_emoji(score)
                    st.session_state['ticker_status'][ticker] = status
                    
                    # [V10.4] 自动触发推送逻辑
                    if push_token and (status in ["🟣", "🟢"]) and (ticker not in st.session_state['pushed_today']):
                        msg = f"<b>{ticker} 触发信号!</b><br>状态: {status}<br>现价: ${curr['Close']:.2f}<br>评分: {score}<br>理由: {', '.join(reasons)}"
                        send_wechat_msg(push_token, f"🚀 {ticker} 信号提醒", msg)
                        st.session_state['pushed_today'].add(ticker)
                        
        except: pass
        
    st.session_state['scan_executed'] = True
    st.session_state['last_scan_time'] = datetime.now()

# --- 主界面 ---
with st.sidebar:
    st.title("🎸 Tod's V10.4")
    st.caption("Cloud | WeChat Push")
    
    with st.expander("📡 微信入耳式监听 (Push)", expanded=False):
        wechat_token = st.text_input("PushPlus Token", type="password", help="去 pushplus.plus 获取 Token 填入此处")
        if wechat_token:
            if st.button("🔔 发送测试信号"):
                if send_wechat_msg(wechat_token, "Tod Studio 测试", "你的入耳式监听系统工作正常！🎤"):
                    st.success("信号发送成功！请看手机。")
                else:
                    st.error("发送失败，请检查 Token。")
    
    st.markdown("### 🔄 自动巡航")
    enable_auto = st.checkbox("沉浸式监控 (60s)", value=False)
    
    if st.button("🔄 强制刷新 (Reset)"):
        st.cache_data.clear()
        st.session_state['scan_executed'] = False
        st.rerun()

    if not st.session_state['scan_executed'] or st.session_state.get('trigger_refresh', False):
        perform_auto_scan(wechat_token if enable_auto else None)
        st.session_state['trigger_refresh'] = False

    # 通道矩阵
    st.markdown("### 🎹 通道选择")
    favs = st.session_state['settings']['favorites']
    for r in range(5):
        cols = st.columns(4)
        for c in range(4):
            idx = r*4+c
            if idx < len(favs):
                t = favs[idx]
                icon = st.session_state['ticker_status'].get(t, "")
                if t:
                    if cols[c].button(f"{icon} {t}", key=f"b_{idx}"):
                        st.session_state['current_ticker'] = t
                        st.rerun()

    # 倒计时逻辑
    if enable_auto:
        placeholder = st.empty()
        with placeholder.container():
            for i in range(60, 0, -1):
                st.markdown(f"<div class='countdown-box'>⏳ 刷新倒计时: {i}s</div>", unsafe_allow_html=True)
                time.sleep(1)
            st.session_state['trigger_refresh'] = True
            st.rerun()

# --- 详情页 ---
ticker = st.session_state['current_ticker']
st.title(f"{ticker} 频谱深度解析")

try:
    with st.spinner('🎵 调谐中...'):
        df, df_b = fetch_pair_data(ticker)
    
    if df is None or df.empty:
        st.markdown(f"<div class='no-signal'>⚠️ 通道 {ticker} 无信号<br><a href='https://finance.yahoo.com/quote/{ticker}' style='color:#ffc107'>手动检查源头</a></div>", unsafe_allow_html=True)
    else:
        atr_mult = st.session_state['settings']['atr_params'].get(ticker, 2.5)
        df = calculate_metrics(df, df_b, atr_mult)
        curr = df.iloc[-1]
        score, reasons = calculate_score(curr, df)
        status = get_emoji(score)
        
        # HUD
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("现价", f"${curr['Close']:.2f}", f"{(curr['Close']-df.iloc[-2]['Close'])/df.iloc[-2]['Close']*100:.2f}%")
        c2.metric("RVol", f"{curr['RVol']:.2f}x")
        c3.metric("RSI", f"{curr['RSI']:.1f}")
        c4.metric("ATR止损", f"${curr['Stop_Loss_Long']:.2f}")
        
        # Advice Box
        color = "#9400D3" if score>=110 else "#28a745" if score>=100 else "#6c757d"
        st.markdown(f"<div class='advice-box' style='background:{color}; color:white'><h3>{status} 评分: {score}</h3><p>依据: {', '.join(reasons)}</p></div>", unsafe_allow_html=True)
        
        # 手动推送按钮
        if wechat_token:
            if st.button("📱 发送当前战报到微信"):
                msg_content = f"<b>{ticker} 分析报告</b><br>现价: ${curr['Close']:.2f}<br>评级: {status} ({score})<br>止损位: ${curr['Stop_Loss_Long']:.2f}"
                if send_wechat_msg(wechat_token, f"{ticker} 实时战报", msg_content):
                    st.toast("已发送到手机! 📱")

        # Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stop_Loss_Long'], line=dict(color='purple', dash='dash'), name='Stop'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='grey'), row=2, col=1)
        fig.update_layout(height=600, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 仓位计算
        with st.expander("🧮 仓位增益 & 止盈目标", expanded=True):
            c_g1, c_g2 = st.columns(2)
            capital = c_g1.number_input("本笔投入", 200000, step=10000)
            rr = c_g2.number_input("盈亏比 (R/R)", 2.0, step=0.5)
            risk_pct = st.slider("最大风控 %", 0.5, 5.0, 2.0)
            
            risk_per_share = curr['Close'] - curr['Stop_Loss_Long']
            if risk_per_share > 0:
                shares = int((capital * risk_pct/100) / max(risk_per_share, curr['ATR']*2))
                target = curr['Close'] + (risk_per_share * rr)
                st.info(f"建议仓位: {shares} 股 | 止盈目标: ${target:.2f}")

except Exception as e:
    st.error(f"Render Error: {e}")
