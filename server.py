from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import math
import urllib.parse
import feedparser
import warnings

warnings.filterwarnings('ignore')
os.environ['YF_VERBOSE'] = '0'

app = FastAPI(title="Minervini Alpha Quant Engine - Live Dynamic Terminal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SECTOR & COMMODITY KEYWORDS FOR LIVE RSS SEARCH
SECTOR_MAP = {
    "SUGAR": ["sugar", "ethanol", "export", "milling", "isma", "cane", "raw sugar"],
    "DEFENCE": ["defence", "defense", "shipyard", "hal", "bel", "outlay", "order", "procurement"],
    "BANKING": ["banking", "nifty bank", "rbi", "credit", "nim", "repo", "psu bank"],
    "AUTO": ["auto sales", "ev policy", "siam", "commercial vehicle", "passenger vehicle"],
    "METALS": ["steel", "iron ore", "aluminum", "lme", "china demand", "copper"],
    "PHARMA": ["usfda", "pharma", "api", "drug clearance", "approval"],
    "REALTY": ["property", "housing", "real estate", "stamp duty", "residential"]
}

STOCK_SECTOR_LOOKUP = {
    "BALRAMCHIN": "SUGAR", "RENUKA": "SUGAR", "DHALPUR": "SUGAR", "TRIVENI": "SUGAR", "UTAMSUGAR": "SUGAR", "AVADHSUGAR": "SUGAR",
    "HAL": "DEFENCE", "BEL": "DEFENCE", "MAZDOCK": "DEFENCE", "COCHINSHIP": "DEFENCE", "BDL": "DEFENCE",
    "SBIN": "BANKING", "CANBK": "BANKING", "MAHABANK": "BANKING", "J&KBANK": "BANKING", "PNB": "BANKING",
    "TATASTEEL": "METALS", "SAIL": "METALS", "JINDALSTEL": "METALS", "HINDALCO": "METALS", "HINDCOPPER": "METALS",
    "TATAMOTORS": "AUTO", "M&M": "AUTO", "MARUTI": "AUTO", "BAJAJ-AUTO": "AUTO", "TVSMOTOR": "AUTO",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA", "GRANULES": "PHARMA", "AUROPHARMA": "PHARMA", "INDSWFTLAB": "PHARMA",
    "DLF": "REALTY", "LODHA": "REALTY", "GODREJPROP": "REALTY", "OBEROIRLTY": "REALTY"
}

POSITIVE_KEYWORDS = [
    "record profit", "order win", "capacity expansion", "margin expansion", 
    "guidance raised", "earnings beat", "acquisition", "contract win", "price hike", "approval"
]

NEGATIVE_KEYWORDS = [
    "earnings miss", "margin compression", "investigation", "default", 
    "pledge", "resignation", "sebi notice", "loss", "downgrade", "export duty hike", "ban"
]

def clean_json_dict(d):
    if isinstance(d, dict):
        return {k: clean_json_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_json_dict(v) for v in d]
    elif isinstance(d, float):
        return 0.0 if (math.isnan(d) or math.isinf(d)) else d
    return d

def fetch_live_google_news_sentiment(clean_symbol: str) -> dict:
    """ REAL-TIME GOOGLE NEWS RSS PIPELINE """
    try:
        sector = STOCK_SECTOR_LOOKUP.get(clean_symbol.upper(), "")
        sector_keywords = SECTOR_MAP.get(sector, [])
        
        # Build Dual Query: Stock + Sector Macro Keywords
        if sector_keywords:
            kw_query = " OR ".join(sector_keywords[:4])
            query = f"{clean_symbol} stock OR ({kw_query})"
        else:
            query = f"{clean_symbol} stock news"
            
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(rss_url)
        total_entries = min(len(feed.entries), 8)
        
        if total_entries == 0:
            return {
                "sentiment": "NEUTRAL", 
                "pos_size_pct": 0.15, 
                "score": 0.0, 
                "headline": "No active news found", 
                "sector": sector if sector else "GENERAL"
            }
            
        pos_hits, neg_hits = 0.0, 0.0
        latest_headline = feed.entries[0].title if feed.entries else "Live news scan complete"
        
        for entry in feed.entries[:total_entries]:
            title = entry.title.lower()
            if any(kw in title for kw in POSITIVE_KEYWORDS): pos_hits += 1.0
            if sector_keywords and any(kw in title for kw in sector_keywords): pos_hits += 1.5
            if any(kw in title for kw in NEGATIVE_KEYWORDS): neg_hits += 1.0
                
        net_score = (pos_hits - neg_hits) / float(total_entries)
        
        if net_score >= 0.20:
            return {
                "sentiment": "POSITIVE", 
                "pos_size_pct": 0.20, 
                "score": round(net_score, 2), 
                "headline": latest_headline, 
                "sector": sector if sector else "GENERAL"
            }
        elif net_score <= -0.15:
            return {
                "sentiment": "NEGATIVE", 
                "pos_size_pct": 0.075, 
                "score": round(net_score, 2), 
                "headline": latest_headline, 
                "sector": sector if sector else "GENERAL"
            }
        else:
            return {
                "sentiment": "NEUTRAL", 
                "pos_size_pct": 0.15, 
                "score": round(net_score, 2), 
                "headline": latest_headline, 
                "sector": sector if sector else "GENERAL"
            }
            
    except Exception:
        return {
            "sentiment": "NEUTRAL", 
            "pos_size_pct": 0.15, 
            "score": 0.0, 
            "headline": "Live news feed error", 
            "sector": "GENERAL"
        }

class MinerviniEngine:
    def __init__(self, csv_path="MINERVINI_FINAL_STOCK_LIST_2126.csv"):
        self.csv_path = csv_path
        self.symbols = []
        self.data_store = {}
        self.is_loading = False
        self.is_ready = False
        self.load_symbols()

    def load_symbols(self):
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            self.symbols = [f"{str(sym).strip()}.NS" for sym in df['SYMBOL'].values if pd.notna(sym) and str(sym).strip() != '']
        else:
            self.symbols = []

    def clean_df(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).capitalize() for col in df.columns]
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def preload_data(self):
        if self.is_ready or self.is_loading:
            return
        self.is_loading = True
        print("[⚡] Fetching Real-Time Market Data into RAM...", flush=True)
        batch_size = 300
        total_batches = (len(self.symbols) + batch_size - 1) // batch_size
        for b in range(total_batches):
            batch_symbols = self.symbols[b * batch_size : (b + 1) * batch_size]
            try:
                bulk_df = yf.download(batch_symbols, period="2y", interval="1d", group_by='ticker', progress=False, threads=True)
                for sym in batch_symbols:
                    try:
                        df = bulk_df if len(batch_symbols) == 1 else (bulk_df[sym] if sym in bulk_df else pd.DataFrame())
                        df = df.dropna(how='all')
                        if not df.empty and 'High' in df.columns:
                            self.data_store[sym] = self.clean_df(df)
                    except Exception:
                        continue
            except Exception:
                continue

        self.is_loading = False
        self.is_ready = True
        print("[+] Live Market Data Synchronized!", flush=True)

    def check_stage2(self, sub_df):
        if len(sub_df) < 180: return False
        close = sub_df['Close'].dropna()
        if close.empty: return False
        curr_p = float(close.iloc[-1])
        
        if math.isnan(curr_p) or curr_p < 30.0 or curr_p > 5000.0: return False
        
        sma_150 = close.rolling(150).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        sma_200_prev = close.rolling(200).mean().iloc[-21]
        sma_50 = close.rolling(50).mean().iloc[-1]
        
        high_52 = sub_df['High'].rolling(252, min_periods=100).max().iloc[-1]
        low_52 = sub_df['Low'].rolling(252, min_periods=100).min().iloc[-1]
        
        window_start = max(0, len(sub_df) - 150)
        base_start_p = close.iloc[window_start]
        base_peak_p = sub_df['High'].iloc[window_start:].max()
        prior_uptrend = (base_peak_p - base_start_p) / base_start_p if base_start_p > 0 else 0
        
        return all([
            pd.notna(sma_150) and pd.notna(sma_200) and pd.notna(sma_50),
            curr_p > sma_150 and curr_p > sma_200,
            sma_150 > sma_200,
            sma_200 > sma_200_prev,
            sma_50 > sma_150 and sma_50 > sma_200,
            curr_p >= low_52 * 1.25,
            curr_p >= high_52 * 0.75,
            prior_uptrend >= 0.30
        ])

    def detect_vcp(self, sub_df):
        if len(sub_df) < 75: return False, {}
        window = sub_df.iloc[-75:]
        closes, highs, lows, vols = window['Close'].values, window['High'].values, window['Low'].values, sub_df['Volume'].values
        
        curr_p = float(closes[-1])
        pivot_high = float(np.max(highs[:-2]))
        
        if math.isnan(pivot_high) or pivot_high <= 0 or math.isnan(curr_p) or curr_p <= 0:
            return False, {}
        
        swing_low_indices = []
        for i in range(2, len(lows) - 2):
            if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                swing_low_indices.append(i)
                
        if len(swing_low_indices) < 2:
            t1_low, t2_low = np.min(lows[:40]), np.min(lows[40:])
            t1_drop, t2_drop = (pivot_high - t1_low) / pivot_high, (pivot_high - t2_low) / pivot_high
            t3_drop = t2_drop * 0.6
        else:
            low_values = [lows[idx] for idx in swing_low_indices]
            t1_drop = (pivot_high - np.min(low_values)) / pivot_high
            t2_drop = (pivot_high - low_values[-1]) / pivot_high
            t3_drop = (pivot_high - np.min(lows[-5:])) / pivot_high if len(lows) >= 5 else t2_drop * 0.7

        if not (t1_drop > t2_drop and t1_drop >= 0.06 and t1_drop <= 0.38):
            return False, {}
            
        dist_to_pivot = ((pivot_high - curr_p) / pivot_high) * 100.0
        if dist_to_pivot > 5.0 or dist_to_pivot < 0.0:
            return False, {}

        vol_sma20 = float(sub_df['Volume'].iloc[-25:-1].mean())
        curr_vol = float(vols[-1])
        vdu_ratio = curr_vol / vol_sma20 if (pd.notna(vol_sma20) and vol_sma20 > 0) else 1.0
        
        if math.isnan(vdu_ratio) or vdu_ratio > 0.55:
            return False, {}
            
        compressed_stop = max(0.020, min(0.035, round(t3_drop + 0.005, 3)))
        
        return True, {
            "Pattern_Type": "TYPE B (VCP)",
            "Pivot_Point": round(pivot_high, 2),
            "Distance_To_Pivot": round(dist_to_pivot, 2),
            "VDU_Ratio": round(vdu_ratio, 2),
            "Compressed_Stop": float(compressed_stop),
            "Vol_SMA20": float(vol_sma20) if pd.notna(vol_sma20) else 0.0
        }

    def detect_htf(self, sub_df):
        if len(sub_df) < 60: return False, {}
        close, highs = sub_df['Close'], sub_df['High']
        pole_start = float(close.iloc[-50:-8].min())
        pole_peak = float(highs.iloc[-12:].max())
        if math.isnan(pole_start) or pole_start <= 0: return False, {}
        pole_gain = (pole_peak - pole_start) / pole_start
        
        if pole_gain < 0.45: return False, {}
        
        flag_window = sub_df.iloc[-8:]
        flag_high = float(flag_window['High'].max())
        flag_low = float(flag_window['Low'].min())
        curr_p = float(close.iloc[-1])
        
        if math.isnan(flag_high) or flag_high <= 0: return False, {}
        
        pullback = (flag_high - flag_low) / flag_high
        vol_sma20 = sub_df['Volume'].iloc[-25:-8].mean()
        vdu_ratio = flag_window['Volume'].mean() / vol_sma20 if (pd.notna(vol_sma20) and vol_sma20 > 0) else 1.0
        
        if pullback <= 0.14 and vdu_ratio <= 0.60:
            dist_to_pivot = ((flag_high - curr_p) / flag_high) * 100
            return True, {
                "Pattern_Type": "HIGH TIGHT FLAG",
                "Pivot_Point": round(flag_high, 2),
                "Distance_To_Pivot": round(dist_to_pivot, 2),
                "VDU_Ratio": round(vdu_ratio, 2),
                "Compressed_Stop": float(max(0.020, min(0.035, round(pullback + 0.005, 3)))),
                "Vol_SMA20": float(vol_sma20) if pd.notna(vol_sma20) else 0.0
            }
        return False, {}

engine = MinerviniEngine()

@app.on_event("startup")
def startup_event():
    engine.preload_data()

@app.get("/status")
def get_status(background_tasks: BackgroundTasks):
    if not engine.is_ready and not engine.is_loading:
        background_tasks.add_task(engine.preload_data)
    return {"ready": engine.is_ready, "loading": engine.is_loading}

@app.get("/scan-live")
def scan_live_market(
    initial_capital: float = Query(10000.0),
    min_vol_surge: float = Query(1.30)
):
    """ LIVE INTRADAY SCANNER WITH LIVE GOOGLE NEWS RSS """
    if not engine.is_ready:
        engine.preload_data()

    now = datetime.now()
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    elapsed_mins = max(15, min(375, int((now - market_start).total_seconds() / 60))) if now > market_start else 375
    
    scanned_signals = []
    
    for sym, df in engine.data_store.items():
        clean_sym = sym.replace('.NS', '')
        if df.empty or len(df) < 180: continue
        
        sub_df = df.iloc[:-1] # Completed bars for pattern detection
        
        if engine.check_stage2(sub_df):
            is_htf, htf_m = engine.detect_htf(sub_df)
            is_vcp, vcp_m = engine.detect_vcp(sub_df) if not is_htf else (False, {})
            meta = htf_m if is_htf else vcp_m
            dist_pivot = meta.get('Distance_To_Pivot', 99)
            
            if (is_htf or is_vcp) and pd.notna(dist_pivot) and float(dist_pivot) <= 5.0:
                today_bar = df.iloc[-1]
                curr_price = float(today_bar['Close'])
                curr_high = float(today_bar['High'])
                curr_vol = float(today_bar['Volume'])
                vol_sma20 = float(meta.get('Vol_SMA20', 0))
                pivot_price = float(meta.get('Pivot_Point', 0.0))
                
                projected_daily_vol = (curr_vol / elapsed_mins) * 375.0
                dynamic_surge = round(projected_daily_vol / vol_sma20, 2) if vol_sma20 > 0 else 0.0
                
                if curr_high >= pivot_price and dynamic_surge >= min_vol_surge:
                    # LIVE RSS SCRAPING FOR ACTIVE SIGNAL
                    news_res = fetch_live_google_news_sentiment(clean_sym)
                    dynamic_pos_pct = news_res["pos_size_pct"]
                    
                    pos_val = float(initial_capital * dynamic_pos_pct)
                    calc_qty = max(1, int(math.floor(pos_val / pivot_price)))
                    
                    scanned_signals.append({
                        "time_scanned": now.strftime('%H:%M:%S'),
                        "symbol": clean_sym,
                        "pattern": str(meta['Pattern_Type']),
                        "sector": news_res["sector"],
                        "news_sentiment": news_res["sentiment"],
                        "news_headline": news_res["headline"],
                        "news_score": news_res["score"],
                        "dynamic_pos_allocation": f"{int(dynamic_pos_pct * 100)}%",
                        "pivot_price": float(pivot_price),
                        "current_price": float(curr_price),
                        "live_vol_surge": f"{dynamic_surge}x",
                        "recommended_qty": int(calc_qty),
                        "capital_required_inr": float(round(calc_qty * pivot_price, 2)),
                        "compressed_stop_loss": f"{meta.get('Compressed_Stop', 0.025)*100}%"
                    })

    return clean_json_dict({"scanned_at": now.strftime('%Y-%m-%d %H:%M:%S'), "active_signals": scanned_signals})

@app.get("/stock-candles")
def get_stock_candles(symbol: str, start_date: str = Query("2026-01-01"), end_date: str = Query("2026-09-10")):
    full_sym = f"{symbol}.NS"
    if full_sym in engine.data_store:
        df = engine.data_store[full_sym]
    else:
        df = yf.download(full_sym, period="1y", interval="1d", progress=False)
        df = engine.clean_df(df)
        
    start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=60)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=30)
    
    sub = df[(df.index >= start_dt) & (df.index <= end_dt)]
    candles = []
    for idx, row in sub.iterrows():
        candles.append({
            "time": idx.strftime('%Y-%m-%d'),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "volume": float(row['Volume'])
        })
    return clean_json_dict(candles)