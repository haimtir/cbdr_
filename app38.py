# CBDR v15 — Fixed Multi-Model Quant Dashboard
# Fixes: None display bug, Sharpe/Sortino/Calmar, risk range 0.1x-1.5x,
#        PDF reports, trade logs, per-model signal predictions
import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesRegressor, AdaBoostRegressor,
    HistGradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, f1_score)
import warnings; warnings.filterwarnings("ignore")
import io

st.set_page_config(page_title="CBDR v15", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown('''<style>
.main .block-container{padding-top:.5rem;max-width:1600px}
.mc{background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #30363d;border-radius:10px;padding:.55rem .7rem;text-align:center;color:#e0e0e0;min-height:72px}
.mc h3{color:#58a6ff;font-size:.6rem;margin-bottom:.05rem;text-transform:uppercase;letter-spacing:.5px}
.mc .val{color:#f0f6fc;font-size:1rem;font-weight:700}
.profit{color:#3fb950!important}.loss{color:#f85149!important}
div[data-testid="stSidebar"]{background-color:#0d1117}
.det{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:.5rem .6rem;margin:.15rem 0}
.det .lbl{color:#8b949e;font-size:.6rem;text-transform:uppercase}.det .vl{color:#f0f6fc;font-size:.85rem;font-weight:600}
.det .sub{color:#58a6ff;font-size:.65rem}
.plv{background:#0d1117;border-radius:8px;padding:.6rem;text-align:center;margin:.15rem 0;border:1px solid #21262d}
.plv .pl{color:#8b949e;font-size:.6rem;text-transform:uppercase}.plv .pp{font-size:1rem;font-weight:700}
.plv .pd{font-size:.65rem;color:#8b949e}
.signal-card{background:linear-gradient(135deg,#0d1117,#161b22);border:2px solid #58a6ff;border-radius:12px;padding:.8rem 1rem;margin:.3rem 0}
.signal-bull{border-color:#3fb950}.signal-bear{border-color:#f85149}
.reason{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:.8rem;color:#c9d1d9;font-size:.82rem;line-height:1.55;margin:.3rem 0}
.fb{display:inline-block;padding:5px 12px;border-radius:8px;margin:2px;font-size:.75rem;font-weight:600}
.fb-r{background:rgba(31,111,235,0.13);color:#58a6ff;border:1px solid #1f6feb}
.fb-b{background:rgba(210,168,40,0.13);color:#d2a828;border:1px solid #d2a828}
.fb-t{background:rgba(163,113,247,0.13);color:#a371f7;border:1px solid #a371f7}
.fb-e{background:rgba(63,185,80,0.13);color:#3fb950;border:1px solid #3fb950}
.fb-s{background:rgba(248,81,73,0.13);color:#f85149;border:1px solid #f85149}
.fa{color:#8b949e;font-size:1rem;margin:0 2px}
</style>''', unsafe_allow_html=True)

def hex_to_rgba(hx, a=0.08):
    hx = hx.lstrip("#"); return f"rgba({int(hx[0:2],16)},{int(hx[2:4],16)},{int(hx[4:6],16)},{a})"

def mcard(col, t, v, fmt="auto", cs=False):
    try:
        if v is None: v = 0
        if fmt=="pct": d=f"{float(v):.1f}%"
        elif fmt=="int": d=str(int(v))
        elif fmt=="dollar": d=f"${float(v):,.0f}"
        else: d=f"{float(v):.2f}" if isinstance(v,(int,float,np.floating,np.integer)) else str(v)
    except: d = str(v)
    cc="profit" if cs and isinstance(v,(int,float)) and v>0 else("loss" if cs and isinstance(v,(int,float)) and v<0 else "")
    col.markdown(f'<div class="mc"><h3>{t}</h3><div class="val {cc}">{d}</div></div>',unsafe_allow_html=True)

PBG=dict(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117")
ASSETS={"Gold (XAUUSD)":{"t":"GC=F","s":0.3},"EUR/USD":{"t":"EURUSD=X","s":0.00012},"GBP/USD":{"t":"GBPUSD=X","s":0.00015},"USD/JPY":{"t":"JPY=X","s":0.015},"GBP/JPY":{"t":"GBPJPY=X","s":0.025},"Silver (XAGUSD)":{"t":"SI=F","s":0.02},"US Oil (WTI)":{"t":"CL=F","s":0.03},"BTC/USD":{"t":"BTC-USD","s":15}}
SESSIONS_GMT={"asia":(0,8),"london":(8,16),"ny":(13,22),"ldn_ny_overlap":(13,16),"asia_ldn_overlap":(7,9)}
ECON={"fomc":["2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18","2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-17","2026-01-28","2026-03-18"],
    "nfp":["2024-01-05","2024-02-02","2024-03-08","2024-04-05","2024-05-03","2024-06-07","2024-07-05","2024-08-02","2024-09-06","2024-10-04","2024-11-01","2024-12-06","2025-01-10","2025-02-07","2025-03-07","2025-04-04","2025-05-02","2025-06-06","2025-07-03","2025-08-01","2025-09-05","2025-10-03","2025-11-07","2025-12-05","2026-01-09","2026-02-06","2026-03-06"],
    "cpi":["2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15","2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10","2024-11-13","2024-12-11","2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13","2025-06-11","2025-07-15","2025-08-12","2025-09-10","2025-10-14","2025-11-12","2025-12-10","2026-01-13","2026-02-11","2026-03-11"]}

def evt_flags(date):
    ds=str(date); f={"fomc":"none","nfp":"none","cpi":"none","any":False}
    for e,dl in ECON.items():
        for ed in dl:
            if abs((pd.Timestamp(ds)-pd.Timestamp(ed)).days)<=1: f[e]="yes"; f["any"]=True; break
    return f

def get_session(hgmt):
    if 13<=hgmt<16: return "ldn_ny_overlap"
    elif 0<=hgmt<8: return "asia"
    elif 8<=hgmt<13: return "london"
    elif 16<=hgmt<22: return "ny"
    return "off_hours"

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_data(ticker,days):
    """Fetch hourly data from yfinance. ALL timestamps normalized to UTC."""
    import yfinance as yf; end=datetime.now()
    for d in [min(days+10,729),365,180,90]:
        try:
            df=yf.download(ticker,start=end-timedelta(days=d),end=end,interval="1h",progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
                # Force to UTC then strip tz label — all CBDR hours are GMT/UTC
                if df.index.tz is not None:
                    df.index=df.index.tz_convert("UTC").tz_localize(None)
                if len(df)>20: return df
        except: continue
    return pd.DataFrame()

# ═══ MACRO DATA: VIX, DXY, US10Y — DAILY CLOSES ═══
# Used as ML features. For each CBDR day (date X), we use
# the SAME DAY's close. US markets close at ~21:00 GMT, CBDR breakout
# happens after 00:00 GMT — so same-day close is known hours before trading.
MACRO_TICKERS = {"vix": "^VIX", "dxy": "DX=F", "us10y": "^TNX", "oil": "CL=F"}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data(days):
    """Fetch daily macro closes with fallback tickers."""
    import yfinance as yf
    end = datetime.now(); macro = {}
    fallbacks = {"vix":["^VIX"], "dxy":["DX=F","DX-Y.NYB","UUP"], "us10y":["^TNX"], "oil":["CL=F"]}
    for name, tickers in fallbacks.items():
        for ticker in tickers:
            try:
                md = yf.download(ticker, start=end-timedelta(days=days+30), end=end, interval="1d", progress=False)
                if md is not None and not md.empty:
                    if isinstance(md.columns, pd.MultiIndex): md.columns = md.columns.get_level_values(0)
                    if md.index.tz is not None: md.index = md.index.tz_convert("UTC").tz_localize(None)
                    s = md["Close"].dropna().copy()
                    s.index = pd.to_datetime(s.index).date
                    if len(s) > 5: macro[name] = s; break
            except: continue
    return macro


def get_macro_features(date, macro_data):
    """
    Get macro features for a CBDR day using SAME DAY's close.

    NOT leakage because: US markets close at ~21:00 GMT. CBDR breakout
    happens AFTER 00:00 GMT (next calendar day). So same-day DXY/VIX/Oil
    close is known 3+ hours before any trade decision.

    Timeline:  [DXY/Oil/VIX close 21:00] → [CBDR forms 20-00] → [Breakout 00:00+]
                     ↑ known here                                    ↑ we trade here

    Features per instrument (VIX, DXY, US10Y, Oil):
      - level: same day's close (raw value)
      - chg_1d: 1-day % change
      - chg_5d: 5-day % change
      - chg_10d: 10-day % change
    """
    feats = {}
    if not macro_data:
        for name in MACRO_TICKERS:
            feats[f"{name}_level"] = 0
            feats[f"{name}_chg1d"] = 0
            feats[f"{name}_chg5d"] = 0
            feats[f"{name}_chg10d"] = 0
        feats["macro_available"] = 0
        return feats

    feats["macro_available"] = 1
    target_date = pd.Timestamp(date).date()

    for name in MACRO_TICKERS:
        series = macro_data.get(name)
        if series is None or len(series) < 2:
            feats[f"{name}_level"] = 0
            feats[f"{name}_chg1d"] = 0
            feats[f"{name}_chg5d"] = 0
            feats[f"{name}_chg10d"] = 0
            continue

        # Same day's close — available before CBDR breakout (US close 21:00 < breakout 00:00+)
        available = series[series.index <= target_date]
        if len(available) == 0:
            feats[f"{name}_level"] = 0
            feats[f"{name}_chg1d"] = 0
            feats[f"{name}_chg5d"] = 0
            feats[f"{name}_chg10d"] = 0
            continue

        today_close = float(available.iloc[-1])
        feats[f"{name}_level"] = round(today_close, 2)

        # 1-day change: today's close vs yesterday's close
        if len(available) >= 2:
            feats[f"{name}_chg1d"] = round((today_close - float(available.iloc[-2])) / max(abs(float(available.iloc[-2])), 0.01) * 100, 3)
        else:
            feats[f"{name}_chg1d"] = 0

        # 5-day change
        if len(available) >= 6:
            feats[f"{name}_chg5d"] = round((today_close - float(available.iloc[-6])) / max(abs(float(available.iloc[-6])), 0.01) * 100, 3)
        else:
            feats[f"{name}_chg5d"] = 0

        # 10-day change
        if len(available) >= 11:
            feats[f"{name}_chg10d"] = round((today_close - float(available.iloc[-11])) / max(abs(float(available.iloc[-11])), 0.01) * 100, 3)
        else:
            feats[f"{name}_chg10d"] = 0

    return feats


def load_csv(f):
    """Smart CSV parser — handles multiple formats:
    1. Tab-separated no header: '2025.12.31 13:00\t4306.37\t4316.31\t4304.08\t4310.98\t40049'
    2. Standard CSV with headers: date,open,high,low,close,volume
    3. Any separator (auto-detected)
    """
    try:
        raw = f.read() if hasattr(f, 'read') else open(f).read()
        if isinstance(raw, bytes): raw = raw.decode('utf-8')
        f.seek(0) if hasattr(f, 'seek') else None

        # Detect separator (tab, semicolon, or comma)
        first_line = raw.split('\n')[0].strip()
        if '\t' in first_line: sep = '\t'
        elif ';' in first_line: sep = ';'
        else: sep = ','

        # Detect if there's a header (first line contains letters beyond column names)
        has_header = any(c.isalpha() for c in first_line.split(sep)[0])

        if has_header:
            df = pd.read_csv(io.StringIO(raw), sep=sep, parse_dates=True)
            # Find date column
            dc = [c for c in df.columns if any(k in c.lower() for k in ["date","time"])]
            if dc:
                df.index = pd.to_datetime(df[dc[0]]); df.drop(columns=dc, inplace=True, errors="ignore")
            else:
                df.index = pd.to_datetime(df.iloc[:,0]); df = df.iloc[:,1:]
            # Map column names
            cm = {}
            for c in df.columns:
                cl = c.lower().strip()
                for kw,nm in [("open","Open"),("high","High"),("low","Low"),("close","Close"),("volume","Volume")]:
                    if kw in cl: cm[c] = nm; break
            df.rename(columns=cm, inplace=True)
        else:
            # No header — positional columns (datetime, O, H, L, C, V)
            df = pd.read_csv(io.StringIO(raw), sep=sep, header=None)
            # Try multiple datetime formats
            dt_col = df.iloc[:,0].astype(str)
            for fmt in ["%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
                        "%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M",
                        "%m/%d/%Y %H:%M", None]:
                try:
                    if fmt: df.index = pd.to_datetime(dt_col, format=fmt)
                    else: df.index = pd.to_datetime(dt_col)
                    break
                except: continue
            else:
                df.index = pd.to_datetime(dt_col)  # last resort
            # Assign OHLCV columns by position
            ncols = len(df.columns)
            if ncols >= 6:
                df = df.iloc[:,1:6]  # skip datetime column, take O,H,L,C,V
                df.columns = ["Open","High","Low","Close","Volume"]
            elif ncols >= 5:
                df = df.iloc[:,1:5]
                df.columns = ["Open","High","Low","Close"]
            else:
                return pd.DataFrame()

        # Ensure numeric
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["Open","High","Low","Close"], inplace=True)
        df.index.name = None
        return df
    except Exception as e:
        st.error(f"CSV parse error: {e}")
        return pd.DataFrame()


def merge_csv_with_yahoo(csv_df, ticker, csv_tz_offset=0):
    """Merge uploaded CSV (historical) with Yahoo data (recent).
    CSV data is used up to its last date, then Yahoo fills in from there to today.
    Handles timezone: CSV shifted by csv_tz_offset to UTC, Yahoo converted to UTC."""
    import yfinance as yf
    # Shift CSV to UTC if needed
    if csv_tz_offset != 0:
        csv_df.index = csv_df.index - timedelta(hours=csv_tz_offset)
    csv_last_date = csv_df.index.max()
    today = datetime.now()
    gap_days = (today - csv_last_date).days
    if gap_days <= 1:
        return csv_df  # CSV is current, no Yahoo needed
    # Fetch Yahoo data from CSV end date to now
    try:
        yf_df = yf.download(ticker, start=csv_last_date + timedelta(hours=1),
                            end=today, interval="1h", progress=False)
        if yf_df is not None and not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            if yf_df.index.tz is not None:
                yf_df.index = yf_df.index.tz_convert("UTC").tz_localize(None)
            # Merge: CSV first, then Yahoo (no overlap)
            combined = pd.concat([csv_df, yf_df[~yf_df.index.isin(csv_df.index)]])
            combined.sort_index(inplace=True)
            return combined
    except: pass
    return csv_df  # fallback: just CSV if Yahoo fails

def clsfy(o,h,l,c):
    body=abs(c-o); rng=max(h-l,0.0001); br=body/rng; green=c>o
    uw=(h-max(o,c))/rng; lw=(min(o,c)-l)/rng; pat="normal"
    if br<.1: pat="doji"
    elif br>.7: pat="large_body"
    elif (not green) and lw>2*br and uw<br*.3 and br>.05: pat="hammer"
    elif green and uw>2*br and lw<br*.3 and br>.05: pat="shooting_star"
    return {"green":green,"body_ratio":round(br,3),"uw":round(uw,3),"lw":round(lw,3),"pattern":pat}

def session_features(dfs, gmt=0):
    """Compute session features. Data is UTC, sessions defined in GMT — direct match."""
    feats={}; hv="Volume" in dfs.columns and dfs["Volume"].sum()>0
    for sn,(ss2,se2) in SESSIONS_GMT.items():
        # No conversion needed — both data and sessions are in UTC/GMT
        if ss2 < se2:
            m = (dfs.index.hour >= ss2) & (dfs.index.hour < se2)
        else:
            m = (dfs.index.hour >= ss2) | (dfs.index.hour < se2)
        s=dfs[m]
        if len(s)==0: feats[f"ses_{sn}_range"]=0; feats[f"ses_{sn}_vol"]=0; feats[f"ses_{sn}_trend"]=0; continue
        sr2=s["High"].max()-s["Low"].min(); dr=max(dfs["High"].max()-dfs["Low"].min(),0.0001)
        feats[f"ses_{sn}_range"]=round(sr2/dr,3)
        feats[f"ses_{sn}_vol"]=round(s["Volume"].sum()/max(dfs["Volume"].sum(),1),3) if hv else 0
        feats[f"ses_{sn}_trend"]=round((s.iloc[-1]["Close"]-s.iloc[0]["Open"])/max(sr2,0.0001),3)
    return feats

def compute_sr(hist,n=8,cp=0.15):
    if len(hist)<20: return [],[],{}
    h,l=hist["High"].values,hist["Low"].values; phi,plo=[],[]
    for i in range(2,len(l)-2):
        if l[i]<=l[i-1] and l[i]<=l[i-2] and l[i]<=l[i+1] and l[i]<=l[i+2]: plo.append(l[i])
        if h[i]>=h[i-1] and h[i]>=h[i-2] and h[i]>=h[i+1] and h[i]>=h[i+2]: phi.append(h[i])
    al=[(p,"s") for p in plo]+[(p,"r") for p in phi]
    if not al: return plo[-5:],phi[-5:],{}
    al.sort(key=lambda x:x[0]); cls2=[]; used=set()
    for i,(pr,tp) in enumerate(al):
        if i in used: continue
        cp2,ct=[pr],[tp]; used.add(i)
        for j in range(i+1,len(al)):
            if j in used: continue
            if abs(al[j][0]-pr)/max(pr,0.01)*100<cp: cp2.append(al[j][0]); ct.append(al[j][1]); used.add(j)
        dt="s" if ct.count("s")>=ct.count("r") else "r"
        cls2.append({"p":round(np.mean(cp2),2),"str":len(cp2),"t":dt})
    cls2.sort(key=lambda x:-x["str"]); si={c["p"]:c for c in cls2[:n]}
    return [c["p"] for c in cls2 if c["t"]=="s"][-5:],[c["p"] for c in cls2 if c["t"]=="r"][-5:],si

def sr_feats(price,sups,ress,si,rs):
    f={}; rs=max(rs,0.0001)
    if sups:
        d=[(price-s)/rs for s in sups if s<price]; f["dist_sup"]=round(min(d),3) if d else 5.0
        ns=min(sups,key=lambda s:abs(price-s)); f["sup_str"]=si.get(ns,{}).get("str",1)
    else: f["dist_sup"]=5.0; f["sup_str"]=0
    if ress:
        d=[(r-price)/rs for r in ress if r>price]; f["dist_res"]=round(min(d),3) if d else 5.0
        nr=min(ress,key=lambda r:abs(price-r)); f["res_str"]=si.get(nr,{}).get("str",1)
    else: f["dist_res"]=5.0; f["res_str"]=0
    f["strong_sr"]=1 if (f["dist_sup"]<0.5 and f["sup_str"]>=3) or (f["dist_res"]<0.5 and f["res_str"]>=3) else 0
    return f

def vol_feats(dday,cbdr,boc,lb):
    hv="Volume" in dday.columns and dday["Volume"].sum()>0
    if not hv: return {"vol_avail":0,"cbdr_rvol":0,"bo_vsurge":0,"vol_trend":0,"vol_hi":0,"vol_lo":0}
    f={"vol_avail":1}; ahv=lb["Volume"].mean() if len(lb)>0 else 1
    f["cbdr_rvol"]=round(cbdr["Volume"].sum()/max(ahv*max(len(cbdr),1),1),3) if len(cbdr)>0 else 1.0
    if boc is not None:
        bv=boc.get("Volume",0) if isinstance(boc,dict) else boc["Volume"]
        f["bo_vsurge"]=round(bv/max(ahv,1),3)
    else: f["bo_vsurge"]=1.0
    if len(cbdr)>=2:
        v=cbdr["Volume"].values; f["vol_trend"]=round((v[-1]-v[0])/max(v[0],1),3) if v[0]>0 else 0
    else: f["vol_trend"]=0
    if len(dday)>2:
        tv=max(dday["Volume"].sum(),1)
        f["vol_hi"]=round(dday.nlargest(max(len(dday)//4,1),"High")["Volume"].sum()/tv,3)
        f["vol_lo"]=round(dday.nsmallest(max(len(dday)//4,1),"Low")["Volume"].sum()/tv,3)
    else: f["vol_hi"]=0.25; f["vol_lo"]=0.25
    return f

# ═══ FIBONACCI RETRACEMENT FEATURES ═══
def fib_features(df_hist, rh, rl, direction, rs):
    """Fibonacci retracement of prior 5-day swing as ML features.
    Measures distance from CBDR boundary to key fib levels.
    0.5 and 0.618/0.786 are known reversal/pullback zones."""
    feats = {"fib_382_dist":0,"fib_500_dist":0,"fib_618_dist":0,"fib_786_dist":0,
             "fib_nearest":0,"fib_nearest_level":0.5,"price_at_fib_zone":0}
    if df_hist is None or len(df_hist)<20 or rs<=0: return feats
    try:
        recent = df_hist.iloc[-120:] if len(df_hist)>=120 else df_hist
        sw_h=recent["High"].max(); sw_l=recent["Low"].min(); sw_r=sw_h-sw_l
        if sw_r<=0: return feats
        fib_lvls = {0.236:sw_h-0.236*sw_r, 0.382:sw_h-0.382*sw_r,
                    0.500:sw_h-0.500*sw_r, 0.618:sw_h-0.618*sw_r, 0.786:sw_h-0.786*sw_r}
        boundary = rh if direction=="bullish" else rl
        for lv,pr in fib_lvls.items():
            d2=abs(boundary-pr)/rs
            if lv==0.382: feats["fib_382_dist"]=round(d2,3)
            elif lv==0.5: feats["fib_500_dist"]=round(d2,3)
            elif lv==0.618: feats["fib_618_dist"]=round(d2,3)
            elif lv==0.786: feats["fib_786_dist"]=round(d2,3)
        nl=min(fib_lvls.keys(), key=lambda k:abs(boundary-fib_lvls[k]))
        feats["fib_nearest"]=round(abs(boundary-fib_lvls[nl])/rs,3)
        feats["fib_nearest_level"]=nl
        for lv,pr in fib_lvls.items():
            if lv in(0.5,0.618,0.786) and abs(boundary-pr)/rs<0.3:
                feats["price_at_fib_zone"]=1; break
    except: pass
    return feats

# ═══ VOLATILITY REGIME DETECTION ═══
# Captures war fears, central bank panics, crisis — through their effect on volatility.
# No external library needed. Uses rolling realized vol + percentile ranking.
def compute_regime_features(df_hist, date, lookback=60):
    """
    Compute volatility regime features from price history.
    Uses data up to (not including) the given date — zero leakage.

    Returns dict with:
      regime: categorical (crisis/high_vol/normal/low_vol)
      realized_vol_20d: 20-day rolling annualized volatility
      vol_zscore: how extreme current vol is vs recent history
      regime_duration: days in current regime
      regime_changed_5d: did regime change recently (binary)
      vol_expansion: is vol expanding or contracting

    WHY THIS WORKS: War fears, central bank surprises, geopolitical shocks
    ALL manifest as volatility spikes BEFORE they affect the CBDR trade.
    A VIX spike from 15 to 30 tells the model "expect larger moves today"
    but the regime feature tells it "we've been in crisis mode for 3 days,
    pullbacks are deeper but runs are also longer."
    """
    feats = {"regime": "normal", "realized_vol_20d": 0, "vol_zscore": 0,
             "regime_duration": 5, "regime_changed_5d": 0, "vol_expansion": 0,
             "vol_ratio_5_20": 1.0}

    if df_hist is None or len(df_hist) < 30:
        return feats

    try:
        target = pd.Timestamp(date)
        # Use only data BEFORE target date
        hist = df_hist[df_hist.index < target].copy()
        if len(hist) < 30:
            return feats

        # Hourly returns
        returns = hist["Close"].pct_change().dropna()
        if len(returns) < 30:
            return feats

        # 20-day rolling realized volatility (annualized from hourly)
        # ~24 candles per day * 20 days = 480 candles, but we have gaps so use 300
        window_20d = min(300, len(returns) - 1)
        window_5d = min(80, len(returns) - 1)
        vol_20d = returns.iloc[-window_20d:].std() * np.sqrt(24 * 252)  # annualized
        vol_5d = returns.iloc[-window_5d:].std() * np.sqrt(24 * 252)

        feats["realized_vol_20d"] = round(float(vol_20d), 4)

        # Ratio of short-term to long-term vol (expansion/contraction)
        feats["vol_ratio_5_20"] = round(float(vol_5d / max(vol_20d, 0.0001)), 3)
        feats["vol_expansion"] = 1 if vol_5d > vol_20d * 1.2 else 0

        # Z-score: how many std devs is current vol from the 60-day rolling mean
        vol_lb = min(lookback * 24, len(returns) - 1)
        vol_series = returns.rolling(window_20d).std() * np.sqrt(24 * 252)
        vol_series = vol_series.dropna()
        if len(vol_series) >= 20:
            vol_mean = vol_series.iloc[-vol_lb:].mean()
            vol_std = vol_series.iloc[-vol_lb:].std()
            if vol_std > 0:
                feats["vol_zscore"] = round(float((vol_20d - vol_mean) / vol_std), 3)

        # Regime classification based on percentile rank
        if len(vol_series) >= 20:
            percentile = (vol_series < vol_20d).mean() * 100
            if percentile >= 90:
                regime = "crisis"
            elif percentile >= 70:
                regime = "high_vol"
            elif percentile <= 30:
                regime = "low_vol"
            else:
                regime = "normal"
            feats["regime"] = regime

            # Regime duration: how many recent candles in same regime bucket
            recent_regimes = []
            for v in vol_series.iloc[-120:]:
                p = (vol_series < v).mean() * 100
                if p >= 90: recent_regimes.append("crisis")
                elif p >= 70: recent_regimes.append("high_vol")
                elif p <= 30: recent_regimes.append("low_vol")
                else: recent_regimes.append("normal")

            if recent_regimes:
                current = recent_regimes[-1]
                duration = 0
                for r in reversed(recent_regimes):
                    if r == current: duration += 1
                    else: break
                feats["regime_duration"] = min(duration // 24, 30)  # convert to approx days

                # Changed in last 5 days (~120 candles)?
                if len(recent_regimes) >= 120:
                    feats["regime_changed_5d"] = 1 if len(set(recent_regimes[-120:])) > 1 else 0
    except:
        pass

    return feats

# ═══ ENGINE ═══
class Engine:
    def __init__(self,df,gmt=0,spread=0.3,sl_mult=1.5,sr_lb=20,macro_data=None,
                 cbdr_start_gmt=20,cbdr_end_gmt=0,csv_tz_offset=0,range_mode="wick"):
        self.df=df.copy(); self.gmt=gmt; self.spread=spread; self.sl_mult=sl_mult; self.sr_lb=sr_lb
        self.macro_data=macro_data or {}
        self.cbdr_start_gmt=cbdr_start_gmt; self.cbdr_end_gmt=cbdr_end_gmt
        self.range_mode=range_mode  # "wick" = High/Low extremes, "close" = max Close/min Close
        # Data is in UTC (from fetch_data). For CSV uploads, user provides csv_tz_offset
        # to shift to UTC. CBDR hours are in GMT/UTC — use directly, no conversion.
        if csv_tz_offset != 0:
            self.df.index = self.df.index - timedelta(hours=csv_tz_offset)
        # CBDR window in UTC — end hour is INCLUSIVE
        # 19-23 means hours 19,20,21,22,23 (5 slots, 4 candles if 22 CME halt)
        # 20-00 means hours 20,21,22,23,0 (cross-midnight, 5 slots)
        self.cs = cbdr_start_gmt   # CBDR start hour (UTC, inclusive)
        self.ce = cbdr_end_gmt     # CBDR end hour (UTC, INCLUSIVE)
        # Session = everything AFTER CBDR ends
        self.ss = (self.ce + 1) % 24   # Session starts 1 hour after CBDR end
        self.se = self.cs              # Session ends when next CBDR starts
    def run(self):
        df=self.df.copy(); df["date"]=df.index.date; df["hour"]=df.index.hour
        dates=sorted(df["date"].unique()); days=[]; prev_dir=None; recent=[]; prev_pb=0; prev_run=0
        for i,date in enumerate(dates):
            # Skip weekends — no valid CBDR on Sat/Sun
            dow=pd.Timestamp(date).dayofweek  # 0=Mon, 5=Sat, 6=Sun
            if dow >= 5: continue
            dd=df[df["date"]==date]
            if len(dd)<4: continue
            # Find previous WEEKDAY for cross-midnight CBDR
            pd2=None
            for pi in range(i-1,-1,-1):
                if pd.Timestamp(dates[pi]).dayofweek < 5: pd2=dates[pi]; break
            if self.cs>self.ce:
                # Cross-midnight: e.g. 20-00 → hours 20..23 prev + 0 today
                late=df[df["date"]==pd2].query("hour>=@self.cs") if pd2 else pd.DataFrame()
                early=dd.query("hour<=@self.ce") if self.ce<23 else pd.DataFrame()
                cbdr=pd.concat([late,early])
            else:
                # Same day: e.g. 19-23 → hours 19,20,21,22,23 (INCLUSIVE)
                cbdr=dd.query("hour>=@self.cs and hour<=@self.ce")
            if len(cbdr)<2: continue
            # Range definition depends on mode
            if self.range_mode == "close":
                rh=cbdr["Close"].max(); rl=cbdr["Close"].min()  # settled prices only
            else:
                rh=cbdr["High"].max(); rl=cbdr["Low"].min()     # full wick extremes
            rs=rh-rl
            if rs<=0: continue
            fc=clsfy(cbdr.iloc[0]["Open"],cbdr.iloc[0]["High"],cbdr.iloc[0]["Low"],cbdr.iloc[0]["Close"])
            lc=clsfy(cbdr.iloc[-1]["Open"],cbdr.iloc[-1]["High"],cbdr.iloc[-1]["Low"],cbdr.iloc[-1]["Close"])
            cbdr_trend=(cbdr.iloc[-1]["Close"]-cbdr.iloc[0]["Open"])/rs
            n_green=sum(1 for ci in range(len(cbdr)) if cbdr.iloc[ci]["Close"]>cbdr.iloc[ci]["Open"])
            uw_avg=np.mean([(cbdr.iloc[ci]["High"]-max(cbdr.iloc[ci]["Open"],cbdr.iloc[ci]["Close"]))/max(cbdr.iloc[ci]["High"]-cbdr.iloc[ci]["Low"],0.0001) for ci in range(len(cbdr))])
            lw_avg=np.mean([(min(cbdr.iloc[ci]["Open"],cbdr.iloc[ci]["Close"])-cbdr.iloc[ci]["Low"])/max(cbdr.iloc[ci]["High"]-cbdr.iloc[ci]["Low"],0.0001) for ci in range(len(cbdr))])
            close_pos=(cbdr.iloc[-1]["Close"]-rl)/rs
            fi = -1
            try:
                fi=df.index.get_indexer([cbdr.index[0]],method="nearest")[0]
                hdf=df.iloc[max(0,fi-self.sr_lb*24):fi]; sups,ress,si=compute_sr(hdf)
            except: sups,ress,si=[],[],{}
            ns=any(abs(rl-s)/max(rl,0.01)*100<0.3 for s in sups) if sups else False
            nr=any(abs(rh-r)/max(rh,0.01)*100<0.3 for r in ress) if ress else False
            srf=sr_feats((rh+rl)/2,sups,ress,si,rs)
            # Bias: use actual PRICE CHANGE over lookback, not direction counts
            # 5-day price change for week bias, 20-day for month bias
            try:
                cur_price = cbdr.iloc[-1]["Close"]
                price_5d_ago = df.iloc[max(0, fi - 5*24)]["Close"] if fi >= 5*24 else df.iloc[0]["Close"]
                price_20d_ago = df.iloc[max(0, fi - 20*24)]["Close"] if fi >= 20*24 else df.iloc[0]["Close"]
                pct_5d = (cur_price - price_5d_ago) / max(price_5d_ago, 0.01) * 100
                pct_20d = (cur_price - price_20d_ago) / max(price_20d_ago, 0.01) * 100
                wb = "bullish" if pct_5d > 0.1 else ("bearish" if pct_5d < -0.1 else "neutral")
                mbi = "bullish" if pct_20d > 0.2 else ("bearish" if pct_20d < -0.2 else "neutral")
            except:
                wb = "neutral"; mbi = "neutral"
                pct_5d = 0; pct_20d = 0
            # Session = where we look for breakouts AFTER CBDR closes.
            # For same-day CBDR (e.g., 19-23): session is NEXT day hours 0-18
            # For cross-midnight CBDR (e.g., 20-00): session is SAME day hours 1-19
            next_day=pd.DataFrame()
            for ndi in range(i+1, min(i+4, len(dates))):
                if pd.Timestamp(dates[ndi]).dayofweek < 5:
                    next_day=df[df["date"]==dates[ndi]]; break
            if self.cs <= self.ce:
                # Same-day CBDR (e.g., 19-23): breakouts happen next day
                sess = next_day.query("hour>=@self.ss and hour<@self.se") if len(next_day)>0 else pd.DataFrame()
                # Get additional next-next day data for post-breakout measurement
                ndd = pd.DataFrame()
            else:
                # Cross-midnight CBDR (e.g., 20-00): breakouts happen same day
                sess = dd.query("hour>=@self.ss and hour<@self.se")
                ndd = next_day.query("hour<@self.se") if len(next_day)>0 else pd.DataFrame()
            if len(sess)<2: continue
            day_name=pd.Timestamp(date).day_name(); dom=pd.Timestamp(date).day
            mpos="start" if dom<=10 else("end" if dom>=21 else "mid")
            ef=evt_flags(date)
            full_day=pd.concat([dd, next_day]) if len(next_day)>0 else dd
            sesf=session_features(full_day)
            lb_vol=df.iloc[max(0,fi-120):fi] if fi>=0 else pd.DataFrame()
            bo_idx=None; direction=None; bo_c=None
            for j in range(len(sess)):
                c2=sess.iloc[j]
                if c2["Close"]>rh: direction="bullish"; bo_idx=j; bo_c=c2; break
                elif c2["Close"]<rl: direction="bearish"; bo_idx=j; bo_c=c2; break
            if direction is None: continue
            bo_cls=clsfy(bo_c["Open"],bo_c["High"],bo_c["Low"],bo_c["Close"])
            bo_hgmt=sess.index[bo_idx].hour; bo_ses=get_session(bo_hgmt)
            vf=vol_feats(full_day,cbdr,bo_c,lb_vol)
            # Fibonacci features from prior 5-day swing (direction is now known)
            fib_hist=df.iloc[max(0,fi-120):fi] if fi>=0 else pd.DataFrame()
            fibf=fib_features(fib_hist, rh, rl, direction, rs)
            all_post=pd.concat([sess.iloc[bo_idx+1:],ndd])
            retest_ses="none"; retest_candles=0
            if len(all_post)>0:
                for ri in range(len(all_post)):
                    rc=all_post.iloc[ri]
                    touched=(direction=="bullish" and rc["Low"]<=rh) or (direction=="bearish" and rc["High"]>=rl)
                    if touched: retest_candles=ri; rt_hgmt=all_post.index[ri].hour; retest_ses=get_session(rt_hgmt); break
            if len(all_post)>0:
                if direction=="bullish": pb=(rh-all_post["Low"].min())/rs; mr=(all_post["High"].max()-rh)/rs
                else: pb=(all_post["High"].max()-rl)/rs; mr=(rl-all_post["Low"].min())/rs
            else: pb=0; mr=0
            pb=max(0,pb); mr=max(0,mr); recent.append(direction)
            # Macro features: VIX, DXY, US10Y — using same day's close (no leakage)
            mf = get_macro_features(date, self.macro_data)
            # Regime detection: volatility regime from price history before this date
            rf = compute_regime_features(df, date)
            # CBDR candle timestamps for audit trail
            cbdr_first_ts=str(cbdr.index[0]); cbdr_last_ts=str(cbdr.index[-1])
            cbdr_hours_found = ",".join([str(cbdr.index[ci].hour) for ci in range(len(cbdr))])
            cbdr_open_price = round(float(cbdr.iloc[0]["Open"]),2)
            cbdr_close_price = round(float(cbdr.iloc[-1]["Close"]),2)
            cbdr_high_price = round(float(cbdr["High"].max()),2)
            cbdr_low_price = round(float(cbdr["Low"].min()),2)
            days.append({"date":date,"day":day_name,"mpos":mpos,"direction":direction,
                "range_size":round(rs,2),"range_high":round(rh,2),"range_low":round(rl,2),
                "range_mode":self.range_mode,
                "cbdr_from":cbdr_first_ts,"cbdr_to":cbdr_last_ts,"cbdr_candles":len(cbdr),
                "cbdr_hours":cbdr_hours_found,
                "cbdr_open":cbdr_open_price,"cbdr_close":cbdr_close_price,
                "cbdr_high":cbdr_high_price,"cbdr_low":cbdr_low_price,
                "fc_green":fc["green"],"fc_pat":fc["pattern"],"fc_br":fc["body_ratio"],"fc_uw":fc["uw"],"fc_lw":fc["lw"],
                "lc_green":lc["green"],"lc_pat":lc["pattern"],"lc_br":lc["body_ratio"],"lc_uw":lc["uw"],"lc_lw":lc["lw"],
                "cbdr_trend":round(cbdr_trend,3),"n_green":n_green,"uw_avg":round(uw_avg,3),"lw_avg":round(lw_avg,3),
                "close_pos":round(close_pos,3),"near_sup":ns,"near_res":nr,**srf,
                "prev_dir":prev_dir,"wbias":wb,"mbias":mbi,
                "bo_pat":bo_cls["pattern"],"bo_br":bo_cls["body_ratio"],"bo_green":bo_cls["green"],
                "evt_fomc":ef["fomc"],"evt_nfp":ef["nfp"],"evt_cpi":ef["cpi"],"evt_any":ef["any"],
                "bo_session":bo_ses,"retest_session":retest_ses,"retest_candles":retest_candles,
                **sesf,**vf,**mf,**rf,**fibf,"pb_depth":round(pb,4),"max_run":round(mr,4),"prev_pb":round(prev_pb,4),"prev_run":round(prev_run,4),
                "pct_5d":round(pct_5d,3),"pct_20d":round(pct_20d,3)})
            prev_dir=direction; prev_pb=pb; prev_run=mr
        return pd.DataFrame(days)

    def detect_latest(self,tdf):
        df=self.df.copy()
        if df.empty: return None
        df["date"]=df.index.date; df["hour"]=df.index.hour; dates=sorted(df["date"].unique())
        if len(dates)<3: return None
        cbdr=pd.DataFrame(); d=None
        # Search recent dates, skip weekends
        for d in reversed(dates[-10:]):
            if pd.Timestamp(d).dayofweek >= 5: continue  # skip Sat/Sun
            dd=df[df["date"]==d]
            # Find previous weekday for cross-midnight CBDR
            pd2=None; di=dates.index(d) if d in dates else -1
            for pi in range(di-1,-1,-1):
                if pd.Timestamp(dates[pi]).dayofweek < 5: pd2=dates[pi]; break
            if self.cs>self.ce:
                late=df[df["date"]==pd2].query("hour>=@self.cs") if pd2 else pd.DataFrame()
                early=dd.query("hour<=@self.ce") if self.ce<23 else pd.DataFrame()
                cbdr=pd.concat([late,early])
            else: cbdr=dd.query("hour>=@self.cs and hour<=@self.ce")
            if len(cbdr)>=2: break
        else: return None
        if d is None: return None
        if self.range_mode == "close":
            rh=cbdr["Close"].max(); rl=cbdr["Close"].min()
        else:
            rh=cbdr["High"].max(); rl=cbdr["Low"].min()
        rs=rh-rl
        if rs<=0: return None
        fc=clsfy(cbdr.iloc[0]["Open"],cbdr.iloc[0]["High"],cbdr.iloc[0]["Low"],cbdr.iloc[0]["Close"])
        lc=clsfy(cbdr.iloc[-1]["Open"],cbdr.iloc[-1]["High"],cbdr.iloc[-1]["Low"],cbdr.iloc[-1]["Close"])
        cbdr_trend=(cbdr.iloc[-1]["Close"]-cbdr.iloc[0]["Open"])/rs
        n_green=sum(1 for ci in range(len(cbdr)) if cbdr.iloc[ci]["Close"]>cbdr.iloc[ci]["Open"])
        uw_avg=np.mean([(cbdr.iloc[ci]["High"]-max(cbdr.iloc[ci]["Open"],cbdr.iloc[ci]["Close"]))/max(cbdr.iloc[ci]["High"]-cbdr.iloc[ci]["Low"],0.0001) for ci in range(len(cbdr))])
        lw_avg=np.mean([(min(cbdr.iloc[ci]["Open"],cbdr.iloc[ci]["Close"])-cbdr.iloc[ci]["Low"])/max(cbdr.iloc[ci]["High"]-cbdr.iloc[ci]["Low"],0.0001) for ci in range(len(cbdr))])
        close_pos=(cbdr.iloc[-1]["Close"]-rl)/rs
        try:
            fi=df.index.get_indexer([cbdr.index[0]],method="nearest")[0]
            hdf=df.iloc[max(0,fi-self.sr_lb*24):fi]; sups,ress,si=compute_sr(hdf)
        except: sups,ress,si=[],[],{}
        nsup=any(abs(rl-s)/max(rl,0.01)*100<0.3 for s in sups) if sups else False
        nres=any(abs(rh-r)/max(rh,0.01)*100<0.3 for r in ress) if ress else False
        srf=sr_feats((rh+rl)/2,sups,ress,si,rs)
        dd2=df[df["date"]==d]
        all_dates=sorted(df["date"].unique())
        di2=all_dates.index(d) if d in all_dates else -1
        # Session: same logic as run() — same-day CBDR uses next day, cross-midnight uses same day
        if self.cs <= self.ce:
            # Same-day CBDR: session is next day
            nd_sess=pd.DataFrame()
            for ndi in range(di2+1, min(di2+4, len(all_dates))):
                if pd.Timestamp(all_dates[ndi]).dayofweek < 5:
                    nd_sess=df[df["date"]==all_dates[ndi]].query("hour>=@self.ss and hour<@self.se")
                    break
            sess=nd_sess
        else:
            # Cross-midnight CBDR: session is same day
            sess=dd2.query("hour>=@self.ss and hour<@self.se")
        bdir=None
        for j in range(len(sess)):
            c2=sess.iloc[j]
            if c2["Close"]>rh: bdir="bullish"; break
            elif c2["Close"]<rl: bdir="bearish"; break
        pdd=None; wb="neutral"; mbi2="neutral"
        if tdf is not None and len(tdf)>0:
            pdd=tdf.iloc[-1].get("direction")
        # Price-based bias (same as engine)
        pct5=0; pct20=0
        try:
            cur_p=cbdr.iloc[-1]["Close"]
            p5=df.iloc[max(0,fi-5*24)]["Close"] if fi>=5*24 else df.iloc[0]["Close"]
            p20=df.iloc[max(0,fi-20*24)]["Close"] if fi>=20*24 else df.iloc[0]["Close"]
            pct5=(cur_p-p5)/max(p5,0.01)*100; pct20=(cur_p-p20)/max(p20,0.01)*100
            wb="bullish" if pct5>0.1 else("bearish" if pct5<-0.1 else "neutral")
            mbi2="bullish" if pct20>0.2 else("bearish" if pct20<-0.2 else "neutral")
        except: pass
        ef=evt_flags(d); sesf=session_features(dd2)
        try: lb_v=df.iloc[max(0,fi-120):fi]
        except: lb_v=pd.DataFrame()
        vf=vol_feats(dd2,cbdr,None,lb_v)
        fib_hist2=df.iloc[max(0,fi-120):fi] if fi>=0 else pd.DataFrame()
        fibf_det=fib_features(fib_hist2, rh, rl, bdir if bdir else "bullish", rs)
        mf_det = get_macro_features(d, self.macro_data)
        rf_det = compute_regime_features(df, d)
        csl=cbdr.index[0]; cel=cbdr.index[-1]
        # Store individual CBDR candle OHLCV for verification
        cbdr_detail = []
        for ci in range(len(cbdr)):
            c = cbdr.iloc[ci]
            cbdr_detail.append({
                "time": cbdr.index[ci].strftime("%Y-%m-%d %H:%M"),
                "hour": cbdr.index[ci].hour,
                "open": round(float(c["Open"]),2),
                "high": round(float(c["High"]),2),
                "low": round(float(c["Low"]),2),
                "close": round(float(c["Close"]),2),
                "volume": int(c.get("Volume",0)) if "Volume" in c.index else 0
            })
        # Expected vs actual candle hours (end is INCLUSIVE)
        if self.cbdr_start_gmt <= self.cbdr_end_gmt:
            expected_hours = list(range(self.cbdr_start_gmt, self.cbdr_end_gmt + 1))
        else:
            expected_hours = list(range(self.cbdr_start_gmt, 24)) + list(range(0, self.cbdr_end_gmt + 1))
        actual_hours = sorted([c["hour"] for c in cbdr_detail])
        missing_hours = [h for h in expected_hours if h not in actual_hours]
        return {"date":d,"day":pd.Timestamp(d).day_name(),"rh":round(rh,2),"rl":round(rl,2),"rs":round(rs,2),
            "range_mode":self.range_mode,
            "cbdr_from":str(csl),"cbdr_to":str(cel),"cbdr_candles":len(cbdr),
            "cbdr_detail":cbdr_detail,"expected_hours":expected_hours,"actual_hours":actual_hours,"missing_hours":missing_hours,
            "cbdr_open":round(float(cbdr.iloc[0]["Open"]),2),"cbdr_close":round(float(cbdr.iloc[-1]["Close"]),2),
            "fc_green":fc["green"],"fc_pat":fc["pattern"],"fc_br":fc["body_ratio"],"fc_uw":fc["uw"],"fc_lw":fc["lw"],
            "lc_green":lc["green"],"lc_pat":lc["pattern"],"lc_br":lc["body_ratio"],"lc_uw":lc["uw"],"lc_lw":lc["lw"],
            "cbdr_trend":round(cbdr_trend,3),"n_green":n_green,"uw_avg":round(uw_avg,3),"lw_avg":round(lw_avg,3),
            "close_pos":round(close_pos,3),"near_sup":nsup,"near_res":nres,**srf,
            "direction":bdir,"prev_dir":pdd,"wbias":wb,"mbias":mbi2,
            "bo_pat":None,"bo_br":0,"bo_green":None,"range_size":round(rs,2),
            "evt_fomc":ef["fomc"],"evt_nfp":ef["nfp"],"evt_cpi":ef["cpi"],"evt_any":ef["any"],
            "mpos":"start" if pd.Timestamp(d).day<=10 else("end" if pd.Timestamp(d).day>=21 else "mid"),
            "bo_session":"unknown","retest_session":"unknown","retest_candles":0,**sesf,**vf,**mf_det,**rf_det,**fibf_det,
            "prev_pb":round(float(tdf.iloc[-1]["pb_depth"]),4) if tdf is not None and len(tdf)>0 else 0,
            "prev_run":round(float(tdf.iloc[-1]["max_run"]),4) if tdf is not None and len(tdf)>0 else 0,
            "pct_5d":round(pct5,3),"pct_20d":round(pct20,3),
            "price":round(df.iloc[-1]["Close"],2),
            "window":f"CBDR {self.cbdr_start_gmt:02d}:00-{self.cbdr_end_gmt:02d}:00 UTC | Range: {'closes' if self.range_mode=='close' else 'wicks'} | Data: {csl.strftime('%Y-%m-%d %H:%M')} to {cel.strftime('%H:%M')} UTC"}

# ═══ ML FEATURES — includes Volume, S/R, Candles, Sessions, Macro ═══
FEAT_N=["fc_green","lc_green","fc_br","lc_br","fc_uw","fc_lw","lc_uw","lc_lw","cbdr_trend","n_green",
    "uw_avg","lw_avg","close_pos","near_sup","near_res","bo_br","bo_green","evt_any","prev_pb","prev_run",
    "dist_sup","dist_res","sup_str","res_str","strong_sr",
    "vol_avail","cbdr_rvol","bo_vsurge","vol_trend","vol_hi","vol_lo",
    "ses_asia_range","ses_asia_vol","ses_asia_trend","ses_london_range","ses_london_vol","ses_london_trend",
    "ses_ny_range","ses_ny_vol","ses_ny_trend","ses_ldn_ny_overlap_range","ses_ldn_ny_overlap_vol","ses_ldn_ny_overlap_trend",
    "ses_asia_ldn_overlap_range","ses_asia_ldn_overlap_vol","ses_asia_ldn_overlap_trend","retest_candles",
    "pct_5d","pct_20d",
    # Macro features (same day's close — US close 21:00 GMT < breakout 00:00+, zero leakage)
    "macro_available","vix_level","vix_chg1d","vix_chg5d","vix_chg10d",
    "dxy_level","dxy_chg1d","dxy_chg5d","dxy_chg10d",
    "us10y_level","us10y_chg1d","us10y_chg5d","us10y_chg10d",
    "oil_level","oil_chg1d","oil_chg5d","oil_chg10d",
    # Regime features (volatility-based, captures war/crisis/CB shocks)
    "realized_vol_20d","vol_zscore","regime_duration","regime_changed_5d","vol_expansion","vol_ratio_5_20",
    # Fibonacci retracement features (0.5 and 0.786 are key reversal zones)
    "fib_382_dist","fib_500_dist","fib_618_dist","fib_786_dist","fib_nearest","fib_nearest_level","price_at_fib_zone"]
CAT_F=["direction","day","wbias","mbias","fc_pat","lc_pat","bo_pat","mpos","bo_session","retest_session","regime"]

def encode(df):
    X=pd.DataFrame(index=df.index)
    for c in FEAT_N:
        if c in df.columns: X[c]=pd.to_numeric(df[c],errors="coerce").fillna(0).astype(float)
    for c in CAT_F:
        if c in df.columns:
            dum=pd.get_dummies(df[c].astype(str),prefix=c,drop_first=False)
            for dc in dum.columns: X[dc]=dum[dc].astype(float)
    return X.fillna(0)

def align_cols(Xtr,Xva,Xte):
    ac=sorted(set(Xtr.columns)|set(Xva.columns)|set(Xte.columns))
    for c in ac:
        if c not in Xtr.columns: Xtr[c]=0
        if c not in Xva.columns: Xva[c]=0
        if c not in Xte.columns: Xte[c]=0
    return Xtr[ac],Xva[ac],Xte[ac],ac

def get_reg_models():
    return {"RF":RandomForestRegressor(n_estimators=200,max_depth=7,min_samples_leaf=5,random_state=42),
        "GBM":GradientBoostingRegressor(n_estimators=150,max_depth=4,min_samples_leaf=5,learning_rate=0.05,random_state=42),
        "XGBoost":HistGradientBoostingRegressor(max_iter=200,max_depth=5,min_samples_leaf=5,learning_rate=0.05,random_state=42),
        "ExtraTrees":ExtraTreesRegressor(n_estimators=200,max_depth=7,min_samples_leaf=5,random_state=42),
        "AdaBoost":AdaBoostRegressor(n_estimators=100,learning_rate=0.1,random_state=42),
        "MLP":MLPRegressor(hidden_layer_sizes=(64,32),max_iter=500,early_stopping=True,validation_fraction=0.15,random_state=42),
        "DeepMLP":MLPRegressor(hidden_layer_sizes=(128,64,32,16),max_iter=800,early_stopping=True,validation_fraction=0.15,learning_rate_init=0.0005,random_state=42),
        "Ridge":Ridge(alpha=1.0)}

def get_cls_models():
    return {"RF_cls":RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=5,class_weight="balanced",random_state=42),
        "GBM_cls":GradientBoostingClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,random_state=42),
        "XGBoost_cls":HistGradientBoostingClassifier(max_iter=200,max_depth=5,learning_rate=0.05,random_state=42),
        "ExtraTrees_cls":ExtraTreesClassifier(n_estimators=200,max_depth=6,class_weight="balanced",random_state=42),
        "MLP_cls":MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500,early_stopping=True,validation_fraction=0.15,random_state=42),
        "DeepMLP_cls":MLPClassifier(hidden_layer_sizes=(128,64,32,16),max_iter=800,early_stopping=True,validation_fraction=0.15,learning_rate_init=0.0005,random_state=42)}

def train_multi_reg(Xtr,ytr,Xva,yva,Xte,yte,tname):
    from sklearn.decomposition import TruncatedSVD
    sc=RobustScaler(); Xtrs=sc.fit_transform(Xtr); Xvas=sc.transform(Xva); Xtes=sc.transform(Xte)
    models=get_reg_models(); res={}; best=None; best_v=999
    for nm,mdl in models.items():
        try:
            mdl.fit(Xtrs,ytr); r={}
            for sn,Xs,ys in [("train",Xtrs,ytr),("val",Xvas,yva),("test",Xtes,yte)]:
                p=mdl.predict(Xs); r[sn]={"mae":round(mean_absolute_error(ys,p),4),"rmse":round(np.sqrt(mean_squared_error(ys,p)),4),"r2":round(r2_score(ys,p),4) if len(ys)>1 else 0,"n":len(ys),"pred":p}
            res[nm]=r
            if r["val"]["mae"]<best_v: best_v=r["val"]["mae"]; best=(nm,mdl)
        except: pass
    # ═══ PCA VARIANTS: TruncatedSVD handles one-hot encoded sparse data ═══
    n_comp = min(20, Xtrs.shape[1]-1, Xtrs.shape[0]-1)
    pca_obj = None
    if n_comp >= 5:
        try:
            pca_obj = TruncatedSVD(n_components=n_comp, random_state=42)
            Xtr_p = pca_obj.fit_transform(Xtrs); Xva_p = pca_obj.transform(Xvas); Xte_p = pca_obj.transform(Xtes)
            for nm2, mdl2 in [("PCA_RF", RandomForestRegressor(200,max_depth=7,min_samples_leaf=5,random_state=42)),
                              ("PCA_XGB", HistGradientBoostingRegressor(max_iter=200,max_depth=5,learning_rate=0.05,random_state=42)),
                              ("PCA_Ridge", Ridge(alpha=1.0)),
                              ("PCA_MLP", MLPRegressor((64,32),max_iter=500,early_stopping=True,validation_fraction=0.15,random_state=42))]:
                try:
                    mdl2.fit(Xtr_p,ytr); r={}
                    for sn,Xs,ys in [("train",Xtr_p,ytr),("val",Xva_p,yva),("test",Xte_p,yte)]:
                        p=mdl2.predict(Xs); r[sn]={"mae":round(mean_absolute_error(ys,p),4),"rmse":round(np.sqrt(mean_squared_error(ys,p)),4),"r2":round(r2_score(ys,p),4) if len(ys)>1 else 0,"n":len(ys),"pred":p}
                    res[nm2]=r; models[nm2]=mdl2
                    if r["val"]["mae"]<best_v: best_v=r["val"]["mae"]; best=(nm2,mdl2)
                except: pass
        except: pass
    ym=ytr.mean()
    for sn,ys in [("train",ytr),("val",yva),("test",yte)]:
        res.setdefault("Baseline",{})[sn]={"mae":round(mean_absolute_error(ys,np.full_like(ys,ym)),4),"rmse":round(np.sqrt(mean_squared_error(ys,np.full_like(ys,ym))),4),"r2":0,"n":len(ys)}
    imp=pd.Series(dtype=float)
    for mn in ["RF","ExtraTrees","GBM"]:
        if mn in models and hasattr(models[mn],"feature_importances_"):
            try: imp=pd.Series(models[mn].feature_importances_,index=Xtr.columns).sort_values(ascending=False); break
            except: pass
    return {"results":res,"best":best,"scaler":sc,"importance":imp,"pca":pca_obj,
        "all_models":{n:m for n,m in models.items() if n in res},
        "Xtr_scaled":Xtrs,"ytr":ytr,
        "y_stats":{"mean":round(float(ytr.mean()),3),"std":round(float(ytr.std()),3),
            "p50":round(float(np.percentile(ytr,50)),3),"p75":round(float(np.percentile(ytr,75)),3),"p90":round(float(np.percentile(ytr,90)),3)}}

def train_dir_cls(Xtr,ytr,Xva,yva,Xte,yte):
    sc=RobustScaler(); Xtrs=sc.fit_transform(Xtr); Xvas=sc.transform(Xva); Xtes=sc.transform(Xte)
    models=get_cls_models(); res={}; best=None; best_f1=-1
    for nm,mdl in models.items():
        try:
            mdl.fit(Xtrs,ytr); r={}
            for sn,Xs,ys in [("train",Xtrs,ytr),("val",Xvas,yva),("test",Xtes,yte)]:
                p=mdl.predict(Xs); r[sn]={"acc":round(accuracy_score(ys,p)*100,1),"f1":round(f1_score(ys,p,pos_label="bullish",average="binary",zero_division=0)*100,1),"n":len(ys),"pred":p}
            res[nm]=r
            if r["val"]["f1"]>best_f1: best_f1=r["val"]["f1"]; best=(nm,mdl)
        except: pass
    imp=pd.Series(dtype=float)
    for mn in ["RF_cls","ExtraTrees_cls","GBM_cls"]:
        if mn in models and hasattr(models[mn],"feature_importances_"):
            try: imp=pd.Series(models[mn].feature_importances_,index=Xtr.columns).sort_values(ascending=False); break
            except: pass
    return {"results":res,"best":best,"scaler":sc,"importance":imp}

def compute_similarity_confidence(Xi_scaled, Xtr_scaled, ytr, prediction, k=20):
    """Compute confidence using cosine similarity to K nearest training neighbors.
    Works for ANY model — not limited to tree ensembles.

    Method:
    1. Find K most similar days in training set (cosine similarity)
    2. Compute prediction accuracy: how close were actual outcomes for similar days?
    3. Compute consistency: how tightly clustered are similar days' outcomes?

    Returns: confidence score 0.2 to 0.95
    """
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        if Xi_scaled.ndim == 1: Xi_scaled = Xi_scaled.reshape(1, -1)
        # Cosine similarity between today and all training days
        sims = cosine_similarity(Xi_scaled, Xtr_scaled)[0]
        # Get top K most similar indices
        top_k = np.argsort(sims)[-k:]
        top_sims = sims[top_k]
        top_targets = ytr[top_k]

        # Weighted mean and std of similar days' outcomes (weight by similarity)
        weights = np.maximum(top_sims, 0.01)  # avoid negative similarities
        weights = weights / weights.sum()
        weighted_mean = np.average(top_targets, weights=weights)
        weighted_std = np.sqrt(np.average((top_targets - weighted_mean)**2, weights=weights))

        # Confidence components:
        # 1. Prediction closeness: is our prediction near the neighbor mean?
        pred_error = abs(prediction - weighted_mean)
        pred_conf = max(0, 1.0 - pred_error / max(weighted_std + 0.01, 0.1))

        # 2. Neighbor agreement: are similar days clustered tightly?
        cv = weighted_std / max(abs(weighted_mean), 0.01)
        agreement_conf = max(0, 1.0 - cv)

        # 3. Similarity quality: are the neighbors actually similar?
        avg_sim = float(top_sims.mean())
        sim_conf = min(1.0, avg_sim * 1.2)  # scale up slightly

        # Combined: geometric mean of components
        conf = float((pred_conf * agreement_conf * sim_conf) ** (1/3))
        return max(0.2, min(0.95, conf))
    except:
        return 0.5

def _model_predict(model, model_name, X_scaled, pca_obj=None):
    """Route prediction through PCA if model is a PCA variant, otherwise use raw scaled data."""
    if model_name.startswith("PCA_") and pca_obj is not None:
        X_input = pca_obj.transform(X_scaled)
    else:
        X_input = X_scaled
    return float(model.predict(X_input)[0])

def sim_trade(actual_pb,actual_run,entry_depth,sl_x,tp_x,spread_cost=0):
    """Simulate a trade with detailed outcome analysis.
    Returns: (outcome, r_mult, rr, detail_dict)
    detail_dict has: loss_reason, max_favorable, max_adverse, beyond_tp, tp_pct_reached"""
    mae=max(0,actual_pb-entry_depth); mfe=actual_run+entry_depth
    mfe = mfe - spread_cost; mae = mae + spread_cost
    rr=tp_x/sl_x if sl_x>0 else 1.0
    detail = {
        "max_adverse": round(mae, 3),      # deepest against you (range multiples)
        "max_favorable": round(mfe, 3),     # furthest for you
        "tp_pct_reached": round(mfe/tp_x*100, 1) if tp_x > 0 else 0,  # how close to TP (%)
        "sl_pct_used": round(mae/sl_x*100, 1) if sl_x > 0 else 0,     # how close to SL (%)
        "beyond_tp": round(mfe - tp_x, 3) if mfe >= tp_x else 0,       # how far past TP
        "loss_reason": "none",
    }
    if mae >= sl_x:
        # LOSS: stopped out
        if mfe < 0.1:
            detail["loss_reason"] = "wrong_direction"  # never went in our favor
        elif mfe >= tp_x * 0.7:
            detail["loss_reason"] = "reversed_near_tp"  # got close to TP then reversed to SL
        else:
            detail["loss_reason"] = "stopped_out"       # normal SL hit
        return "loss", -1.0, rr, detail
    elif mfe >= tp_x:
        detail["loss_reason"] = "none"
        return "win", rr, rr, detail
    else:
        net = (mfe - mae) / sl_x
        detail["loss_reason"] = "expired_negative" if net < 0 else "none"
        return ("win" if net > 0 else "loss"), round(net, 3), rr, detail

def build_ml(tdf, spread=0.3):
    RT=0.05
    if len(tdf)<40: return None
    tdf=tdf.copy()

    # ═══ DATA QUALITY: filter noise days ═══
    # Days with tiny range produce extreme multiples (0.2pt range → 151x run = garbage)
    # Minimum viable range: ~0.1% of price (e.g., 3pts on $3000 gold)
    median_price = tdf["range_high"].median()
    min_range = max(median_price * 0.001, 2.0)  # 0.1% of price or 2pts minimum
    noise_days = tdf["range_size"] < min_range
    n_noise = int(noise_days.sum())
    if n_noise > 0:
        tdf = tdf[~noise_days].copy()
    if len(tdf) < 40: return None

    # ═══ TARGET CAPPING: clip extreme outliers before training ═══
    # Cap at 95th percentile of training data to prevent outlier poisoning
    pb_cap = np.percentile(tdf["pb_depth"], 95)
    run_cap = np.percentile(tdf["max_run"], 95)
    tdf["pb_depth_raw"] = tdf["pb_depth"]
    tdf["max_run_raw"] = tdf["max_run"]
    tdf["pb_depth"] = tdf["pb_depth"].clip(0, pb_cap)
    tdf["max_run"] = tdf["max_run"].clip(0, run_cap)

    tdf["had_retest"]=tdf["pb_depth_raw"]>=RT
    n=len(tdf); tre=int(n*0.6); vae=int(n*0.8)
    tr=tdf.iloc[:tre]; va=tdf.iloc[tre:vae]; te=tdf.iloc[vae:]
    Xtr=encode(tr); Xva=encode(va); Xte=encode(te)
    Xtr,Xva,Xte,acols=align_cols(Xtr,Xva,Xte)
    dir_m=None
    if len(tr["direction"].unique())>=2:
        dir_m=train_dir_cls(Xtr,tr["direction"].values,Xva,va["direction"].values,Xte,te["direction"].values)
    sc_cls=RobustScaler(); Xtr_c=sc_cls.fit_transform(Xtr); Xva_c=sc_cls.transform(Xva); Xte_c=sc_cls.transform(Xte)
    yctr=tr["had_retest"].astype(int).values; ycva=va["had_retest"].astype(int).values; ycte=te["had_retest"].astype(int).values
    rt_clf=RandomForestClassifier(n_estimators=200,max_depth=6,class_weight="balanced",random_state=42)
    rt_res={}
    if len(np.unique(yctr))>=2:
        rt_clf.fit(Xtr_c,yctr)
        for sn,Xs,ys in [("train",Xtr_c,yctr),("val",Xva_c,ycva),("test",Xte_c,ycte)]:
            p=rt_clf.predict(Xs); rt_res[sn]={"acc":round(accuracy_score(ys,p)*100,1),"n":len(ys)}
    rt_rate={"train":tr["had_retest"].mean(),"val":va["had_retest"].mean(),"test":te["had_retest"].mean()}
    trr=tr[tr["had_retest"]]; var=va[va["had_retest"]]; ter=te[te["had_retest"]]
    entry_m=exit_m=sl_m=None
    if len(trr)>=20 and len(var)>=5 and len(ter)>=5:
        Xtrr=encode(trr); Xvar=encode(var); Xter=encode(ter)
        for c in acols:
            if c not in Xtrr.columns: Xtrr[c]=0
            if c not in Xvar.columns: Xvar[c]=0
            if c not in Xter.columns: Xter[c]=0
        Xtrr=Xtrr[acols]; Xvar=Xvar[acols]; Xter=Xter[acols]
        # ═══ LOG TRANSFORM: reduces outlier influence, improves model fit ═══
        # Train on log1p(target), predict log, then expm1 back to real scale
        pb_tr_log = np.log1p(trr["pb_depth"].values)
        pb_va_log = np.log1p(var["pb_depth"].values)
        pb_te_log = np.log1p(ter["pb_depth"].values)
        run_tr_log = np.log1p(trr["max_run"].values)
        run_va_log = np.log1p(var["max_run"].values)
        run_te_log = np.log1p(ter["max_run"].values)
        entry_m=train_multi_reg(Xtrr,pb_tr_log,Xvar,pb_va_log,Xter,pb_te_log,"Entry")
        exit_m=train_multi_reg(Xtrr,run_tr_log,Xvar,run_va_log,Xter,run_te_log,"Exit")
        # ═══ 3RD TARGET: ML-PREDICTED SL from MAE (Maximum Adverse Excursion) ═══
        # Trains on ALL days (not just retest) to predict max adverse move
        # Optimal SL = predicted_MAE * 1.2 safety buffer
        mae_tr_log = np.log1p(tr["pb_depth"].values)
        mae_va_log = np.log1p(va["pb_depth"].values)
        mae_te_log = np.log1p(te["pb_depth"].values)
        sl_m = train_multi_reg(Xtr, mae_tr_log, Xva, mae_va_log, Xte, mae_te_log, "SL_MAE")
    # ═══ 5 STRATEGIES ═══
    S={"Baseline":[],"ML_Retest":[],"Breakout_Only":[],"Ensemble":[],"Ensemble_Guard":[]}
    has_ml=entry_m and entry_m["best"] and exit_m and exit_m["best"] and len(np.unique(yctr))>=2
    has_sl_ml = sl_m and sl_m.get("best")
    if has_ml: en,em2=entry_m["best"]; xn,xm2=exit_m["best"]; esc=entry_m["scaler"]; xsc=exit_m["scaler"]
    if has_sl_ml: sln,slm2=sl_m["best"]; slsc=sl_m["scaler"]
    # Spread in range multiples for each trade (computed per-trade since range varies)
    for idx in range(len(te)):
        row=te.iloc[idx]; rsv=row["range_size"]; d=row["direction"]
        if rsv<=0: continue
        apb=row.get("pb_depth_raw", row["pb_depth"]); amr=row.get("max_run_raw", row["max_run"])
        art=row["had_retest"]; dt=row["date"]
        rh_price=row["range_high"]; rl_price=row["range_low"]
        Xi=encode(pd.DataFrame([row]))
        for c in acols:
            if c not in Xi.columns: Xi[c]=0
        Xi=Xi[acols].fillna(0)
        pe=0.3; px=2.0; conf=0.5; dconf=0.5; rtprob=0.5; rmult=1.0; pred_sl=1.3
        # Spread cost in range multiples (e.g., 0.3pt spread / 30pt range = 0.01)
        sprd_cost = spread / rsv if rsv > 0 else 0
        if has_ml:
            rtprob=float(rt_clf.predict_proba(Xte_c[idx:idx+1])[0][1])
            try:
                Xes=esc.transform(Xi.values); Xxs=xsc.transform(Xi.values)
                pe_raw=_model_predict(em2, en, Xes, entry_m.get("pca"))
                px_raw=_model_predict(xm2, xn, Xxs, exit_m.get("pca"))
                pe=np.expm1(pe_raw); px=np.expm1(px_raw)
                pe=max(0, min(pe, 2.0))
                px=max(0.3, min(px, 8.0))
                conf=compute_similarity_confidence(Xes, entry_m["Xtr_scaled"], entry_m["ytr"], pe)
            except: pass
            # ML-predicted SL: MAE model predicts max adverse excursion
            if has_sl_ml:
                try:
                    Xsl=slsc.transform(Xi.values)
                    sl_raw=_model_predict(slm2, sln, Xsl, sl_m.get("pca"))
                    pred_mae=np.expm1(sl_raw)
                    pred_sl=max(0.5, min(pred_mae * 1.3, 3.0))
                except: pred_sl=1.3
            if dir_m and dir_m["best"]:
                try: dp=dir_m["best"][1].predict_proba(dir_m["scaler"].transform(Xi.values))[0]; dconf=float(max(dp))
                except: pass
        # RISK: 0.1x to 1.5x
        rmult=round(max(0.1,min(1.5, 0.1 + conf*0.9 + (dconf-0.5)*0.5)),2)

        # Helper: compute real prices for a given entry_depth, sl_x, tp_x
        def real_prices(entry_depth, sl_x, tp_x, direction, rh_p, rl_p, rs):
            if direction=="bullish":
                ep = rh_p - entry_depth * rs  # entry below range high
                slp = ep - sl_x * rs          # SL below entry
                tpp = ep + tp_x * rs          # TP above entry
            else:
                ep = rl_p + entry_depth * rs  # entry above range low
                slp = ep + sl_x * rs          # SL above entry
                tpp = ep - tp_x * rs          # TP below entry
            return round(ep,2), round(slp,2), round(tpp,2)

        bt={"date":str(dt),"direction":d,"range_high":rh_price,"range_low":rl_price,"range_size":rsv,
            "retest_prob":round(rtprob,2),"pred_entry":round(pe,3),"pred_exit":round(px,3),"pred_sl":round(pred_sl,3),
            "actual_pb":round(apb,3),"actual_run":round(amr,3),"actual_retest":bool(art),"confidence":round(conf,2),
            "dir_conf":round(dconf,2),"risk_mult":rmult,"spread_cost":round(sprd_cost,4)}
        # 1) BASELINE — fixed SL/TP, 1.0x risk, with spread
        if art:
            o,r,rr,td=sim_trade(apb,amr,0,1.5,3.0,sprd_cost)
            ep,slp,tpp=real_prices(0, 1.5, 3.0, d, rh_price, rl_price, rsv)
            S["Baseline"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"boundary","sl_used":1.5,"risk_mult":1.0,
                "entry_price":ep,"sl_price":slp,"tp_price":tpp,**td})
        else:
            S["Baseline"].append({**bt,"action":"skip","outcome":"skipped","r":0,"rr":0,"entry_type":"none","sl_used":0,"risk_mult":1.0,
                "entry_price":0,"sl_price":0,"tp_price":0})
        if not has_ml: continue
        # 2) ML RETEST — ML SL + ML TP + spread
        ml_sl_limit = max(0.5, pred_sl)  # ML SL for limit entries
        if rtprob>=0.4 and apb>=pe:
            o,r,rr,td=sim_trade(apb,amr,pe,ml_sl_limit,px+pe,sprd_cost)
            ep,slp,tpp=real_prices(pe, ml_sl_limit, px+pe, d, rh_price, rl_price, rsv)
            S["ML_Retest"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"limit","sl_used":round(ml_sl_limit,2),
                "entry_price":ep,"sl_price":slp,"tp_price":tpp,**td})
        else:
            S["ML_Retest"].append({**bt,"action":"skip","outcome":"skipped","r":0,"rr":0,"entry_type":"none","sl_used":0,
                "entry_price":0,"sl_price":0,"tp_price":0})
        # 3) BREAKOUT ONLY — ML SL + ML TP + spread
        ml_sl_bo = max(0.5, pred_sl * 0.8)  # Tighter SL for breakout (less room needed)
        o,r,rr,td=sim_trade(apb,amr,0,ml_sl_bo,px,sprd_cost)
        ep,slp,tpp=real_prices(0, ml_sl_bo, px, d, rh_price, rl_price, rsv)
        S["Breakout_Only"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"breakout","sl_used":round(ml_sl_bo,2),
            "entry_price":ep,"sl_price":slp,"tp_price":tpp,**td})
        # 4) ENSEMBLE — ML SL + ML TP + spread, NO leakage
        if rtprob >= 0.5:
            ens_depth = pe * 0.3
            if apb >= ens_depth:
                o,r,rr,td = sim_trade(apb, amr, ens_depth, ml_sl_limit, px + ens_depth, sprd_cost)
                ep,slp,tpp=real_prices(ens_depth, ml_sl_limit, px+ens_depth, d, rh_price, rl_price, rsv)
                S["Ensemble"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"limit_retest","sl_used":round(ml_sl_limit,2),
                    "entry_price":ep,"sl_price":slp,"tp_price":tpp,**td})
            else:
                S["Ensemble"].append({**bt,"action":"limit_not_filled","outcome":"skipped","r":0,"rr":0,"entry_type":"limit_missed","sl_used":0,
                    "entry_price":0,"sl_price":0,"tp_price":0})
        else:
            o,r,rr,td = sim_trade(apb, amr, 0, ml_sl_bo, px, sprd_cost)
            ep,slp,tpp=real_prices(0, ml_sl_bo, px, d, rh_price, rl_price, rsv)
            S["Ensemble"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"breakout","sl_used":round(ml_sl_bo,2),
                "entry_price":ep,"sl_price":slp,"tp_price":tpp,**td})
        # 5) ENSEMBLE GUARD — same + skip low conf
        cconf=(conf+dconf)/2
        if cconf < 0.4:
            S["Ensemble_Guard"].append({**bt,"action":"skip_lowconf","outcome":"skipped","r":0,"rr":0,"entry_type":"none","sl_used":0,
                "entry_price":0,"sl_price":0,"tp_price":0})
        elif rtprob >= 0.5:
            ens_depth = pe * 0.3
            if apb >= ens_depth:
                o,r,rr,td = sim_trade(apb, amr, ens_depth, ml_sl_limit, px + ens_depth, sprd_cost)
                ep,slp,tpp=real_prices(ens_depth, ml_sl_limit, px+ens_depth, d, rh_price, rl_price, rsv)
                S["Ensemble_Guard"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"limit_retest","sl_used":round(ml_sl_limit,2),
                    "entry_price":ep,"sl_price":slp,"tp_price":tpp,**td})
            else:
                S["Ensemble_Guard"].append({**bt,"action":"limit_not_filled","outcome":"skipped","r":0,"rr":0,"entry_type":"limit_missed","sl_used":0,
                    "entry_price":0,"sl_price":0,"tp_price":0})
        else:
            o,r,rr,td = sim_trade(apb, amr, 0, ml_sl_bo, px, sprd_cost)
            ep,slp,tpp=real_prices(0, ml_sl_bo, px, d, rh_price, rl_price, rsv)
            S["Ensemble_Guard"].append({**bt,"action":"filled","outcome":o,"r":r,"rr":round(rr,1),"entry_type":"breakout","sl_used":round(ml_sl_bo,2),
                "entry_price":ep,"sl_price":slp,"tp_price":tpp,**td})
    # ═══ PER-MODEL INDIVIDUAL BACKTESTS ═══
    # For each entry/exit model combo, simulate Ensemble using that model's predictions
    per_model_results = {}
    if has_ml and entry_m and exit_m:
        entry_models = {nm: entry_m["results"][nm]["test"]["pred"] for nm in entry_m["results"]
                       if nm != "Baseline" and "test" in entry_m["results"].get(nm,{}) and "pred" in entry_m["results"][nm].get("test",{})}
        exit_models = {nm: exit_m["results"][nm]["test"]["pred"] for nm in exit_m["results"]
                      if nm != "Baseline" and "test" in exit_m["results"].get(nm,{}) and "pred" in exit_m["results"][nm].get("test",{})}
        for e_nm, e_preds in entry_models.items():
            for x_nm, x_preds in exit_models.items():
                if e_nm == x_nm or e_nm.replace("PCA_","") == x_nm.replace("PCA_",""):  # same-model combos only to limit
                    combo_name = e_nm if e_nm == x_nm else f"{e_nm}+{x_nm}"
                    combo_trades = []
                    for ci in range(min(len(te), len(e_preds), len(x_preds))):
                        crow = te.iloc[ci]; crsv = crow["range_size"]
                        if crsv <= 0: continue
                        c_apb = crow.get("pb_depth_raw", crow["pb_depth"])
                        c_amr = crow.get("max_run_raw", crow["max_run"])
                        c_pe = max(0, min(np.expm1(float(e_preds[ci])), 2.0))
                        c_px = max(0.3, min(np.expm1(float(x_preds[ci])), 8.0))
                        c_sprd = spread / crsv if crsv > 0 else 0
                        # Simple ensemble: breakout entry with ML TP
                        co,cr,crr,ctd = sim_trade(c_apb, c_amr, 0, 1.3, c_px, c_sprd)
                        combo_trades.append({"outcome":co, "r":cr})
                    if combo_trades:
                        ct_df = pd.DataFrame(combo_trades)
                        ct_filled = ct_df[ct_df["outcome"].isin(["win","loss"])]
                        if len(ct_filled) > 0:
                            ct_wr = (ct_filled["outcome"]=="win").mean()*100
                            ct_pf = ct_filled[ct_filled["r"]>0]["r"].sum() / max(abs(ct_filled[ct_filled["r"]<0]["r"].sum()),0.01)
                            ct_avg_r = ct_filled["r"].mean()
                            per_model_results[combo_name] = {
                                "trades":len(ct_filled), "wr":round(ct_wr,1),
                                "pf":round(ct_pf,2), "avg_r":round(ct_avg_r,3),
                                "total_r":round(ct_filled["r"].sum(),1)}

    return {"entry":entry_m,"exit":exit_m,"sl_model":sl_m,"dir_model":dir_m,
        "retest_clf":rt_clf if len(np.unique(yctr))>=2 else None,"retest_clf_results":rt_res,
        "retest_rate":rt_rate,"retest_threshold":RT,"strategies":{k:pd.DataFrame(v) for k,v in S.items()},
        "splits":{"train":len(tr),"val":len(va),"test":len(te),"train_rt":len(trr),"val_rt":len(var),"test_rt":len(ter)},
        "all_cols":acols,"cls_scaler":sc_cls,
        "caps":{"pb_cap":round(pb_cap,3),"run_cap":round(run_cap,3),"min_range":round(min_range,2),
                "noise_filtered":n_noise},
        "per_model_pnl":per_model_results}

def predict_today(ml,det):
    if ml is None or det is None: return None
    em=ml.get("entry"); xm=ml.get("exit"); rc=ml.get("retest_clf")
    if not em or not em.get("best") or not xm or not xm.get("best"): return None
    X=encode(pd.DataFrame([det]))
    for c in ml["all_cols"]:
        if c not in X.columns: X[c]=0
    X=X[ml["all_cols"]].fillna(0)
    rtprob=0.5
    if rc and ml.get("cls_scaler"):
        Xc=ml["cls_scaler"].transform(X.values); rtprob=float(rc.predict_proba(Xc)[0][1])
    dconf=0.5; dpred=det.get("direction","unknown")
    dm=ml.get("dir_model")
    if dm and dm["best"]:
        Xd=dm["scaler"].transform(X.values); dpred=dm["best"][1].predict(Xd)[0]
        dp=dm["best"][1].predict_proba(Xd)[0]; dconf=float(max(dp))
    esc2=em["scaler"]; xsc2=xm["scaler"]; Xes=esc2.transform(X.values); Xxs=xsc2.transform(X.values)
    # Predictions in LOG space → expm1 back + caps (route through PCA if best model is PCA variant)
    ep_log=_model_predict(em["best"][1], em["best"][0], Xes, em.get("pca"))
    xp_log=_model_predict(xm["best"][1], xm["best"][0], Xxs, xm.get("pca"))
    ep=max(0, min(np.expm1(ep_log), 2.0))
    xp=max(0.3, min(np.expm1(xp_log), 8.0))
    conf=compute_similarity_confidence(Xes, em["Xtr_scaled"], em["ytr"], np.log1p(ep)) if em.get("Xtr_scaled") is not None else 0.5
    # ML SL prediction
    pred_sl = 1.3
    sl_model = ml.get("sl_model")
    if sl_model and sl_model.get("best"):
        try:
            Xslp = sl_model["scaler"].transform(X.values)
            sl_raw = _model_predict(sl_model["best"][1], sl_model["best"][0], Xslp, sl_model.get("pca"))
            pred_sl = max(0.5, min(np.expm1(sl_raw) * 1.3, 3.0))
        except: pass
    # Get ALL model predictions (route PCA variants correctly)
    entry_preds={}; exit_preds={}; sl_preds={}
    for mn,mo in em.get("all_models",{}).items():
        try:
            raw=_model_predict(mo, mn, esc2.transform(X.values), em.get("pca"))
            entry_preds[mn]=round(max(0, min(np.expm1(raw), 2.0)), 3)
        except: pass
    for mn,mo in xm.get("all_models",{}).items():
        try:
            raw=_model_predict(mo, mn, xsc2.transform(X.values), xm.get("pca"))
            exit_preds[mn]=round(max(0.3, min(np.expm1(raw), 8.0)), 3)
        except: pass
    if sl_model and sl_model.get("all_models"):
        for mn,mo in sl_model.get("all_models",{}).items():
            try:
                raw=_model_predict(mo, mn, sl_model["scaler"].transform(X.values), sl_model.get("pca"))
                sl_preds[mn]=round(max(0.5, min(np.expm1(raw)*1.3, 3.0)), 3)
            except: pass
    rmult=round(max(0.1,min(1.5,0.1+conf*0.9+(dconf-0.5)*0.5)),2)
    return {"pred_entry":round(ep,3),"pred_exit":round(xp,3),"pred_sl":round(pred_sl,3),"retest_prob":round(rtprob,2),
        "confidence":round(conf,2),"dir_conf":round(dconf,2),"dir_pred":dpred,"risk_mult":rmult,
        "entry_model":em["best"][0],"exit_model":xm["best"][0],
        "sl_model":sl_model["best"][0] if sl_model and sl_model.get("best") else "fixed",
        "entry_by_model":entry_preds,"exit_by_model":exit_preds,"sl_by_model":sl_preds}

# ═══ METRICS — FIXED Sharpe/Sortino/Calmar ═══
EMPTY_STATS={"n":0,"active":0,"skipped":0,"wins":0,"losses":0,"wr":0,"pf":0,"avg_r":0,
    "sharpe":0,"sortino":0,"calmar":0,"total_r":0,"dollar_pnl":0,"final_eq":0,
    "return_pct":0,"max_dd_pct":0,"eq_curve":[],"dd_series":[],"max_win_streak":0,"max_loss_streak":0,
    "avg_win":0,"avg_loss":0}

def calc_stats(outcomes,rs_arr,capital=10000,risk_pct=1.0,risk_mults=None):
    active=[(i,o,r) for i,(o,r) in enumerate(zip(outcomes,rs_arr)) if o in ("win","loss")]
    n=len(outcomes); ns=n-len(active)
    if not active: return {**EMPTY_STATS,"n":n,"skipped":ns}
    idxs,outs,ra=zip(*active); ra=np.array(ra,dtype=float)
    wins=sum(1 for o in outs if o=="win"); na=len(outs); wr=wins/na*100 if na>0 else 0
    wr2=ra[ra>0]; lr2=np.abs(ra[ra<0])
    pf=float(wr2.sum()/lr2.sum()) if lr2.sum()>0 else 99.0
    # Sharpe: per-trade mean/std (no annualization for short backtests)
    sh=float(ra.mean()/ra.std()) if ra.std()>0 else 0
    # Sortino: mean / downside_deviation
    neg=ra[ra<0]
    downside_dev=np.sqrt(np.mean(neg**2)) if len(neg)>0 else 0
    so=float(ra.mean()/downside_dev) if downside_dev>0 else 0
    # Equity curve with adaptive risk
    eq=float(capital); eqc=[eq]; mx=eq; mdd=0; dds=[0]
    for ii,r in enumerate(ra):
        rm=1.0
        if risk_mults is not None:
            oi=idxs[ii]
            if oi<len(risk_mults): rm=float(risk_mults[oi])
        eq+=eq*(risk_pct*rm/100)*r; eqc.append(max(eq,0.01))
        if eq>mx: mx=eq
        dd2=(mx-eq)/mx*100 if mx>0 else 0; dds.append(dd2)
        if dd2>mdd: mdd=dd2
    dp=eq-capital; rp=dp/capital*100
    # Calmar: total return / max drawdown (not annualized — cleaner for backtests)
    cal=round(rp/(mdd) if mdd>0 else 0, 2)
    sw,sl3,cw,cl3=[],[],0,0
    for o in outs:
        if o=="win":
            cw+=1
            if cl3>0: sl3.append(cl3); cl3=0
        else:
            cl3+=1
            if cw>0: sw.append(cw); cw=0
    if cw>0: sw.append(cw)
    if cl3>0: sl3.append(cl3)
    return {"n":n,"active":na,"skipped":ns,"wins":wins,"losses":na-wins,"wr":round(wr,1),"pf":round(pf,2),
        "avg_r":round(float(ra.mean()),3),"sharpe":round(sh,3),"sortino":round(so,3),"calmar":cal,
        "total_r":round(float(ra.sum()),1),"dollar_pnl":round(dp,0),"final_eq":round(eq,0),
        "return_pct":round(rp,1),"max_dd_pct":round(mdd,1),"eq_curve":eqc,"dd_series":dds,
        "max_win_streak":max(sw) if sw else 0,"max_loss_streak":max(sl3) if sl3 else 0,
        "avg_win":round(float(wr2.mean()),3) if len(wr2)>0 else 0,"avg_loss":round(float(lr2.mean()),3) if len(lr2)>0 else 0}

# ═══ PDF REPORT ═══
def generate_pdf(det,today_pred,ml,tdf,asset_name):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter,topMargin=40,bottomMargin=40)
    styles=getSampleStyleSheet(); story=[]
    ts=ParagraphStyle('Title2',parent=styles['Title'],fontSize=18,textColor=colors.HexColor('#1a73e8'))
    story.append(Paragraph(f"CBDR Signal Report - {asset_name}",ts))
    story.append(Spacer(1,12))
    if det:
        story.append(Paragraph(f"<b>Date:</b> {det.get('date','')} ({det.get('day','')}) | <b>Price:</b> {det.get('price','N/A')}",styles['Normal']))
        story.append(Paragraph(f"<b>CBDR Range:</b> {det.get('rh',0)} - {det.get('rl',0)} ({det.get('rs',0)} pts)",styles['Normal']))
        story.append(Paragraph(f"<b>1st Candle:</b> {'Green' if det.get('fc_green') else 'Red'} {det.get('fc_pat','')} | <b>Last:</b> {'Green' if det.get('lc_green') else 'Red'} {det.get('lc_pat','')}",styles['Normal']))
        story.append(Paragraph(f"<b>Trend:</b> {det.get('cbdr_trend',0):+.2f} | <b>Week:</b> {det.get('wbias','?')} | <b>Month:</b> {det.get('mbias','?')}",styles['Normal']))
        story.append(Paragraph(f"<b>S/R:</b> Sup dist {det.get('dist_sup',0):.1f}x (str {det.get('sup_str',0)}) | Res dist {det.get('dist_res',0):.1f}x (str {det.get('res_str',0)})",styles['Normal']))
        story.append(Paragraph(f"<b>Volume:</b> CBDR rel vol {det.get('cbdr_rvol','N/A')}x",styles['Normal']))
        if det.get("macro_available",0)==1:
            story.append(Paragraph(f"<b>Macro (same day close):</b> VIX {det.get('vix_level',0):.1f} (1d {det.get('vix_chg1d',0):+.1f}%, 5d {det.get('vix_chg5d',0):+.1f}%) | DXY {det.get('dxy_level',0):.2f} (1d {det.get('dxy_chg1d',0):+.2f}%, 5d {det.get('dxy_chg5d',0):+.2f}%) | US10Y {det.get('us10y_level',0):.2f}% (5d {det.get('us10y_chg5d',0):+.2f}%) | Oil ${det.get('oil_level',0):.1f} (5d {det.get('oil_chg5d',0):+.1f}%)",styles['Normal']))
        story.append(Spacer(1,12))
    if today_pred:
        tp=today_pred; d=det.get("direction","?") if det else "?"
        story.append(Paragraph("<b>ML Signal Decision</b>",styles['Heading2']))
        story.append(Paragraph(f"Direction: <b>{d.upper()}</b> | Retest prob: <b>{tp['retest_prob']*100:.0f}%</b>",styles['Normal']))
        story.append(Paragraph(f"Entry model ({tp['entry_model']}): pullback <b>{tp['pred_entry']:.2f}x</b> range",styles['Normal']))
        story.append(Paragraph(f"Exit model ({tp['exit_model']}): run <b>{tp['pred_exit']:.1f}x</b> range",styles['Normal']))
        story.append(Paragraph(f"Confidence: <b>{tp['confidence']*100:.0f}%</b> | Dir conf: <b>{tp['dir_conf']*100:.0f}%</b> | Risk mult: <b>{tp['risk_mult']}x</b>",styles['Normal']))
        story.append(Spacer(1,8))
        # Per-model predictions table
        if tp.get("entry_by_model") or tp.get("exit_by_model"):
            story.append(Paragraph("<b>All Model Predictions</b>",styles['Heading3']))
            tdata=[["Model","Entry Pred (x range)","Exit Pred (x range)"]]
            all_m=set(list(tp.get("entry_by_model",{}).keys())+list(tp.get("exit_by_model",{}).keys()))
            for mn in sorted(all_m):
                ep2=tp.get("entry_by_model",{}).get(mn,"N/A")
                xp2=tp.get("exit_by_model",{}).get(mn,"N/A")
                tdata.append([mn,str(ep2),str(xp2)])
            t=Table(tdata,colWidths=[120,150,150])
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1a73e8')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
                ('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),9)]))
            story.append(t); story.append(Spacer(1,8))
    # Historical context
    if tdf is not None and det and det.get("direction"):
        d2=det["direction"]; day2=det["day"]
        sim=tdf[(tdf["day"]==day2)&(tdf["direction"]==d2)]
        if len(sim)>=3:
            story.append(Paragraph(f"<b>Historical ({day2}+{d2}, n={len(sim)}):</b> Avg PB {sim['pb_depth'].mean():.2f}x | Avg Run {sim['max_run'].mean():.1f}x | Med Run {sim['max_run'].median():.1f}x",styles['Normal']))
    story.append(Spacer(1,12))
    story.append(Paragraph("<b>Features Used by ML Models</b>",styles['Heading3']))
    story.append(Paragraph("Candle patterns (body ratio, upper/lower wick, doji/hammer/large_body), CBDR trend, close position, S/R distance and strength, Volume (CBDR relative, breakout surge, trend), Session data (Asia/London/NY/overlap range and volume ratios), Day of week, Week/Month price trend (%), Event flags (FOMC/NFP/CPI), Previous pullback and run values, <b>Macro: VIX level and changes (fear gauge), DXY level and changes (dollar strength, inverse to gold), US10Y level and changes (yield pressure), Oil level and changes (inflation proxy)</b>. All macro features use previous day close — zero data leakage.",styles['Normal']))
    story.append(Spacer(1,12))
    story.append(Paragraph("<b>Methodology</b>",styles['Heading3']))
    story.append(Paragraph("60/20/20 chronological train/val/test split. 7 regression models (RF, GBM, HistGBM, ExtraTrees, AdaBoost, MLP, Ridge) compete for entry depth and exit run prediction. Direction classifier uses 5 models with balanced class weights. Best model selected on validation MAE (regression) or F1 (classifier). SL optimized at 1.3x range (0.231R expectancy from historical analysis). Adaptive risk sizing: 0.1x-1.5x base risk per trade based on model confidence.",styles['Normal']))
    doc.build(story); buf.seek(0); return buf.getvalue()

# ═══ SIDEBAR ═══
st.sidebar.markdown("## CBDR v15 Quant"); st.sidebar.markdown("---")
asset_name=st.sidebar.selectbox("**Asset**",list(ASSETS.keys()),index=0); asset=ASSETS[asset_name]
dsrc=st.sidebar.radio("**Data Source**",["Yahoo Finance","Upload CSV","CSV + Yahoo (merge)"],index=0,
    help="CSV+Yahoo: upload historical CSV, system auto-fetches recent data from Yahoo to fill the gap to today.")
uf=st.sidebar.file_uploader("CSV/TXT file",type=["csv","txt"]) if dsrc in ["Upload CSV","CSV + Yahoo (merge)"] else None
csv_tz_offset=0
if dsrc in ["Upload CSV","CSV + Yahoo (merge)"]:
    csv_tz_offset=st.sidebar.number_input("**CSV Timezone (hours from UTC)**",-12,14,0,
        help="If your CSV timestamps are in GMT+2, enter 2. Data will be shifted to UTC.")
st.sidebar.markdown("---")
st.sidebar.markdown("##### CBDR Window (UTC/GMT)")
cbdr_window=st.sidebar.selectbox("**Window**",
    ["20:00-00:00 UTC (DST summer)","19:00-23:00 UTC (winter)","Custom"],index=0)
if cbdr_window.startswith("20"):
    cbdr_start_gmt=20; cbdr_end_gmt=0
elif cbdr_window.startswith("19"):
    cbdr_start_gmt=19; cbdr_end_gmt=23
else:
    cc1,cc2=st.sidebar.columns(2)
    cbdr_start_gmt=cc1.number_input("Start (UTC)",0,23,20)
    cbdr_end_gmt=cc2.number_input("End (UTC)",0,23,0)
range_mode=st.sidebar.selectbox("**Range Definition**",
    ["Wick (High/Low extremes)","Close (settled prices only)"],index=0,
    help="Wick: range = highest high to lowest low of CBDR candles (wider). "
         "Close: range = highest close to lowest close (tighter, more breakouts).")
range_mode_val="wick" if range_mode.startswith("Wick") else "close"
st.sidebar.markdown("---")
st.sidebar.markdown("##### Data Range")
date_mode=st.sidebar.selectbox("**Period Mode**",["Recent (rolling)","Custom Date Range"],index=0)
if date_mode=="Recent (rolling)":
    per_o={"1 Month":30,"3 Months":90,"6 Months":180,"1 Year":365,"2 Years":730}
    pl2=st.sidebar.selectbox("**Period**",list(per_o.keys()),index=4); pdays=per_o[pl2]
    custom_start=None; custom_end=None
else:
    st.sidebar.caption("Pick any historical range. Test = last 20% of selected range.")
    custom_start=st.sidebar.date_input("**Start Date**",datetime(2023,1,1))
    custom_end=st.sidebar.date_input("**End Date**",datetime(2024,12,31))
    pdays=(custom_end-custom_start).days
st.sidebar.markdown("---")
spread=st.sidebar.number_input("**Spread**",0.0,10.0,float(asset["s"]),0.01)
sr_lb=st.sidebar.slider("**S/R Days**",5,60,20,5)
st.sidebar.markdown("---")
base_risk=st.sidebar.slider("**Base Risk %**",0.25,10.0,1.0,0.25)
capital=st.sidebar.number_input("**Capital ($)**",1000,1000000,10000,1000)
run_btn=st.sidebar.button("Run Full Analysis",type="primary",use_container_width=True)

# ═══ MAIN ═══
st.markdown("# CBDR v15 — Multi-Model Quant Dashboard")
st.caption("Ensemble | Risk 0.1x-1.5x | 7 models | Session+Volume+S/R+Macro(VIX/DXY/US10Y/Oil) | Zero leakage")

if run_btn or "tdf" in st.session_state:
    if run_btn:
        with st.spinner("Fetching price data..."):
            if dsrc=="Upload CSV" and uf:
                df=load_csv(uf)
            elif dsrc=="CSV + Yahoo (merge)" and uf:
                csv_df=load_csv(uf)
                if csv_df.empty: st.error("Could not parse CSV."); st.stop()
                df=merge_csv_with_yahoo(csv_df, asset["t"], csv_tz_offset)
                csv_tz_offset=0  # already handled in merge
            elif date_mode=="Custom Date Range" and custom_start and custom_end:
                import yfinance as yf
                df=yf.download(asset["t"],start=str(custom_start),end=str(custom_end),interval="1h",progress=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
                    if df.index.tz is not None: df.index=df.index.tz_convert("UTC").tz_localize(None)
                else: df=pd.DataFrame()
            else:
                df=fetch_data(asset["t"],pdays)
            if df.empty: st.error("No data. Check file format or try shorter range."); st.stop()
            # Apply date range filter for ALL sources
            if date_mode=="Custom Date Range" and custom_start and custom_end:
                cs_ts=pd.Timestamp(str(custom_start)); ce_ts=pd.Timestamp(str(custom_end))+timedelta(days=1)
                df=df[(df.index>=cs_ts)&(df.index<ce_ts)]
                if df.empty: st.error(f"No data between {custom_start} and {custom_end}."); st.stop()
            elif date_mode=="Recent (rolling)" and custom_start is None:
                df=df[df.index>=datetime.now()-timedelta(days=pdays)]
            if df.empty: st.error("No data in selected range."); st.stop()
            is_historical = date_mode=="Custom Date Range"
            data_info = f"{df.index.min().date()} to {df.index.max().date()} ({len(df):,} candles)"
        with st.spinner("Fetching macro data (VIX, DXY, US10Y, Oil)..."):
            macro_data={}
            try:
                macro_days = max(pdays, 730)  # at least 2 years of macro history
                macro_data=fetch_macro_data(macro_days)
            except: pass
            macro_status=f"VIX/DXY/US10Y/Oil loaded ({len(macro_data)} instruments)" if macro_data else "Macro unavailable"
        with st.spinner(f"Running engine (CBDR {cbdr_start_gmt:02d}:00-{cbdr_end_gmt:02d}:00 UTC)..."):
            eng=Engine(df,gmt=0,spread=spread,sl_mult=1.5,sr_lb=sr_lb,macro_data=macro_data,
                      cbdr_start_gmt=cbdr_start_gmt,cbdr_end_gmt=cbdr_end_gmt,
                      csv_tz_offset=csv_tz_offset,range_mode=range_mode_val)
            tdf=eng.run()
            if tdf.empty: st.error("No trades found. Check CBDR window and data range."); st.stop()
            # detect_latest only for live/recent mode
            det = eng.detect_latest(tdf) if not is_historical else None
        with st.spinner("Training ML..."):
            ml=build_ml(tdf, spread=spread); today_pred=predict_today(ml,det) if ml and det else None
        st.session_state.update({"tdf":tdf,"det":det,"ml":ml,"today_pred":today_pred,
            "is_historical":is_historical,"data_info":data_info})
    tdf=st.session_state["tdf"]; det=st.session_state.get("det")
    ml=st.session_state.get("ml"); today_pred=st.session_state.get("today_pred")
    is_historical=st.session_state.get("is_historical",False)
    data_info=st.session_state.get("data_info","")
    rm_display = "wicks (H/L)" if range_mode_val == "wick" else "closes (C/C)"
    st.markdown(f"### {len(tdf)} breakout days | CBDR {cbdr_start_gmt:02d}:00-{cbdr_end_gmt:02d}:00 UTC | Range: {rm_display}")
    if data_info: st.caption(f"Data: {data_info}")
    if is_historical: st.info("Historical analysis mode — Signal tab shows test period summary instead of live signal.")
    if ml:
        sp=ml["splits"]
        st.markdown(f"**Split:** Train {sp['train']} | Val {sp['val']} | **Test {sp['test']}** | "
                   f"Train dates: {tdf.iloc[:sp['train']]['date'].iloc[0]} to {tdf.iloc[:sp['train']]['date'].iloc[-1]} | "
                   f"Test dates: {tdf.iloc[sp['train']+sp['val']:]['date'].iloc[0]} to {tdf.iloc[-1]['date']}")
        if 'macro_status' in dir(): st.caption(macro_status)
        caps=ml.get("caps",{})
        if caps:
            st.caption(f"ML quality: {caps.get('noise_filtered',0)} noise days filtered (range < {caps.get('min_range',0)}pts) | "
                      f"Targets capped: PB < {caps.get('pb_cap',0):.1f}x, Run < {caps.get('run_cap',0):.1f}x | "
                      f"Predictions capped: entry < 2.0x, exit < 8.0x | Log-transformed training")

    tabs=st.tabs(["Signal","Arena","Models","Sessions","Risk","Distributions","Trade Logs"])

    # ═══ SIGNAL ═══
    with tabs[0]:
        if is_historical:
            # Historical analysis mode — no live signal, show test period summary
            st.markdown("### Historical Analysis Mode")
            st.info("Custom date range selected. This shows test period performance, not a live signal.")
            if ml and ml.get("strategies"):
                te_start = tdf.iloc[ml["splits"]["train"]+ml["splits"]["val"]:]["date"].iloc[0] if len(tdf)>ml["splits"]["train"]+ml["splits"]["val"] else "?"
                te_end = tdf.iloc[-1]["date"]
                st.markdown(f"**Test period:** {te_start} to {te_end} ({ml['splits']['test']} days)")
                # Show test period direction breakdown
                test_df = tdf.iloc[ml["splits"]["train"]+ml["splits"]["val"]:]
                c1,c2,c3,c4 = st.columns(4)
                n_bull = (test_df["direction"]=="bullish").sum()
                n_bear = (test_df["direction"]=="bearish").sum()
                avg_pb = test_df["pb_depth"].mean()
                avg_run = test_df["max_run"].mean()
                mcard(c1,"Test Days",ml["splits"]["test"],"int")
                mcard(c2,"Bull/Bear",f"{n_bull}/{n_bear}")
                mcard(c3,"Avg PB",avg_pb)
                mcard(c4,"Avg Run",avg_run)
                # Show best strategy summary
                st.markdown("#### Best Strategy in Test Period")
                best_ret = -999; best_name = ""
                for key, tds in ml["strategies"].items():
                    if tds is None or tds.empty: continue
                    filled = tds[tds["outcome"].isin(["win","loss"])]
                    if len(filled) == 0: continue
                    rm_v = filled["risk_mult"].values if "risk_mult" in filled.columns else None
                    s = calc_stats(filled["outcome"].tolist(), filled["r"].tolist(), capital, base_risk, rm_v)
                    if s["return_pct"] > best_ret: best_ret = s["return_pct"]; best_name = key
                if best_name:
                    st.success(f"Best: **{best_name}** with **{best_ret:.1f}%** return. See Arena tab for full comparison.")
        elif det:
            st.markdown("### Latest Signal")
            sig_date = pd.Timestamp(det["date"])
            today = pd.Timestamp(datetime.now().date())
            days_ago = (today - sig_date).days
            cbdr_from = det.get("cbdr_from","?")
            cbdr_to = det.get("cbdr_to","?")
            rm_label = "closes" if det.get("range_mode")=="close" else "wicks"
            n_candles = det.get("cbdr_candles","?")

            # Main header: trading day + CBDR source info
            st.markdown(f"**Trade day: {det['day']} {det['date']}** | Price: **{det.get('price','N/A')}**")
            st.markdown(f"CBDR data: **{cbdr_from} to {cbdr_to} UTC** ({n_candles} candles, {rm_label}) | Open: {det.get('cbdr_open','?')} | Close: {det.get('cbdr_close','?')}")

            # CBDR candle detail table — shows exactly what data built the range
            cbdr_det = det.get("cbdr_detail",[])
            missing_h = det.get("missing_hours",[])
            if missing_h:
                st.warning(f"Missing CBDR candles at hours: {missing_h} UTC. Expected {len(det.get('expected_hours',[]))} candles, got {n_candles}. Range may be narrower than actual.")
            if cbdr_det:
                with st.expander(f"CBDR Candle Detail ({len(cbdr_det)} candles)", expanded=True):
                    cd_rows = []
                    for c in cbdr_det:
                        cd_rows.append({"Hour (UTC)":f"{c['hour']:02d}:00","Open":c["open"],"High":c["high"],"Low":c["low"],"Close":c["close"],"Volume":c["volume"]})
                    cd_df = pd.DataFrame(cd_rows)
                    st.dataframe(cd_df, use_container_width=True, hide_index=True)
                    # Show range calculation
                    if rm_label == "wicks":
                        st.caption(f"Range = Highest High ({det['rh']}) - Lowest Low ({det['rl']}) = {det['rs']} pts")
                    else:
                        st.caption(f"Range = Highest Close ({det['rh']}) - Lowest Close ({det['rl']}) = {det['rs']} pts")

            if days_ago > 0:
                if days_ago <= 3:
                    st.info(f"This is {det['day']}'s signal ({days_ago} day{'s' if days_ago>1 else ''} ago). "
                            f"Today's CBDR ({cbdr_start_gmt:02d}:00-{cbdr_end_gmt:02d}:00 UTC) hasn't formed yet. "
                            f"Run again after {cbdr_end_gmt:02d}:00 UTC tonight.")
            dc=st.columns(6)
            dc[0].markdown(f'<div class="det"><div class="lbl">Range ({rm_label})</div><div class="vl">{det["rh"]} - {det["rl"]}</div><div class="sub">{det["rs"]} pts | {n_candles} candles</div></div>',unsafe_allow_html=True)
            dc[1].markdown(f'<div class="det"><div class="lbl">1st</div><div class="vl">{"G" if det["fc_green"] else "R"} {det["fc_pat"]}</div></div>',unsafe_allow_html=True)
            dc[2].markdown(f'<div class="det"><div class="lbl">Last</div><div class="vl">{"G" if det["lc_green"] else "R"} {det["lc_pat"]}</div></div>',unsafe_allow_html=True)
            dc[3].markdown(f'<div class="det"><div class="lbl">Trend</div><div class="vl">{det["cbdr_trend"]:+.2f}</div><div class="sub">5d:{det["wbias"]} ({det.get("pct_5d",0):+.1f}%) 20d:{det["mbias"]} ({det.get("pct_20d",0):+.1f}%)</div></div>',unsafe_allow_html=True)
            dc[4].markdown(f'<div class="det"><div class="lbl">S/R</div><div class="vl">{"Sup" if det["near_sup"] else("Res" if det["near_res"] else "Clear")}</div><div class="sub">Str {det.get("sup_str",0)}/{det.get("res_str",0)}</div></div>',unsafe_allow_html=True)
            dc[5].markdown(f'<div class="det"><div class="lbl">Vol</div><div class="vl">{det.get("cbdr_rvol","?")}x</div><div class="sub">{"Event!" if det["evt_any"] else "Normal"}</div></div>',unsafe_allow_html=True)
            # Macro data row (VIX, DXY, US10Y, Oil — previous day's close = zero leakage)
            if det.get("macro_available",0)==1:
                mc2=st.columns(5)
                mc2[0].markdown(f'<div class="det"><div class="lbl">VIX (today)</div><div class="vl">{det.get("vix_level",0):.1f}</div><div class="sub">1d:{det.get("vix_chg1d",0):+.1f}% 5d:{det.get("vix_chg5d",0):+.1f}%</div></div>',unsafe_allow_html=True)
                mc2[1].markdown(f'<div class="det"><div class="lbl">DXY (today)</div><div class="vl">{det.get("dxy_level",0):.2f}</div><div class="sub">1d:{det.get("dxy_chg1d",0):+.2f}% 5d:{det.get("dxy_chg5d",0):+.2f}%</div></div>',unsafe_allow_html=True)
                mc2[2].markdown(f'<div class="det"><div class="lbl">US10Y (today)</div><div class="vl">{det.get("us10y_level",0):.2f}%</div><div class="sub">1d:{det.get("us10y_chg1d",0):+.2f}% 5d:{det.get("us10y_chg5d",0):+.2f}%</div></div>',unsafe_allow_html=True)
                mc2[3].markdown(f'<div class="det"><div class="lbl">Oil (today)</div><div class="vl">${det.get("oil_level",0):.1f}</div><div class="sub">1d:{det.get("oil_chg1d",0):+.1f}% 5d:{det.get("oil_chg5d",0):+.1f}%</div></div>',unsafe_allow_html=True)
                dxy5=det.get("dxy_chg5d",0); vix_l=det.get("vix_level",0)
                sig_parts=[]
                if vix_l>25: sig_parts.append("High fear")
                if dxy5<-0.3: sig_parts.append("DXY weak=Gold+")
                elif dxy5>0.3: sig_parts.append("DXY strong=Gold-")
                mc2[4].markdown(f'<div class="det"><div class="lbl">Macro Signal</div><div class="vl">{"Fear" if vix_l>20 else "Calm"}</div><div class="sub">{" | ".join(sig_parts) if sig_parts else "Neutral"}</div></div>',unsafe_allow_html=True)
            # Regime row
            regime=det.get("regime","normal"); vol20=det.get("realized_vol_20d",0); vz=det.get("vol_zscore",0)
            reg_dur=det.get("regime_duration",0); vr=det.get("vol_ratio_5_20",1.0)
            reg_colors={"crisis":"#f85149","high_vol":"#d29922","normal":"#8b949e","low_vol":"#3fb950"}
            rc3=st.columns(4)
            rc3[0].markdown(f'<div class="det"><div class="lbl">Volatility Regime</div><div class="vl" style="color:{reg_colors.get(regime,"#8b949e")}">{regime.upper()}</div><div class="sub">{reg_dur} days in regime</div></div>',unsafe_allow_html=True)
            rc3[1].markdown(f'<div class="det"><div class="lbl">Realized Vol (20d)</div><div class="vl">{vol20*100:.1f}%</div><div class="sub">Z-score: {vz:+.1f}</div></div>',unsafe_allow_html=True)
            rc3[2].markdown(f'<div class="det"><div class="lbl">Vol 5d/20d Ratio</div><div class="vl">{vr:.2f}x</div><div class="sub">{"Expanding" if vr>1.2 else("Contracting" if vr<0.8 else "Stable")}</div></div>',unsafe_allow_html=True)
            regime_impact = "Expect larger moves, deeper PB, longer runs" if regime in ("crisis","high_vol") else ("Tight ranges, smaller moves" if regime=="low_vol" else "Normal conditions")
            rc3[3].markdown(f'<div class="det"><div class="lbl">Regime Impact</div><div class="vl">{"Caution" if regime=="crisis" else "Normal"}</div><div class="sub">{regime_impact}</div></div>',unsafe_allow_html=True)
            if today_pred and det.get("direction"):
                tp2=today_pred; rs=det["rs"]; d=det["direction"]; pe=tp2["pred_entry"]; px=tp2["pred_exit"]
                conf=tp2["confidence"]; dconf=tp2["dir_conf"]; rmult=tp2["risk_mult"]; rtprob=tp2["retest_prob"]
                if rtprob>=0.5: etype="LIMIT (Retest)"; ud=pe*0.3; slx=1.3
                else: etype="MARKET (Breakout)"; ud=0; slx=1.0
                tpx=px+ud
                if d=="bullish": ep=det["rh"]-ud*rs+spread; slp=ep-slx*rs; tpp=ep+tpx*rs
                else: ep=det["rl"]+ud*rs-spread; slp=ep+slx*rs; tpp=ep-tpx*rs
                sld=slx*rs; tpd=tpx*rs; rrv=tpx/slx if slx>0 else 0
                adj_risk=round(base_risk*rmult,2); ramt=capital*(adj_risk/100); pos=ramt/sld if sld>0 else 0
                scls="signal-bull" if d=="bullish" else "signal-bear"
                st.markdown(f'<div class="signal-card {scls}"><div style="display:flex;justify-content:space-between;flex-wrap:wrap"><div><h2 style="color:white;margin:0">{d.upper()} {etype}</h2><p style="color:#8b949e;margin:0">Retest: {rtprob*100:.0f}% | PB: {pe:.2f}x | Run: {px:.1f}x</p></div><div style="text-align:right"><h2 style="color:#58a6ff;margin:0">{conf*100:.0f}% Conf</h2><p style="color:#8b949e;margin:0">Dir: {dconf*100:.0f}% | Risk: {adj_risk:.2f}% ({rmult}x)</p></div></div></div>',unsafe_allow_html=True)
                tc=st.columns(6)
                tc[0].markdown(f'<div class="plv" style="border-color:#58a6ff"><div class="pl">Entry</div><div class="pp" style="color:#58a6ff">{ep:.2f}</div><div class="pd">{ud:.2f}x deep</div></div>',unsafe_allow_html=True)
                tc[1].markdown(f'<div class="plv" style="border-color:#f85149"><div class="pl">SL</div><div class="pp" style="color:#f85149">{slp:.2f}</div><div class="pd">{sld:.1f}pts ({slx}x)</div></div>',unsafe_allow_html=True)
                tc[2].markdown(f'<div class="plv" style="border-color:#3fb950"><div class="pl">TP</div><div class="pp" style="color:#3fb950">{tpp:.2f}</div><div class="pd">{tpd:.1f}pts</div></div>',unsafe_allow_html=True)
                tc[3].markdown(f'<div class="plv"><div class="pl">R:R</div><div class="pp">{rrv:.1f}:1</div></div>',unsafe_allow_html=True)
                tc[4].markdown(f'<div class="plv"><div class="pl">Pos</div><div class="pp">{pos:,.2f}</div></div>',unsafe_allow_html=True)
                tc[5].markdown(f'<div class="plv"><div class="pl">$ Risk</div><div class="pp">${ramt:,.0f}</div></div>',unsafe_allow_html=True)
                # Per-model predictions
                st.markdown("#### All Model Predictions")
                mrows=[]
                all_mn=sorted(set(list(tp2.get("entry_by_model",{}).keys())+list(tp2.get("exit_by_model",{}).keys())))
                for mn in all_mn:
                    ep2=tp2.get("entry_by_model",{}).get(mn,"-")
                    xp2=tp2.get("exit_by_model",{}).get(mn,"-")
                    mrows.append({"Model":mn,"Entry (PB depth)":ep2,"Exit (Run)":xp2})
                if mrows: st.dataframe(pd.DataFrame(mrows),use_container_width=True,hide_index=True)
                # ═══ FULL SIGNAL NARRATIVE ═══
                st.markdown("#### Signal Analysis")
                # Build comprehensive narrative
                nar = []
                # 1. Entry decision
                if rtprob>=0.5:
                    nar.append(f"The retest classifier assigns a {rtprob*100:.0f}% probability that price will pull back to the {det['rh'] if d=='bullish' else det['rl']} boundary after breakout. "
                        f"Based on this, the system places a <b>limit order</b> at {pe:.2f}x range depth (${ep:.2f}), waiting for the pullback rather than chasing the breakout.")
                else:
                    nar.append(f"Retest probability is only {rtprob*100:.0f}% — the model expects price to continue without returning to the boundary. "
                        f"The system enters at <b>market</b> (breakout close) at approximately ${det['rh'] if d=='bullish' else det['rl']:.2f}.")
                # 2. SL reasoning
                pred_sl_val = tp2.get("pred_sl", 1.3)
                nar.append(f"The SL model ({tp2.get('sl_model','?')}) predicts maximum adverse excursion of {pred_sl_val/1.3:.2f}x range. "
                    f"With a 1.3x safety buffer, SL is set at <b>{slx:.2f}x range = {sld:.1f} pts</b> from entry (${slp:.2f}). "
                    f"{'This is tighter than the default 1.3x because conditions suggest a shallow pullback.' if pred_sl_val<1.2 else ('This is wider than default because the model sees elevated risk.' if pred_sl_val>1.5 else 'This is near the historically optimal 1.3x level.')}")
                # 3. TP reasoning
                nar.append(f"The exit model ({tp2['exit_model']}) predicts a maximum run of <b>{px:.1f}x range = {tpd:.0f} pts</b>. TP target: ${tpp:.2f}. R:R = {rrv:.1f}:1.")
                # 4. S/R context
                dsup=det.get("dist_sup",5); dres=det.get("dist_res",5); sstr=det.get("sup_str",0); rstr=det.get("res_str",0)
                if dsup<1.0 and sstr>=2:
                    nar.append(f"<b>Support nearby:</b> {dsup:.1f}x range away with {sstr} touches — this acts as a floor, suggesting shallower pullbacks for bullish trades.")
                if dres<1.0 and rstr>=2:
                    nar.append(f"<b>Resistance nearby:</b> {dres:.1f}x range away with {rstr} touches — may cap the upside run.")
                # 5. Fibonacci
                fib_zone=det.get("price_at_fib_zone",0); fib_near=det.get("fib_nearest",0); fib_lvl=det.get("fib_nearest_level",0.5)
                if fib_zone:
                    nar.append(f"<b>Fibonacci:</b> Price is sitting in a key fib zone (nearest level: {fib_lvl:.1%}, distance: {fib_near:.2f}x range). "
                        f"The 50% and 61.8% retracement levels are historically strong reversal points — the model factors this into entry depth.")
                elif fib_near<2:
                    nar.append(f"Fibonacci: nearest level is {fib_lvl:.1%} at {fib_near:.2f}x range distance. Not in a critical zone but close enough to influence pullback depth.")
                # 6. Macro
                if det.get("macro_available",0)==1:
                    vix_l=det.get("vix_level",0); dxy_5d=det.get("dxy_chg5d",0); oil_5d=det.get("oil_chg5d",0); us10y_5d=det.get("us10y_chg5d",0)
                    mp=[]
                    if vix_l>25: mp.append(f"VIX at {vix_l:.0f} signals high fear — safe-haven demand supports gold, but expect larger swings")
                    elif vix_l<15: mp.append(f"VIX at {vix_l:.0f} is calm — expect tighter ranges and smaller moves")
                    if dxy_5d<-0.3: mp.append(f"the dollar has weakened {dxy_5d:+.1f}% over 5 days, which is a tailwind for gold")
                    elif dxy_5d>0.3: mp.append(f"the dollar has strengthened {dxy_5d:+.1f}% over 5 days, creating headwinds for gold")
                    if us10y_5d>0.5: mp.append(f"yields are rising ({us10y_5d:+.1f}% 5d), increasing opportunity cost of holding gold")
                    elif us10y_5d<-0.5: mp.append(f"yields are falling ({us10y_5d:+.1f}% 5d), making gold relatively more attractive")
                    if abs(oil_5d)>2: mp.append(f"oil moved {oil_5d:+.1f}% over 5 days — {'rising inflation expectations support gold' if oil_5d>0 else 'falling inflation eases gold demand'}")
                    if mp: nar.append("<b>Macro environment:</b> " + "; ".join(mp) + ".")
                # 7. Regime
                reg=det.get("regime","normal"); vz2=det.get("vol_zscore",0)
                if reg=="crisis": nar.append(f"<b>REGIME WARNING:</b> Volatility is at crisis levels (z-score {vz2:+.1f}). Expect 2-3x larger moves than normal. The ML SL is widened accordingly. Position sizing is reduced via the confidence-based risk multiplier.")
                elif reg=="high_vol": nar.append(f"Volatility is elevated (z-score {vz2:+.1f}). Larger moves expected — both pullbacks and runs will be deeper/longer than average.")
                elif reg=="low_vol": nar.append(f"Volatility is low (z-score {vz2:+.1f}). Expect tight ranges and smaller targets. The SL model should produce tighter stops.")
                # 8. Candle patterns
                nar.append(f"<b>CBDR candles:</b> First candle was {'green' if det['fc_green'] else 'red'} {det['fc_pat']}, last candle was {'green' if det['lc_green'] else 'red'} {det['lc_pat']}. "
                    f"{'Strong bullish close (close near range high at ' + str(det.get('close_pos',0)) + ') suggests momentum.' if det.get('close_pos',0)>0.7 else ('Weak close (near range low) suggests selling pressure.' if det.get('close_pos',0)<0.3 else 'Neutral close position.')}")
                # 9. Historical context
                sim=tdf[(tdf["day"]==det["day"])&(tdf["direction"]==d)]
                if len(sim)>=3:
                    nar.append(f"<b>Historical pattern:</b> On {det['day']}s with {d} breakouts (n={len(sim)}), average pullback is {sim['pb_depth'].mean():.2f}x and average run is {sim['max_run'].mean():.1f}x. "
                        f"Median run is {sim['max_run'].median():.1f}x (more reliable than mean). Win rate at 1.3x SL: {(sim['pb_depth']<1.3).mean()*100:.0f}%.")
                # 10. Confidence
                nar.append(f"<b>Confidence:</b> Entry model agreement is {conf*100:.0f}%, direction confidence is {dconf*100:.0f}%. "
                    f"Combined risk multiplier: <b>{rmult}x</b> of base risk ({adj_risk:.2f}%). {'High conviction — full sizing.' if rmult>1.0 else ('Moderate conviction.' if rmult>0.5 else 'Low conviction — reduced sizing to protect capital.')}")
                # 11. Price trend
                p5d=det.get("pct_5d",0); p20d=det.get("pct_20d",0)
                nar.append(f"<b>Trend:</b> Gold is {'up' if p5d>0 else 'down'} {abs(p5d):.1f}% over 5 days and {'up' if p20d>0 else 'down'} {abs(p20d):.1f}% over 20 days. "
                    f"{'Both timeframes aligned ' + d + ' — momentum is with the trade.' if (p5d>0)==(d=='bullish') and (p20d>0)==(d=='bullish') else 'Trend divergence — counter-trend risk.'}")

                st.markdown('<div class="reason">' + "<br><br>".join(nar) + '</div>', unsafe_allow_html=True)
                # PDF download
                try:
                    pdf_bytes=generate_pdf(det,today_pred,ml,tdf,asset_name)
                    st.download_button("Download Signal PDF",pdf_bytes,f"cbdr_signal_{det['date']}.pdf","application/pdf")
                except Exception as e:
                    st.caption(f"PDF generation: {e}")
            elif det.get("direction") is None: st.info("No breakout yet.")
            else: st.info("Need 40+ trades for ML.")
        else: st.warning("Could not detect CBDR.")

    # ═══ ARENA — FIXED: robust display, no None values ═══
    with tabs[1]:
        st.markdown("### Strategy Arena — Test Period")
        st.markdown("Ensemble uses **adaptive risk** (0.1x-1.5x per trade). Sharpe/Sortino = per-trade. Calmar = return/maxDD.")
        if ml and ml.get("strategies"):
            stdata=ml["strategies"]; colors_list=["#8b949e","#58a6ff","#f85149","#3fb950","#d2a828"]
            strat_labels={"Baseline":"Baseline (boundary, 1.5x SL, 3SD TP)","ML_Retest":"ML Retest (limit, 1.3x SL)",
                "Breakout_Only":"Breakout-Only (bo close, ML TP)","Ensemble":"Ensemble (retest OR bo, adaptive)",
                "Ensemble_Guard":"Ensemble Guard (skip low-conf)"}
            comp_rows=[]; strat_stats={}
            for key,tds in stdata.items():
                if tds is None or (isinstance(tds,pd.DataFrame) and tds.empty): continue
                if not isinstance(tds,pd.DataFrame): continue
                try:
                    rm_vals=tds["risk_mult"].values if "risk_mult" in tds.columns else None
                    s=calc_stats(tds["outcome"].tolist(),tds["r"].tolist(),capital,base_risk,rm_vals)
                    strat_stats[key]=s
                    if s.get("active",0)>0:
                        fl=tds[tds["outcome"].isin(["win","loss"])]
                        arv=fl["rr"].mean() if "rr" in fl.columns and len(fl)>0 else 0
                        et=fl["entry_type"].value_counts().to_dict() if "entry_type" in fl.columns else {}
                        comp_rows.append({"Strategy":strat_labels.get(key,key),"Trades":int(s["active"]),"Skip":int(s["skipped"]),
                            "WR":f'{s["wr"]:.1f}%',"PF":float(s["pf"]),"Avg R:R":f'{arv:.1f}:1',
                            "Sharpe":float(s["sharpe"]),"Sortino":float(s["sortino"]),"Calmar":float(s["calmar"]),
                            "P&L":f'${s["dollar_pnl"]:,.0f}',"Return":f'{s["return_pct"]:.1f}%',
                            "Max DD":f'{s["max_dd_pct"]:.1f}%',"Loss Streak":int(s["max_loss_streak"])})
                except Exception as ex:
                    st.warning(f"Error computing {key}: {ex}")
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows),use_container_width=True,hide_index=True)
            else:
                st.warning("No strategies have active trades. Check data or period.")
            # Equity curves
            fig=go.Figure()
            for i2,(key,_) in enumerate(stdata.items()):
                s2=strat_stats.get(key)
                if s2 and s2.get("eq_curve") and len(s2["eq_curve"])>1:
                    fig.add_trace(go.Scatter(y=s2["eq_curve"],mode="lines",name=strat_labels.get(key,key)[:25],line=dict(color=colors_list[i2%len(colors_list)],width=2)))
            fig.add_hline(y=capital,line_dash="dash",line_color="#30363d")
            fig.update_layout(**PBG,height=400,yaxis_title="$",yaxis_tickprefix="$",title="Equity Curves")
            st.plotly_chart(fig,use_container_width=True)
            # Drawdown
            fig2=go.Figure()
            for i2,(key,_) in enumerate(stdata.items()):
                s2=strat_stats.get(key)
                if s2 and s2.get("dd_series") and len(s2["dd_series"])>1:
                    c=colors_list[i2%len(colors_list)]
                    fig2.add_trace(go.Scatter(y=[-d for d in s2["dd_series"]],mode="lines",name=strat_labels.get(key,key)[:25],line=dict(color=c,width=1.5),fill="tozeroy",fillcolor=hex_to_rgba(c,0.08)))
            fig2.update_layout(**PBG,height=300,yaxis_title="DD %",title="Drawdowns")
            st.plotly_chart(fig2,use_container_width=True)
            # ═══ PER-MODEL INDIVIDUAL PnL ═══
            pmr = ml.get("per_model_pnl", {})
            if pmr:
                st.markdown("---")
                st.markdown("#### Individual Model Performance (Test Period)")
                st.caption("Each model's predictions simulated as standalone Ensemble-style trades (breakout entry, 1.3x SL, model-predicted TP).")
                pm_rows = []
                for mn, stats in sorted(pmr.items(), key=lambda x: -x[1].get("pf",0)):
                    pm_rows.append({
                        "Model": mn, "Trades": stats["trades"], "WR": f'{stats["wr"]:.1f}%',
                        "PF": stats["pf"], "Avg R": stats["avg_r"], "Total R": stats["total_r"],
                        "Type": "PCA" if "PCA" in mn else "Standard"
                    })
                if pm_rows:
                    st.dataframe(pd.DataFrame(pm_rows), use_container_width=True, hide_index=True)
                    # Highlight best
                    best_pm = max(pmr.items(), key=lambda x: x[1].get("pf",0))
                    st.success(f"Best individual model: **{best_pm[0]}** — PF {best_pm[1]['pf']}, WR {best_pm[1]['wr']}%, Avg R {best_pm[1]['avg_r']}")
        else: st.info("Need 40+ trades.")

    # ═══ MODELS ═══
    with tabs[2]:
        st.markdown("### Model Arena")
        if ml:
            dm=ml.get("dir_model")
            if dm and dm.get("results"):
                st.markdown("#### Direction Classifier")
                rows=[]
                for nm,r in dm["results"].items():
                    for sn in ["train","val","test"]:
                        if sn in r: rows.append({"Model":nm,"Split":sn.title(),"Acc":f'{r[sn]["acc"]}%',"F1":f'{r[sn].get("f1",0)}%',"N":r[sn]["n"]})
                if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                if dm.get("best"): st.success(f"Best: **{dm['best'][0]}**")
                if not dm["importance"].empty:
                    top=dm["importance"].head(15)
                    fig=go.Figure(data=[go.Bar(x=top.values,y=top.index,orientation="h",marker_color="#d2a828")])
                    fig.update_layout(**PBG,height=380,yaxis=dict(autorange="reversed")); st.plotly_chart(fig,use_container_width=True)
            st.markdown("---")
            cr=ml.get("retest_clf_results",{})
            if cr: st.dataframe(pd.DataFrame([{"Split":k.title(),"Acc":f'{v["acc"]:.1f}%',"N":v["n"]} for k,v in cr.items()]),use_container_width=True,hide_index=True)
            for mk,title,clr in [("entry","Entry Depth","#58a6ff"),("exit","Max Run","#3fb950")]:
                em2=ml.get(mk)
                if not em2: continue
                st.markdown(f"#### {title}")
                if em2["best"]: st.success(f"Best: **{em2['best'][0]}**")
                rows=[]
                for nm,r in em2["results"].items():
                    for sn in ["train","val","test"]:
                        if sn in r and isinstance(r[sn],dict): rows.append({"Model":nm,"Split":sn.title(),"MAE":r[sn].get("mae","?"),"R2":r[sn].get("r2","?"),"N":r[sn].get("n","?")})
                if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                if not em2["importance"].empty:
                    top=em2["importance"].head(15)
                    fig=go.Figure(data=[go.Bar(x=top.values,y=top.index,orientation="h",marker_color=clr)])
                    fig.update_layout(**PBG,height=380,yaxis=dict(autorange="reversed")); st.plotly_chart(fig,use_container_width=True)
                st.markdown("---")

    # ═══ SESSIONS ═══
    with tabs[3]:
        st.markdown("### Session Analysis")
        if "bo_session" in tdf.columns:
            bg=tdf.groupby("bo_session").agg(n=("max_run","count"),avg_run=("max_run","mean"),avg_pb=("pb_depth","mean")).round(3).reset_index()
            bg.columns=["Session","N","Avg Run","Avg PB"]; st.dataframe(bg,use_container_width=True,hide_index=True)
            scm={"asia":"#58a6ff","london":"#3fb950","ny":"#f85149","ldn_ny_overlap":"#d2a828","off_hours":"#8b949e"}
            fig=make_subplots(rows=1,cols=2,subplot_titles=("Run","PB"))
            cl2=[scm.get(s,"#8b949e") for s in bg["Session"]]
            fig.add_trace(go.Bar(x=bg["Session"],y=bg["Avg Run"],marker_color=cl2),row=1,col=1)
            fig.add_trace(go.Bar(x=bg["Session"],y=bg["Avg PB"],marker_color=cl2),row=1,col=2)
            fig.update_layout(**PBG,height=350,showlegend=False); st.plotly_chart(fig,use_container_width=True)

    # ═══ RISK ═══
    with tabs[4]:
        st.markdown("### Risk Analytics")
        if ml and ml.get("strategies"):
            keys=list(ml["strategies"].keys())
            ssel=st.selectbox("Strategy",keys,index=min(3,len(keys)-1))
            tds=ml["strategies"][ssel]
            if tds is not None and not tds.empty:
                fl=tds[tds["outcome"].isin(["win","loss"])]
                if len(fl)>=2:
                    rm=fl["risk_mult"].values if "risk_mult" in fl.columns else None
                    s=calc_stats(fl["outcome"].tolist(),fl["r"].tolist(),capital,base_risk,rm)
                    c1,c2,c3,c4,c5,c6=st.columns(6)
                    mcard(c1,"Sharpe",s["sharpe"]); mcard(c2,"Sortino",s["sortino"]); mcard(c3,"Calmar",s["calmar"])
                    mcard(c4,"Max DD",s["max_dd_pct"],"pct"); mcard(c5,"Loss Streak",s["max_loss_streak"],"int"); mcard(c6,"PF",s["pf"])
                    ra=fl["r"].values
                    fig=go.Figure(); fig.add_trace(go.Histogram(x=ra,nbinsx=30,marker_color="#58a6ff",opacity=0.8))
                    fig.add_vline(x=0,line_dash="dash",line_color="#f85149"); fig.update_layout(**PBG,height=300,title="R Distribution")
                    st.plotly_chart(fig,use_container_width=True)
                    # Reversal analysis
                    ls15=tdf[tdf["pb_depth"]>=1.5]
                    if len(ls15)>0:
                        rv=ls15[ls15["max_run"]>=1.0]; rp=len(rv)/len(ls15)*100
                        st.markdown(f"**Post-SL Reversal:** {len(ls15)} trades hit 1.5x SL, **{rp:.0f}%** reversed to profit.")
                    # Loss breakdown
                    st.markdown("#### Loss Analysis")
                    losses=fl[fl["outcome"]=="loss"]
                    if len(losses)>0 and "loss_reason" in losses.columns:
                        lr_counts=losses["loss_reason"].value_counts()
                        lr_rows=[]
                        for reason, cnt in lr_counts.items():
                            pct=cnt/len(losses)*100
                            desc={"wrong_direction":"Direction was wrong — price never went in our favor",
                                  "stopped_out":"Normal SL hit — pullback exceeded our stop level",
                                  "reversed_near_tp":"Got within 70% of TP then reversed to hit SL",
                                  "expired_negative":"Neither TP nor SL hit, closed at a loss"}.get(reason, reason)
                            lr_rows.append({"Reason":reason,"Count":cnt,"% of Losses":f"{pct:.0f}%","Description":desc})
                        st.dataframe(pd.DataFrame(lr_rows),use_container_width=True,hide_index=True)
                    # Win detail
                    wins=fl[fl["outcome"]=="win"]
                    if len(wins)>0 and "beyond_tp" in wins.columns:
                        avg_beyond=wins["beyond_tp"].mean()
                        st.markdown(f"**Win Analysis:** {len(wins)} wins. Average move beyond TP: **{avg_beyond:.2f}x** range. "
                            f"{'Significant continuation — consider trailing stops.' if avg_beyond>0.5 else 'Most wins captured near full move.'}")
                    # ═══ SL SIMULATOR — interactive slider ═══
                    st.markdown("---")
                    st.markdown("#### SL Sensitivity Simulator")
                    st.caption("Adjust the base SL multiplier and see its effect on performance. This re-simulates all test trades with the new SL.")
                    sim_sl=st.slider("SL Multiplier (x range)",0.5,3.0,1.3,0.1,key="sl_sim")
                    # Re-simulate with new SL
                    sim_results=[]
                    for _,trow in fl.iterrows():
                        apb2=trow.get("actual_pb",0); amr2=trow.get("actual_run",0)
                        ed2=trow.get("pred_entry",0) if trow.get("entry_type","")=="limit" else 0
                        tp2x=trow.get("pred_exit",2.0)+ed2
                        sc2=trow.get("spread_cost",0)
                        o2,r2,rr2,td2=sim_trade(apb2,amr2,ed2,sim_sl,tp2x,sc2)
                        sim_results.append({"outcome":o2,"r":r2})
                    sim_df=pd.DataFrame(sim_results)
                    if len(sim_df)>0:
                        sim_wins=(sim_df["outcome"]=="win").sum(); sim_n=len(sim_df)
                        sim_wr=sim_wins/sim_n*100; sim_pf=sim_df[sim_df["r"]>0]["r"].sum()/max(abs(sim_df[sim_df["r"]<0]["r"].sum()),0.01)
                        sim_avg_r=sim_df["r"].mean()
                        c1s,c2s,c3s,c4s=st.columns(4)
                        mcard(c1s,f"WR @ {sim_sl}x SL",sim_wr,"pct")
                        mcard(c2s,"Avg R",sim_avg_r)
                        mcard(c3s,"PF",round(sim_pf,2))
                        mcard(c4s,"Wins/Total",f"{sim_wins}/{sim_n}")

    # ═══ DISTRIBUTIONS ═══
    with tabs[5]:
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(); fig.add_trace(go.Histogram(x=tdf["pb_depth"],nbinsx=30,marker_color="#58a6ff",opacity=.8))
            fig.update_layout(**PBG,height=300,title="Pullback Depth"); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=go.Figure(); fig.add_trace(go.Histogram(x=tdf["max_run"],nbinsx=30,marker_color="#3fb950",opacity=.8))
            fig.update_layout(**PBG,height=300,title="Max Run"); st.plotly_chart(fig,use_container_width=True)
        for col,lb2 in [("direction","Direction"),("day","Day"),("bo_session","BO Session"),("fc_pat","1st Candle"),("wbias","Week Bias"),("mpos","Month Pos")]:
            if col not in tdf.columns: continue
            g=tdf.groupby(col).agg(n=("pb_depth","count"),pb=("pb_depth","mean"),run=("max_run","mean")).round(3).reset_index()
            g.columns=[lb2,"N","Avg PB","Avg Run"]; st.markdown(f"#### {lb2}"); st.dataframe(g,use_container_width=True,hide_index=True)

    # ═══ TRADE LOGS — CSV downloads for ALL strategies ═══
    with tabs[6]:
        st.markdown("### Trade Logs")
        # Raw data log
        sc2=["date","day","cbdr_from","cbdr_to","cbdr_candles","cbdr_hours","cbdr_open","cbdr_high","cbdr_low","cbdr_close","range_mode","range_high","range_low","range_size","direction","bo_session","retest_session","pb_depth","max_run","fc_pat","cbdr_trend","near_sup","near_res","dist_sup","sup_str","cbdr_rvol","bo_vsurge","wbias","mbias","mpos","evt_any","pct_5d","pct_20d","regime","realized_vol_20d","vol_zscore","vol_ratio_5_20","fib_500_dist","fib_618_dist","fib_786_dist","price_at_fib_zone","vix_level","dxy_level","oil_level","us10y_level"]
        sc2=[c for c in sc2 if c in tdf.columns]
        st.markdown("#### Raw CBDR Data (all breakout days)")
        st.dataframe(tdf[sc2].round(3),use_container_width=True,hide_index=True,height=300)
        st.download_button("Download Raw Data CSV",tdf.to_csv(index=False),"cbdr_v15_raw.csv","text/csv")
        st.markdown("---")
        # Strategy trade logs
        if ml and ml.get("strategies"):
            strat_labels2={"Baseline":"Baseline","ML_Retest":"ML Retest","Breakout_Only":"Breakout-Only","Ensemble":"Ensemble","Ensemble_Guard":"Ensemble Guard"}
            st.markdown("#### Strategy Trade Logs")
            for key,tds in ml["strategies"].items():
                if tds is None or (isinstance(tds,pd.DataFrame) and tds.empty): continue
                label2=strat_labels2.get(key,key)
                n_filled=len(tds[tds["outcome"].isin(["win","loss"])]) if "outcome" in tds.columns else 0
                with st.expander(f"{label2} — {n_filled} filled / {len(tds)} total"):
                    show_cols=["date","direction","action","entry_type","outcome","loss_reason","r","rr",
                        "entry_price","sl_price","tp_price","sl_used","pred_sl","risk_mult","spread_cost",
                        "max_adverse","max_favorable","tp_pct_reached","beyond_tp",
                        "pred_entry","pred_exit","actual_pb","actual_run","retest_prob","confidence","dir_conf"]
                    show_cols=[c for c in show_cols if c in tds.columns]
                    st.dataframe(tds[show_cols].round(3),use_container_width=True,hide_index=True,height=300)
                    csv=tds.to_csv(index=False)
                    st.download_button(f"Download {label2} CSV",csv,f"cbdr_{key}_trades.csv","text/csv",key=f"dl_{key}")
            # Combined CSV with all strategies
            st.markdown("---")
            all_trades=[]
            for key,tds in ml["strategies"].items():
                if tds is None or (isinstance(tds,pd.DataFrame) and tds.empty): continue
                t2=tds.copy(); t2["strategy"]=strat_labels2.get(key,key)
                all_trades.append(t2)
            if all_trades:
                combined=pd.concat(all_trades,ignore_index=True)
                st.download_button("Download ALL Strategies CSV",combined.to_csv(index=False),"cbdr_all_strategies.csv","text/csv",key="dl_all")
else:
    st.info("Click Run Full Analysis to start.")
    st.markdown('''### v15 — Ensemble + Multi-Model

**Core:** 70%+ of 1.5x SL stops reversed to profit. Ensemble ALWAYS trades every day.

| Scenario | Entry | SL |
|---|---|---|
| Retest prob >= 50% | Limit at 30% predicted depth | 1.3x range |
| No retest | Market at breakout close | 1.0x range |
| Low confidence < 40% | Skip (Guard only) | - |

**Risk:** 0.1x-1.5x base per trade by confidence. 7 models compete. Zero leakage.

**Features:** Candle patterns, S/R strength+distance, Volume (CBDR relative, BO surge, trend), Session (Asia/London/NY/overlap), Day/Week/Month bias, Events, **Macro: VIX (fear), DXY (dollar — inverse to gold), US10Y (yield pressure), Oil (inflation proxy)**. All macro uses previous day close = zero leakage.
    ''')
