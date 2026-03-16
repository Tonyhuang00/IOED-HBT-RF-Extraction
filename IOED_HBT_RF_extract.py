# 核心升級:
# 1. 拔除複雜的 st.secrets，採用全域統一密碼，消除環境報錯。
# 2. 萃取演算法 extract_limit 完全遵守側邊欄的 freq_min 與 freq_max 設定。
# 3. 自動遮蔽指定低頻以下的所有雜訊（預設 0.4 GHz 以下不列入萃取計算）。
# 4. 保留負頻率發散保護、三態智慧判定，以及完美的 ADS 視覺佈局。
# ==============================================================================

import io, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="IOED HBT RF EXTRACTION ver5.2", layout="wide", page_icon="📡")

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 🔐 實驗室專屬密碼鎖 (全域統一密碼版)
# ═══════════════════════════════════════════════════════════════════════════════
# 👇 你可以直接在這裡修改實驗室的專屬密碼
LAB_PASSWORD = "CHWUCHWUCHWU"

def check_password():
    def password_entered():
        if st.session_state["pwd_input"] == LAB_PASSWORD:
            st.session_state["authenticated"] = True
            del st.session_state["pwd_input"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated", False):
        return True

    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("請輸入 IOED 實驗室專屬密碼以解鎖工具。")
        st.text_input("Access Password", type="password", on_change=password_entered, key="pwd_input")
        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("❌ 密碼錯誤 (Access Denied)")
    return False

if not check_password():
    st.stop()

# ── 解鎖後的主介面 ──
st.title("📡 IOED HBT RF Tool (Function Only · ver5.2)")
st.caption("""
**Core Settings:** Using NumPy, Extraction Range = Plot Range
* If all below 0dB, return None
* If cross 0dB before 50GHz, return cross section point
* If no cross section above 50GHz, then using single extrapolate or UIUC method depending on the slope.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Touchstone S2P Parser & Matrices
# ═══════════════════════════════════════════════════════════════════════════════
def parse_s2p(content: str):
    freq_unit, fmt, z0, data_lines = 'hz', 'ma', 50.0, []
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith('!'): continue
        if s.startswith('#'):
            parts = s[1:].lower().split()
            for i, p in enumerate(parts):
                if p in ('hz', 'khz', 'mhz', 'ghz'): freq_unit = p
                elif p in ('ma', 'db', 'ri'): fmt = p
                elif p == 'r' and i + 1 < len(parts):
                    try: z0 = float(parts[i + 1])
                    except: pass
            continue
        data_lines.append(s)

    vals = np.array([float(x) for x in ' '.join(data_lines).split()])
    n = len(vals) // 9
    vals = vals[:n * 9].reshape(n, 9)
    freq = vals[:, 0] * {'hz': 1., 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}[freq_unit]

    def to_c(ca, cb):
        a, b = vals[:, ca], vals[:, cb]
        if fmt == 'db': return 10 ** (a / 20.) * np.exp(1j * np.deg2rad(b))
        if fmt == 'ma': return a * np.exp(1j * np.deg2rad(b))
        return a + 1j * b

    S = np.zeros((n, 2, 2), dtype=complex)
    for (r, c), (ca, cb) in zip([(0, 0), (1, 0), (0, 1), (1, 1)], [(1, 2), (3, 4), (5, 6), (7, 8)]):
        S[:, r, c] = to_c(ca, cb)
    return freq, S, z0

def s_to_y(S, z0=50.):
    s11, s12, s21, s22 = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]
    d = (1 + s11) * (1 + s22) - s12 * s21
    Y = np.zeros_like(S)
    Y[:, 0, 0] = ((1 - s11) * (1 + s22) + s12 * s21) / (d * z0)
    Y[:, 0, 1] = -2. * s12 / (d * z0)
    Y[:, 1, 0] = -2. * s21 / (d * z0)
    Y[:, 1, 1] = ((1 + s11) * (1 - s22) + s12 * s21) / (d * z0)
    return Y

def _inv2(M):
    out = np.zeros_like(M)
    for i in range(len(M)):
        try: out[i] = np.linalg.inv(M[i])
        except: out[i] = np.full((2, 2), np.nan + 0j)
    return out

y_to_z = z_to_y = _inv2

def deembed_open_short(Y_dut, Y_open, Y_short): return z_to_y(y_to_z(Y_dut - Y_open) - y_to_z(Y_short - Y_open))
def deembed_thru_half(Y_dut, Y_thru_deemb): return z_to_y(y_to_z(Y_dut) - 0.5 * y_to_z(Y_thru_deemb))
def strict_freq_check(f_dut, f_dummy, dummy_name):
    if len(f_dut) != len(f_dummy) or not np.allclose(f_dut, f_dummy, rtol=1e-5):
        raise ValueError(f"❌ 頻率網格不匹配：DUT 與 {dummy_name} 點數/範圍不同，嚴禁插值。")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. RF Metrics & 範圍同步智慧萃取邏輯
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(Y, freq_hz):
    f = freq_hz * 1e-9
    y11, y12, y21, y22 = Y[:, 0, 0], Y[:, 0, 1], Y[:, 1, 0], Y[:, 1, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        h21 = -y21 / y11
        num_u = np.abs(y21 - y12) ** 2
        den_u = 4. * (y11.real * y22.real - y12.real * y21.real)
        U = np.where(den_u > 0, num_u / den_u, np.nan)

        num_k = 2. * y11.real * y22.real - (y12 * y21).real
        K = num_k / (np.abs(y12 * y21) + 1e-60)
        MSG = np.abs(y21) / (np.abs(y12) + 1e-30)
        MAG = MSG * (K - np.sqrt(np.clip(K ** 2 - 1., 0, None)))
        MAG_MSG = np.where(K > 1., MAG, MSG)

    return pd.DataFrame({
        "Freq (GHz)": f,
        "|h21|² (dB)": 10 * np.log10(np.abs(h21) ** 2 + 1e-30),
        "Mason U (dB)": 10 * np.log10(np.abs(U) + 1e-30),
        "MAG/MSG (dB)": 10 * np.log10(np.abs(MAG_MSG) + 1e-30),
        "K Factor": K,
        "fT Plateau (GHz)": f * np.abs(h21),
        "fmax U Plateau (GHz)": f * np.sqrt(np.abs(U)),
        "fmax MAG Plateau (GHz)": f * np.sqrt(np.abs(MAG_MSG)),
    })

def extract_limit(freq_ghz, gain_db, plateau_arr, n_pts, f_min, f_max):
    valid_mask = (freq_ghz >= f_min) & (freq_ghz <= f_max) & ~np.isnan(gain_db)
    if not np.any(valid_mask): return np.nan, np.nan, "No Data"

    f_v = freq_ghz[valid_mask]
    g_v = gain_db[valid_mask]
    p_v = plateau_arr[valid_mask]
    N   = len(f_v)

    if np.nanmax(g_v) <= 0: return np.nan, np.nan, "No Gain"

    above = g_v >= 0
    crossings = np.where(above[:-1] & ~above[1:])[0]

    genuine_idx = None
    for idx in crossings[::-1]:
        cnt = 0
        for j in range(idx, -1, -1):
            if above[j]: cnt += 1
            else: break
        if cnt < 10: continue
        seg_start = max(0, idx - cnt + 1)
        if seg_start > int(0.80 * N) and cnt < 20: continue
        genuine_idx = idx
        break

    if genuine_idx is None:
        if np.median(g_v) > 0:
            v_plat = np.nanmax(p_v) if not np.isnan(p_v).all() else np.nan
            n_use  = min(n_pts, len(f_v))
            x_log  = np.log10(f_v[-n_use:])
            y_gain = g_v[-n_use:]
            v_extrap = np.nan
            if len(x_log) >= 2:
                with np.errstate(all='ignore'):
                    m, c = np.polyfit(x_log, y_gain, 1)
                    if m < 0: v_extrap = 10**(-c / m)
            return v_extrap, v_plat, "Extrap & Plat."
        return np.nan, np.nan, "No Gain"

    idx = genuine_idx
    s = max(0, idx - n_pts // 2 + 1)
    e = min(N, idx + n_pts // 2 + 1 + (n_pts % 2))
    if (e - s) < 2: s, e = max(0, idx), min(N, idx + 2)

    with np.errstate(all='ignore'):
        deg   = min(2, e - s - 1)
        poly  = np.polyfit(g_v[s:e], f_v[s:e], deg)
        v_cross = np.polyval(poly, 0.0)
        if v_cross <= 0 or v_cross < f_v[s] or v_cross > f_v[e - 1]:
            x1, x2 = g_v[idx], g_v[idx + 1]
            y1, y2 = f_v[idx], f_v[idx + 1]
            v_cross = y1 + (0 - x1) * (y2 - y1) / (x2 - x1)

    return v_cross, np.nan, "0dB Cross"

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Processing Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def process_dut(content, filename, s1_o, s1_s, s2_o, s2_s, s3_t, n_pts, f_min, f_max):
    freq, S, z0 = parse_s2p(content)
    Y_raw = s_to_y(S, z0)
    df_raw = compute_metrics(Y_raw, freq)

    Y_fin = Y_raw
    stages = []
    d1_o = d1_s = None

    if s1_o and s1_s:
        f1o, S1o, z1o = s1_o; f1s, S1s, z1s = s1_s
        strict_freq_check(freq, f1o, "Probe Open")
        Y1o, Y1s = s_to_y(S1o, z1o), s_to_y(S1s, z1s)
        d1_o, d1_s = Y1o, Y1s
        Y_fin = deembed_open_short(Y_fin, Y1o, Y1s)
        stages.append("Probe")

    if s2_o and s2_s:
        f2o, S2o, z2o = s2_o; f2s, S2s, z2s = s2_s
        strict_freq_check(freq, f2o, "Dev Open")
        Y2o, Y2s = s_to_y(S2o, z2o), s_to_y(S2s, z2s)
        if d1_o is not None:
            Y2o = deembed_open_short(Y2o, d1_o, d1_s)
            Y2s = deembed_open_short(Y2s, d1_o, d1_s)
        Y_fin = deembed_open_short(Y_fin, Y2o, Y2s)
        stages.append("Dev(O/S)")

    if s3_t:
        f3t, S3t, z3t = s3_t
        strict_freq_check(freq, f3t, "Dev Thru")
        Y3t = s_to_y(S3t, z3t)
        if d1_o is not None: Y3t = deembed_open_short(Y3t, d1_o, d1_s)
        if 'Dev(O/S)' in stages:
            Y3t_raw = s_to_y(S3t, z3t)
            if d1_o is not None: Y3t_raw = deembed_open_short(Y3t_raw, d1_o, d1_s)
            Y3t = deembed_open_short(Y3t_raw, Y2o, Y2s)
        Y_fin = deembed_thru_half(Y_fin, Y3t)
        stages.append("Dev(Thru)")

    note = " + ".join(stages) if stages else "None"
    df_fin = compute_metrics(Y_fin, freq) if stages else None

    df_e = df_fin if df_fin is not None else df_raw
    f_arr = df_e["Freq (GHz)"].values

    fT_cr, fT_pl, ft_m = extract_limit(f_arr, df_e["|h21|² (dB)"].values, df_e["fT Plateau (GHz)"].values, n_pts, f_min, f_max)
    fmU_cr, fmU_pl, fmU_m = extract_limit(f_arr, df_e["Mason U (dB)"].values, df_e["fmax U Plateau (GHz)"].values, n_pts, f_min, f_max)
    fmM_cr, fmM_pl, fmM_m = extract_limit(f_arr, df_e["MAG/MSG (dB)"].values, df_e["fmax MAG Plateau (GHz)"].values, n_pts, f_min, f_max)

    stem = re.sub(r'\.s2p$', '', filename, flags=re.IGNORECASE)
    m = re.search(r'[Vv][Cc][Ee][_\-]?([\d]+(?:p\d+)?)\s*[Vv]', stem)
    vce = float(m.group(1).replace('p', '.')) if m else None
    m = re.search(r'[Ii][Bb][_\-]?([\d]+(?:p\d+)?)\s*([pnuUmM]?)[Aa]?', stem)

    def _si(v, p):
        return float(v.replace('p', '.')) * {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'U': 1e-6, 'm': 1e-3, 'M': 1e-3, '': 1.}.get(p, 1.)

    ib = _si(m.group(1), m.group(2)) if m else None

    return df_raw, df_fin, {
        "Label": stem, "Vce (V)": vce, "Ib (A)": ib,
        "De-embedding": note,
        "fT Cross/Extrap (GHz)": fT_cr, "fT Plateau (GHz)": fT_pl, "fT Method": ft_m,
        "fmax U Cross/Extrap (GHz)": fmU_cr, "fmax U Plateau (GHz)": fmU_pl, "fmax U Method": fmU_m,
        "fmax MAG Cross/Extrap (GHz)": fmM_cr, "fmax MAG Plateau (GHz)": fmM_pl, "fmax MAG Method": fmM_m,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Plotting & UI Helpers
# ═══════════════════════════════════════════════════════════════════════════════
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def _darken(c):
    try:
        h = c.lstrip("#"); r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{max(0, r - 45):02x}{max(0, g - 45):02x}{max(0, b - 45):02x}"
    except: return c

def _layout(title, ytitle, yr, xr, legend_pos="right"):
    lx = 1.0 if legend_pos == "right" else 0.02
    la = "right" if legend_pos == "right" else "left"
    return dict(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Frequency (GHz)", type="log", range=[np.log10(max(xr[0], 1e-4)), np.log10(xr[1])], showgrid=True, gridcolor="#ebebeb", minor_showgrid=True),
        yaxis=dict(title=ytitle, range=list(yr), showgrid=True, gridcolor="#ebebeb"),
        legend=dict(x=lx, y=1.0, xanchor=la, yanchor="top", bgcolor="rgba(255,255,255,0.88)", bordercolor="#ccc", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white", height=440, margin=dict(l=55, r=25, t=45, b=50), hovermode="x unified", template="plotly_white"
    )

def _hline_plateau(fig, fval, color, label):
    if fval and np.isfinite(fval) and fval > 0:
        fig.add_hline(y=fval, line=dict(color=color, width=1.2, dash="dot"), annotation_text=f"{label}={fval:.3f} GHz", annotation_position="right", annotation_font_size=9)

def make_bode(df, title, xr, yr, sh21, su, smag, color):
    fig = go.Figure()
    f = df["Freq (GHz)"]
    hov = "Freq:%{x:.4f}GHz<br>Gain:%{y:.4f}dB<extra></extra>"
    if sh21: fig.add_trace(go.Scatter(x=f, y=df["|h21|² (dB)"], name="|h21|²", line=dict(color=color, width=2.5), hovertemplate=hov))
    if su: fig.add_trace(go.Scatter(x=f, y=df["Mason U (dB)"], name="Mason U", line=dict(color=_darken(color), width=2.5, dash="dash"), hovertemplate=hov))
    if smag: fig.add_trace(go.Scatter(x=f, y=df["MAG/MSG (dB)"], name="MAG/MSG", line=dict(color="#2ca02c", width=2.5, dash="dot"), hovertemplate=hov))
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="0 dB", annotation_position="bottom right")
    fig.update_layout(**_layout(f"Bode: {title}", "Gain (dB)", yr, xr))
    return fig

def make_plateau(df, res, title, xr, sh21, su, smag, color):
    f = df["Freq (GHz)"]
    cols = ([df["fT Plateau (GHz)"].tolist()] if sh21 else []) + ([df["fmax U Plateau (GHz)"].tolist()] if su else []) + ([df["fmax MAG Plateau (GHz)"].tolist()] if smag else [])
    arr = np.array([v for sub in cols for v in sub if np.isfinite(v) and v > 0])
    y_max = float(np.quantile(arr, 0.97)) * 1.3 if len(arr) else 100
    hov = "Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
    fig = go.Figure()
    if sh21:
        fig.add_trace(go.Scatter(x=f, y=df["fT Plateau (GHz)"], name="fT (f×|h21|)", line=dict(color=color, width=2.5), hovertemplate=hov))
        v_ft = res.get("fT Plateau (GHz)") if pd.notna(res.get("fT Plateau (GHz)")) else res.get("fT Cross/Extrap (GHz)")
        _hline_plateau(fig, v_ft, color, "fT")
    if su:
        fig.add_trace(go.Scatter(x=f, y=df["fmax U Plateau (GHz)"], name="fmax (f×√U)", line=dict(color=_darken(color), width=2.5, dash="dash"), hovertemplate=hov))
        v_fu = res.get("fmax U Plateau (GHz)") if pd.notna(res.get("fmax U Plateau (GHz)")) else res.get("fmax U Cross/Extrap (GHz)")
        _hline_plateau(fig, v_fu, _darken(color), "fmax(U)")
    if smag:
        fig.add_trace(go.Scatter(x=f, y=df["fmax MAG Plateau (GHz)"], name="fmax (f×√MAG)", line=dict(color="#2ca02c", width=2.5, dash="dot"), hovertemplate=hov))
    layout = _layout(f"Plateau: {title}", "GBP (GHz)", [0, y_max], xr)
    layout["annotations"] = [dict(x=0.5, y=1.06, xref="paper", yref="paper", showarrow=False, text="<b>判讀：</b>中高頻趨於平坦的水平值即為 fT/fmax", font=dict(size=10), bgcolor="rgba(255,255,0,0.3)", bordercolor="#aaa", borderwidth=1)]
    fig.update_layout(**layout)
    return fig

def _card(col, title, value_str, sub_str, color="#4A90D9"):
    col.markdown(
        f'<div style="padding:10px 14px;border-radius:8px;border-left:4px solid {color}; background:#f7f9fc;min-height:70px;margin-bottom:10px;">'
        f'<div style="font-size:0.74rem;color:#666;white-space:nowrap;">{title}</div>'
        f'<div style="font-size:1.15rem;font-weight:700;color:#1a2e4a;">{value_str}</div>'
        f'<div style="font-size:0.70rem;color:#888;margin-top:1px;">{sub_str}</div></div>', unsafe_allow_html=True)

def build_excel(summary_df, all_data):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        for k, v in all_data.items():
            df_p = v["df_fin"] if v["df_fin"] is not None else v["df_raw"]
            base = re.sub(r'[:\\/*?\[\]]', '_', Path(k).stem)[:28]
            df_p.to_excel(w, sheet_name=base, index=False)
            if v["df_fin"] is not None: v["df_raw"].to_excel(w, sheet_name=base[:24] + "_raw", index=False)
    return buf.getvalue()

def _load_cal(fobj):
    if fobj is None: return None
    try: return parse_s2p(fobj.getvalue().decode("utf-8", errors="ignore"))
    except Exception as e: st.sidebar.error(f"解析 {fobj.name} 失敗：{e}"); return None

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Sidebar & Main
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    st.markdown("#### 🔧 3段式 De-embedding")
    sw1 = st.toggle("① Probe / SLOT (Open-Short)", value=False)
    c1, c2 = st.columns(2)
    f1o = st.file_uploader("Probe Open", type=["s2p"]) if sw1 else None
    f1s = st.file_uploader("Probe Short", type=["s2p"]) if sw1 else None

    st.divider()
    sw2 = st.toggle("② Device Dummy (Open-Short)", value=False)
    c3, c4 = st.columns(2)
    f2o = st.file_uploader("Dev Open", type=["s2p"]) if sw2 else None
    f2s = st.file_uploader("Dev Short", type=["s2p"]) if sw2 else None

    st.divider()
    sw3 = st.toggle("③ Device Thru (Koolen Half-Z)", value=False)
    f3t = st.file_uploader("Dev Thru", type=["s2p"]) if sw3 else None

    st.divider()
    st.markdown("#### 📊 圖表與萃取控制")
    freq_min = st.number_input("頻率下限 (GHz)", value=0.4, min_value=0.0001, format="%.4f")
    freq_max = st.number_input("頻率上限 (GHz)", value=50.0, min_value=1.0)
    db_min = st.number_input("Bode Y 下限 (dB)", value=-50.0)
    db_max = st.number_input("Bode Y 上限 (dB)", value=50.0)
    n_pts = st.slider("內外插點數 (n_pts)", 2, 10, 2)
    show_raw = st.checkbox("疊加 Raw（de-emb 前）", value=True, disabled=not (sw1 or sw2 or sw3))

st.markdown("---")
dut_files = st.file_uploader("上傳 DUT .s2p 檔案（可多選）", type=["s2p"], accept_multiple_files=True)
if not dut_files: st.stop()

s1o, s1s = _load_cal(f1o) if sw1 else None, _load_cal(f1s) if sw1 else None
s2o, s2s = _load_cal(f2o) if sw2 else None, _load_cal(f2s) if sw2 else None
s3t = _load_cal(f3t) if sw3 else None

all_data, errors = {}, {}
for f in dut_files:
    try:
        content = f.getvalue().decode("utf-8", errors="ignore")
        df_raw, df_fin, res = process_dut(content, f.name, s1o, s1s, s2o, s2s, s3t, n_pts, freq_min, freq_max)
        all_data[f.name] = {"df_raw": df_raw, "df_fin": df_fin, **res}
    except Exception as e:
        errors[f.name] = str(e)

for fname, err in errors.items(): st.error(f"**{fname}**: {err}")
if not all_data: st.stop()

file_names = list(all_data.keys())
left, right = st.columns([1, 4], gap="medium")

with left:
    st.markdown("### 📂 檔案清單")
    selected = [n for n in file_names if st.checkbox(Path(n).stem, value=True, key=f"cb_{n}")]
    st.divider()
    st.markdown("##### 📈 Trace 選擇")
    sh21 = st.checkbox("|h21|² → fT", value=True, key="sh21")
    su = st.checkbox("Mason U → fmax(U)", value=True, key="su")
    smag = st.checkbox("MAG/MSG → fmax(MAG)", value=True, key="smag")
    st.divider()
    for i, n in enumerate(file_names):
        c = PALETTE[i % len(PALETTE)]
        st.markdown(
            f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:{c};margin-right:6px;vertical-align:middle"></span><small>{Path(n).stem[:20]}</small>',
            unsafe_allow_html=True)

xr, yr = (freq_min, freq_max), (db_min, db_max)

with right:
    tab_ov, tab_ind, tab_sum = st.tabs(["📊 Overlay", "📁 個別檔案", "📋 Summary"])

    # ── Overlay ─────────────────────────────────────────────────────────────
    with tab_ov:
        if not selected:
            st.info("請勾選至少一個檔案。")
        elif not (sh21 or su or smag):
            st.info("請至少勾選一個 Trace。")
        else:
            def _ov_bode():
                fig = go.Figure()
                for i, n in enumerate(file_names):
                    if n not in selected: continue
                    d = all_data[n]
                    df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                    c, lbl = PALETTE[i % len(PALETTE)], Path(n).stem
                    if show_raw and d["df_fin"] is not None and sh21:
                        fig.add_trace(go.Scatter(x=d["df_raw"]["Freq (GHz)"], y=d["df_raw"]["|h21|² (dB)"],
                                                 name=f"|h21|² raw–{lbl}", line=dict(color=c, width=1.2, dash="dot"),
                                                 opacity=0.35, hovertemplate="Freq:%{x:.4f}GHz<br>%{y:.4f}dB<extra></extra>"))
                    hov2 = "Freq:%{x:.4f}GHz<br>%{y:.4f}dB<extra></extra>"
                    if sh21: fig.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["|h21|² (dB)"], name=f"|h21|²–{lbl}",
                                                      line=dict(color=c, width=2.5), hovertemplate=hov2))
                    if su: fig.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["Mason U (dB)"], name=f"U–{lbl}",
                                                    line=dict(color=_darken(c), width=2.5, dash="dash"), hovertemplate=hov2))
                    if smag: fig.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["MAG/MSG (dB)"], name=f"MAG–{lbl}",
                                                      line=dict(color=c, width=2, dash="dot"), opacity=0.7, hovertemplate=hov2))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(**_layout("Overlay — Bode", "Gain (dB)", yr, xr))
                return fig


            def _ov_plat():
                fig = go.Figure();
                all_v = []
                for i, n in enumerate(file_names):
                    if n not in selected: continue
                    d = all_data[n]
                    df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                    c, lbl = PALETTE[i % len(PALETTE)], Path(n).stem
                    hov = "Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
                    if sh21:
                        fig.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["fT Plateau (GHz)"], name=f"fT–{lbl}",
                                                 line=dict(color=c, width=2.5), hovertemplate=hov))
                        valid_ft = df_p.loc[(df_p["Freq (GHz)"] >= freq_min) & (df_p["Freq (GHz)"] <= freq_max), "fT Plateau (GHz)"].dropna().tolist()
                        all_v.extend(valid_ft)
                    if su:
                        fig.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["fmax U Plateau (GHz)"], name=f"fmax(U)–{lbl}",
                                       line=dict(color=_darken(c), width=2.5, dash="dash"), hovertemplate=hov))
                        valid_fu = df_p.loc[(df_p["Freq (GHz)"] >= freq_min) & (df_p["Freq (GHz)"] <= freq_max), "fmax U Plateau (GHz)"].dropna().tolist()
                        all_v.extend(valid_fu)
                arr = np.array([v for v in all_v if np.isfinite(v) and v > 0])
                ym = float(np.quantile(arr, 0.97)) * 1.3 if len(arr) else 100
                fig.update_layout(**_layout("Overlay — UIUC Plateau", "GBP (GHz)", [0, ym], xr))
                return fig


            sub1, sub2 = st.tabs(["Bode Plot", "UIUC Plateau Plot"])
            with sub1:
                st.plotly_chart(_ov_bode(), use_container_width=True)
            with sub2:
                st.plotly_chart(_ov_plat(), use_container_width=True)

    # ── Individual Files ─────────────────────────────────────────────────────
    with tab_ind:
        if not selected:
            st.info("請勾選至少一個檔案。")
        else:
            def _fmt_card(v_cr, v_pl, method):
                if method == "No Gain" or method == "No Data": return method
                if method == "0dB Cross": return f"{v_cr:.3f} GHz" if np.isfinite(v_cr) else "N/A"
                if method == "Extrap & Plat.": return f"{v_pl:.3f} GHz" if np.isfinite(v_pl) else "N/A"
                return "N/A"

            def _sub_card(v_cr, method):
                if method == "Extrap & Plat.": return f"Extrap: {v_cr:.3f} GHz" if np.isfinite(v_cr) else "Extrap: N/A"
                return method

            stabs = st.tabs([Path(n).stem for n in selected])
            for stab, n in zip(stabs, selected):
                c = PALETTE[file_names.index(n) % len(PALETTE)]
                d = all_data[n]
                df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                with stab:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    _card(c1, "De-embedding", d["De-embedding"], "mode", "#888")
                    _card(c2, "fT (GHz)", _fmt_card(d["fT Cross/Extrap (GHz)"], d["fT Plateau (GHz)"], d["fT Method"]), _sub_card(d["fT Cross/Extrap (GHz)"], d["fT Method"]))
                    _card(c3, "fmax U (GHz)", _fmt_card(d["fmax U Cross/Extrap (GHz)"], d["fmax U Plateau (GHz)"], d["fmax U Method"]), _sub_card(d["fmax U Cross/Extrap (GHz)"], d["fmax U Method"]), "#d62728")
                    _card(c4, "fmax MAG (GHz)", _fmt_card(d["fmax MAG Cross/Extrap (GHz)"], d["fmax MAG Plateau (GHz)"], d["fmax MAG Method"]), _sub_card(d["fmax MAG Cross/Extrap (GHz)"], d["fmax MAG Method"]), "#2ca02c")
                    if d["Vce (V)"] is not None:
                        _card(c5, "Vce", f"{d['Vce (V)']} V", "bias", "#9467bd")
                    elif d["Ib (A)"] is not None:
                        _card(c5, "Ib", f"{d['Ib (A)'] * 1e6:.1f} µA", "bias", "#9467bd")

                    p1, p2 = st.columns(2)
                    with p1:
                        st.plotly_chart(make_bode(df_p, Path(n).stem, xr, yr, sh21, su, smag, c), use_container_width=True)
                    with p2:
                        st.plotly_chart(make_plateau(df_p, d, Path(n).stem, xr, sh21, su, smag, c), use_container_width=True)

                    with st.expander("📋 數據表"):
                        if d["df_fin"] is not None:
                            ta, tb = st.tabs(["De-embedded", "Raw"])
                            with ta:
                                st.dataframe(df_p.round(4), use_container_width=True, hide_index=True)
                            with tb:
                                st.dataframe(d["df_raw"].round(4), use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(df_p.round(4), use_container_width=True, hide_index=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    with tab_sum:
        rows = []
        for k, d in all_data.items():
            rows.append({
                "File": k,
                "De-embedding": d["De-embedding"],
                "Vce (V)": d["Vce (V)"],
                "Ib (µA)": round(d["Ib (A)"] * 1e6, 1) if d["Ib (A)"] else None,
                "fT Cross/Extp": d["fT Cross/Extrap (GHz)"],
                "fT Plateau": d["fT Plateau (GHz)"],
                "fT Method": d["fT Method"],
                "fmax U Cross/Extp": d["fmax U Cross/Extrap (GHz)"],
                "fmax U Plateau": d["fmax U Plateau (GHz)"],
                "fmax U Method": d["fmax U Method"],
                "fmax MAG Cross/Extp": d["fmax MAG Cross/Extrap (GHz)"],
                "fmax MAG Plateau": d["fmax MAG Plateau (GHz)"],
            })

        sum_df = pd.DataFrame(rows)
        st.markdown(f"### fT & fmax 摘要 (萃取範圍：{freq_min} ~ {freq_max} GHz)")

        fmt = {"Vce (V)": "{:.3f}", "Ib (µA)": "{:.1f}"}
        for col in sum_df.columns:
            if "Cross/Extp" in col or "Plateau" in col: fmt[col] = "{:.4f}"

        styled = sum_df.style.format(fmt, na_rep="—")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("📥 Excel", data=build_excel(sum_df, all_data), file_name="HBT_PureMath_v5.2.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with d2:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("Summary.csv", sum_df.to_csv(index=False).encode())
                for k, d in all_data.items():
                    df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                    stem = Path(k).stem
                    zf.writestr(f"{stem}.csv", df_p.to_csv(index=False).encode())
            st.download_button("📦 ZIP (CSV)", data=zbuf.getvalue(), file_name="HBT_PureMath_v5.2.zip",
                               mime="application/zip", use_container_width=True)