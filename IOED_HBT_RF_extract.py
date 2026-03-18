import io, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="IOED HBT RF EXTRACTION", layout="wide", page_icon="📡")


def check_password():
    def password_entered():
        try:
            correct_pwd = st.secrets["APP_PASSWORD"]
        except Exception:
            correct_pwd = "IOED"
        if st.session_state["pwd_input"] == correct_pwd:
            st.session_state["authenticated"] = True
            del st.session_state["pwd_input"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated", False):
        return True

    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Please enter IOED Lab Password or ask HBT team members. / 請輸入 IOED 實驗室專屬密碼或向團隊成員索取。")
        st.text_input("Access Password / 存取密碼", type="password", on_change=password_entered, key="pwd_input")
        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("❌ Password Incorrect (Access Denied) / 密碼錯誤 (拒絕存取)")
    return False


if not check_password():
    st.stop()

# ── Uploader key（用來一鍵清除所有已上傳檔案）──────────────────────────────
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

st.title("📡 IOED HBT RF Tool")

st.caption("""
**Core Settings / 核心設定:** Using NumPy, Extraction Range = Plot Range (萃取範圍與圖表同步)  
**Version 7.0:** Full-Width UI layout, Pre-rendered Plots, and Vertical Stacked Overlay / 滿版視覺重構、無檔案預載圖表、垂直疊圖
""")


def parse_s2p(content: str):
    freq_unit, fmt, z0, data_lines = 'hz', 'ma', 50.0, []
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith('!'): continue
        if s.startswith('#'):
            parts = s[1:].lower().split()
            for i, p in enumerate(parts):
                if p in ('hz', 'khz', 'mhz', 'ghz'):
                    freq_unit = p
                elif p in ('ma', 'db', 'ri'):
                    fmt = p
                elif p == 'r' and i + 1 < len(parts):
                    try:
                        z0 = float(parts[i + 1])
                    except:
                        pass
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


def y_to_s(Y, z0=50.):
    S = np.zeros_like(Y)
    I = np.eye(2)
    for i in range(len(Y)):
        yn = Y[i] * z0
        try:
            S[i] = np.dot(I - yn, np.linalg.inv(I + yn))
        except:
            S[i] = np.full((2, 2), np.nan + 0j)
    return S


def _inv2(M):
    out = np.zeros_like(M)
    for i in range(len(M)):
        try:
            out[i] = np.linalg.inv(M[i])
        except:
            out[i] = np.full((2, 2), np.nan + 0j)
    return out


y_to_z = z_to_y = _inv2


def deembed_open_short(Y_dut, Y_open, Y_short): return z_to_y(y_to_z(Y_dut - Y_open) - y_to_z(Y_short - Y_open))


def deembed_thru_half(Y_dut, Y_thru_deemb): return z_to_y(y_to_z(Y_dut) - 0.5 * y_to_z(Y_thru_deemb))


def strict_freq_check(f_dut, f_dummy, dummy_name):
    if len(f_dut) != len(f_dummy) or not np.allclose(f_dut, f_dummy, rtol=1e-5):
        raise ValueError(f"Frequency mismatch: DUT and {dummy_name} differ.")


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
        "Freq (GHz)": f, "|h21|² (dB)": 10 * np.log10(np.abs(h21) ** 2 + 1e-30),
        "Mason U (dB)": 10 * np.log10(np.abs(U) + 1e-30), "MAG/MSG (dB)": 10 * np.log10(np.abs(MAG_MSG) + 1e-30),
        "K Factor": K, "fT Plateau (GHz)": f * np.abs(h21), "fmax U Plateau (GHz)": f * np.sqrt(np.abs(U)),
        "fmax MAG Plateau (GHz)": f * np.sqrt(np.abs(MAG_MSG)),
    })


def extract_limit(freq_ghz, gain_db, plateau_arr, n_pts, f_min, f_max):
    valid_mask = (freq_ghz >= f_min) & (freq_ghz <= f_max) & ~np.isnan(gain_db)
    if not np.any(valid_mask): return np.nan, np.nan, "No Data"
    f_v, g_v, p_v, N = freq_ghz[valid_mask], gain_db[valid_mask], plateau_arr[valid_mask], len(freq_ghz[valid_mask])
    if np.nanmax(g_v) <= 0: return np.nan, np.nan, "No Gain"
    above = g_v >= 0
    crossings = np.where(above[:-1] & ~above[1:])[0]
    genuine_idx = None
    for idx in crossings[::-1]:
        cnt = 0
        for j in range(idx, -1, -1):
            if above[j]:
                cnt += 1
            else:
                break
        if cnt < 10: continue
        if max(0, idx - cnt + 1) > int(0.80 * N) and cnt < 20: continue
        genuine_idx = idx
        break
    if genuine_idx is None:
        if np.median(g_v) > 0:
            v_plat = np.nanmax(p_v) if not np.isnan(p_v).all() else np.nan
            n_use, v_extrap = min(n_pts, len(f_v)), np.nan
            if len(np.log10(f_v[-n_use:])) >= 2:
                with np.errstate(all='ignore'):
                    m, c = np.polyfit(np.log10(f_v[-n_use:]), g_v[-n_use:], 1)
                    if m < 0: v_extrap = 10 ** (-c / m)
            return v_extrap, v_plat, "Extrap & Plat."
        return np.nan, np.nan, "No Gain"
    idx = genuine_idx
    s, e = max(0, idx - n_pts // 2 + 1), min(N, idx + n_pts // 2 + 1 + (n_pts % 2))
    if (e - s) < 2: s, e = max(0, idx), min(N, idx + 2)
    with np.errstate(all='ignore'):
        v_cross = np.polyval(np.polyfit(g_v[s:e], f_v[s:e], min(2, e - s - 1)), 0.0)
        if v_cross <= 0 or v_cross < f_v[s] or v_cross > f_v[e - 1]:
            v_cross = f_v[idx] + (0 - g_v[idx]) * (f_v[idx + 1] - f_v[idx]) / (g_v[idx + 1] - g_v[idx])
    return v_cross, np.nan, "0dB Cross"


def process_dut(content, filename, s1_o, s1_s, s2_o, s2_s, s3_t, n_pts, f_min, f_max):
    freq, S_raw, z0 = parse_s2p(content)
    Y_raw = s_to_y(S_raw, z0)
    df_raw = compute_metrics(Y_raw, freq)
    Y_fin, stages, d1_o, d1_s = Y_raw, [], None, None
    if s1_o and s1_s:
        f1o, S1o, z1o = s1_o
        f1s, S1s, z1s = s1_s
        strict_freq_check(freq, f1o, "Probe Open")
        d1_o, d1_s = s_to_y(S1o, z1o), s_to_y(S1s, z1s)
        Y_fin = deembed_open_short(Y_fin, d1_o, d1_s)
        stages.append("Probe")
    if s2_o and s2_s:
        f2o, S2o, z2o = s2_o
        f2s, S2s, z2s = s2_s
        strict_freq_check(freq, f2o, "Dev Open")
        Y2o, Y2s = s_to_y(S2o, z2o), s_to_y(S2s, z2s)
        if d1_o is not None:
            Y2o, Y2s = deembed_open_short(Y2o, d1_o, d1_s), deembed_open_short(Y2s, d1_o, d1_s)
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
    S_fin = y_to_s(Y_fin, z0) if stages else S_raw
    df_e = df_fin if df_fin is not None else df_raw
    f_arr = df_e["Freq (GHz)"].values
    fT_cr, fT_pl, ft_m = extract_limit(f_arr, df_e["|h21|² (dB)"].values, df_e["fT Plateau (GHz)"].values, n_pts, f_min,
                                       f_max)
    fmU_cr, fmU_pl, fmU_m = extract_limit(f_arr, df_e["Mason U (dB)"].values, df_e["fmax U Plateau (GHz)"].values,
                                          n_pts, f_min, f_max)
    fmM_cr, fmM_pl, fmM_m = extract_limit(f_arr, df_e["MAG/MSG (dB)"].values, df_e["fmax MAG Plateau (GHz)"].values,
                                          n_pts, f_min, f_max)
    stem = re.sub(r'\.s2p$', '', filename, flags=re.IGNORECASE)
    m = re.search(r'[Vv][Cc][Ee][_\-]?([\d]+(?:p\d+)?)\s*[Vv]', stem)
    vce = float(m.group(1).replace('p', '.')) if m else None
    m = re.search(r'[Ii][Bb][_\-]?([\d]+(?:p\d+)?)\s*([pnuUmM]?)[Aa]?', stem)
    ib = float(m.group(1).replace('p', '.')) * {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'U': 1e-6, 'm': 1e-3, 'M': 1e-3,
                                                '': 1.}.get(m.group(2) if m else '', 1.) if m else None
    return df_raw, df_fin, S_fin, {
        "Label": stem, "Vce (V)": vce, "Ib (A)": ib, "De-embedding": note,
        "fT Cross/Extrap (GHz)": fT_cr, "fT Plateau (GHz)": fT_pl, "fT Method": ft_m,
        "fmax U Cross/Extrap (GHz)": fmU_cr, "fmax U Plateau (GHz)": fmU_pl, "fmax U Method": fmU_m,
        "fmax MAG Cross/Extrap (GHz)": fmM_cr, "fmax MAG Plateau (GHz)": fmM_pl, "fmax MAG Method": fmM_m,
    }


PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _darken(c):
    try:
        h = c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{max(0, r - 45):02x}{max(0, g - 45):02x}{max(0, b - 45):02x}"
    except:
        return c


def _layout(title, ytitle, yr, xr, legend_pos="right"):
    lx = 1.0 if legend_pos == "right" else 0.02
    la = "right" if legend_pos == "right" else "left"
    return dict(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Frequency (GHz)", type="log", range=[np.log10(max(xr[0], 1e-4)), np.log10(xr[1])],
                   showgrid=True, gridcolor="#ebebeb", minor_showgrid=True),
        yaxis=dict(title=ytitle, range=list(yr), showgrid=True, gridcolor="#ebebeb"),
        legend=dict(x=lx, y=1.0, xanchor=la, yanchor="top", bgcolor="rgba(255,255,255,0.88)", bordercolor="#ccc",
                    borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white", height=500, margin=dict(l=55, r=25, t=45, b=50),
        hovermode="x unified", template="plotly_white"
    )


def _hline_plateau(fig, fval, color, label):
    if fval and np.isfinite(fval) and fval > 0:
        fig.add_hline(y=fval, line=dict(color=color, width=1.2, dash="dot"), annotation_text=f"{label}={fval:.3f} GHz",
                      annotation_position="right", annotation_font_size=9)


def make_bode(df, title, xr, yr, sh21, su, smag, color):
    fig = go.Figure()
    f = df["Freq (GHz)"]
    hov = "Freq:%{x:.4f}GHz<br>Gain:%{y:.4f}dB<extra></extra>"
    if sh21: fig.add_trace(
        go.Scatter(x=f, y=df["|h21|² (dB)"], name="|h21|²", line=dict(color=color, width=2.5), hovertemplate=hov))
    if su: fig.add_trace(
        go.Scatter(x=f, y=df["Mason U (dB)"], name="Mason U", line=dict(color=_darken(color), width=2.5, dash="dash"),
                   hovertemplate=hov))
    if smag: fig.add_trace(
        go.Scatter(x=f, y=df["MAG/MSG (dB)"], name="MAG/MSG", line=dict(color="#2ca02c", width=2.5, dash="dot"),
                   hovertemplate=hov))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(**_layout(f"Bode Plot — {title}", "Gain (dB)", yr, xr))
    return fig


def make_plateau(df, res, title, xr, sh21, su, smag, color):
    f = df["Freq (GHz)"]
    cols = ([df["fT Plateau (GHz)"].tolist()] if sh21 else []) + (
        [df["fmax U Plateau (GHz)"].tolist()] if su else []) + ([df["fmax MAG Plateau (GHz)"].tolist()] if smag else [])
    arr = np.array([v for sub in cols for v in sub if np.isfinite(v) and v > 0])
    y_max = float(np.quantile(arr, 0.97)) * 1.3 if len(arr) else 100
    hov = "Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
    fig = go.Figure()
    if sh21:
        fig.add_trace(
            go.Scatter(x=f, y=df["fT Plateau (GHz)"], name="fT", line=dict(color=color, width=2.5), hovertemplate=hov))
        _hline_plateau(fig, res.get("fT Plateau (GHz)") if pd.notna(res.get("fT Plateau (GHz)")) else res.get(
            "fT Cross/Extrap (GHz)"), color, "fT")
    if su:
        fig.add_trace(go.Scatter(x=f, y=df["fmax U Plateau (GHz)"], name="fmax(U)",
                                 line=dict(color=_darken(color), width=2.5, dash="dash"), hovertemplate=hov))
        _hline_plateau(fig, res.get("fmax U Plateau (GHz)") if pd.notna(res.get("fmax U Plateau (GHz)")) else res.get(
            "fmax U Cross/Extrap (GHz)"), _darken(color), "fmax(U)")
    if smag:
        fig.add_trace(go.Scatter(x=f, y=df["fmax MAG Plateau (GHz)"], name="fmax(MAG)",
                                 line=dict(color="#2ca02c", width=2.5, dash="dot"), hovertemplate=hov))
    layout = _layout(f"Plateau Plot — {title}", "GBP (GHz)", [0, y_max], xr)
    fig.update_layout(**layout)
    return fig


def _extended_smith_grid(max_r=3.0):
    traces = []
    t = np.linspace(0, 2 * np.pi, 500)
    skip_kw = dict(mode='lines', showlegend=False, hoverinfo='skip')
    for r_outer in np.arange(1.0, max_r + 0.5, 1.0):
        lw = 1.6 if r_outer == 1.0 else 0.9
        col = 'rgba(60,60,60,0.85)' if r_outer == 1.0 else 'rgba(170,170,170,0.6)'
        traces.append(
            go.Scatter(x=np.cos(t) * r_outer, y=np.sin(t) * r_outer, line=dict(color=col, width=lw), **skip_kw))
    traces.append(
        go.Scatter(x=[-max_r, max_r], y=[0., 0.], line=dict(color='rgba(100,100,100,0.6)', width=0.8), **skip_kw))
    gray = 'rgba(155,155,155,0.5)'
    lw = 0.8
    for r in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]:
        cx = r / (r + 1)
        rad = 1.0 / (r + 1)
        xc = cx + rad * np.cos(t)
        yc = rad * np.sin(t)
        mag = np.sqrt(xc ** 2 + yc ** 2)
        xc[mag > max_r] = np.nan
        yc[mag > max_r] = np.nan
        traces.append(go.Scatter(x=xc, y=yc, line=dict(color=gray, width=lw), **skip_kw))
    for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
        for sign in [1, -1]:
            xv = sign * x
            rad_x = abs(1.0 / xv)
            xc = 1.0 + rad_x * np.cos(t)
            yc = (1.0 / xv) + rad_x * np.sin(t)
            mag = np.sqrt(xc ** 2 + yc ** 2)
            xc[mag > max_r] = np.nan
            yc[mag > max_r] = np.nan
            traces.append(go.Scatter(x=xc, y=yc, line=dict(color=gray, width=lw), **skip_kw))
    label_r = [0.0, 0.5, 1.0, 2.0, 5.0]
    label_x = [r / (r + 1) + 1 / (r + 1) for r in label_r]
    for lx, lr in zip(label_x, label_r):
        traces.append(go.Scatter(x=[lx], y=[0.01], mode='text', text=[f" {lr}"],
                                 textfont=dict(size=8, color='rgba(100,100,100,0.7)'), showlegend=False,
                                 hoverinfo='skip'))
    return traces


def make_smith(S, f_array, f_min, f_max, toggles, scales, title, max_r=3.0):
    mask = (f_array >= f_min) & (f_array <= f_max)
    S_plot = S[mask].copy()
    f_plot = f_array[mask]
    if len(S_plot) == 0:
        fig = go.Figure()
        fig.update_layout(title=dict(text="No Data in Range", font=dict(size=13)))
        return fig
    fig = go.Figure()
    for tr in _extended_smith_grid(max_r):
        fig.add_trace(tr)
    params = [
        ('S11', (0, 0), '#1f77b4', 'solid'),
        ('S22', (1, 1), '#ff7f0e', 'dash'),
        ('S21', (1, 0), '#2ca02c', 'dot'),
        ('S12', (0, 1), '#d62728', 'dashdot'),
    ]
    for key, (r, c), color, dash in params:
        if not toggles.get(key, False): continue
        s_val = S_plot[:, r, c] * scales.get(key, 1.0)
        mag_mask = np.abs(s_val) > max_r
        s_val[mag_mask] = np.nan + 1j * np.nan
        sc = scales.get(key, 1.0)
        if sc == 1.0:
            label = f"{key}  ({f_min:.2g}–{f_max:.2g} GHz)"
        elif sc > 1:
            label = f"{key} ×{sc:g}  ({f_min:.2g}–{f_max:.2g} GHz)"
        else:
            label = f"{key} ÷{1 / sc:g}  ({f_min:.2g}–{f_max:.2g} GHz)"
        hov = [f"f = {fv:.3f} GHz<br>Re(Γ) = {rv:.4f}<br>Im(Γ) = {iv:.4f}<br>|Γ| = {abs(complex(rv, iv)):.4f}" for
               fv, rv, iv in zip(f_plot, s_val.real, s_val.imag)]
        fig.add_trace(go.Scatter(x=s_val.real, y=s_val.imag, mode='lines', line=dict(color=color, width=2.2, dash=dash),
                                 name=label, text=hov, hoverinfo='text'))
        if len(s_val) > 0 and not np.isnan(s_val.real[0]):
            fig.add_trace(go.Scatter(x=[s_val.real[0]], y=[s_val.imag[0]], mode='markers',
                                     marker=dict(size=6, symbol='circle', color=color,
                                                 line=dict(color='white', width=1)), showlegend=False,
                                     hoverinfo='skip'))
    lim = max_r * 1.05
    fig.update_layout(
        title=dict(text=f"Smith Chart — {title}  (|Γ| up to {max_r})", font=dict(size=13)),
        xaxis=dict(title="Re(Γ)", range=[-lim, lim], showgrid=False, zeroline=False, scaleanchor="y", scaleratio=1,
                   tickvals=[-3, -2, -1, 0, 1, 2, 3]),
        yaxis=dict(title="Im(Γ)", range=[-lim, lim], showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=540, margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1),
        hovermode="closest",
        annotations=[dict(x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
                          text=f"freq ({f_min * 1000:.0f}MHz – {f_max:.1f}GHz)  ·  ●=低頻起始點",
                          font=dict(size=10, color="gray"), align="center")]
    )
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
    try:
        return parse_s2p(fobj.getvalue().decode("utf-8", errors="ignore"))
    except Exception as e:
        st.sidebar.error(f"Parse Failed / 解析失敗 {fobj.name}: {e}")
        return None


with st.sidebar:
    st.markdown("## ⚙️ Settings / 設定")
    st.markdown("#### 🔧 3-Step De-embedding / 三段式去嵌入")
    sw1 = st.toggle("① Probe (Open-Short)", value=False)
    c1, c2 = st.columns(2)
    _uk = st.session_state["uploader_key"]
    f1o = st.file_uploader("Probe Open",  type=["s2p"], key=f"f1o_{_uk}") if sw1 else None
    f1s = st.file_uploader("Probe Short", type=["s2p"], key=f"f1s_{_uk}") if sw1 else None
    st.divider()
    sw2 = st.toggle("② Device Dummy (Open-Short)", value=False)
    c3, c4 = st.columns(2)
    f2o = st.file_uploader("Dev Open",  type=["s2p"], key=f"f2o_{_uk}") if sw2 else None
    f2s = st.file_uploader("Dev Short", type=["s2p"], key=f"f2s_{_uk}") if sw2 else None
    st.divider()
    sw3 = st.toggle("③ Device Thru (Half-Z)", value=False)
    f3t = st.file_uploader("Dev Thru", type=["s2p"], key=f"f3t_{_uk}") if sw3 else None

    st.divider()
    st.markdown("#### 📊 Chart Control / 圖表控制")
    freq_min = st.number_input("Freq Min (GHz)", value=0.4, min_value=0.0001, format="%.4f")
    freq_max = st.number_input("Freq Max (GHz)", value=50.0, min_value=1.0)
    db_min = st.number_input("Bode Y Min (dB)", value=-50.0)
    db_max = st.number_input("Bode Y Max (dB)", value=50.0)
    n_pts = st.slider("Interpolation pts / 內外插點數", 2, 10, 2)
    show_raw = st.checkbox("Overlay Raw / 疊加原始數據", value=True, disabled=not (sw1 or sw2 or sw3))

    st.markdown("##### 📈 Trace Selection / 曲線顯示")
    sh21 = st.checkbox("|h21|² → fT", value=True, key="sh21")
    su = st.checkbox("Mason U → fmax(U)", value=True, key="su")
    smag = st.checkbox("MAG/MSG → fmax(MAG)", value=True, key="smag")

    st.divider()
    st.markdown("#### 🍩 Smith Chart / 史密斯圖設定")
    smith_f_min = st.number_input("Smith Freq Min (GHz)", value=0.4, min_value=0.0001, format="%.4f")
    smith_f_max = st.number_input("Smith Freq Max (GHz)", value=50.0, min_value=1.0)
    smith_max_r = st.slider("|Γ| 顯示範圍（1 = 標準單位圓，3 = 含外側 2 圈）",
                            min_value=1.0, max_value=5.0, value=1.0, step=0.5,
                            help="大於 1 時可顯示 |Γ|>1 的不穩定區域，與 ADS 的 Extended Smith Chart 一致")

    st.markdown("##### Display & Scale / 顯示與倍率")
    col_a, col_b = st.columns(2)
    show_s11 = col_a.checkbox("Show S11", value=True)
    scale_s11 = col_b.number_input("S11 倍率", value=1.0, step=0.1)

    col_c, col_d = st.columns(2)
    show_s22 = col_c.checkbox("Show S22", value=True)
    scale_s22 = col_d.number_input("S22 倍率", value=1.0, step=0.1)

    col_e, col_f = st.columns(2)
    show_s21 = col_e.checkbox("Show S21", value=True)
    scale_s21 = col_f.number_input("S21 倍率", value=1.0, step=0.1)

    col_g, col_h = st.columns(2)
    show_s12 = col_g.checkbox("Show S12", value=True)
    scale_s12 = col_h.number_input("S12 倍率", value=1.0, step=0.1)

col_up_rf, col_clr_rf = st.columns([5, 1])
with col_up_rf:
    dut_files = st.file_uploader(
        "Upload DUT .s2p files / 上傳 DUT",
        type=["s2p"], accept_multiple_files=True,
        key=f"rf_dut_{st.session_state['uploader_key']}",
    )
with col_clr_rf:
    st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All", key="rf_clear",
                 help="清除所有已上傳的 DUT 與校準結構檔案"):
        st.session_state["uploader_key"] += 1
        st.rerun()

s1o, s1s = _load_cal(f1o) if sw1 else None, _load_cal(f1s) if sw1 else None
s2o, s2s = _load_cal(f2o) if sw2 else None, _load_cal(f2s) if sw2 else None
s3t = _load_cal(f3t) if sw3 else None

all_data, errors = {}, {}
if dut_files:
    for f in dut_files:
        try:
            df_raw, df_fin, S_fin, res = process_dut(f.getvalue().decode("utf-8", errors="ignore"), f.name, s1o, s1s,
                                                     s2o,
                                                     s2s, s3t, n_pts, freq_min, freq_max)
            all_data[f.name] = {"df_raw": df_raw, "df_fin": df_fin, "S_fin": S_fin, **res}
        except Exception as e:
            errors[f.name] = str(e)

for fname, err in errors.items(): st.error(f"**{fname}**: {err}")

file_names = list(all_data.keys())

if all_data:
    file_options = list(all_data.keys())
    selected_files = st.multiselect(
        "📂 選擇要分析與疊加的檔案 (支援輸入關鍵字搜尋、可一鍵清除)：",
        options=file_options,
        default=file_options,
        format_func=lambda x: Path(x).stem
    )
else:
    selected_files = []
    st.info("💡 等待上傳 DUT 模擬或量測檔案中...下方為預設的空白圖表區塊。")

xr, yr = (freq_min, freq_max), (db_min, db_max)

tab_ov, tab_ind, tab_sum = st.tabs(["📊 Overlay / 疊圖", "📁 Individual / 個別檔案", "📋 Summary / 總結"])

with tab_ov:
    st.markdown("### 📊 Bode Plot Overlay")
    f_bode = go.Figure()

    if all_data and selected_files and (sh21 or su or smag):
        for i, n in enumerate(selected_files):
            d, c, lbl = all_data[n], PALETTE[i % len(PALETTE)], Path(n).stem
            df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
            hov2 = "Freq:%{x:.4f}GHz<br>%{y:.4f}dB<extra></extra>"
            if show_raw and d["df_fin"] is not None and sh21:
                f_bode.add_trace(
                    go.Scatter(x=d["df_raw"]["Freq (GHz)"], y=d["df_raw"]["|h21|² (dB)"], name=f"|h21|² raw–{lbl}",
                               line=dict(color=c, width=1.2, dash="dot"), opacity=0.35, hovertemplate=hov2))
            if sh21: f_bode.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["|h21|² (dB)"], name=f"|h21|²–{lbl}",
                                                 line=dict(color=c, width=2.5), hovertemplate=hov2))
            if su: f_bode.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["Mason U (dB)"], name=f"U–{lbl}",
                                               line=dict(color=_darken(c), width=2.5, dash="dash"), hovertemplate=hov2))
            if smag: f_bode.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["MAG/MSG (dB)"], name=f"MAG–{lbl}",
                                                 line=dict(color=c, width=2, dash="dot"), opacity=0.7,
                                                 hovertemplate=hov2))

    f_bode.add_hline(y=0, line_dash="dash", line_color="black")
    f_bode.update_layout(**_layout("Overlay — Bode Plot", "Gain (dB)", yr, xr))
    f_bode.update_layout(height=550)
    st.plotly_chart(f_bode, use_container_width=True)

    st.divider()

    st.markdown("### 📊 Plateau Plot Overlay")
    f_plat = go.Figure()
    all_v = []

    if all_data and selected_files and (sh21 or su or smag):
        for i, n in enumerate(selected_files):
            d, c, lbl = all_data[n], PALETTE[i % len(PALETTE)], Path(n).stem
            df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
            hov = "Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
            if sh21:
                f_plat.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["fT Plateau (GHz)"], name=f"fT–{lbl}",
                                            line=dict(color=c, width=2.5), hovertemplate=hov))
                valid_ft = df_p.loc[(df_p["Freq (GHz)"] >= freq_min) & (
                            df_p["Freq (GHz)"] <= freq_max), "fT Plateau (GHz)"].dropna().tolist()
                all_v.extend(valid_ft)
            if su:
                f_plat.add_trace(go.Scatter(x=df_p["Freq (GHz)"], y=df_p["fmax U Plateau (GHz)"], name=f"fmax(U)–{lbl}",
                                            line=dict(color=_darken(c), width=2.5, dash="dash"), hovertemplate=hov))
                valid_fu = df_p.loc[(df_p["Freq (GHz)"] >= freq_min) & (
                            df_p["Freq (GHz)"] <= freq_max), "fmax U Plateau (GHz)"].dropna().tolist()
                all_v.extend(valid_fu)

    arr = np.array([v for v in all_v if np.isfinite(v) and v > 0])
    ym = float(np.quantile(arr, 0.97)) * 1.3 if len(arr) else 100
    f_plat.update_layout(**_layout("Overlay — Plateau Plot", "GBP (GHz)", [0, ym], xr))
    f_plat.update_layout(height=550)
    st.plotly_chart(f_plat, use_container_width=True)

with tab_ind:
    if not all_data or not selected_files:
        st.info("請上傳並選擇檔案以查看個別分析。")
    else:
        stabs = st.tabs([Path(n).stem for n in selected_files])
        for stab, n in zip(stabs, selected_files):
            c = PALETTE[file_names.index(n) % len(PALETTE)]
            d = all_data[n]
            df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]


            def _fmt_card(v_cr, v_pl, method):
                if method in ["No Gain", "No Data"]: return method
                if method == "0dB Cross": return f"{v_cr:.3f} GHz" if np.isfinite(v_cr) else "N/A"
                if method == "Extrap & Plat.": return f"{v_pl:.3f} GHz" if np.isfinite(v_pl) else "N/A"
                return "N/A"


            def _sub_card(v_cr, method):
                if method == "Extrap & Plat.": return f"Extrap: {v_cr:.3f} GHz" if np.isfinite(v_cr) else "Extrap: N/A"
                return method


            with stab:
                c1, c2, c3, c4, c5 = st.columns(5)
                _card(c1, "De-embedding", d["De-embedding"], "mode", "#888")
                _card(c2, "fT (GHz)", _fmt_card(d["fT Cross/Extrap (GHz)"], d["fT Plateau (GHz)"], d["fT Method"]),
                      d["fT Method"])
                _card(c3, "fmax U",
                      _fmt_card(d["fmax U Cross/Extrap (GHz)"], d["fmax U Plateau (GHz)"], d["fmax U Method"]),
                      d["fmax U Method"], "#d62728")
                _card(c4, "fmax MAG",
                      _fmt_card(d["fmax MAG Cross/Extrap (GHz)"], d["fmax MAG Plateau (GHz)"], d["fmax MAG Method"]),
                      d["fmax MAG Method"], "#2ca02c")
                if d["Vce (V)"] is not None:
                    _card(c5, "Vce", f"{d['Vce (V)']} V", "bias", "#9467bd")
                elif d["Ib (A)"] is not None:
                    _card(c5, "Ib", f"{d['Ib (A)'] * 1e6:.1f} µA", "bias", "#9467bd")

                ta, tb, tc = st.tabs(["Bode Plot", "Plateau Plot", "Smith Chart"])
                with ta:
                    st.plotly_chart(make_bode(df_p, Path(n).stem, xr, yr, sh21, su, smag, c), use_container_width=True)
                with tb:
                    st.plotly_chart(make_plateau(df_p, d, Path(n).stem, xr, sh21, su, smag, c),
                                    use_container_width=True)

                toggles = {'S11': show_s11, 'S22': show_s22, 'S21': show_s21, 'S12': show_s12}
                scales = {'S11': scale_s11, 'S22': scale_s22, 'S21': scale_s21, 'S12': scale_s12}
                with tc:
                    st.plotly_chart(
                        make_smith(d["S_fin"], df_p["Freq (GHz)"].values, smith_f_min, smith_f_max, toggles, scales,
                                   Path(n).stem, max_r=smith_max_r), use_container_width=True)

                with st.expander("📋 Data Table / 數據表"):
                    if d["df_fin"] is not None:
                        ta_data, tb_data = st.tabs(["De-embedded / 已去嵌入", "Raw / 原始數據"])
                        with ta_data:
                            st.dataframe(df_p.round(4), use_container_width=True, hide_index=True)
                        with tb_data:
                            st.dataframe(d["df_raw"].round(4), use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(df_p.round(4), use_container_width=True, hide_index=True)

with tab_sum:
    if not all_data:
        st.info("請上傳檔案以產生摘要報表。")
    else:
        rows = [{"File": k, "De-embedding": d["De-embedding"], "Vce (V)": d["Vce (V)"],
                 "Ib (µA)": round(d["Ib (A)"] * 1e6, 1) if d["Ib (A)"] else None,
                 "fT Cross": d["fT Cross/Extrap (GHz)"], "fT Plat": d["fT Plateau (GHz)"],
                 "fmax U Cross": d["fmax U Cross/Extrap (GHz)"], "fmax U Plat": d["fmax U Plateau (GHz)"]} for k, d in
                all_data.items()]
        sum_df = pd.DataFrame(rows)
        fmt = {c: "{:.4f}" for c in sum_df.columns if "Cross" in c or "Plat" in c}
        fmt["Vce (V)"] = "{:.3f}"
        fmt["Ib (µA)"] = "{:.1f}"
        st.dataframe(sum_df.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("📥 Excel", data=build_excel(sum_df, all_data), file_name="HBT_PureMath_v7.0.xlsx",
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
            st.download_button("📦 ZIP (CSV)", data=zbuf.getvalue(), file_name="HBT_PureMath_v7.0.zip",
                               mime="application/zip", use_container_width=True)