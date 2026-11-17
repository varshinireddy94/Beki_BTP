"""
BEKI FLOOD WATCH – ML-Based Flood Hazard Prediction and Decision Support System

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import map_utils as mp
import metrics_utils as me
import model_utils as mu
import report_utils as ru
from app_config import (
    COVERAGE_DOMAIN, COVERAGE_LABELS, COVERAGE_SIMULATED, CRS_NAME, CRS_NOTE,
    HAZARD_CLASSES, HAZARD_COLORS, HAZARD_NAMES, MODEL_INFO, N_CELLS, N_COLS,
    NOPRED_INFO, Q_MAX, Q_MIN,
)

st.set_page_config(
    page_title="Beki Flood Watch",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#1f2933"
MUTED = "#52606d"
NAVY = "#0b3954"
VIEW_GEO = "Geographic map (basemap)"
VIEW_XY = "Grid coordinates (X / Y axes)"

LIMITATIONS = [
    "The model is an FHI regression model. It was trained only on cells that "
    "were inundated deeper than 0.1 m and had NDWI < 0 (h_avg > 0.1 AND "
    "NDWI_avg < 0) in the hydraulic simulations.",
    "It does not predict which cells flood. There is no separate wet/dry model, "
    "and inundation at an unsimulated discharge cannot be determined from the "
    "six model inputs, so the dashboard does not invent an eligibility mask.",
    "In the 'Model domain' view, every cell with NDWI < 0 receives a predicted "
    "FHI. Read it as the hazard the cell would have IF it were inundated, not as "
    "a forecast that it will flood.",
    "Grey cells have no valid prediction. They are not 'Very Low', not 'dry' and "
    "must not be treated as safe.",
    "The 'Simulated inundation extent' view uses the hydraulic model's own "
    "inundation mask and is available only at the 48 simulated discharges.",
    "Validation accuracy: MAE 0.861, RMSE 14.31, R² 0.593. Errors are largest "
    "for very high FHI values. Straight horizontal/vertical bands in the map are "
    "an artefact of the tree model splitting on the X and Y coordinates.",
    "Hazard classes follow the report's FHI thresholds (0.5, 1, 2, 3).",
]


# ==========================================================================
# Styling
# ==========================================================================
def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: #0b2537; }
[data-testid="stSidebar"] * { color: #e6eef5; }
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #102a43 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #ffffff; }
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background: linear-gradient(90deg,#e4572e,#c51b1b); border: 0; color: #fff;
  font-weight: 700; padding: .7rem .5rem; font-size: 1.0rem;
  box-shadow: 0 4px 14px rgba(197,27,27,.35);
}
[data-testid="stSidebar"] .stButton button[kind="secondary"] {
  background: #13405c; border: 1px solid #2f6f93; color: #fff; font-weight: 600;
}
[data-testid="stSidebar"] hr { border-color: #24475f; }
.sb-brand { font-weight: 800; font-size: 1.25rem; letter-spacing: .04em; }
.sb-sub { font-size: .78rem; color: #9fb3c8 !important; margin-top: -4px; }
.sb-step { font-size: .72rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: #7cc4e4 !important; margin: 1rem 0 .2rem; }
.hero { background: linear-gradient(115deg,#0b2537 0%,#0b3954 55%,#0f6e8c 100%);
  border-radius: 14px; padding: 18px 26px; color: #fff; margin-bottom: 14px;
  display:flex; justify-content:space-between; align-items:center; gap: 16px; flex-wrap: wrap; }
.hero h1 { font-size: 1.65rem; margin: 0; font-weight: 800; letter-spacing: .03em; color:#fff; }
.hero p { margin: 2px 0 0; color: #bcd4e6; font-size: .92rem; }
.hero .pill { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
  border-radius: 999px; padding: 4px 12px; font-size: .78rem; color:#e6f2fa; white-space: nowrap; }
.strip { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin: 2px 0 10px; }
.badge { border-radius: 999px; padding: 4px 12px; font-size: .8rem; font-weight: 600;
  border: 1px solid #c3ccd6; background:#fff; color:#1f2933; }
.badge.q { background:#0b3954; color:#fff; border-color:#0b3954; font-size:.95rem; }
.badge.sim { background:#e3f4ea; border-color:#9fd3b0; color:#145a32; }
.badge.interp { background:#fff4d6; border-color:#f0cf7a; color:#7a5300; }
.caveat { background:#fff8e6; border:1px solid #f3d27a; border-left:5px solid #e0a100;
  border-radius:10px; padding:10px 14px; font-size:.86rem; color:#4a3b00; margin-bottom: 10px; }
.caveat b { color:#3a2e00; }
.card { background:#fff; border:1px solid #e1e7ee; border-radius:12px; padding:14px 16px;
  box-shadow: 0 1px 3px rgba(16,42,67,.06); height:100%; }
.card .lbl { font-size:.74rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:#52606d; }
.card .val { font-size:1.55rem; font-weight:800; color:#102a43; line-height:1.25; margin-top:4px; }
.card .val small { font-size:.85rem; font-weight:600; color:#52606d; }
.card .sub { font-size:.8rem; color:#52606d; margin-top:2px; }
.card.accent-high { border-top:4px solid #f68d2e; }
.card.accent-vhigh { border-top:4px solid #c51b1b; }
.card.accent-navy { border-top:4px solid #0b3954; }
.card.accent-teal { border-top:4px solid #0f6e8c; }
.cls { background:#fff; border:1px solid #e1e7ee; border-radius:12px; padding:10px 12px 10px 14px;
  position:relative; overflow:hidden; }
.cls:before { content:""; position:absolute; left:0; top:0; bottom:0; width:6px; background: var(--c); }
.cls .n { font-weight:700; font-size:.85rem; color:#1f2933; }
.cls .r { font-size:.72rem; color:#7b8794; }
.cls .a { font-size:1.2rem; font-weight:800; color:#102a43; margin-top:2px; }
.cls .p { font-size:.8rem; color:#52606d; }
.section { font-size:1.15rem; font-weight:800; color:#0b3954; margin: 22px 0 4px;
  display:flex; align-items:center; gap:8px; }
.section-sub { color:#52606d; font-size:.86rem; margin-bottom:10px; }
.cell-card { background:#fff; border:1px solid #e1e7ee; border-radius:12px; padding:14px 16px; }
.cell-card h4 { margin:0 0 8px; font-size:.95rem; color:#0b3954; }
.cell-card table { width:100%; font-size:.84rem; border-collapse:collapse; }
.cell-card td { padding:4px 0; border-bottom:1px solid #f0f3f6; color:#1f2933; }
.cell-card td:first-child { color:#52606d; }
.cell-card td:last-child { text-align:right; font-weight:600; }
.cat { display:inline-block; padding:3px 10px; border-radius:6px; font-weight:700; font-size:.85rem; }
.legend-row { display:flex; align-items:center; gap:8px; font-size:.82rem; margin:3px 0; color:#1f2933; }
.legend-sw { width:16px; height:16px; border-radius:3px; flex:0 0 16px; border:1px solid rgba(0,0,0,.12); }
.small-note { font-size:.78rem; color:#52606d; }
.step-card { background:#fff; border:1px solid #e1e7ee; border-radius:12px; padding:16px; height:100%; }
.step-card .k { width:30px; height:30px; border-radius:50%; background:#0b3954; color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:800; margin-bottom:8px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background:#eef2f6; border-radius:8px 8px 0 0; padding:8px 18px; font-weight:600; }
.stTabs [aria-selected="true"] { background:#0b3954 !important; color:#fff !important; }
</style>
""",
        unsafe_allow_html=True,
    )


# ==========================================================================
# Cached resources
# ==========================================================================
@st.cache_resource(show_spinner=False)
def get_model():
    return mu.load_model()


@st.cache_resource(show_spinner=False)
def get_static():
    grid = mu.load_grid()
    geom = mp.build_geometry(grid)
    masks = mu.load_simulated_masks()
    return grid, geom, masks, grid["cell_area_m2"].to_numpy()


@st.cache_data(show_spinner=False, max_entries=32)
def domain_fhi(q: float) -> np.ndarray:
    grid, _, _, _ = get_static()
    return mu.predict_domain_fhi(get_model(), grid, q)


def get_prediction(q: float, coverage: str) -> mu.Prediction:
    grid, _, masks, _ = get_static()
    return mu.make_prediction(domain_fhi(q), grid, q, coverage, masks)


@st.cache_data(show_spinner=False, max_entries=48)
def overlay_uri(q: float, coverage: str, view: str, opacity: float) -> str:
    _, geom, _, _ = get_static()
    pred = get_prediction(q, coverage)
    status2d = pred.grid2d(pred.status)
    if view == VIEW_GEO:
        png = mp.render_status_png(status2d, geom.NMx, geom.NMy,
                                   geom.merc_bbox, opacity=opacity)
    else:
        bbox = (geom.NX.min(), geom.NX.max(), geom.NY.min(), geom.NY.max())
        png = mp.render_status_png(status2d, geom.NX, geom.NY, bbox,
                                   opacity=opacity)
    return mp.png_data_uri(png)


@st.cache_resource(show_spinner=False)
def get_cell_index_raster():
    grid, geom, _, _ = get_static()
    return ru.build_cell_index_raster(grid, geom.NX, geom.NY, res=20.0)


@st.cache_data(show_spinner=False, max_entries=8)
def csv_bytes(q, coverage, predicted_only):
    grid, _, _, _ = get_static()
    return ru.prediction_csv(get_prediction(q, coverage), grid, predicted_only)


@st.cache_data(show_spinner=False, max_entries=8)
def native_tif_bytes(q, coverage):
    grid, _, _, _ = get_static()
    return ru.native_grid_geotiff(get_prediction(q, coverage), grid)


@st.cache_data(show_spinner=False, max_entries=8)
def resampled_tif_bytes(q, coverage):
    idx, transform = get_cell_index_raster()
    return ru.resampled_geotiff(get_prediction(q, coverage), idx, transform)


@st.cache_data(show_spinner=False, max_entries=8)
def map_png_bytes(q, coverage):
    _, geom, _, _ = get_static()
    pred = get_prediction(q, coverage)
    return ru.static_map_png(
        pred, geom, f"Predicted flood hazard – Q = {fmt_q(q)} cumec", dpi=250)


@st.cache_data(show_spinner=False, max_entries=8)
def pdf_bytes(q, coverage):
    grid, geom, masks, area = get_static()
    pred = get_prediction(q, coverage)
    summ = me.summary(pred, area)
    status, _ = mu.scenario_status(q, sorted(masks))
    return ru.pdf_report(
        pred, summ, me.top_cells(pred, grid, 10), geom, status,
        COVERAGE_LABELS[coverage],
        me.interpretation(summ, status, COVERAGE_LABELS[coverage]),
        LIMITATIONS,
    )


# ==========================================================================
# Helpers
# ==========================================================================
def fmt_q(q: float) -> str:
    return f"{q:,.0f}" if abs(q - round(q)) < 1e-9 else f"{q:,.1f}"


def is_simulated(q: float, masks) -> bool:
    return abs(q - round(q)) < 1e-9 and int(round(q)) in masks


def effective_coverage(q: float, wanted: str, masks) -> tuple[str, str | None]:
    if wanted == COVERAGE_SIMULATED and not is_simulated(q, masks):
        return COVERAGE_DOMAIN, (
            f"No simulated inundation extent exists for {fmt_q(q)} cumec "
            "(only the 48 simulated discharges have one), so the model-domain "
            "coverage is shown instead.")
    return wanted, None


def card(label, value, sub="", accent="accent-navy"):
    return (f'<div class="card {accent}"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def section(title, sub=None):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>',
                    unsafe_allow_html=True)


def text_color_for(bg: str) -> str:
    h = bg.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#111" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#fff"


def plotly_config(name: str):
    return {
        "scrollZoom": True, "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        "toImageButtonOptions": {"format": "png", "filename": name, "scale": 2},
    }


def curve_lookup(pred: mu.Prediction) -> list[np.ndarray]:
    """Flat cell indices for each hover trace (same order as map_utils)."""
    out = []
    for code in [1, 2, 3, 4, 5, 6, 7]:
        m = pred.status == code
        if m.any():
            out.append(np.flatnonzero(m))
    return out


def cell_from_selection(state, lookup) -> tuple[int, int] | None:
    """(row, col) of a clicked cell. Hover traces are the first curves of the
    figure, one per status code, so (curve, point index) maps exactly to a
    grid cell; customdata is used as a fallback."""
    try:
        pts = state["selection"]["points"]
    except (KeyError, TypeError):
        pts = []
    for p in pts or []:
        cn = p.get("curve_number")
        pi = p.get("point_index", p.get("point_number"))
        try:
            if cn is not None and pi is not None and int(cn) < len(lookup):
                flat = int(lookup[int(cn)][int(pi)])
                return divmod(flat, N_COLS)
        except (IndexError, TypeError, ValueError):
            continue
    return mp.selected_cell_from_event(state)


# ==========================================================================
# Sidebar
# ==========================================================================
def sidebar(masks):
    ss = st.session_state
    with st.sidebar:
        st.markdown('<div class="sb-brand">🌊 BEKI FLOOD WATCH</div>'
                    '<div class="sb-sub">Flood hazard decision support</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="sb-step">① River discharge</div>',
                    unsafe_allow_html=True)
        q_in = st.number_input(
            "Discharge Q (cumec, m³/s)", min_value=float(Q_MIN),
            max_value=float(Q_MAX), value=5350.0,
            step=50.0, format="%.0f", key="q_input",
            help=f"Any value from {Q_MIN:,} to {Q_MAX:,} cumec. You are not "
                 "limited to the simulated scenarios (e.g. 5,350 or 7,250).")
        if is_simulated(q_in, masks):
            st.caption("✔ A hydraulic simulation exists for this discharge.")
        else:
            st.caption("◆ Unsimulated discharge – the ML model interpolates.")
        gen = st.button("Generate Flood Hazard Map", type="primary",
                        width="stretch")

        st.markdown('<div class="sb-step">② Map settings</div>',
                    unsafe_allow_html=True)
        view = st.radio("Map view", [VIEW_GEO, VIEW_XY], key="view")
        basemap = st.selectbox("Basemap", list(mp.BASEMAPS), key="basemap",
                               disabled=view != VIEW_GEO)
        opacity = st.slider("Hazard layer opacity", 0.3, 1.0, 0.8, 0.05,
                            key="opacity")
        coverage = st.radio(
            "Prediction coverage", [COVERAGE_DOMAIN, COVERAGE_SIMULATED],
            format_func=lambda k: COVERAGE_LABELS[k], key="coverage",
            help="Model domain: all cells with NDWI < 0 (hazard IF inundated). "
                 "Simulated extent: only cells inundated > 0.1 m in the "
                 "hydraulic simulation – available at the 48 simulated "
                 "discharges only.")
        outline = st.checkbox("Show model grid boundary", True, key="outline")

        st.markdown('<div class="sb-step">③ Compare two discharges</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        qa = c1.number_input("Scenario A", float(Q_MIN), float(Q_MAX),
                             2000.0, 100.0, format="%.0f", key="qa")
        qb = c2.number_input("Scenario B", float(Q_MIN), float(Q_MAX),
                             8000.0, 100.0, format="%.0f", key="qb")
        cmp_btn = st.button("Compare scenarios", width="stretch")

        st.markdown("---")
        st.markdown(
            f'<div class="small-note" style="color:#9fb3c8">Model: '
            f'{MODEL_INFO["Algorithm"]} · {MODEL_INFO["Configuration"]}<br>'
            f'Validation MAE {MODEL_INFO["Validation MAE"]} · R² '
            f'{MODEL_INFO["Validation R²"]}<br>Grid: 286 × 600 curvilinear '
            f'cells</div>', unsafe_allow_html=True)
    return dict(q_in=q_in, gen=gen, view=view, basemap=basemap,
                opacity=opacity, coverage=coverage, outline=outline,
                qa=qa, qb=qb, cmp=cmp_btn)


# ==========================================================================
# Page sections
# ==========================================================================
def hero():
    st.markdown(
        '<div class="hero"><div><h1>BEKI FLOOD WATCH</h1>'
        '<p>ML-Based Flood Hazard Prediction and Decision Support System · '
        'Beki River Basin, Assam</p></div>'
        '<div style="display:flex;gap:8px;flex-wrap:wrap">'
        '<span class="pill">XGBoost FHI model</span>'
        '<span class="pill">171,600 grid cells</span>'
        '<span class="pill">Q 200 – 10,000 cumec</span></div></div>',
        unsafe_allow_html=True)


def caveat_box(pred: mu.Prediction):
    if pred.coverage == COVERAGE_DOMAIN:
        msg = ("<b>How to read this map:</b> colours show the hazard each cell "
               "would face <b>if it is inundated</b> at this discharge. The model "
               "does <b>not</b> predict the flood boundary (which cells actually "
               "get wet). <b>Grey cells have no prediction and are not safe by "
               "default.</b>")
    else:
        msg = ("<b>How to read this map:</b> predictions are shown only inside the "
               "<b>hydraulically simulated inundation extent</b> (depth > 0.1 m) "
               "for this discharge. <b>Grey cells were not assessed – do not treat "
               "them as safe.</b>")
    st.markdown(f'<div class="caveat">⚠️ {msg} '
                f'See <i>About the model</i> for all limitations.</div>',
                unsafe_allow_html=True)


def welcome(geom, grid, settings):
    st.markdown('<div class="section">Get started</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    steps = [
        ("1", "Enter a discharge",
         f"Type any river discharge between {Q_MIN:,} and {Q_MAX:,} cumec in "
         "the sidebar, e.g. 5,350 cumec."),
        ("2", "Generate the map",
         "Press <b>Generate Flood Hazard Map</b>. The tuned XGBoost model "
         "predicts the Flood Hazard Index (FHI) for every eligible cell."),
        ("3", "Explore and download",
         "Zoom, pan and click cells, review hazard areas and high-risk "
         "locations, compare scenarios, and download CSV, GeoTIFF or PDF."),
    ]
    for c, (k, t, d) in zip(cols, steps):
        c.markdown(f'<div class="step-card"><div class="k">{k}</div>'
                   f'<b>{t}</b><div class="small-note" style="margin-top:4px">'
                   f'{d}</div></div>', unsafe_allow_html=True)
    section("Study area",
            "Outline of the 286 × 600 curvilinear model grid on the Beki River. "
            "Generate a map to see predicted hazard.")
    if settings["view"] == VIEW_GEO:
        fig = mp.build_map_figure(None, grid, geom, None, settings["basemap"],
                                  height=640, width_hint=1300)
    else:
        fig = mp.build_xy_figure(None, grid, geom, None, height=640)
    st.plotly_chart(fig, width="stretch",
                    config=plotly_config("beki_study_area"))


def selected_cell_panel(pred, grid, sel):
    if sel is None:
        st.markdown(
            '<div class="cell-card"><h4>📍 Selected cell</h4>'
            '<div class="small-note">Hover over the map to inspect cells. '
            '<b>Click a cell</b> (or pick one in the High-risk table below) to '
            'pin its details here.</div></div>', unsafe_allow_html=True)
        return
    r, c = sel
    i = r * N_COLS + c
    code = int(pred.status[i])
    if code <= 5:
        name, color = HAZARD_NAMES[code], HAZARD_COLORS[code]
        fhi = f"{pred.fhi[i]:,.3f}"
    else:
        name, color = NOPRED_INFO[code][0], NOPRED_INFO[code][1]
        fhi = "no prediction"
    rows = [
        ("Discharge", f"{fmt_q(pred.q)} cumec"),
        ("Predicted FHI", fhi),
        ("X (easting)", f"{grid['X'].iat[i]:,.1f} m"),
        ("Y (northing)", f"{grid['Y'].iat[i]:,.1f} m"),
        ("Grid row / column", f"{r} / {c}"),
        ("Cell area", f"{grid['cell_area_m2'].iat[i]:,.0f} m²"),
        ("Bed elevation", f"{grid['bed_elevation'].iat[i]:,.2f} m"),
        ("Slope", f"{grid['slope_degrees'].iat[i]:.2f}°"),
        ("NDWI (avg)", f"{grid['NDWI_avg'].iat[i]:.3f}"),
    ]
    trs = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)
    extra = ("" if code <= 5 else
             f'<div class="small-note" style="margin-top:6px">'
             f'{NOPRED_INFO[code][2]}</div>')
    st.markdown(
        f'<div class="cell-card"><h4>📍 Selected cell</h4>'
        f'<span class="cat" style="background:{color};color:'
        f'{text_color_for(color)}">{name}</span>'
        f'<table style="margin-top:8px">{trs}</table>{extra}</div>',
        unsafe_allow_html=True)
    if st.button("Clear selection", key="clear_sel"):
        st.session_state.selected_cell = None
        st.rerun()


def legend_panel(pred):
    items = "".join(
        f'<div class="legend-row"><span class="legend-sw" style="background:'
        f'{col}"></span><b>{n}</b>&nbsp;<span class="small-note">{r}</span></div>'
        for _, n, col, r in HAZARD_CLASSES)
    for k, (label, col, _) in NOPRED_INFO.items():
        if (pred.status == k).any():
            items += (f'<div class="legend-row"><span class="legend-sw" style='
                      f'"background:{col}"></span>{label}</div>')
    st.markdown(f'<div class="cell-card" style="margin-top:10px"><h4>Legend – '
                f'Flood Hazard Index</h4>{items}<div class="small-note" '
                f'style="margin-top:6px">FHI = h × (v + 0.5); thresholds from '
                f'the project report.</div></div>', unsafe_allow_html=True)


def summary_cards(s: dict, pred):
    c = st.columns(4)
    c[0].markdown(card("Selected discharge", f"{fmt_q(pred.q)} <small>cumec</small>",
                       "River discharge Q (m³/s)", "accent-navy"),
                  unsafe_allow_html=True)
    c[1].markdown(card("Cells with predictions", f"{s['cells_predicted']:,}",
                       f"of {s['cells_total']:,} grid cells "
                       f"({100 * s['cells_predicted'] / s['cells_total']:.1f}%)",
                       "accent-teal"), unsafe_allow_html=True)
    c[2].markdown(card("Total predicted area",
                       f"{s['predicted_area_km2']:,.1f} <small>km²</small>",
                       f"of {s['grid_area_km2']:,.1f} km² grid "
                       f"({s['coverage_pct']:.1f}% coverage)", "accent-teal"),
                  unsafe_allow_html=True)
    c[3].markdown(card("High + Very High share",
                       f"{s['high_vhigh_pct']:.1f}<small>%</small>",
                       f"{s['high_area_km2'] + s['very_high_area_km2']:,.1f} km² "
                       "of predicted area", "accent-vhigh"),
                  unsafe_allow_html=True)
    st.write("")
    c = st.columns([1, 1, 2])
    c[0].markdown(card("High hazard area",
                       f"{s['high_area_km2']:,.1f} <small>km²</small>",
                       "2 < FHI ≤ 3", "accent-high"), unsafe_allow_html=True)
    c[1].markdown(card("Very High hazard area",
                       f"{s['very_high_area_km2']:,.1f} <small>km²</small>",
                       "FHI > 3", "accent-vhigh"), unsafe_allow_html=True)
    nop = [f"{k.replace('No prediction – ', '')}: {v['cells']:,} cells, "
           f"{v['area_km2']:,.1f} km²"
           for k, v in s["no_prediction"].items() if v["cells"]]
    c[2].markdown(card("Not assessed (no valid prediction)",
                       f"{s['grid_area_km2'] - s['predicted_area_km2']:,.1f} "
                       "<small>km²</small>",
                       " · ".join(nop) + " — hazard unknown, not safe.",
                       "accent-navy"), unsafe_allow_html=True)

    section("Area under each hazard category",
            "Areas use the true surface area of every curvilinear cell "
            "(cells range from about 1,100 m² to 15,300 m²), not a cell count.")
    cols = st.columns(5)
    for col, (_, r) in zip(cols, s["table"].iterrows()):
        col.markdown(
            f'<div class="cls" style="--c:{r["color"]}"><div class="n">'
            f'{r["Hazard category"]}</div><div class="r">{r["FHI range"]}</div>'
            f'<div class="a">{r["Area (km²)"]:,.1f} km²</div><div class="p">'
            f'{r["% of predicted area"]:.1f}% · {r["Cells"]:,} cells</div></div>',
            unsafe_allow_html=True)


def distribution_charts(s: dict):
    t = s["table"]
    section("Hazard distribution",
            "Share of the predicted area in each hazard category.")
    c1, c2 = st.columns([1, 1.35])
    donut = go.Figure(go.Pie(
        labels=t["Hazard category"], values=t["Area (km²)"], hole=0.6,
        sort=False, direction="clockwise",
        marker=dict(colors=t["color"], line=dict(color="white", width=2)),
        texttemplate="%{percent:.1%}", textposition="outside",
        customdata=t["Cells"],
        hovertemplate="<b>%{label}</b><br>%{value:,.2f} km²<br>%{percent:.1%} of "
                      "predicted area<br>%{customdata:,} cells<extra></extra>"))
    donut.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10), showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=12)),
        annotations=[dict(text=f"<b>{s['predicted_area_km2']:,.0f}</b><br>"
                               "km² predicted", showarrow=False,
                          font=dict(size=14, color=INK))],
        title=dict(text="Area share (%)", x=0.02, font=dict(size=13, color=MUTED)))
    c1.plotly_chart(donut, width="stretch",
                    config=plotly_config("hazard_share"))

    bar = go.Figure(go.Bar(
        y=t["Hazard category"], x=t["Area (km²)"], orientation="h",
        marker=dict(color=t["color"], line=dict(color="white", width=1)),
        text=[f"{a:,.1f} km² · {p:.1f}%" for a, p in
              zip(t["Area (km²)"], t["% of predicted area"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:,.2f} km²<extra></extra>"))
    bar.update_layout(
        height=360, margin=dict(l=10, r=40, t=30, b=40),
        xaxis=dict(title="Area (km²)", gridcolor="#eef2f6",
                   range=[0, t["Area (km²)"].max() * 1.35]),
        yaxis=dict(autorange="reversed"), plot_bgcolor="white",
        title=dict(text="Area by hazard category (km² and %)", x=0.02,
                   font=dict(size=13, color=MUTED)))
    c2.plotly_chart(bar, width="stretch",
                    config=plotly_config("hazard_area"))


def top10_section(top10: pd.DataFrame, table_key: str):
    section("Top 10 high-risk locations",
            "Cells with the highest predicted FHI. <b>Select a row</b> to "
            "highlight and zoom to that cell on the map above.")
    if top10.empty:
        st.info("No cells have valid predictions.")
        return

    def colour(v):
        code = {n: c for c, n in HAZARD_NAMES.items()}[v]
        bg = HAZARD_COLORS[code]
        return f"background-color:{bg};color:{text_color_for(bg)};font-weight:600"

    styled = top10.style.map(colour, subset=["Hazard category"])
    st.dataframe(
        styled, key=table_key, on_select="rerun", selection_mode="single-row",
        hide_index=True, width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Row": st.column_config.NumberColumn(width="small"),
            "Column": st.column_config.NumberColumn(width="small"),
            "X (m)": st.column_config.NumberColumn(format="%.1f"),
            "Y (m)": st.column_config.NumberColumn(format="%.1f"),
            "Predicted FHI": st.column_config.NumberColumn(format="%.3f"),
        })
    st.markdown('<div class="small-note">Very large FHI values come from cells '
                'where the model predicts deep, fast flow; prediction error is '
                'largest in this range (validation RMSE 14.3).</div>',
                unsafe_allow_html=True)


def downloads_section(q, coverage, pred):
    section("Downloads",
            "Export the current prediction for GIS, spreadsheets or reporting.")
    tag = f"Q{fmt_q(q).replace(',', '')}_{coverage}"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**A · CSV table**")
        st.caption("Row, column, X, Y, discharge, predicted FHI, hazard "
                   "category, prediction status and cell area.")
        only = st.toggle("Predicted cells only", False, key="csv_only")
        with st.spinner("Preparing CSV…"):
            data = csv_bytes(q, coverage, only)
        st.download_button("⬇ Download CSV", data,
                           file_name=f"beki_hazard_{tag}.csv",
                           mime="text/csv", width="stretch")
    with c2:
        st.markdown("**B · Hazard map image (PNG)**")
        st.caption("High-resolution map with legend, X/Y axes and title. "
                   "Opens in any image viewer, Word or PowerPoint.")
        with st.spinner("Preparing PNG map…"):
            png = map_png_bytes(q, coverage)
        st.download_button(
            "⬇ Download hazard map (PNG)", png,
            file_name=f"beki_hazard_map_{tag}.png", mime="image/png",
            width="stretch", type="primary")
        with st.expander("Preview map image"):
            st.image(png, width="stretch")
    with c3:
        st.markdown("**C · PDF report**")
        st.caption("Discharge, hazard map, distribution chart, area "
                   "statistics, top-10 high-risk cells, interpretation and "
                   "limitations.")
        with st.spinner("Building PDF report…"):
            pdf = pdf_bytes(q, coverage)
        st.download_button("⬇ Download PDF report", pdf,
                           file_name=f"beki_hazard_report_{tag}.pdf",
                           mime="application/pdf", width="stretch",
                           type="primary")


# ==========================================================================
# Comparison
# ==========================================================================
def comparison_tab(settings, grid, geom, masks, area):
    ss = st.session_state
    section("Compare two discharge scenarios",
            "See how flood hazard changes as discharge increases. Both maps use "
            "the same colour scale and legend.")
    if not ss.get("compare"):
        st.info("Choose **Scenario A** and **Scenario B** in the sidebar "
                "(e.g. 2,000 and 8,000 cumec) and press **Compare scenarios**.")
        return
    qa, qb = ss.compare
    cov_a, na = effective_coverage(qa, settings["coverage"], masks)
    cov_b, nb = effective_coverage(qb, settings["coverage"], masks)
    if cov_a != cov_b:
        cov_a = cov_b = COVERAGE_DOMAIN
    for n in {na, nb} - {None}:
        st.warning(n)
    with st.spinner(f"Predicting {fmt_q(qa)} and {fmt_q(qb)} cumec…"):
        pa, pb = get_prediction(qa, cov_a), get_prediction(qb, cov_b)
        cmp = me.compare(pa, pb, area)

    st.markdown(
        f'<div class="strip"><span class="badge q">A: {fmt_q(qa)} cumec</span>'
        f'<span class="badge q" style="background:#0f6e8c;border-color:#0f6e8c">'
        f'B: {fmt_q(qb)} cumec</span><span class="badge">Coverage: '
        f'{COVERAGE_LABELS[cov_a]}</span></div>', unsafe_allow_html=True)

    c = st.columns(4)
    sign = lambda v: f"{v:+,.1f}"  # noqa: E731
    c[0].markdown(card("Change in High area",
                       f"{sign(cmp['high_change'])} <small>km²</small>",
                       f"{cmp['table_a'].loc[3, 'Area (km²)']:,.1f} → "
                       f"{cmp['table_b'].loc[3, 'Area (km²)']:,.1f} km²",
                       "accent-high"), unsafe_allow_html=True)
    c[1].markdown(card("Change in Very High area",
                       f"{sign(cmp['very_high_change'])} <small>km²</small>",
                       f"{cmp['table_a'].loc[4, 'Area (km²)']:,.1f} → "
                       f"{cmp['table_b'].loc[4, 'Area (km²)']:,.1f} km²",
                       "accent-vhigh"), unsafe_allow_html=True)
    c[2].markdown(card("High + Very High",
                       f"{sign(cmp['hv_b'] - cmp['hv_a'])} <small>km²</small>",
                       f"{cmp['hv_a']:,.1f} → {cmp['hv_b']:,.1f} km²",
                       "accent-vhigh"), unsafe_allow_html=True)
    c[3].markdown(card("Cells moving to a higher class",
                       f"{cmp['area_up']:,.1f} <small>km²</small>",
                       f"lower class: {cmp['area_down']:,.1f} km² · unchanged: "
                       f"{cmp['area_same']:,.1f} km²", "accent-teal"),
                  unsafe_allow_html=True)
    if abs(cmp["pred_area_a"] - cmp["pred_area_b"]) > 1e-6:
        st.caption(f"Predicted area differs between scenarios: "
                   f"{cmp['pred_area_a']:,.1f} km² (A) vs "
                   f"{cmp['pred_area_b']:,.1f} km² (B), so compare the "
                   "percentages as well as the km² values.")

    m1, m2 = st.columns(2)
    for col, p, cov, lbl in [(m1, pa, cov_a, "A"), (m2, pb, cov_b, "B")]:
        with col:
            with st.spinner("Rendering map…"):
                uri = overlay_uri(p.q, cov, settings["view"], settings["opacity"])
            title = f"<b>Scenario {lbl}</b> · Q = {fmt_q(p.q)} cumec"
            if settings["view"] == VIEW_GEO:
                fig = mp.build_map_figure(p, grid, geom, uri,
                                          settings["basemap"], height=660,
                                          width_hint=650, title=title,
                                          compact=True, uirevision="cmp",
                                          show_outline=settings["outline"])
            else:
                fig = mp.build_xy_figure(p, grid, geom, uri, height=660,
                                         title=title, compact=True,
                                         uirevision="cmp",
                                         show_outline=settings["outline"])
            st.plotly_chart(fig, width="stretch",
                            config=plotly_config(f"beki_compare_{lbl}"),
                            key=f"cmp_map_{lbl}")

    section("Change in hazard-class distribution")
    ta, tb = cmp["table_a"], cmp["table_b"]
    g = go.Figure()
    for t, name, color in [(ta, f"A: {fmt_q(qa)} cumec", "#9aa5b1"),
                           (tb, f"B: {fmt_q(qb)} cumec", NAVY)]:
        g.add_trace(go.Bar(
            x=t["Hazard category"], y=t["Area (km²)"], name=name,
            marker=dict(color=color), customdata=t["% of predicted area"],
            text=[f"{a:,.0f}" for a in t["Area (km²)"]], textposition="outside",
            hovertemplate="%{x}<br>%{y:,.1f} km² (%{customdata:.1f}%)"
                          "<extra>" + name + "</extra>"))
    g.update_layout(barmode="group", height=380, plot_bgcolor="white",
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(title="Area (km²)", gridcolor="#eef2f6"),
                    legend=dict(orientation="h", y=1.12, x=0),
                    bargap=0.25, bargroupgap=0.08)
    c1, c2 = st.columns([1.3, 1])
    c1.plotly_chart(g, width="stretch",
                    config=plotly_config("compare_distribution"))
    with c2:
        d = cmp["delta"].copy()
        st.dataframe(
            d, hide_index=True, width="stretch",
            column_config={k: st.column_config.NumberColumn(format="%.2f")
                           for k in d.columns if k != "Hazard category"})

    section("Where does the hazard move? (class transitions, km²)",
            f"Rows: class at A ({fmt_q(qa)} cumec) · Columns: class at B "
            f"({fmt_q(qb)} cumec). Cells above the diagonal become more "
            "hazardous.")
    tr = cmp["transitions"]
    hm = go.Figure(go.Heatmap(
        z=tr.values, x=[f"B: {c}" for c in tr.columns],
        y=[f"A: {r}" for r in tr.index], colorscale="Blues",
        text=[[f"{v:,.1f}" for v in row] for row in tr.values],
        texttemplate="%{text}", hovertemplate="%{y} → %{x}<br>%{z:,.2f} km²"
                                              "<extra></extra>",
        colorbar=dict(title="km²")))
    hm.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                     yaxis=dict(autorange="reversed"))
    st.plotly_chart(hm, width="stretch",
                    config=plotly_config("transitions"))


# ==========================================================================
# About tab
# ==========================================================================
def about_tab(build_info, masks):
    section("About the model")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(pd.DataFrame(MODEL_INFO.items(), columns=["Item", "Value"]),
                     hide_index=True, width="stretch")
        st.markdown(
            "**Inputs (in order):** `Q, X, Y, bed_elevation, NDWI_avg, "
            "slope_degrees`. Only **Q** is entered by you; the five spatial "
            "features are read automatically for every cell from the stored "
            "grid dataset.")
    with c2:
        st.markdown("**Important limitations**")
        for s in LIMITATIONS:
            st.markdown(f"- {s}")
    section("Data and geometry")
    st.markdown(
        f"- Grid: **286 rows × 600 columns = 171,600 curvilinear cells**. The "
        "original X/Y cell centres are used unchanged; cell corners are the "
        "mean of the four surrounding centres (edges extrapolated).\n"
        f"- Cell areas are computed from each quadrilateral (shoelace formula): "
        f"{build_info.get('cell_area_m2_min', float('nan')):,.0f} – "
        f"{build_info.get('cell_area_m2_max', float('nan')):,.0f} m² "
        f"(median {build_info.get('cell_area_m2_median', float('nan')):,.0f} m²); "
        f"total grid area {build_info.get('total_grid_area_km2', float('nan')):,.1f} km². "
        "All cells were checked to be convex with consistent orientation.\n"
        f"- Hydraulically simulated discharges ({len(masks)}): "
        f"{', '.join(f'{q:,}' for q in sorted(masks))} cumec.\n"
        f"- Coordinate system: **{CRS_NAME}**. {CRS_NOTE}")
    section("Hazard classes (project report thresholds)")
    st.dataframe(pd.DataFrame(
        [(c, n, r) for c, n, _, r in HAZARD_CLASSES],
        columns=["Class", "Hazard category", "FHI range"]),
        hide_index=True)


# ==========================================================================
# Main
# ==========================================================================
def main():
    inject_css()
    ss = st.session_state
    ss.setdefault("active", None)
    ss.setdefault("selected_cell", None)
    ss.setdefault("focus", False)
    ss.setdefault("focus_token", 0)
    ss.setdefault("compare", None)

    try:
        with st.spinner("Loading grid data…"):
            grid, geom, masks, area = get_static()
        with st.spinner("Loading the tuned XGBoost model (first run only)…"):
            get_model()
    except mu.AppDataError as e:
        st.error(f"**The dashboard cannot start:** {e}")
        st.stop()
    except Exception as e:  # pragma: no cover
        st.error(f"Unexpected error while loading the model or data: {e}")
        st.stop()

    settings = sidebar(masks)
    hero()

    if settings["gen"]:
        try:
            q = mu.validate_discharge(settings["q_in"])
        except ValueError as e:
            st.sidebar.error(str(e))
        else:
            t0 = time.time()
            with st.spinner(f"Running the XGBoost model for {fmt_q(q)} cumec "
                            "on 171,600 grid cells…"):
                domain_fhi(q)
            ss.active = q
            ss.selected_cell = None
            ss.focus = False
            ss.last_run_s = time.time() - t0
    if settings["cmp"]:
        try:
            qa = mu.validate_discharge(settings["qa"])
            qb = mu.validate_discharge(settings["qb"])
        except ValueError as e:
            st.sidebar.error(str(e))
        else:
            if qa == qb:
                st.sidebar.warning("Choose two different discharges.")
            else:
                ss.compare = (qa, qb)

    tab_main, tab_cmp, tab_about = st.tabs(
        ["🗺️  Hazard dashboard", "⚖️  Compare scenarios", "ℹ️  About the model"])

    with tab_main:
        if ss.active is None:
            welcome(geom, grid, settings)
        else:
            dashboard(ss.active, settings, grid, geom, masks, area)
    with tab_cmp:
        comparison_tab(settings, grid, geom, masks, area)
    with tab_about:
        about_tab(mu.load_build_info(), masks)


def dashboard(q, settings, grid, geom, masks, area):
    ss = st.session_state
    coverage, note = effective_coverage(q, settings["coverage"], masks)
    if note:
        st.warning(note)
    try:
        with st.spinner("Predicting flood hazard…"):
            pred = get_prediction(q, coverage)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return
    summ = me.summary(pred, area)
    top10 = me.top_cells(pred, grid, 10)
    status, status_long = mu.scenario_status(q, sorted(masks))
    pred_key = f"{fmt_q(q)}_{coverage}"
    map_keys = {VIEW_GEO: f"map_geo_{pred_key}", VIEW_XY: f"map_xy_{pred_key}"}
    map_key = map_keys[settings["view"]]
    table_key = f"top10_{pred_key}"

    # ---- process selections from the previous interaction ---------------
    lookup = curve_lookup(pred)
    for key in map_keys.values():
        state = ss.get(key)
        cell = cell_from_selection(state, lookup) if state else None
        if cell is not None and cell != ss.get("_last_" + key):
            ss.selected_cell, ss.focus = cell, False
        ss["_last_" + key] = cell
    tstate = ss.get(table_key)
    rows = []
    if tstate:
        try:
            rows = tstate["selection"]["rows"]
        except (KeyError, TypeError):
            rows = getattr(getattr(tstate, "selection", None), "rows", [])
    tsel = rows[0] if rows else None
    if tsel is not None and tsel != ss.get("_last_" + table_key) \
            and tsel < len(top10):
        r = top10.iloc[tsel]
        ss.selected_cell = (int(r["Row"]), int(r["Column"]))
        ss.focus = True
        ss.focus_token += 1
    ss["_last_" + table_key] = tsel

    # ---- scenario strip ---------------------------------------------------
    sim_cls = "sim" if "Simulated" in status else "interp"
    st.markdown(
        f'<div class="strip"><span class="badge q">Q = {fmt_q(q)} cumec</span>'
        f'<span class="badge {sim_cls}" title="{status_long}">{status}</span>'
        f'<span class="badge">Coverage: {COVERAGE_LABELS[coverage]}</span>'
        f'<span class="badge">{summ["cells_predicted"]:,} cells predicted'
        f'</span></div>', unsafe_allow_html=True)
    st.caption(status_long + " " + pred.coverage_note)
    caveat_box(pred)

    # ---- map --------------------------------------------------------------
    mcol, pcol = st.columns([3.3, 1])
    with mcol:
        with st.spinner("Drawing the hazard map…"):
            uri = overlay_uri(q, coverage, settings["view"], settings["opacity"])
            sel = ss.selected_cell
            uirev = f"{settings['view']}-{ss.focus_token}"
            title = (f"<b>Predicted flood hazard</b> · Q = {fmt_q(q)} cumec "
                     f"<span style='color:{MUTED};font-size:12px'>· hover to "
                     "inspect, click to select, scroll to zoom</span>")
            if settings["view"] == VIEW_GEO:
                fig = mp.build_map_figure(
                    pred, grid, geom, uri, settings["basemap"], height=780,
                    width_hint=1050, selected=sel, focus=ss.focus,
                    uirevision=uirev, title=title,
                    show_outline=settings["outline"])
            else:
                fig = mp.build_xy_figure(
                    pred, grid, geom, uri, height=780, selected=sel,
                    focus=ss.focus, uirevision=uirev, title=title,
                    show_outline=settings["outline"])
        st.plotly_chart(fig, width="stretch", key=map_key,
                        on_select="rerun", selection_mode="points",
                        config=plotly_config(f"beki_hazard_{pred_key}"))
    with pcol:
        selected_cell_panel(pred, grid, ss.selected_cell)
        legend_panel(pred)
        if settings["view"] == VIEW_GEO and \
                mp.BASEMAPS[settings["basemap"]] != "white-bg":
            st.markdown('<div class="small-note" style="margin-top:8px">'
                        'Basemap tiles need an internet connection. Choose '
                        '“No basemap” when working offline.</div>',
                        unsafe_allow_html=True)

    # ---- cards / charts / table / downloads -----------------------------
    section("Scenario summary")
    summary_cards(summ, pred)
    distribution_charts(summ)
    top10_section(top10, table_key)
    downloads_section(q, coverage, pred)


main()
