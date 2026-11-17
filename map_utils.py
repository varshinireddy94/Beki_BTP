"""Interactive hazard maps for the curvilinear Beki grid.

The 286 x 600 grid is curvilinear, so it is never treated as a regular
raster. Each cell is a quadrilateral whose corners are derived from the
surrounding cell centres. The hazard surface is drawn by rendering those
quadrilaterals (matplotlib pcolormesh) into an image that is placed exactly
on the map, while invisible WebGL points at the true X/Y cell centres provide
hover and click information.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba  # noqa: E402
from pyproj import Transformer  # noqa: E402

from app_config import (  # noqa: E402
    CRS_EPSG, HAZARD_CLASSES, HAZARD_COLORS, HAZARD_NAMES, N_COLS, N_ROWS,
    NOPRED_INFO, NOPRED_OPEN_WATER, NOPRED_OUTSIDE_SIM,
)

EARTH_R = 6378137.0
BASEMAPS = {
    "Light map": "carto-positron",
    "Street map": "open-street-map",
    "Satellite imagery": "satellite",
    "No basemap (offline)": "white-bg",
}
ESRI_IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}")


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def corner_nodes(A: np.ndarray) -> np.ndarray:
    """(R+1, C+1) corner nodes from (R, C) cell centres (same rule as the
    cell-area calculation in scripts/24_build_app_data.py)."""
    p = np.pad(A, 1, mode="edge").astype(float)
    p[0, :] = 2 * p[1, :] - p[2, :]
    p[-1, :] = 2 * p[-2, :] - p[-3, :]
    p[:, 0] = 2 * p[:, 1] - p[:, 2]
    p[:, -1] = 2 * p[:, -2] - p[:, -3]
    return 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])


@dataclass
class GridGeometry:
    X: np.ndarray            # (R, C) cell-centre easting (m)
    Y: np.ndarray            # (R, C) cell-centre northing (m)
    NX: np.ndarray           # (R+1, C+1) corner nodes, projected
    NY: np.ndarray
    lon: np.ndarray          # (R, C) cell centres, WGS84
    lat: np.ndarray
    NLon: np.ndarray         # (R+1, C+1) corner nodes, WGS84
    NLat: np.ndarray
    NMx: np.ndarray          # (R+1, C+1) corner nodes, Web Mercator
    NMy: np.ndarray
    merc_bbox: tuple         # (xmin, xmax, ymin, ymax)
    image_corners: list      # MapLibre image coordinates [TL, TR, BR, BL]

    @property
    def center(self) -> dict:
        return {"lon": float(self.lon.mean()), "lat": float(self.lat.mean())}

    def outline(self, which: str = "lonlat") -> tuple[np.ndarray, np.ndarray]:
        A, B = ((self.NLon, self.NLat) if which == "lonlat"
                else (self.NX, self.NY))
        xa = np.concatenate([A[0, :], A[1:, -1], A[-1, -2::-1], A[-2::-1, 0]])
        ya = np.concatenate([B[0, :], B[1:, -1], B[-1, -2::-1], B[-2::-1, 0]])
        return xa, ya

    def cell_polygon(self, r: int, c: int, which: str = "lonlat"):
        A, B = ((self.NLon, self.NLat) if which == "lonlat"
                else (self.NX, self.NY))
        idx = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c), (r, c)]
        return [A[i] for i in idx], [B[i] for i in idx]


def build_geometry(grid: pd.DataFrame) -> GridGeometry:
    X = grid["X"].to_numpy().reshape(N_ROWS, N_COLS)
    Y = grid["Y"].to_numpy().reshape(N_ROWS, N_COLS)
    NX, NY = corner_nodes(X), corner_nodes(Y)
    to_ll = Transformer.from_crs(CRS_EPSG, 4326, always_xy=True)
    to_merc = Transformer.from_crs(CRS_EPSG, 3857, always_xy=True)
    lon, lat = to_ll.transform(X, Y)
    NLon, NLat = to_ll.transform(NX, NY)
    NMx, NMy = to_merc.transform(NX, NY)
    bbox = (NMx.min(), NMx.max(), NMy.min(), NMy.max())
    merc_to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
    xmin, xmax, ymin, ymax = bbox
    corners = [merc_to_ll.transform(x, y) for x, y in
               [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]]
    return GridGeometry(X, Y, NX, NY, lon, lat, NLon, NLat, NMx, NMy, bbox,
                        [list(c) for c in corners])


# --------------------------------------------------------------------------
# Raster rendering of the curvilinear cells
# --------------------------------------------------------------------------
def _status_cmap(opacity: float, nopred_opacity: float):
    colors = [(0, 0, 0, 0)]  # code 0 unused
    for code, _, col, _ in HAZARD_CLASSES:
        colors.append(to_rgba(col, opacity))
    colors.append(to_rgba(NOPRED_INFO[NOPRED_OPEN_WATER][1], nopred_opacity))
    colors.append(to_rgba(NOPRED_INFO[NOPRED_OUTSIDE_SIM][1], nopred_opacity))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)
    return cmap, norm


def render_status_png(status2d: np.ndarray, NXr: np.ndarray, NYr: np.ndarray,
                      bbox: tuple, height_px: int = 3600,
                      opacity: float = 0.85,
                      nopred_opacity: float = 0.45) -> bytes:
    """Render every curvilinear cell as a filled quadrilateral into a
    transparent PNG whose edges are exactly `bbox` (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = bbox
    width_px = max(1, int(round(height_px * (xmax - xmin) / (ymax - ymin))))
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    cmap, norm = _status_cmap(opacity, nopred_opacity)
    ax.pcolormesh(NXr, NYr, status2d, cmap=cmap, norm=norm,
                  shading="flat", antialiased=False, linewidth=0,
                  edgecolors="none", rasterized=True)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(fig)
    return buf.getvalue()


def png_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# --------------------------------------------------------------------------
# Hover / click layer
# --------------------------------------------------------------------------
def _hover_template(code: int, q: float) -> str:
    qtxt = f"{q:,.0f} cumec"
    loc = ("X: %{customdata[0]:,.0f} m<br>Y: %{customdata[1]:,.0f} m<br>"
           "Grid row %{customdata[3]:.0f}, column %{customdata[4]:.0f}<br>")
    if code <= 5:
        return (f"<b>{HAZARD_NAMES[code]} hazard</b><br>"
                "Predicted FHI: <b>%{customdata[2]:.3f}</b><br>"
                f"{loc}Discharge: {qtxt}<extra></extra>")
    label, _, why = NOPRED_INFO[code]
    return (f"<b>{label}</b><br>{loc}Discharge: {qtxt}<br>"
            f"<i>{why}</i><extra></extra>")


def _customdata(grid: pd.DataFrame, fhi: np.ndarray, m: np.ndarray):
    return np.column_stack([
        grid["X"].to_numpy()[m], grid["Y"].to_numpy()[m], fhi[m],
        grid["row"].to_numpy()[m], grid["col"].to_numpy()[m],
    ]).astype(np.float32)


def _hover_traces(pred, grid, geom, kind: str):
    """Invisible markers at true cell centres, one trace per status code."""
    traces = []
    status = pred.status
    lon, lat = geom.lon.ravel(), geom.lat.ravel()
    X, Y = grid["X"].to_numpy(), grid["Y"].to_numpy()
    for code in [1, 2, 3, 4, 5, NOPRED_OPEN_WATER, NOPRED_OUTSIDE_SIM]:
        m = status == code
        if not m.any():
            continue
        common = dict(
            mode="markers", showlegend=False, hoverinfo="text",
            name=(HAZARD_NAMES.get(code) or NOPRED_INFO[code][0]),
            customdata=_customdata(grid, pred.fhi, m),
            hovertemplate=_hover_template(code, pred.q),
            hoverlabel=dict(bgcolor="white", bordercolor=(
                HAZARD_COLORS.get(code) or NOPRED_INFO[code][1]),
                font=dict(color="#1f2933", size=13)),
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=0)),
        )
        if kind == "map":
            traces.append(go.Scattermap(
                lon=lon[m].astype(np.float32), lat=lat[m].astype(np.float32),
                marker=dict(size=7, opacity=0, color="#000"), **common))
        else:
            traces.append(go.Scattergl(
                x=X[m].astype(np.float32), y=Y[m].astype(np.float32),
                marker=dict(size=5, opacity=0, color="#000"), **common))
    return traces


def _legend_traces(pred, kind: str):
    present_nopred = [k for k in NOPRED_INFO if (pred.status == k).any()]
    items = [(HAZARD_NAMES[c] + f"  ({r})", HAZARD_COLORS[c])
             for c, _, _, r in HAZARD_CLASSES]
    items += [(NOPRED_INFO[k][0], NOPRED_INFO[k][1]) for k in present_nopred]
    out = []
    for name, color in items:
        kw = dict(mode="markers", name=name, hoverinfo="skip",
                  marker=dict(size=14, color=color, symbol="square"
                              if kind != "map" else "circle"),
                  showlegend=True)
        if kind == "map":
            out.append(go.Scattermap(lon=[None], lat=[None], **kw))
        else:
            out.append(go.Scattergl(x=[None], y=[None], **kw))
    return out


def _legend_layout(compact: bool):
    return dict(
        orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
        bgcolor="rgba(255,255,255,0.92)", bordercolor="#d0d7de", borderwidth=1,
        font=dict(size=11 if compact else 12, color="#1f2933"),
        itemclick=False, itemdoubleclick=False,
        title=dict(text="<b>Hazard category</b>  ", side="left"),
    )


def _auto_zoom(geom: GridGeometry, height_px: int, width_px: int) -> float:
    xmin, xmax, ymin, ymax = geom.merc_bbox
    world = 2 * math.pi * EARTH_R
    zy = math.log2(height_px / (512 * (ymax - ymin) / world))
    zx = math.log2(width_px / (512 * (xmax - xmin) / world))
    return min(zx, zy) - 0.15


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def build_map_figure(pred, grid, geom: GridGeometry, image_uri: str | None,
                     basemap: str = "Light map", height: int = 720,
                     width_hint: int = 900, selected: tuple | None = None,
                     focus: bool = False, uirevision: str = "beki",
                     title: str | None = None, compact: bool = False,
                     show_outline: bool = True) -> go.Figure:
    fig = go.Figure()
    if pred is not None:
        fig.add_traces(_hover_traces(pred, grid, geom, "map"))
        fig.add_traces(_legend_traces(pred, "map"))
    if show_outline:
        ox, oy = geom.outline("lonlat")
        fig.add_trace(go.Scattermap(
            lon=ox, lat=oy, mode="lines", line=dict(color="#0b3954", width=2),
            name="Model grid boundary (286 × 600 cells)", hoverinfo="skip",
            showlegend=pred is None))
    if selected is not None:
        r, c = selected
        px, py = geom.cell_polygon(r, c, "lonlat")
        fig.add_trace(go.Scattermap(
            lon=[geom.lon[r, c]], lat=[geom.lat[r, c]], mode="markers",
            marker=dict(size=26, color="#00c2ff", opacity=0.35),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattermap(
            lon=px, lat=py, mode="lines", line=dict(color="#002b45", width=3),
            hoverinfo="skip", showlegend=False, name="Selected cell"))

    style = BASEMAPS.get(basemap, "carto-positron")
    layers = []
    if style == "satellite":
        style = "white-bg"
        layers.append(dict(sourcetype="raster", source=[ESRI_IMAGERY],
                           below="traces", sourceattribution=(
                               "Imagery © Esri, Maxar, Earthstar Geographics")))
    if image_uri:
        layers.append(dict(sourcetype="image", source=image_uri,
                           coordinates=geom.image_corners, below="traces"))

    if selected is not None and focus:
        center = {"lon": float(geom.lon[selected]),
                  "lat": float(geom.lat[selected])}
        zoom = 13.2
    else:
        center = geom.center
        # usable map height = figure height minus title/legend margin
        zoom = _auto_zoom(geom, height - (70 if title else 40) - 40,
                          width_hint)

    fig.update_layout(
        map=dict(style=style, center=center, zoom=zoom, layers=layers),
        height=height, margin=dict(l=0, r=0, t=70 if title else 40, b=0),
        legend=_legend_layout(compact), uirevision=uirevision,
        hovermode="closest", clickmode="event+select", dragmode="pan",
        paper_bgcolor="white",
        title=dict(text=title, x=0.01, y=0.985, font=dict(size=15))
        if title else None,
    )
    return fig


def build_xy_figure(pred, grid, geom: GridGeometry, image_uri: str | None,
                    height: int = 720, selected: tuple | None = None,
                    focus: bool = False, uirevision: str = "beki",
                    title: str | None = None, compact: bool = False,
                    show_outline: bool = True) -> go.Figure:
    """Projected X/Y view with real coordinate axes (no basemap)."""
    fig = go.Figure()
    xmin, xmax = geom.NX.min(), geom.NX.max()
    ymin, ymax = geom.NY.min(), geom.NY.max()
    if image_uri:
        fig.add_layout_image(dict(
            source=image_uri, xref="x", yref="y", x=xmin, y=ymax,
            sizex=xmax - xmin, sizey=ymax - ymin, sizing="stretch",
            layer="below", opacity=1.0))
    if pred is not None:
        fig.add_traces(_hover_traces(pred, grid, geom, "xy"))
        fig.add_traces(_legend_traces(pred, "xy"))
    if show_outline:
        ox, oy = geom.outline("xy")
        fig.add_trace(go.Scattergl(
            x=ox, y=oy, mode="lines", line=dict(color="#0b3954", width=1.5),
            name="Model grid boundary", hoverinfo="skip",
            showlegend=pred is None))
    if selected is not None:
        r, c = selected
        px, py = geom.cell_polygon(r, c, "xy")
        fig.add_trace(go.Scattergl(
            x=[geom.X[r, c]], y=[geom.Y[r, c]], mode="markers",
            marker=dict(size=22, color="rgba(0,194,255,0.35)",
                        line=dict(color="#002b45", width=2)),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattergl(
            x=px, y=py, mode="lines", line=dict(color="#002b45", width=2.5),
            hoverinfo="skip", showlegend=False))
    if selected is not None and focus:
        r, c = selected
        half = 2500
        xr = [geom.X[r, c] - half, geom.X[r, c] + half]
        yr = [geom.Y[r, c] - half, geom.Y[r, c] + half]
    else:
        pad = 800
        xr, yr = [xmin - pad, xmax + pad], [ymin - pad, ymax + pad]
    axis = dict(showgrid=True, gridcolor="#e5e9ef", zeroline=False,
                tickformat=",.0f", ticks="outside", color="#3e4c59",
                mirror=True, showline=True, linecolor="#c3ccd6")
    fig.update_layout(
        xaxis=dict(title="Easting X (m)", range=xr, **axis),
        yaxis=dict(title="Northing Y (m)", range=yr, scaleanchor="x",
                   scaleratio=1, **axis),
        height=height, margin=dict(l=70, r=10, t=70 if title else 40, b=50),
        legend=_legend_layout(compact), uirevision=uirevision,
        hovermode="closest", clickmode="event+select", dragmode="pan",
        plot_bgcolor="#f7f9fb", paper_bgcolor="white",
        title=dict(text=title, x=0.01, y=0.985, font=dict(size=15))
        if title else None,
    )
    return fig


def selected_cell_from_event(event) -> tuple[int, int] | None:
    """Extract (row, col) from a Streamlit plotly selection event."""
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        try:
            points = event.selection.points
        except AttributeError:
            return None
    for p in points or []:
        cd = p.get("customdata")
        try:
            if isinstance(cd, dict):
                cd = [cd[k] for k in sorted(cd, key=lambda k: int(k))]
            if cd is not None and len(cd) >= 5:
                return int(round(float(cd[3]))), int(round(float(cd[4])))
        except (KeyError, TypeError, ValueError):
            continue
    return None
