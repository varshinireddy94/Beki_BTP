"""CSV, GeoTIFF and PDF exports."""

from __future__ import annotations

import datetime as dt
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rasterio.control import GroundControlPoint  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from app_config import (  # noqa: E402
    CRS_EPSG, CRS_NAME, HAZARD_CLASSES, HAZARD_COLORS, HAZARD_NAMES,
    MODEL_INFO, N_COLS, N_ROWS, NOPRED_INFO,
)
from map_utils import _status_cmap  # noqa: E402

NODATA_FHI = -9999.0


# --------------------------------------------------------------------------
# A. CSV
# --------------------------------------------------------------------------
def prediction_csv(pred, grid: pd.DataFrame, predicted_only: bool = False) -> bytes:
    status = pred.status
    lookup = {s: (HAZARD_NAMES.get(s) or NOPRED_INFO[s][0])
              for s in list(HAZARD_NAMES) + list(NOPRED_INFO)}
    df = pd.DataFrame({
        "row": grid["row"].to_numpy(),
        "col": grid["col"].to_numpy(),
        "X": grid["X"].to_numpy(),
        "Y": grid["Y"].to_numpy(),
        "discharge_cumec": pred.q,
        "predicted_FHI": np.where(pred.valid, pred.fhi.astype(float), np.nan),
        "hazard_class": np.where(pred.valid, status, np.nan),
        "hazard_category": pd.Series(status).map(lookup).to_numpy(),
        "prediction_valid": pred.valid,
        "cell_area_m2": grid["cell_area_m2"].to_numpy(),
    })
    df["hazard_class"] = df["hazard_class"].astype("Int8")
    if predicted_only:
        df = df[df["prediction_valid"]]
    return df.to_csv(index=False, float_format="%.6f").encode("utf-8")


# --------------------------------------------------------------------------
# B. GeoTIFF
# --------------------------------------------------------------------------
def _hazard_colormap() -> dict:
    cmap = {0: (255, 255, 255, 0)}
    for code, _, col, _ in HAZARD_CLASSES:
        h = col.lstrip("#")
        cmap[code] = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    return cmap


def native_grid_geotiff(pred, grid: pd.DataFrame, gcp_step: int = 10) -> bytes:
    """Lossless export on the native 286 x 600 curvilinear grid.

    No affine transform is imposed. Georeferencing is carried by
      * bands 3-4: the exact X and Y cell-centre coordinates, and
      * ground control points (every `gcp_step` cells plus all grid edges)
        tagged with the CRS, which GIS software can use to warp the grid.
    """
    X = grid["X"].to_numpy().reshape(N_ROWS, N_COLS)
    Y = grid["Y"].to_numpy().reshape(N_ROWS, N_COLS)
    cls = np.where(pred.valid, pred.status, 0).reshape(N_ROWS, N_COLS)
    fhi = np.where(pred.valid, pred.fhi, NODATA_FHI).reshape(N_ROWS, N_COLS)

    rows = sorted(set(range(0, N_ROWS, gcp_step)) | {N_ROWS - 1})
    cols = sorted(set(range(0, N_COLS, gcp_step)) | {N_COLS - 1})
    gcps = [GroundControlPoint(row=r + 0.5, col=c + 0.5,
                               x=float(X[r, c]), y=float(Y[r, c]),
                               id=f"r{r}c{c}")
            for r in rows for c in cols]

    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=N_COLS, height=N_ROWS, count=4,
                      dtype="float64", compress="deflate",
                      nodata=NODATA_FHI) as dst:
            dst.gcps = (gcps, CRS.from_epsg(CRS_EPSG))
            dst.write(cls.astype("float64"), 1)
            dst.write(fhi.astype("float64"), 2)
            dst.write(X, 3)
            dst.write(Y, 4)
            for i, d in enumerate(["hazard_class (1-5, 0 = no prediction)",
                                   "predicted_FHI", "X_cell_centre_m",
                                   "Y_cell_centre_m"], start=1):
                dst.set_band_description(i, d)
            dst.update_tags(
                discharge_cumec=f"{pred.q:g}", crs=CRS_NAME,
                grid="Native curvilinear grid; pixel (row, col) = model cell. "
                     "Exact cell-centre coordinates are in bands 3-4.",
                coverage=pred.coverage_note,
                hazard_thresholds="1:<=0.5 2:<=1 3:<=2 4:<=3 5:>3",
            )
        buf.write(mem.read())
    return buf.getvalue()


def build_cell_index_raster(grid: pd.DataFrame, NX: np.ndarray,
                            NY: np.ndarray, res: float = 20.0):
    """Rasterise every curvilinear cell polygon onto a north-up grid.

    Returns (index raster int32 with -1 outside the grid, affine transform).
    Built once; any scenario can then be mapped onto it by lookup.
    """
    xmin, xmax = NX.min(), NX.max()
    ymin, ymax = NY.min(), NY.max()
    width = int(np.ceil((xmax - xmin) / res))
    height = int(np.ceil((ymax - ymin) / res))
    transform = from_origin(xmin, ymax, res, res)

    def shapes():
        k = 0
        for r in range(N_ROWS):
            for c in range(N_COLS):
                ring = [(NX[r, c], NY[r, c]), (NX[r, c + 1], NY[r, c + 1]),
                        (NX[r + 1, c + 1], NY[r + 1, c + 1]),
                        (NX[r + 1, c], NY[r + 1, c]), (NX[r, c], NY[r, c])]
                yield {"type": "Polygon", "coordinates": [ring]}, k
                k += 1

    idx = rasterize(shapes(), out_shape=(height, width), transform=transform,
                    fill=-1, dtype="int32", all_touched=False)
    return idx, transform


def resampled_geotiff(pred, index_raster: np.ndarray, transform) -> bytes:
    """Map-ready, north-up GeoTIFF: each pixel takes the hazard class of the
    curvilinear cell containing the pixel centre (nearest-cell resampling)."""
    inside = index_raster >= 0
    status = np.where(pred.valid, pred.status, 0).astype("uint8")
    band = np.zeros(index_raster.shape, dtype="uint8")
    band[inside] = status[index_raster[inside]]
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=band.shape[1],
                      height=band.shape[0], count=1, dtype="uint8",
                      crs=CRS.from_epsg(CRS_EPSG), transform=transform,
                      nodata=0, compress="deflate") as dst:
            dst.write(band, 1)
            dst.write_colormap(1, _hazard_colormap())
            dst.set_band_description(1, "hazard_class (1-5, 0 = no prediction)")
            dst.update_tags(
                discharge_cumec=f"{pred.q:g}", crs=CRS_NAME,
                resampling="Rasterised from curvilinear cell polygons at "
                           f"{transform.a:g} m; value = class of containing cell",
                coverage=pred.coverage_note,
            )
        return mem.read()


# --------------------------------------------------------------------------
# C. PDF report
# --------------------------------------------------------------------------
def static_map_png(pred, geom, title: str, dpi: int = 170) -> bytes:
    fig, ax = plt.subplots(figsize=(6.2, 9.6), dpi=dpi)
    cmap, norm = _status_cmap(1.0, 0.8)
    ax.pcolormesh(geom.NX, geom.NY, pred.grid2d(pred.status), cmap=cmap,
                  norm=norm, shading="flat", antialiased=False, linewidth=0,
                  rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting X (m)", fontsize=8)
    ax.set_ylabel("Northing Y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_title(title, fontsize=10, fontweight="bold")
    handles = [Patch(color=HAZARD_COLORS[c], label=f"{n} ({r})")
               for c, n, _, r in HAZARD_CLASSES]
    handles += [Patch(color=NOPRED_INFO[k][1], label=NOPRED_INFO[k][0])
                for k in NOPRED_INFO if (pred.status == k).any()]
    ax.legend(handles=handles, loc="upper center", fontsize=6.5, ncol=2,
              bbox_to_anchor=(0.5, -0.07), frameon=False,
              title="Hazard category", title_fontsize=7)
    ax.grid(color="#dde3ea", lw=0.4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def distribution_png(table: pd.DataFrame, dpi: int = 170) -> bytes:
    fig, ax = plt.subplots(figsize=(7.0, 2.8), dpi=dpi)
    names = table["Hazard category"][::-1]
    areas = table["Area (km²)"][::-1]
    pcts = table["% of predicted area"][::-1]
    ax.barh(names, areas, color=table["color"][::-1], height=0.62,
            edgecolor="white")
    for y, (a, p) in enumerate(zip(areas, pcts)):
        ax.text(a + areas.max() * 0.01, y, f"{a:,.1f} km²  ({p:.1f}%)",
                va="center", fontsize=8, color="#1f2933")
    ax.set_xlim(0, areas.max() * 1.32)
    ax.set_xlabel("Area (km²)", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Predicted area by hazard category", fontsize=10,
                 fontweight="bold", loc="left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def _register_fonts() -> tuple[str, str]:
    """Unicode-capable fonts (≤, ≥, ², ³) bundled with matplotlib."""
    from pathlib import Path

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    try:
        fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(
                TTFont("DejaVu", str(fdir / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(
                TTFont("DejaVu-Bold", str(fdir / "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def pdf_report(pred, summ: dict, top10: pd.DataFrame, geom,
               status_text: str, coverage_label: str,
               interpretation: list[str], limitations: list[str]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    font, bold = _register_fonts()
    ss = getSampleStyleSheet()
    for s in ss.byName.values():
        s.fontName = bold if "Heading" in s.name or s.name == "Title" else font
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17,
                        textColor=colors.HexColor("#0b3954"), alignment=TA_LEFT,
                        spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5,
                        textColor=colors.HexColor("#0b3954"), spaceBefore=8)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12.5)
    small = ParagraphStyle("s", parent=body, fontSize=7.8, leading=10,
                           textColor=colors.HexColor("#52606d"))

    def tbl(data, widths, header_bg="#0b3954"):
        t = Table(data, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTSIZE", (0, 0), (-1, -1), 8.2),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f2f5f8")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3ccd6")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.7 * cm,
                            rightMargin=1.7 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm,
                            title=f"Beki Flood Watch – Q {pred.q:,.0f} cumec")
    story = [
        Paragraph("BEKI FLOOD WATCH", h1),
        Paragraph("ML-based flood hazard prediction – scenario report", body),
        Paragraph(f"Generated {dt.datetime.now():%d %b %Y, %H:%M}", small),
        Spacer(1, 8),
    ]
    key = [["Item", "Value"],
           ["Selected discharge", f"{pred.q:,.0f} cumec (m³/s)"],
           ["Scenario status", status_text],
           ["Prediction coverage", coverage_label],
           ["Cells with valid predictions",
            f"{summ['cells_predicted']:,} of {summ['cells_total']:,}"],
           ["Predicted area", f"{summ['predicted_area_km2']:,.2f} km² of "
                              f"{summ['grid_area_km2']:,.2f} km² grid "
                              f"({summ['coverage_pct']:.1f}%)"],
           ["High hazard area", f"{summ['high_area_km2']:,.2f} km²"],
           ["Very High hazard area", f"{summ['very_high_area_km2']:,.2f} km²"],
           ["High + Very High share", f"{summ['high_vhigh_pct']:.1f}% of predicted area"],
           ["Model", f"{MODEL_INFO['Algorithm']} ({MODEL_INFO['Configuration']}), "
                     f"validation MAE {MODEL_INFO['Validation MAE']}, "
                     f"R² {MODEL_INFO['Validation R²']}"]]
    key = [key[0]] + [[a, Paragraph(b, body)] for a, b in key[1:]]
    kt = tbl(key, [5 * cm, 12.3 * cm])
    kt.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "LEFT")]))
    story += [Paragraph("Scenario summary", h2), kt]

    story += [Paragraph("Interpretation", h2)]
    story += [Paragraph("• " + s, body) for s in interpretation]

    story += [Paragraph("Area statistics", h2)]
    t = summ["table"]
    data = [["Hazard category", "FHI range", "Cells", "Area (km²)", "% of predicted area"]]
    for _, r in t.iterrows():
        data.append([r["Hazard category"], r["FHI range"], f"{r['Cells']:,}",
                     f"{r['Area (km²)']:,.2f}", f"{r['% of predicted area']:.1f}%"])
    data.append(["Total predicted", "", f"{t['Cells'].sum():,}",
                 f"{t['Area (km²)'].sum():,.2f}", "100.0%"])
    for label, v in summ["no_prediction"].items():
        if v["cells"]:
            data.append([Paragraph(label.replace("No prediction – ",
                                                 "No prediction: "), small),
                         "–", f"{v['cells']:,}", f"{v['area_km2']:,.2f}", "–"])
    st_ = tbl(data, [5.6 * cm, 3.0 * cm, 2.4 * cm, 2.6 * cm, 3.6 * cm])
    st_.setStyle(TableStyle([("FONTNAME", (0, len(t) + 1), (-1, len(t) + 1),
                              bold)]))
    story += [st_, Spacer(1, 4),
              Image(io.BytesIO(distribution_png(t)), width=14 * cm,
                    height=5.6 * cm)]

    story += [PageBreak(), Paragraph("Predicted hazard map", h2),
              Image(io.BytesIO(static_map_png(
                  pred, geom, f"Predicted flood hazard – Q = {pred.q:,.0f} cumec")),
                  width=12.4 * cm, height=19.2 * cm),
              Paragraph(f"Coordinates: {CRS_NAME}. Curvilinear 286 × 600 model "
                        "grid drawn cell by cell. Grey cells have no valid "
                        "prediction and must not be read as safe.", small)]

    story += [PageBreak(), Paragraph("Top 10 highest-hazard cells", h2)]
    d = [["Rank", "Row", "Col", "X (m)", "Y (m)", "Predicted FHI", "Category"]]
    for _, r in top10.iterrows():
        d.append([int(r["Rank"]), int(r["Row"]), int(r["Column"]),
                  f"{r['X (m)']:,.1f}", f"{r['Y (m)']:,.1f}",
                  f"{r['Predicted FHI']:,.3f}", r["Hazard category"]])
    story += [tbl(d, [1.2 * cm, 1.3 * cm, 1.3 * cm, 3 * cm, 3.3 * cm,
                      2.8 * cm, 2.6 * cm])]

    story += [Paragraph("Important limitations", h2)]
    story += [Paragraph("• " + s, body) for s in limitations]
    doc.build(story)
    return buf.getvalue()
