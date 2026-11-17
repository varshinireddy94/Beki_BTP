"""Area statistics, hazard summaries and scenario comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app_config import HAZARD_CLASSES, HAZARD_NAMES, NOPRED_INFO
from model_utils import Prediction

M2_PER_KM2 = 1e6


def class_area_table(pred: Prediction, cell_area_m2: np.ndarray) -> pd.DataFrame:
    """Cells and area per hazard class (percent of predicted area)."""
    rows = []
    valid = pred.valid
    total = cell_area_m2[valid].sum()
    for code, name, color, rule in HAZARD_CLASSES:
        m = pred.status == code
        a = cell_area_m2[m].sum()
        rows.append({
            "code": code, "Hazard category": name, "FHI range": rule,
            "Cells": int(m.sum()), "Area (km²)": a / M2_PER_KM2,
            "% of predicted area": 100 * a / total if total > 0 else 0.0,
            "color": color,
        })
    return pd.DataFrame(rows)


def summary(pred: Prediction, cell_area_m2: np.ndarray) -> dict:
    tbl = class_area_table(pred, cell_area_m2)
    valid = pred.valid
    pred_area = cell_area_m2[valid].sum() / M2_PER_KM2
    grid_area = cell_area_m2.sum() / M2_PER_KM2
    high = tbl.loc[tbl.code == 4, "Area (km²)"].item()
    vhigh = tbl.loc[tbl.code == 5, "Area (km²)"].item()
    nopred = {
        NOPRED_INFO[k][0]: {
            "cells": int((pred.status == k).sum()),
            "area_km2": cell_area_m2[pred.status == k].sum() / M2_PER_KM2,
        }
        for k in NOPRED_INFO
    }
    fhi_valid = pred.fhi[valid]
    return {
        "q": pred.q,
        "cells_total": int(len(pred.status)),
        "cells_predicted": int(valid.sum()),
        "grid_area_km2": grid_area,
        "predicted_area_km2": pred_area,
        "coverage_pct": 100 * pred_area / grid_area if grid_area else 0.0,
        "high_area_km2": high,
        "very_high_area_km2": vhigh,
        "high_vhigh_pct": 100 * (high + vhigh) / pred_area if pred_area else 0.0,
        "fhi_median": float(np.median(fhi_valid)) if fhi_valid.size else np.nan,
        "fhi_max": float(fhi_valid.max()) if fhi_valid.size else np.nan,
        "no_prediction": nopred,
        "table": tbl,
    }


def top_cells(pred: Prediction, grid: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    valid_idx = np.flatnonzero(pred.valid)
    if valid_idx.size == 0:
        return pd.DataFrame(columns=["Rank", "Row", "Column", "X (m)", "Y (m)",
                                     "Predicted FHI", "Hazard category"])
    order = valid_idx[np.argsort(-pred.fhi[valid_idx], kind="stable")[:n]]
    return pd.DataFrame({
        "Rank": np.arange(1, len(order) + 1),
        "Row": grid["row"].to_numpy()[order],
        "Column": grid["col"].to_numpy()[order],
        "X (m)": grid["X"].to_numpy()[order],
        "Y (m)": grid["Y"].to_numpy()[order],
        "Predicted FHI": pred.fhi[order].astype(float),
        "Hazard category": [HAZARD_NAMES[int(s)] for s in pred.status[order]],
    })


def compare(pred_a: Prediction, pred_b: Prediction,
            cell_area_m2: np.ndarray) -> dict:
    """Compare two scenarios over the cells predicted in BOTH."""
    ta = class_area_table(pred_a, cell_area_m2)
    tb = class_area_table(pred_b, cell_area_m2)
    delta = pd.DataFrame({
        "Hazard category": ta["Hazard category"],
        f"Q = {pred_a.q:,.0f}  area (km²)": ta["Area (km²)"],
        f"Q = {pred_b.q:,.0f}  area (km²)": tb["Area (km²)"],
        "Change (km²)": tb["Area (km²)"] - ta["Area (km²)"],
        f"Q = {pred_a.q:,.0f}  share (%)": ta["% of predicted area"],
        f"Q = {pred_b.q:,.0f}  share (%)": tb["% of predicted area"],
        "Change (percentage points)": (tb["% of predicted area"]
                                       - ta["% of predicted area"]),
    })

    both = pred_a.valid & pred_b.valid
    sa, sb = pred_a.status[both], pred_b.status[both]
    ar = cell_area_m2[both]
    up = sb > sa
    down = sb < sa
    trans = pd.crosstab(
        pd.Categorical([HAZARD_NAMES[s] for s in sa],
                       categories=list(HAZARD_NAMES.values())),
        pd.Categorical([HAZARD_NAMES[s] for s in sb],
                       categories=list(HAZARD_NAMES.values())),
        values=ar / M2_PER_KM2, aggfunc="sum", dropna=False,
    ).fillna(0.0)
    trans.index.name = f"Q = {pred_a.q:,.0f}"
    trans.columns.name = f"Q = {pred_b.q:,.0f}"

    hv_a = ta.loc[ta.code >= 4, "Area (km²)"].sum()
    hv_b = tb.loc[tb.code >= 4, "Area (km²)"].sum()
    return {
        "table_a": ta, "table_b": tb, "delta": delta, "transitions": trans,
        "pred_area_a": cell_area_m2[pred_a.valid].sum() / M2_PER_KM2,
        "pred_area_b": cell_area_m2[pred_b.valid].sum() / M2_PER_KM2,
        "high_change": tb.loc[tb.code == 4, "Area (km²)"].item()
                       - ta.loc[ta.code == 4, "Area (km²)"].item(),
        "very_high_change": tb.loc[tb.code == 5, "Area (km²)"].item()
                            - ta.loc[ta.code == 5, "Area (km²)"].item(),
        "hv_a": hv_a, "hv_b": hv_b,
        "common_area": ar.sum() / M2_PER_KM2,
        "area_up": ar[up].sum() / M2_PER_KM2,
        "area_down": ar[down].sum() / M2_PER_KM2,
        "area_same": ar[~up & ~down].sum() / M2_PER_KM2,
    }


def interpretation(s: dict, status_text: str, coverage_label: str) -> list[str]:
    """Short plain-language interpretation of a scenario."""
    tbl = s["table"]
    dom = tbl.sort_values("Area (km²)", ascending=False).iloc[0]
    out = [
        f"At a river discharge of {s['q']:,.0f} cumec, the model produced valid "
        f"FHI predictions for {s['cells_predicted']:,} of {s['cells_total']:,} "
        f"grid cells ({s['predicted_area_km2']:,.1f} km² of the "
        f"{s['grid_area_km2']:,.1f} km² grid; coverage rule: {coverage_label}).",
        f"{s['high_vhigh_pct']:.1f}% of the predicted area "
        f"({s['high_area_km2'] + s['very_high_area_km2']:,.1f} km²) falls in the "
        f"High or Very High hazard categories, including "
        f"{s['very_high_area_km2']:,.1f} km² of Very High hazard (FHI > 3).",
        f"The most widespread category is {dom['Hazard category']} "
        f"({dom['% of predicted area']:.1f}% of the predicted area). The median "
        f"predicted FHI is {s['fhi_median']:.2f} and the maximum is "
        f"{s['fhi_max']:,.1f}.",
        f"Scenario status: {status_text}.",
    ]
    return out
