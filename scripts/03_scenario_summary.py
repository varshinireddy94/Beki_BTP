
import pandas as pd

from config import DATA_PATH

df = pd.read_parquet(
    DATA_PATH,
    columns=["Q", "report_eligible", "report_hazard_class"]
)

summary = (
    df[df["report_eligible"]]
    .groupby("Q")
    .agg(
        eligible_cells=("report_hazard_class", "size"),
        mean_hazard=("report_hazard_class", "mean"),
        very_high_cells=(
            "report_hazard_class",
            lambda x: (x == 5).sum()
        )
    )
    .reset_index()
)

summary["very_high_percentage"] = (
    summary["very_high_cells"]
    / summary["eligible_cells"] * 100
)

print(summary.to_string(index=False))

summary.to_csv(
    "scenario_summary_ml.csv",
    index=False
)

print("\nScenario summary saved!")
