"""
Random-forest factor analysis for explaining judge and audience-performance metrics.

The script trains one RandomForestRegressor per target variable, reports feature
importance, computes marginal correlation signs, and exports both factor reports
and row-level predictions/residuals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from data_preparation import find_column, parse_follower_count, read_table, write_table


COASTAL_STATES = {
    "California", "New York", "Florida", "Massachusetts", "New Jersey", "Washington", "Oregon",
    "Connecticut", "Maryland", "District of Columbia", "Hawaii", "Rhode Island", "Virginia",
}


def map_region(row: pd.Series) -> str:
    """Map home location to a compact region bucket."""
    region = str(row.get("Home_Region", "")).strip()
    state = str(row.get("Home_State", "")).strip()
    if region and region.lower() != "united states":
        return "International"
    if state in COASTAL_STATES:
        return "US_Coastal"
    if state and state.lower() not in {"nan", "none"}:
        return "US_Heartland"
    return "Unknown"


def map_industry(industry: object) -> str:
    """Map fine-grained occupations into modeling groups."""
    text = str(industry).lower()
    if any(term in text for term in ("actor", "actress", "singer", "rapper", "pop", "model", "influencer")):
        return "Entertainment"
    if any(term in text for term in ("athlete", "nfl", "nba", "olympian", "gymnast", "fighter", "skater")):
        return "Sports"
    if any(term in text for term in ("reality", "bachelor", "survivor", "bravo")):
        return "Reality_TV"
    if any(term in text for term in ("host", "comedian", "anchor", "journalist")):
        return "TV_Host_Comedy"
    return "Other"


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build numeric and one-hot encoded model features."""
    df = df.copy()

    # Harmonize common column names.
    age_col = find_column(df, ["Age", "age"], required=False)
    partner_col = find_column(df, ["Partner_Ability_Score", "partner_score", "partner"], required=False)
    industry_col = find_column(df, ["Industry", "industry"], required=False)
    follower_col = find_column(df, ["instagram_followers", "followers", "x_followers", "follower"], required=False)

    if age_col:
        df["Age_Feature"] = pd.to_numeric(df[age_col], errors="coerce").fillna(pd.to_numeric(df[age_col], errors="coerce").median())
    else:
        df["Age_Feature"] = 0

    if partner_col:
        df["Partner_Ability_Feature"] = pd.to_numeric(df[partner_col], errors="coerce").fillna(0)
    else:
        df["Partner_Ability_Feature"] = 0

    if follower_col:
        df["Followers_Log"] = df[follower_col].map(parse_follower_count)
        if df["Followers_Log"].isna().all():
            df["Followers_Log"] = pd.to_numeric(df[follower_col], errors="coerce")
        df["Followers_Log"] = np.log1p(df["Followers_Log"].fillna(0))
    else:
        df["Followers_Log"] = 0

    if "Home_Region" not in df.columns:
        df["Home_Region"] = ""
    if "Home_State" not in df.columns:
        df["Home_State"] = ""
    df["Region_Group"] = df.apply(map_region, axis=1)

    if industry_col:
        df["Industry_Group"] = df[industry_col].map(map_industry)
    else:
        df["Industry_Group"] = "Other"

    base_features = ["Age_Feature", "Partner_Ability_Feature", "Followers_Log"]
    encoded = pd.get_dummies(df[["Region_Group", "Industry_Group"]], drop_first=False)
    X = pd.concat([df[base_features], encoded], axis=1).fillna(0)
    return X, X.columns.tolist()


def train_one_target(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str | Path,
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a tuned random forest and export factor-level and row-level results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    working = df.copy()
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    working = working.dropna(subset=[target_col])
    if len(working) < max(10, cv_splits):
        raise ValueError(f"Not enough valid rows for target {target_col}: {len(working)}")

    X, feature_names = prepare_features(working)
    y = working[target_col].astype(float)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "min_samples_split": [2, 5, 10],
    }
    cv = KFold(n_splits=min(cv_splits, len(working)), shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)
    model = grid.best_estimator_
    predictions = model.predict(X)

    correlations = X.corrwith(y).replace([np.inf, -np.inf], np.nan).fillna(0)
    factor_report = pd.DataFrame(
        {
            "Target": target_col,
            "Feature": feature_names,
            "Importance": model.feature_importances_,
            "Correlation": correlations.reindex(feature_names).to_numpy(),
            "Best_Params": str(grid.best_params_),
            "R2_InSample": r2_score(y, predictions),
            "RMSE_InSample": mean_squared_error(y, predictions, squared=False),
            "MAE_InSample": mean_absolute_error(y, predictions),
            "Sample_Size": len(working),
        }
    ).sort_values("Importance", ascending=False)

    prediction_report = working.copy()
    prediction_report[f"Predicted_{target_col}"] = predictions
    prediction_report[f"Residual_{target_col}"] = y - predictions

    safe_target = target_col.replace("/", "_").replace(" ", "_")
    write_table(factor_report, output_dir / f"factor_report_{safe_target}.csv")
    write_table(prediction_report, output_dir / f"predictions_{safe_target}.csv")
    return factor_report, prediction_report


def run_factor_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    target_cols: Sequence[str] | None = None,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """Run random-forest analysis for multiple target columns."""
    df = read_table(input_path)
    if target_cols is None or len(target_cols) == 0:
        target_cols = [c for c in df.columns if c.startswith("Avg_Relative_")]
        if not target_cols:
            target_cols = [c for c in df.columns if c in {"weekly_judge_perf", "weekly_fan_perf"}]
    if not target_cols:
        raise ValueError("No target columns were provided or automatically detected.")

    reports = []
    for target in target_cols:
        if target not in df.columns:
            print(f"Skipping missing target column: {target}")
            continue
        report, _ = train_one_target(df, target, output_dir, cv_splits=cv_splits)
        reports.append(report)

    if not reports:
        raise ValueError("No valid target could be analyzed.")

    combined = pd.concat(reports, ignore_index=True)
    write_table(combined, Path(output_dir) / "factor_report_all_targets.csv")
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random-forest factor analysis for contest performance metrics.")
    parser.add_argument("--input", required=True, help="Metadata file containing features and target metrics.")
    parser.add_argument("--output-dir", default="outputs/factor_analysis")
    parser.add_argument("--targets", default=None, help="Comma-separated target columns. Defaults to Avg_Relative_* columns.")
    parser.add_argument("--cv-splits", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    combined = run_factor_analysis(args.input, args.output_dir, targets, args.cv_splits)
    print(combined.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
