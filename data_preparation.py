"""
Data preparation utilities for the audience-vote modeling project.

This module consolidates the one-off scripts used for:
1. merging celebrity metadata with social-media follower counts,
2. cleaning inconsistent name and follower-count formats,
3. creating late-season relative performance metrics that are later used by
   the statistical and machine-learning models.

The code is intentionally file-path agnostic: all input/output paths are passed
from the command line instead of being hard-coded to a local machine.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------------


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file with a small amount of encoding fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    encodings = ("utf-8-sig", "utf-8", "gbk", "utf-16")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown", b"", 0, 1, f"Unable to read {path} with {encodings}: {last_error}"
    )


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV or Excel based on the output suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from column names without changing the user's schema."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    """
    Find a column case-insensitively.

    Parameters
    ----------
    df:
        Input table.
    candidates:
        Preferred column names or keywords. Exact matches are tried first,
        then substring matches.
    required:
        If True, raise a KeyError when no matching column is found.
    """
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lookup:
            return lookup[key]

    for candidate in candidates:
        key = candidate.lower()
        for normalized, original in lookup.items():
            if key in normalized:
                return original

    if required:
        raise KeyError(f"Could not find any of {candidates}. Available columns: {df.columns.tolist()}")
    return None


# ---------------------------------------------------------------------------
# Name and follower cleaning
# ---------------------------------------------------------------------------


def clean_name(value: object) -> str:
    """Create a robust merge key for celebrity names."""
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9 '&.-]", "", value)
    return value


def parse_follower_count(value: object) -> float:
    """
    Convert follower strings such as '1.2M', '850K', '12,345', or 'Not Found'
    into numeric counts. Unparseable values are returned as NaN.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace('"', "").replace(",", "")
    if not text or text.lower() in {"na", "nan", "none", "not found", "n/a", "--"}:
        return np.nan

    match = re.search(r"([-+]?\d*\.?\d+)\s*([kmb]?)", text, flags=re.IGNORECASE)
    if not match:
        return np.nan

    number = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"": 1.0, "k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}[suffix]
    return number * multiplier


def split_semicolon_social_cell(value: object) -> tuple[str, float]:
    """
    Parse cells of the form 'Celebrity Name; 123K'.

    Some scraped X/Twitter files store a name and follower count in one cell.
    This helper separates them in a deterministic way.
    """
    text = "" if pd.isna(value) else str(value)
    if ";" not in text:
        return text.strip(), np.nan
    name, followers = text.split(";", 1)
    return name.strip(), parse_follower_count(followers)


# ---------------------------------------------------------------------------
# Merge social-media metadata
# ---------------------------------------------------------------------------


def merge_social_followers(
    metadata_path: str | Path,
    instagram_path: str | Path | None,
    x_path: str | Path | None,
    output_path: str | Path,
    celebrity_col: str = "Celebrity",
) -> pd.DataFrame:
    """
    Merge Instagram and X/Twitter follower counts into the celebrity metadata.

    The function preserves every row in the metadata table and leaves unmatched
    follower counts blank, which is safer than dropping contestants with missing
    public profile data.
    """
    metadata = standardize_columns(read_table(metadata_path))
    celebrity_col = find_column(metadata, [celebrity_col, "celebrity", "celebrity_name", "name"])
    metadata["_merge_name"] = metadata[celebrity_col].map(clean_name)

    result = metadata.copy()

    if instagram_path:
        instagram = standardize_columns(read_table(instagram_path))
        ins_name_col = find_column(instagram, ["celebrity_name", "Celebrity", "name"])
        ins_follower_col = find_column(instagram, ["followers", "instagram_followers", "follower"])
        instagram_subset = instagram[[ins_name_col, ins_follower_col]].copy()
        instagram_subset["_merge_name"] = instagram_subset[ins_name_col].map(clean_name)
        instagram_subset["instagram_followers"] = instagram_subset[ins_follower_col].map(parse_follower_count)
        instagram_subset = instagram_subset[["_merge_name", "instagram_followers"]].drop_duplicates("_merge_name")
        result = result.merge(instagram_subset, on="_merge_name", how="left")

    if x_path:
        x_df = standardize_columns(read_table(x_path))
        x_name_col = find_column(x_df, ["celebrity_name", "Celebrity", "name", "clean_name"], required=False)
        x_follower_col = find_column(x_df, ["x_followers", "twitter_followers", "followers", "follower"], required=False)

        if x_name_col is None or x_follower_col is None:
            # Fallback for one-cell scraped format, e.g. 'name; followers'.
            candidate_col = x_df.columns[1] if len(x_df.columns) > 1 else x_df.columns[0]
            parsed = x_df[candidate_col].apply(split_semicolon_social_cell)
            x_subset = pd.DataFrame(parsed.tolist(), columns=["_raw_name", "x_followers"])
            x_subset["_merge_name"] = x_subset["_raw_name"].map(clean_name)
        else:
            x_subset = x_df[[x_name_col, x_follower_col]].copy()
            x_subset["_merge_name"] = x_subset[x_name_col].map(clean_name)
            x_subset["x_followers"] = x_subset[x_follower_col].map(parse_follower_count)

        x_subset = x_subset[["_merge_name", "x_followers"]].drop_duplicates("_merge_name")
        result = result.merge(x_subset, on="_merge_name", how="left")

    result = result.drop(columns=["_merge_name"])
    write_table(result, output_path)
    return result


def merge_existing_follower_columns(
    target_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    name_col: str = "Celebrity",
    season_col: str = "Season",
    follower_cols: Sequence[str] = ("instagram_followers", "x_followers"),
) -> pd.DataFrame:
    """
    Left-join follower columns from one metadata file into another by season and name.
    """
    target = standardize_columns(read_table(target_path))
    source = standardize_columns(read_table(source_path))

    target_name = find_column(target, [name_col, "celebrity", "celebrity_name", "name"])
    source_name = find_column(source, [name_col, "celebrity", "celebrity_name", "name"])
    target_season = find_column(target, [season_col, "season"])
    source_season = find_column(source, [season_col, "season"])

    target["_merge_name"] = target[target_name].map(clean_name)
    source["_merge_name"] = source[source_name].map(clean_name)
    target["_merge_season"] = pd.to_numeric(target[target_season], errors="coerce").astype("Int64")
    source["_merge_season"] = pd.to_numeric(source[source_season], errors="coerce").astype("Int64")

    keep_cols = [c for c in follower_cols if c in source.columns]
    if not keep_cols:
        raise KeyError(f"None of the requested follower columns {follower_cols} were found in {source_path}")

    subset = source[["_merge_name", "_merge_season", *keep_cols]].copy()
    merged = target.merge(subset, on=["_merge_name", "_merge_season"], how="left")
    merged = merged.drop(columns=["_merge_name", "_merge_season"])
    write_table(merged, output_path)
    return merged


# ---------------------------------------------------------------------------
# Late-window relative metrics
# ---------------------------------------------------------------------------


def build_late_week_index(
    weekly_context: pd.DataFrame,
    seasons: Iterable[int] | None = None,
    weeks_count: int = 5,
) -> pd.DataFrame:
    """Return the final `weeks_count` weeks for each requested season."""
    season_col = find_column(weekly_context, ["Season", "season"])
    week_col = find_column(weekly_context, ["Week", "week"])

    weekly = weekly_context[[season_col, week_col]].copy()
    weekly.columns = ["Season", "Week"]
    weekly["Season"] = pd.to_numeric(weekly["Season"], errors="coerce")
    weekly["Week"] = pd.to_numeric(weekly["Week"], errors="coerce")
    weekly = weekly.dropna().astype({"Season": int, "Week": int})

    if seasons is not None:
        season_set = {int(s) for s in seasons}
        weekly = weekly[weekly["Season"].isin(season_set)]

    records: list[dict[str, int]] = []
    for season, group in weekly.groupby("Season"):
        max_week = int(group["Week"].max())
        start_week = max(1, max_week - weeks_count + 1)
        records.extend({"Season": int(season), "Week": int(w)} for w in range(start_week, max_week + 1))

    return pd.DataFrame(records)


def add_relative_late_metrics(
    metadata_path: str | Path,
    performance_path: str | Path,
    weekly_context_path: str | Path,
    posterior_path: str | Path,
    output_path: str | Path,
    seasons: Iterable[int] | None = None,
    weeks_count: int = 5,
) -> pd.DataFrame:
    """
    Add late-season relative judge-score and fan-share metrics to metadata.

    The relative metric is normalized by the number of active couples in a week:
        contestant_share / weekly_sum * active_couples
    Values above 1 indicate above-average performance within that week.
    """
    metadata = standardize_columns(read_table(metadata_path))
    performance = standardize_columns(read_table(performance_path))
    weekly = standardize_columns(read_table(weekly_context_path))
    posterior = standardize_columns(read_table(posterior_path))

    valid_weeks = build_late_week_index(weekly, seasons=seasons, weeks_count=weeks_count)

    season_col = find_column(performance, ["Season", "season"])
    week_col = find_column(performance, ["Week", "week"])
    celeb_col = find_column(performance, ["Celebrity", "celebrity", "celebrity_name"])
    score_col = find_column(performance, ["Average_Score", "Total_Score", "score"])
    active_col = find_column(weekly, ["Active_Couples", "active_couples", "contestant_count"])

    perf = performance[[season_col, week_col, celeb_col, score_col]].copy()
    perf.columns = ["Season", "Week", "Celebrity", "Average_Score"]
    perf["Average_Score"] = pd.to_numeric(perf["Average_Score"], errors="coerce")

    weekly_small = weekly[[find_column(weekly, ["Season", "season"]), find_column(weekly, ["Week", "week"]), active_col]].copy()
    weekly_small.columns = ["Season", "Week", "Active_Couples"]
    weekly_small["Active_Couples"] = pd.to_numeric(weekly_small["Active_Couples"], errors="coerce")

    perf = perf.merge(weekly_small, on=["Season", "Week"], how="left")
    perf["Weekly_Score_Sum"] = perf.groupby(["Season", "Week"])["Average_Score"].transform("sum")
    perf["Relative_Score_Share"] = (
        perf["Average_Score"] / perf["Weekly_Score_Sum"].replace(0, np.nan) * perf["Active_Couples"]
    )
    perf_late = perf.merge(valid_weeks, on=["Season", "Week"], how="inner")
    score_metric_col = f"Avg_Relative_Score_Last_{weeks_count}_Weeks"
    score_agg = (
        perf_late.groupby(["Season", "Celebrity"], as_index=False)["Relative_Score_Share"]
        .mean()
        .rename(columns={"Relative_Score_Share": score_metric_col})
    )

    post_season_col = find_column(posterior, ["Season", "season"])
    post_week_col = find_column(posterior, ["Week", "week"])
    post_celeb_col = find_column(posterior, ["Celebrity", "celebrity", "celebrity_name"])
    fan_col = find_column(posterior, ["Est_Fan_Share_Mean", "fan_share", "vote_share"])
    post = posterior[[post_season_col, post_week_col, post_celeb_col, fan_col]].copy()
    post.columns = ["Season", "Week", "Celebrity", "Est_Fan_Share_Mean"]
    post["Est_Fan_Share_Mean"] = pd.to_numeric(post["Est_Fan_Share_Mean"], errors="coerce")
    post = post.merge(weekly_small, on=["Season", "Week"], how="left")
    post["Weekly_Fan_Sum"] = post.groupby(["Season", "Week"])["Est_Fan_Share_Mean"].transform("sum")
    post["Relative_Fan_Share"] = (
        post["Est_Fan_Share_Mean"] / post["Weekly_Fan_Sum"].replace(0, np.nan) * post["Active_Couples"]
    )
    post_late = post.merge(valid_weeks, on=["Season", "Week"], how="inner")
    fan_metric_col = f"Avg_Relative_Fan_Share_Last_{weeks_count}_Weeks"
    fan_agg = (
        post_late.groupby(["Season", "Celebrity"], as_index=False)["Relative_Fan_Share"]
        .mean()
        .rename(columns={"Relative_Fan_Share": fan_metric_col})
    )

    meta_season = find_column(metadata, ["Season", "season"])
    meta_celeb = find_column(metadata, ["Celebrity", "celebrity", "celebrity_name"])
    result = metadata.copy()
    result["_merge_name"] = result[meta_celeb].map(clean_name)
    result["_merge_season"] = pd.to_numeric(result[meta_season], errors="coerce").astype("Int64")

    for agg in (score_agg, fan_agg):
        agg = agg.copy()
        agg["_merge_name"] = agg["Celebrity"].map(clean_name)
        agg["_merge_season"] = pd.to_numeric(agg["Season"], errors="coerce").astype("Int64")
        metric_cols = [c for c in agg.columns if c.startswith("Avg_Relative_")]
        result = result.merge(agg[["_merge_name", "_merge_season", *metric_cols]], on=["_merge_name", "_merge_season"], how="left")

    result = result.drop(columns=["_merge_name", "_merge_season"])
    write_table(result, output_path)
    return result


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_seasons(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    seasons: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if "-" in token:
            start, end = token.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(token))
    return sorted(set(seasons))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare metadata and engineered metrics for the vote-modeling pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_social = subparsers.add_parser("merge-social", help="Merge Instagram/X follower counts into metadata.")
    merge_social.add_argument("--metadata", required=True)
    merge_social.add_argument("--instagram", default=None)
    merge_social.add_argument("--x", default=None)
    merge_social.add_argument("--output", required=True)

    merge_followers = subparsers.add_parser("merge-followers", help="Merge existing follower columns into a target metadata file.")
    merge_followers.add_argument("--target", required=True)
    merge_followers.add_argument("--source", required=True)
    merge_followers.add_argument("--output", required=True)

    metrics = subparsers.add_parser("add-relative-metrics", help="Add late-season relative score and fan-share metrics.")
    metrics.add_argument("--metadata", required=True)
    metrics.add_argument("--performance", required=True)
    metrics.add_argument("--weekly-context", required=True)
    metrics.add_argument("--posterior", required=True)
    metrics.add_argument("--weeks-count", type=int, default=5)
    metrics.add_argument("--seasons", default=None, help="Comma/range format, e.g. '18-34' or '30,31,32'.")
    metrics.add_argument("--output", required=True)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "merge-social":
        merge_social_followers(args.metadata, args.instagram, args.x, args.output)
    elif args.command == "merge-followers":
        merge_existing_follower_columns(args.target, args.source, args.output)
    elif args.command == "add-relative-metrics":
        add_relative_late_metrics(
            metadata_path=args.metadata,
            performance_path=args.performance,
            weekly_context_path=args.weekly_context,
            posterior_path=args.posterior,
            output_path=args.output,
            seasons=parse_seasons(args.seasons),
            weeks_count=args.weeks_count,
        )


if __name__ == "__main__":
    main()
