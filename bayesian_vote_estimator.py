"""
Bayesian reconstruction of unobserved audience vote shares.

The estimator combines three sources of information:
1. observed judge scores,
2. observed elimination outcomes,
3. prior popularity/baseline estimates from a separate machine-learning model.

For each week, the sampler proposes latent fan-vote shares and keeps samples
that are consistent with the observed elimination result under the competition's
combined-score rule. A dynamic prior carries momentum across weeks and applies a
small "lifeboat" boost to contestants who were previously in the low-score zone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from tqdm import tqdm

from data_preparation import clean_name, find_column, read_table, write_table


@dataclass
class MCMCConfig:
    """Sampling and dynamic-prior parameters."""

    n_iterations: int = 2_000
    burn_in: int = 500
    start_rhos: tuple[float, ...] = (0.5, -0.2, 0.0, 0.2, -0.5)
    rho_step: float = 0.10
    prior_sigma: float = 0.35
    momentum_factor: float = 0.20
    lifeboat_factor: float = 1.10
    random_seed: int = 42


@dataclass
class WeekEstimate:
    season: int
    week: int
    records: list[dict]
    bottom_zone_names: list[str]
    momentum_shift: dict[str, float]


def calculate_rhat(chains_samples: list[np.ndarray]) -> np.ndarray | None:
    """Gelman-Rubin R-hat diagnostic for equal-shaped chains."""
    valid_chains = [c for c in chains_samples if len(c) > 10]
    if len(valid_chains) < 2:
        return None

    min_len = min(len(c) for c in valid_chains)
    if min_len < 10:
        return None

    data = np.array([c[:min_len] for c in valid_chains])
    m_chains, n_samples, _ = data.shape
    chain_means = np.mean(data, axis=1)
    between = n_samples * np.var(chain_means, axis=0, ddof=1)
    within = np.mean(np.var(data, axis=1, ddof=1), axis=0)
    within = np.where(within == 0, 1e-9, within)
    variance_plus = ((n_samples - 1) / n_samples) * within + (between / n_samples)
    return np.sqrt(variance_plus / within)


class DynamicVoteEstimator:
    """Estimate weekly fan-vote posterior distributions from score/outcome data."""

    def __init__(
        self,
        performance_path: str | Path,
        weekly_context_path: str | Path,
        prior_path: str | Path | None = None,
        config: MCMCConfig | None = None,
    ):
        self.performance_path = Path(performance_path)
        self.weekly_context_path = Path(weekly_context_path)
        self.prior_path = Path(prior_path) if prior_path else None
        self.config = config or MCMCConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        self.performance = self._load_performance()
        self.weekly_context = self._load_weekly_context()
        self.priors = self._load_priors()
        self.prior_sigma = self._infer_prior_sigma()

    # ------------------------------------------------------------------
    # Loading and schema normalization
    # ------------------------------------------------------------------

    def _load_performance(self) -> pd.DataFrame:
        df = read_table(self.performance_path)
        season = find_column(df, ["Season", "season"])
        week = find_column(df, ["Week", "week"])
        celeb = find_column(df, ["Celebrity", "celebrity", "celebrity_name", "name"])
        score = find_column(df, ["Total_Score", "Average_Score", "current_total", "score"])
        status = find_column(df, ["Status_In_Week", "Actual_Status", "status", "results"], required=False)

        keep_cols = [season, week, celeb, score] + ([status] if status else [])
        out = df[keep_cols].copy()
        out.columns = ["Season", "Week", "Celebrity", "Judge_Score"] + (["Status"] if status else [])
        out["Season"] = pd.to_numeric(out["Season"], errors="coerce").astype("Int64")
        out["Week"] = pd.to_numeric(out["Week"], errors="coerce").astype("Int64")
        out["Judge_Score"] = pd.to_numeric(out["Judge_Score"], errors="coerce")
        out["Celebrity_Key"] = out["Celebrity"].map(clean_name)
        if "Status" not in out.columns:
            out["Status"] = "Safe"
        out = out.dropna(subset=["Season", "Week", "Judge_Score", "Celebrity_Key"])
        return out.astype({"Season": int, "Week": int})

    def _load_weekly_context(self) -> pd.DataFrame:
        df = read_table(self.weekly_context_path)
        season = find_column(df, ["Season", "season"])
        week = find_column(df, ["Week", "week"])
        elim = find_column(df, ["Elimination_Count", "elimination_count", "n_elim"], required=False)
        active = find_column(df, ["Active_Couples", "active_couples", "contestant_count"], required=False)

        cols = [season, week] + ([elim] if elim else []) + ([active] if active else [])
        out = df[cols].copy()
        out.columns = ["Season", "Week"] + (["Elimination_Count"] if elim else []) + (["Active_Couples"] if active else [])
        out["Season"] = pd.to_numeric(out["Season"], errors="coerce").astype("Int64")
        out["Week"] = pd.to_numeric(out["Week"], errors="coerce").astype("Int64")
        if "Elimination_Count" in out.columns:
            out["Elimination_Count"] = pd.to_numeric(out["Elimination_Count"], errors="coerce").fillna(0).astype(int)
        else:
            out["Elimination_Count"] = 1
        out = out.dropna(subset=["Season", "Week"])
        out = out.astype({"Season": int, "Week": int})
        return out.sort_values(["Season", "Week"])

    def _load_priors(self) -> pd.DataFrame | None:
        if self.prior_path is None:
            return None
        df = read_table(self.prior_path)
        celeb = find_column(df, ["Celebrity", "celebrity", "celebrity_name", "name"])
        baseline = find_column(df, ["Predicted_Baseline", "Predicted", "prediction", "baseline", "Star_Power_Residual"], required=False)
        if baseline is None:
            return None
        out = df[[celeb, baseline]].copy()
        out.columns = ["Celebrity", "Predicted_Baseline"]
        out["Celebrity_Key"] = out["Celebrity"].map(clean_name)
        out["Predicted_Baseline"] = pd.to_numeric(out["Predicted_Baseline"], errors="coerce")
        return out.dropna(subset=["Celebrity_Key"])

    def _infer_prior_sigma(self) -> float:
        if self.prior_path is None:
            return self.config.prior_sigma
        df = read_table(self.prior_path)
        residual_col = find_column(df, ["Star_Power_Residual", "Residual", "error"], required=False)
        if residual_col is None:
            return self.config.prior_sigma
        values = pd.to_numeric(df[residual_col], errors="coerce").dropna()
        return float(values.std()) if len(values) > 1 and values.std() > 0 else self.config.prior_sigma

    # ------------------------------------------------------------------
    # Weekly estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _status_to_eliminated(status: object) -> bool:
        text = str(status).lower()
        return any(token in text for token in ("exited", "eliminated", "bottom", "out")) and "safe" not in text

    def _week_data(self, season: int, week: int) -> pd.DataFrame:
        data = self.performance[(self.performance["Season"] == season) & (self.performance["Week"] == week)].copy()
        if data.empty:
            return data
        data["Eliminated"] = data["Status"].map(self._status_to_eliminated)

        if self.priors is not None:
            data = data.merge(
                self.priors[["Celebrity_Key", "Predicted_Baseline"]].drop_duplicates("Celebrity_Key"),
                on="Celebrity_Key",
                how="left",
            )
        else:
            data["Predicted_Baseline"] = np.nan

        if data["Predicted_Baseline"].isna().all():
            # Fallback: judge score rank as a weak prior.
            ranks = rankdata(data["Judge_Score"].to_numpy(), method="average")
            data["Predicted_Baseline"] = ranks / ranks.max()
        else:
            data["Predicted_Baseline"] = data["Predicted_Baseline"].fillna(data["Predicted_Baseline"].mean())
        return data

    def _judge_share_and_z(self, judge_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        judge_scores = np.asarray(judge_scores, dtype=float)
        if judge_scores.sum() <= 0:
            judge_share = np.full(len(judge_scores), 1.0 / len(judge_scores))
        else:
            judge_share = judge_scores / judge_scores.sum()

        ranks = rankdata(judge_scores, method="average")
        quantiles = np.clip((ranks - 0.5) / len(judge_scores), 1e-6, 1 - 1e-6)
        judge_z = norm.ppf(quantiles)
        return judge_share, judge_z

    def _fan_share_from_latent(self, latent_votes: np.ndarray) -> np.ndarray:
        relu_votes = np.maximum(latent_votes, 0)
        total = relu_votes.sum()
        if total <= 0:
            return np.full(len(latent_votes), 1.0 / len(latent_votes))
        return relu_votes / total

    @staticmethod
    def _passes_elimination_constraint(
        judge_share: np.ndarray,
        fan_share: np.ndarray,
        actual_elim_indices: np.ndarray,
        n_elim: int,
    ) -> bool:
        combined = judge_share + fan_share
        simulated_elims = np.argsort(combined)[:n_elim]
        return set(simulated_elims.tolist()) == set(actual_elim_indices.tolist())

    def _adjust_priors(
        self,
        names: Iterable[str],
        raw_priors: np.ndarray,
        momentum_memory: dict[str, float],
        bottom_zone_names: list[str],
    ) -> np.ndarray:
        adjusted = []
        bottom_set = set(bottom_zone_names)
        for name, base_value in zip(names, raw_priors):
            value = float(base_value)
            value += momentum_memory.get(name, 0.0) * 5.0 * self.config.momentum_factor
            if name in bottom_set:
                value = value * self.config.lifeboat_factor if value > 0 else value + 0.1
            adjusted.append(value)
        return np.array(adjusted, dtype=float)

    def estimate_week(
        self,
        season: int,
        week: int,
        n_elim: int,
        momentum_memory: dict[str, float],
        bottom_zone_names: list[str],
    ) -> WeekEstimate | None:
        week_data = self._week_data(season, week)
        if week_data.empty:
            return None

        names = week_data["Celebrity"].astype(str).tolist()
        name_keys = week_data["Celebrity_Key"].astype(str).tolist()
        judge_scores = week_data["Judge_Score"].to_numpy(dtype=float)
        judge_share, judge_z = self._judge_share_and_z(judge_scores)
        raw_priors = week_data["Predicted_Baseline"].to_numpy(dtype=float)
        prior_means = self._adjust_priors(name_keys, raw_priors, momentum_memory, bottom_zone_names)

        actual_elim_indices = np.flatnonzero(week_data["Eliminated"].to_numpy(dtype=bool))
        if len(actual_elim_indices) == 0:
            return None
        n_elim = int(len(actual_elim_indices)) if n_elim <= 0 else min(int(n_elim), len(actual_elim_indices))

        chains: list[np.ndarray] = []
        rho_samples: list[float] = []
        total_attempts = 0

        for start_rho in self.config.start_rhos:
            current_rho = float(start_rho)
            chain_shares: list[np.ndarray] = []

            for iteration in range(self.config.n_iterations):
                proposed_rho = float(np.clip(current_rho + self.rng.normal(0, self.config.rho_step), -0.99, 0.99))
                conditional_mu = prior_means + proposed_rho * self.prior_sigma * judge_z
                conditional_sigma = self.prior_sigma * np.sqrt(1 - proposed_rho**2 + 1e-6)
                latent = self.rng.normal(conditional_mu, conditional_sigma)
                fan_share = self._fan_share_from_latent(latent)
                valid = self._passes_elimination_constraint(judge_share, fan_share, actual_elim_indices, n_elim)

                if iteration > self.config.burn_in:
                    total_attempts += 1

                if valid:
                    current_rho = proposed_rho
                    if iteration > self.config.burn_in:
                        chain_shares.append(fan_share)
                        rho_samples.append(current_rho)

            if chain_shares:
                chains.append(np.array(chain_shares))

        if not chains:
            return None

        all_shares = np.vstack(chains)
        posterior_mean = all_shares.mean(axis=0)
        rho_mean = float(np.mean(rho_samples)) if rho_samples else np.nan
        rhat = calculate_rhat(chains)
        max_rhat = float(np.nanmax(rhat)) if rhat is not None else np.nan

        prior_relu = np.maximum(prior_means, 0)
        prior_share = prior_relu / (prior_relu.sum() + 1e-9)
        shift = posterior_mean - prior_share
        new_momentum = {name: float(value) for name, value in zip(name_keys, shift)}

        unique_scores = sorted(set(judge_scores.tolist()))
        if len(unique_scores) >= 2:
            cutoff = unique_scores[1]
        elif unique_scores:
            cutoff = unique_scores[0]
        else:
            cutoff = -np.inf
        new_bottom_zone = [name_keys[i] for i, score in enumerate(judge_scores) if score <= cutoff]

        ci_lower = np.percentile(all_shares, 2.5, axis=0)
        ci_upper = np.percentile(all_shares, 97.5, axis=0)
        records = []
        for idx, name in enumerate(names):
            records.append(
                {
                    "Season": season,
                    "Week": week,
                    "Celebrity": name,
                    "Actual_Status": "Eliminated" if idx in actual_elim_indices else "Safe",
                    "Judge_Share_Pct": judge_share[idx],
                    "Est_Fan_Share_Mean": posterior_mean[idx],
                    "Total_Score_Pct": judge_share[idx] + posterior_mean[idx],
                    "Est_Rho_Mean": rho_mean,
                    "R_hat_Convergence": max_rhat,
                    "Shift_Surprise": shift[idx],
                    "RF_Baseline_Share": prior_share[idx],
                    "CI_Lower": ci_lower[idx],
                    "CI_Upper": ci_upper[idx],
                    "Acceptance_Rate": len(all_shares) / max(total_attempts, 1),
                }
            )

        return WeekEstimate(season, week, records, new_bottom_zone, new_momentum)

    def run(self, seasons: Iterable[int] | None = None) -> pd.DataFrame:
        """Estimate posterior fan shares for all weeks with elimination events."""
        weeks = self.weekly_context[self.weekly_context["Elimination_Count"] > 0].copy()
        if seasons is not None:
            season_set = {int(s) for s in seasons}
            weeks = weeks[weeks["Season"].isin(season_set)]

        records: list[dict] = []
        current_season: int | None = None
        momentum_memory: dict[str, float] = {}
        bottom_zone: list[str] = []
        failed_weeks: list[tuple[int, int]] = []

        for row in tqdm(weeks.itertuples(index=False), total=len(weeks), desc="Estimating weekly posteriors"):
            season = int(getattr(row, "Season"))
            week = int(getattr(row, "Week"))
            n_elim = int(getattr(row, "Elimination_Count"))

            if season != current_season:
                current_season = season
                momentum_memory = {}
                bottom_zone = []

            estimate = self.estimate_week(season, week, n_elim, momentum_memory, bottom_zone)
            if estimate is None:
                failed_weeks.append((season, week))
                continue

            records.extend(estimate.records)
            momentum_memory.update(estimate.momentum_shift)
            bottom_zone = estimate.bottom_zone_names

        result = pd.DataFrame(records)
        result.attrs["failed_weeks"] = failed_weeks
        return result


def parse_seasons(text: str | None) -> list[int] | None:
    if not text:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate posterior audience vote shares with dynamic Bayesian sampling.")
    parser.add_argument("--performance", required=True, help="Long-format weekly performance CSV/Excel file.")
    parser.add_argument("--weekly-context", required=True, help="Weekly context file with elimination counts.")
    parser.add_argument("--priors", default=None, help="Optional RF prior/baseline predictions file.")
    parser.add_argument("--output", default="outputs/fan_vote_posterior.csv")
    parser.add_argument("--seasons", default=None, help="Comma/range format, e.g. '3-27' or '30,31,32'.")
    parser.add_argument("--n-iterations", type=int, default=MCMCConfig.n_iterations)
    parser.add_argument("--burn-in", type=int, default=MCMCConfig.burn_in)
    parser.add_argument("--seed", type=int, default=MCMCConfig.random_seed)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MCMCConfig(n_iterations=args.n_iterations, burn_in=args.burn_in, random_seed=args.seed)
    estimator = DynamicVoteEstimator(args.performance, args.weekly_context, args.priors, config)
    posterior = estimator.run(seasons=parse_seasons(args.seasons))
    write_table(posterior, args.output)
    failed = posterior.attrs.get("failed_weeks", [])
    print(f"Saved posterior estimates to {args.output}")
    if failed:
        print(f"Weeks skipped because no valid posterior sample was found: {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
