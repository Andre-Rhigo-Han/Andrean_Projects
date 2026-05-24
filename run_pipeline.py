"""
Convenience entry point for the full modeling workflow.

Each step can still be run independently, but this script documents the intended
project order:
1. estimate fan-vote posterior shares,
2. add late-window relative metrics to metadata,
3. train factor-analysis models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_vote_estimator import DynamicVoteEstimator, MCMCConfig, parse_seasons
from data_preparation import add_relative_late_metrics, write_table
from factor_analysis import run_factor_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the audience-preference modeling pipeline.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--performance", required=True)
    parser.add_argument("--weekly-context", required=True)
    parser.add_argument("--priors", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seasons", default=None, help="Comma/range format, e.g. '3-27' or '18-34'.")
    parser.add_argument("--weeks-count", type=int, default=5)
    parser.add_argument("--n-iterations", type=int, default=2_000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seasons = parse_seasons(args.seasons)

    posterior_path = output_dir / "fan_vote_posterior.csv"
    config = MCMCConfig(n_iterations=args.n_iterations, burn_in=args.burn_in, random_seed=args.seed)
    estimator = DynamicVoteEstimator(args.performance, args.weekly_context, args.priors, config)
    posterior = estimator.run(seasons=seasons)
    write_table(posterior, posterior_path)
    print(f"[1/3] Posterior vote estimates saved to {posterior_path}")

    metadata_metrics_path = output_dir / "metadata_with_late_metrics.csv"
    add_relative_late_metrics(
        metadata_path=args.metadata,
        performance_path=args.performance,
        weekly_context_path=args.weekly_context,
        posterior_path=posterior_path,
        output_path=metadata_metrics_path,
        seasons=seasons,
        weeks_count=args.weeks_count,
    )
    print(f"[2/3] Late-window metrics saved to {metadata_metrics_path}")

    factor_output_dir = output_dir / "factor_analysis"
    run_factor_analysis(metadata_metrics_path, factor_output_dir)
    print(f"[3/3] Factor-analysis reports saved to {factor_output_dir}")


if __name__ == "__main__":
    main()
