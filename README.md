# Audience Preference Modeling for Reality-Competition Outcomes

This project builds a data-science pipeline for estimating hidden audience voting behavior in a televised dance-competition setting. The central challenge is that public data usually includes judge scores and elimination results, but not the audience vote shares. The project reconstructs plausible fan-vote distributions with Bayesian sampling, then uses machine-learning analysis to study which contestant attributes are associated with judge performance and audience support.

The repository is organized as a clean, script-based implementation rather than a collection of exploratory notebooks. Each module can be run independently, and the full workflow can also be executed through one pipeline entry point.

## Project Goals

The project answers three modeling questions:

1. **Can hidden fan-vote shares be inferred from observed outcomes?**  
   The Bayesian sampler estimates weekly fan-vote distributions by combining judge scores, elimination results, and prior popularity estimates.

2. **How do judge support and audience support differ?**  
   The pipeline computes normalized late-season metrics for both judge-score share and estimated fan-share, making performance comparable across seasons with different numbers of active contestants.

3. **Which contestant-level factors help explain performance?**  
   Random-forest factor analysis evaluates the importance and direction of features such as age, region, occupation group, partner ability, and social-media popularity.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── data_preparation.py
│   ├── bayesian_vote_estimator.py
│   ├── factor_analysis.py
│   ├── social_media_scraper.py
│   ├── run_pipeline.py
│   └── __init__.py
└── outputs/                  # generated locally, ignored by Git
```

## What Each File Does

### `src/data_preparation.py`

Handles data cleaning and feature construction.

Main responsibilities:

- read CSV or Excel files with encoding fallback;
- standardize contestant names for safer joins;
- parse follower strings such as `1.2M`, `850K`, or `12,345` into numeric values;
- merge Instagram and X/Twitter follower counts into metadata;
- compute late-season relative metrics:
  - `Avg_Relative_Score_Last_N_Weeks`
  - `Avg_Relative_Fan_Share_Last_N_Weeks`

Implementation path:

1. Keep the metadata table as the left-side base table so missing follower data does not remove contestants.
2. Normalize names into merge keys using lowercasing, whitespace trimming, and punctuation cleanup.
3. Select the final `N` weeks of each season from the weekly context table.
4. Normalize each contestant's judge score and estimated fan share by the weekly total and the number of active couples.
5. Aggregate the normalized metrics back to the contestant-season level.

Example:

```bash
python src/data_preparation.py add-relative-metrics \
  --metadata data/processed/celebrity_metadata.csv \
  --performance data/processed/performance_data.csv \
  --weekly-context data/processed/weekly_context.csv \
  --posterior outputs/fan_vote_posterior.csv \
  --weeks-count 5 \
  --seasons 18-34 \
  --output outputs/metadata_with_late_metrics.csv
```

### `src/bayesian_vote_estimator.py`

Reconstructs weekly fan-vote shares using Bayesian simulation.

Main responsibilities:

- convert observed judge scores into weekly judge-score shares;
- build a prior fan-support baseline from an external prediction file when available;
- sample latent fan-vote vectors;
- accept only samples consistent with observed elimination outcomes;
- estimate weekly posterior vote share, uncertainty intervals, acceptance rate, and a judge-audience alignment parameter `rho`;
- carry dynamic momentum across weeks and apply a small boost to low-score survivors from the previous week.

Implementation path:

1. For each elimination week, load all active contestants and their judge scores.
2. Convert judge scores into a comparable weekly share.
3. Build a prior baseline from random-forest predictions or, if unavailable, from score rank as a weak fallback.
4. Sample latent vote strengths from a conditional normal distribution controlled by `rho`.
5. Convert latent vote strengths into fan-vote shares using a ReLU-style nonnegative normalization.
6. Apply the hard elimination constraint: the contestants with the lowest combined judge-plus-fan score must match the observed eliminated contestants.
7. Store posterior means, 95% intervals, convergence diagnostics, and surprise shifts from the baseline.

Example:

```bash
python src/bayesian_vote_estimator.py \
  --performance data/processed/performance_data.csv \
  --weekly-context data/processed/weekly_context.csv \
  --priors outputs/celebrity_baseline_predictions.csv \
  --seasons 3-27 \
  --n-iterations 2000 \
  --burn-in 500 \
  --output outputs/fan_vote_posterior.csv
```

### `src/factor_analysis.py`

Runs interpretable machine-learning analysis on the engineered metrics.

Main responsibilities:

- map raw location and occupation fields into modeling categories;
- log-transform social-media follower counts;
- one-hot encode categorical features;
- train one random-forest model per target variable;
- export feature importance, marginal correlation, in-sample error metrics, predictions, and residuals.

Implementation path:

1. Load contestant-level metadata containing target metrics.
2. Build model features: age, partner ability, follower count, region group, and industry group.
3. Train a tuned `RandomForestRegressor` using cross-validated grid search.
4. Compare feature importance with simple correlation direction to separate predictive weight from positive/negative association.
5. Save one factor report per target and a combined summary report.

Example:

```bash
python src/factor_analysis.py \
  --input outputs/metadata_with_late_metrics.csv \
  --targets Avg_Relative_Score_Last_5_Weeks,Avg_Relative_Fan_Share_Last_5_Weeks \
  --output-dir outputs/factor_analysis
```

### `src/social_media_scraper.py`

Optional utility for refreshing public follower-count data.

Main responsibilities:

- connect to a manually opened Chrome browser through the debugging port;
- search names on SocialBlade;
- write raw follower strings for later cleaning by `data_preparation.py`.

This script is intentionally separate from the core modeling pipeline because website layouts and anti-bot controls can change. The model does not require this script if follower-count files are already available.

Example Chrome startup on Windows:

```powershell
chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\temp\chrome-debug"
```

Then run:

```bash
python src/social_media_scraper.py \
  --input data/raw/namelist.xlsx \
  --output data/interim/social_media_results.xlsx
```

### `src/run_pipeline.py`

Runs the full workflow in the intended order.

Pipeline order:

1. estimate weekly posterior fan-vote shares;
2. merge posterior estimates into late-season contestant metrics;
3. run random-forest factor analysis.

Example:

```bash
python src/run_pipeline.py \
  --metadata data/processed/celebrity_metadata.csv \
  --performance data/processed/performance_data.csv \
  --weekly-context data/processed/weekly_context.csv \
  --priors outputs/celebrity_baseline_predictions.csv \
  --seasons 18-34 \
  --weeks-count 5 \
  --output-dir outputs
```

## Expected Data Inputs

The code is designed to be flexible, but the following column names are recommended.

### `performance_data.csv`

| Column | Meaning |
|---|---|
| `Season` | season identifier |
| `Week` | week identifier |
| `Celebrity` | contestant name |
| `Total_Score` or `Average_Score` | judge score for the week |
| `Status_In_Week` | weekly status, such as `Safe`, `Exited`, or `Eliminated` |

### `weekly_context.csv`

| Column | Meaning |
|---|---|
| `Season` | season identifier |
| `Week` | week identifier |
| `Elimination_Count` | number of contestants eliminated in the week |
| `Active_Couples` | number of active contestants/couples |

### `celebrity_metadata.csv`

| Column | Meaning |
|---|---|
| `Season` | season identifier |
| `Celebrity` | contestant name |
| `Age` | age at competition time |
| `Industry` | occupation or public identity |
| `Home_Region` | country or broad region |
| `Home_State` | U.S. state if applicable |
| `Partner_Ability_Score` | partner-quality proxy |
| `instagram_followers` / `followers` | public popularity proxy |

### Optional prior prediction file

| Column | Meaning |
|---|---|
| `Celebrity` | contestant name |
| `Predicted_Baseline` | baseline popularity/fan-support prediction |
| `Star_Power_Residual` | optional residual column used to infer prior uncertainty |

## Method Summary

### Bayesian vote reconstruction

The audience vote is treated as a hidden weekly vector. The sampler proposes a fan-share vector, combines it with the observed judge-share vector, and checks whether the implied elimination result matches the real elimination. Accepted samples form a posterior distribution over possible fan-vote shares.

The model also estimates a weekly `rho` parameter:

- higher `rho` means fan support is more aligned with judge scores;
- lower or negative `rho` means fan voting is more detached from, or opposed to, judge rankings.

### Dynamic prior adjustment

The estimator includes two time-dependent effects:

- **Momentum:** contestants who outperform their baseline fan share in one week receive a small prior shift in the next week.
- **Lifeboat effect:** contestants who survive despite being near the bottom of the judge ranking receive a small next-week boost, representing possible supporter mobilization.

### Factor analysis

After posterior fan shares are estimated, the project compares audience support and judge performance against contestant-level features. Random forests are used because they can capture nonlinear relationships and interactions without requiring a manually specified functional form.

The output reports both:

- **importance**, which measures predictive contribution;
- **correlation**, which gives a simple direction of association.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Output Files

Typical generated files include:

```text
outputs/
├── fan_vote_posterior.csv
├── metadata_with_late_metrics.csv
└── factor_analysis/
    ├── factor_report_all_targets.csv
    ├── factor_report_Avg_Relative_Score_Last_5_Weeks.csv
    ├── factor_report_Avg_Relative_Fan_Share_Last_5_Weeks.csv
    ├── predictions_Avg_Relative_Score_Last_5_Weeks.csv
    └── predictions_Avg_Relative_Fan_Share_Last_5_Weeks.csv
```

## Notes on Data Availability

Raw datasets are not included in this repository. The scripts assume that cleaned CSV or Excel files are placed under `data/` locally. Generated outputs are ignored by Git to keep the repository lightweight and reproducible.

## Why This Project Is Useful

This project demonstrates an end-to-end analytical workflow for a partially observed ranking system:

- data cleaning from heterogeneous sources;
- probabilistic inference under outcome constraints;
- dynamic modeling across time;
- uncertainty estimation;
- interpretable machine-learning analysis;
- reproducible command-line execution.
