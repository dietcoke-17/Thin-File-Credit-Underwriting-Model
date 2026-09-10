"""Central configuration for the pipeline. Keeping these in one place makes
it easy to re-tune the synthetic data or the scorecard scaling without
hunting through every module.
"""

# ---- Data generation ----
N_CUSTOMERS = 10_000
RANDOM_SEED = 42

# ---- Feature selection ----
IV_THRESHOLD = 0.02  # features below this Information Value are dropped
WOE_BINS = 10

# ---- Scorecard (Points-to-Double-the-Odds) ----
PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50  # good:bad odds at the base score
SCORE_FLOOR = 300
SCORE_CAP = 900

# ---- Paths ----
RESULTS_DIR = "results"
FIGURES_DIR = "reports/figures"
