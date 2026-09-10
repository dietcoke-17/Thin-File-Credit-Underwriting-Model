"""CLI entry point: run the full thin-file credit underwriting pipeline.

Usage:
    python main.py

Produces:
    results/scored_customers.csv   - customer_id, PD, final score
    results/iv_summary.csv         - Information Value per candidate feature
    results/model_summary.txt      - statsmodels Logit summary + ROC-AUC
    reports/figures/*.png          - diagnostic plots
"""

from src.pipeline import run

if __name__ == "__main__":
    run()
