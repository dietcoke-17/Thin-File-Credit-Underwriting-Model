"""Live scoring API + demo form for the thin-file credit scorecard.

Loads the bundle produced by train.py (WoE tables, logistic regression
coefficients, PDO scaling constants, KS-optimal cutoff) once at startup
and scores individual customers on request, instead of the batch of
10,000 synthetic customers main.py scores at once.

Usage:
    python train.py   # one-time: fit the model, save model/bundle.joblib
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.inference import load_bundle, score_customer

BUNDLE_PATH = os.environ.get("BUNDLE_PATH", "model/bundle.joblib")

app = FastAPI(title="Thin-File Credit Scorecard API")
_bundle = None


@app.on_event("startup")
def _load_model() -> None:
    global _bundle
    if not os.path.exists(BUNDLE_PATH):
        raise RuntimeError(
            f"{BUNDLE_PATH} not found. Run `python train.py` to build it first."
        )
    _bundle = load_bundle(BUNDLE_PATH)


class CustomerFeatures(BaseModel):
    mobile_recharge_freq_30d: float = Field(..., ge=0, description="Mobile recharges in the last 30 days")
    utility_days_past_due: float = Field(..., ge=0, description="Days past due on utility bills")
    agri_yield_stability_index: float = Field(..., ge=0, le=1, description="Agricultural yield stability, 0-1")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.post("/score")
def score(features: CustomerFeatures):
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return score_customer(features.model_dump(), _bundle)


@app.get("/", response_class=HTMLResponse)
def demo_form():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Thin-File Credit Scorecard</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 1.3rem; }
  label { display: block; margin-top: 16px; font-weight: 600; }
  small { display: block; color: #666; font-weight: 400; }
  input { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
  button { margin-top: 20px; padding: 10px 20px; cursor: pointer; }
  #result { margin-top: 24px; padding: 16px; border-radius: 8px; display: none; }
  #result.approve { background: #e6f4ea; border: 1px solid #55A868; }
  #result.decline { background: #fbe9e7; border: 1px solid #C44E52; }
</style>
</head>
<body>
  <h1>Thin-File Credit Scorecard - live demo</h1>
  <p>Enter alternate-data values for a synthetic applicant and score them against the fitted logistic-regression scorecard.</p>
  <form id="f">
    <label>Mobile recharges in last 30 days
      <small>typical range 0-12</small>
      <input type="number" name="mobile_recharge_freq_30d" value="4" min="0" required>
    </label>
    <label>Utility bill days past due
      <small>typical range 0-40</small>
      <input type="number" name="utility_days_past_due" value="5" min="0" required>
    </label>
    <label>Agricultural yield stability index
      <small>0 (unstable) to 1 (stable)</small>
      <input type="number" name="agri_yield_stability_index" value="0.75" min="0" max="1" step="0.01" required>
    </label>
    <button type="submit">Score applicant</button>
  </form>
  <div id="result"></div>
<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const body = {
    mobile_recharge_freq_30d: parseFloat(form.mobile_recharge_freq_30d.value),
    utility_days_past_due: parseFloat(form.utility_days_past_due.value),
    agri_yield_stability_index: parseFloat(form.agri_yield_stability_index.value),
  };
  const res = await fetch('/score', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const div = document.getElementById('result');
  if (!res.ok) {
    div.className = 'decline';
    div.style.display = 'block';
    div.textContent = 'Error scoring applicant.';
    return;
  }
  const data = await res.json();
  div.className = data.decision === 'Approve' ? 'approve' : 'decline';
  div.style.display = 'block';
  div.innerHTML = `
    <strong>${data.decision}</strong><br>
    Credit score: ${data.credit_score} (300-900 scale)<br>
    Probability of default: ${(data.probability_of_default * 100).toFixed(1)}%<br>
    Cutoff: score &ge; ${data.cutoff_score}
  `;
});
</script>
</body>
</html>
"""
