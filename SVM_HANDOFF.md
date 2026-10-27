# SVM Handoff — Start Here
Date: 2026-10-27 (about 2:30 AM ET).
Owner: Mark. Mark returns tomorrow to finish.
Status: PLAN ONLY. No code changed. No model trained.
Full detail: `SCOUT_REGIME_PLAN.md` (same folder).

## 1. The new repo
- https://github.com/monty313/-Trading-with-Machine-Learning-Classification-and-SVM
- It holds:
  - The Quantra "Classification and SVM" course files. Mark bought the course.
  - The Copilot "Regime Model Workshop" app (`app.py`, `regime_workshop/`, `tests/`).
  - Perplexity's handoff: `Handoff projects/START_HERE_Regime_Model_Builder_Handoff.md`.

## 2. The goal
- Scouts identify market regimes.
- They do not just copy the rules.
- They report to the boss (Jev).

## 3. How the repo will fix "copying the rules"
- **SVM is the training backbone.**
  - Mark's strategies are teachers. They pick examples.
  - The model learns from separate clues.
  - Use fast SVM (linear, or approximate kernel). Full RBF is too slow on years of 5m bars.
  - XGBoost stays as the comparison.
- **Labels map onto the 13 scouts.**
  - Scout 1 Trend up is the pilot: CCI 2 of 3, Power all 3, RSI all 3. All state checks.
  - Table in plan section 2.2.
- **Three labels: active, inactive, unknown.**
  - Rule off is "unknown", not "no".
  - Active also needs the regime to last (Mark's spans: +5m, +15m, +1h, +1d).
  - Helps with flicker and early starts.
- **FROZEN and CHALLENGER.**
  - Live uses the FROZEN model.
  - The CHALLENGER keeps training.
  - It is promoted only if it passes the checklist (plan 4.3) and Mark says yes.
  - IDs: `FT-S<set>-SC<scout>-<YYYYMMDD>-R<run>`.
- **Confidence on every label.**
  - Calibrated probability.
  - Boss gets: rule yes/no, scout probability, confidence, state age, model ID.
- **Repo changes** (full list in plan section 7):
  - Add CCI, RSI, Power, SMA, StdDev, ATR on all 3 charts.
  - MT5 CSV loader.
  - Forming bars as known at the time.
  - Rule match is a candidate, not "active".
  - Walk-forward with purge and embargo.
  - Unseen symbols in unseen time.
  - Rules-only baseline.
  - Calibration.
  - Model ID bundle.
  - Remove the price CSV. Disable `SVM.py` order code.

## 4. Later phase
- Boss feedback loop. Boss outcomes help correct the scouts.
- Mark said: regime identification first. Build this later.

## 5. Known issues
- Fallout `build.py` peeks ahead.
  - It uses final values of forming 30m/1h bars.
  - About 3.9% of 5m bars differ on US500 (5 weeks).
- Trend Up has no written trigger, exit or invalidation.
- The repo is public. Course pages say "© QuantInsti". Mark's call.
- The ICICI price CSV is on GitHub. That breaks rule 7. Mark's call.
- `Handoff projects/SVM/SVM.py` places orders. Never run it.

## 6. Open decisions for Mark
- SVM for the scouts, or SVM as a test next to XGBoost?
- Pilot scout and set (suggested: Scout 1, Set 2).
- What counts as "active" across the 4 spans.
- How to merge short off-blinks.
- Which clues each scout keeps.
- Embargo length. Held-out symbols.
- Pass numbers for promotion.
- Repo private? Remove the CSV? Rewrite history?

## 7. Start here tomorrow
1. Read this file. Then plan sections 8 and 9.
2. Decide the repo privacy and the CSV.
3. Pick the pilot scout and set.
4. Answer the label questions.
5. OK the peek-ahead fix.
6. OK the repo changes, safety items first.
7. Build the Scout 1 pilot. Rules-only baseline first.
8. Check results against the checklist. Then decide.
