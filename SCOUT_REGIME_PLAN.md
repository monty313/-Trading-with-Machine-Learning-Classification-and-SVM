# Scout Regime Plan — SVM Workshop Foundation
Date: 2026-10-27 (written about 2:30 AM ET).
Status: PLAN ONLY. No code changed. No model trained.
Owner: Mark. Mark decides every open item.

Copies of this file (same text):
- `C:\Users\C2K\Desktop\FALLOUT TRADER\SCOUT_REGIME_PLAN.md`
- SVM repo root: `SCOUT_REGIME_PLAN.md`

Short handoff: `SVM_HANDOFF.md` (same two places).

Note on folder rule 5 (one master plan):
- `MASTER_PLAN_2026-10-26.md` is still the only master plan.
- This file is a detail plan for Step 1 (Scouts) and Step 4 (Flow).
- Mark decides if the master plan should link to it.

---

## 0. The goal in one line
- Scouts identify market regimes.
- They do not just copy the hard-coded rules.
- They send what they see to the boss (Jev).

## 1. What we start from

### 1.1 The SVM repo
Repo: https://github.com/monty313/-Trading-with-Machine-Learning-Classification-and-SVM
It holds three things:
- The Quantra course files (Mark bought the course).
  - Notebook: `Handoff projects/.../Prediction and Strategy/Trading Strategy Classification.ipynb`
  - Course script: `Handoff projects/SVM/SVM.py` (it places orders).
  - Price data: `Handoff projects/.../data_modules/ICICI Minute Data.csv`
- The Copilot "Regime Model Workshop" (merged PR #1).
  - `app.py` — Streamlit app. Add strategy → Add data → Review → Train → Test → Export.
  - `regime_workshop/config.py` — strategy forms, timeframe sets.
  - `regime_workshop/data.py` — CSV reader, timeframe alignment, rule labels.
  - `regime_workshop/model.py` — SVM training, bundle export, predict.
  - `tests/test_workshop.py` — 7 tests.
- Perplexity's handoff: `Handoff projects/START_HERE_Regime_Model_Builder_Handoff.md`.

### 1.2 What the workshop does well
- Rule non-matches become "unknown", not "no" (`label_examples` in `data.py`).
- Unknown rows are left out of training.
- Bigger-chart bars are used only after they close (`align_timeframes`).
- The scaler is fit on the training period only.
- The bundle saves a model ID, feature order and a data fingerprint.
- The bundle says honestly: "raw SVM margins are not calibrated probabilities".

### 1.3 What it does not do yet
- Inputs are only candle shapes: return, body, wicks, range.
- No CCI, RSI, Bulls/Bears Power, SMA, StdDev, ATR.
- Rule conditions use the SAME features the model sees. So the model can copy the rule.
- Rule matches go straight in as "active". That is still circular.
- Strategy groups can only be AND of groups. A "2 of 3 charts" vote is hard to write.
- One time split. No purge. No embargo.
- No unseen-symbol test. No rules-only baseline.
- No confidence number (`confidence: None`).
- Reader wants one CSV with symbol, timeframe, timestamp, OHLCV. Not Mark's MT5 files.

### 1.4 Fallout Trader today
- 13 scouts are locked (`scouts/lineup.py`).
- Each scout is 3 XGBoost mini-models (COMPLETE line 40).
- NOW label = the rule detector (COMPLETE line 64).
- LATER = "Is that same detector still on in 5 min, 15 min, 1 hour, and 1 day?" (line 65).
- Checks: "walk-forward, leave-one-symbol-out, and the newest 183 days" (line 67).
- "Training target is regime NOW and LATER. Trade-outcome code stays off the main path." (line 495).
- The hard test (`shortcut_stress_run.py`) showed the scouts copy their rules.

---

## 2. Part 1 — SVM as the training backbone

### 2.1 The idea
- Mark's strategies are the teachers.
- They pick out historical examples of each regime.
- A separate set of clues (observations) is what the model learns from.
- The model learns the pattern. Not the exact rule.

### 2.2 The 13 scouts, mapped to workshop regimes
Names are from `1_SCOUTS\FALLOUT_TRADER_COMPLETE.md`, section 2.

| Scout | Name in COMPLETE | Teacher source | Workshop plan |
|---|---|---|---|
| 1 | Trend up | `REGIME_03_TREND_UP.md` | Regime `trend_up`. 3 strategies. Pilot. |
| 2 | Trend down | `REGIME_04_TREND_DOWN.md` | Mirror of 1. |
| 3 | Pullback / Counter-trend up | `REGIME_05_PULLBACK_UP.md` + live screener rule | 3 strategies (RSI, CCI, Heikin-Ashi). |
| 4 | Pullback / Counter-trend down | `REGIME_06_PULLBACK_DOWN.md` | Mirror of 3. |
| 5 | Range / chop / Quiet vol | `REGIME_07_CONFIRMED_LIFECYCLE.md` | 2 strategies (bands, ATR). |
| 6 | Reversal top | STRATEGY_ANSWERS 2026-10-25 | DEPRIORITIZED. Keep. Do not train now. |
| 7 | Reversal bottom | mirror of 6 | DEPRIORITIZED. Keep. Do not train now. |
| 8 | Wild / High vol | StdDev states 1, 2, 9 | Teacher = states. |
| 9 | Re-entry | `reentry.py` | Teacher = RSI 4 cross + trend state. |
| 10 | Expansion | StdDev states 1, 2 | Teacher = states. |
| 11 | Vol pullback / Slowdown / Contracting | StdDev states 3, 4 | Teacher = states. |
| 12 | Undefined | none | No detector. Skip. |
| 13 | Closing / Fading momentum | 3 fading clues | Not one label. Skip for now. |

### 2.3 Scout 1 teacher, quoted
Source: `1_SCOUTS\regime_notes\REGIME_03_TREND_UP.md` and COMPLETE lines 114–122.
- Strategy 1, CCI: CCI 30 > 0. CCI 100 > 0. Each above its own SMA 50. Vote: at least 2 of 3 charts.
- Strategy 2, Power: Bulls Power 13 > 0. Bulls Power 33 > 0. Typical price > SMA 50 of typical price. Vote: all 3 charts.
- Strategy 3, RSI: RSI 4 > 50. RSI 14 > 50. Vote: all 3 charts.
- "The scout is yes if strategy 1 or 2 or 3 passes."
- REGIME_03 line 78: "These are state checks, not crossover events."
- So every Scout 1 condition is a STATE. None is a trigger.

### 2.4 Honest notes about SVM
- SVM is one choice. It is not magic.
- Speed:
  - A kernel SVM (RBF) slows down badly as rows grow.
  - Years of 5m bars on 5 symbols is millions of rows.
  - A full RBF SVM on that is not practical.
- Ways around it:
  - Linear SVM (`LinearSVC` or `SGDClassifier` with hinge loss). Fast on big data.
  - Approximate kernel (`Nystroem` or `RBFSampler`) + linear SVM.
  - Subsample: fewer rows from long, quiet stretches. Keep all edge rows.
- Probabilities:
  - SVM gives a margin, not a probability.
  - It needs calibration on separate, later data (see Part 4).
- Scaling:
  - SVM needs scaled inputs. Fit the scaler on training data only.
- Keep XGBoost:
  - COMPLETE line 40 locks "3 gradient-boosted mini-models".
  - So XGBoost stays as the comparison baseline.
  - SVM must beat it, or match it with a clear benefit.
  - Switching to SVM is Mark's call.
- Switching the model does NOT fix copying by itself.
  - The fix is in the labels and inputs (Part 2).

---

## 3. Part 2 — Active, inactive, unknown

### 3.1 Why
- If label = rule, the model copies the rule.
- Rule-off bars are not all "no". Some are flicker. Some are early starts.

### 3.2 The three labels
- **Active**
  - The rule is on now.
  - AND the regime kept going afterward (checked at Mark's LATER spans).
- **Inactive**
  - The rule is off now.
  - AND it stays off over the LATER spans.
  - Or: a person reviewed it and said "no".
- **Unknown**
  - Everything else. Examples:
    - Rule on, but it blinks off right away (flicker).
    - Rule off, but it turns on soon (possible early start).
    - Bars near a data gap or during indicator warm-up.
  - Unknown is never counted as "no".
  - Unknown rows are left out of training, or reviewed by a person.

### 3.3 Grading on what happened next
- Use Mark's spans: +5m, +15m, +1h, +1d (COMPLETE line 65).
- For each bar, ask: "Is the regime still on at each span?"
- That is a LATER answer. It is not the same as "rule on now".
- Mark decides how many spans must agree for "active".
- Mark decides how to treat short off-blinks.
  - The Scout 1 test found on-runs have a median of only 5 bars (US500, 5 weeks).

### 3.4 Two jobs, kept separate
- Job A (start): "The regime just turned on. Does it last?"
  - One row per start.
- Job B (during): "We are in the regime. Does it continue or change?"
  - One row per on-bar.
  - Rows overlap. They need overlap weights.
- Both jobs can share the same inputs.
- Each job keeps its own output and its own scorecard.

### 3.5 Inputs vs teacher
- Teacher clues build the label.
- Observation clues are the model inputs.
- Do not feed the exact rule flags that build the label.
  - Example: no `regime_*` columns for scouts 8, 10, 11.
- Raw indicator values may stay. The hard test must check for copying.
- Mark decides which clues each scout may keep (scout-fix-guide.md).

### 3.6 Scout 1 real example (US500, Wed 2026-09-30, ET)
From the Scout 1 deep dive. One symbol, one day. Not proof.
- 08:30 start. Close 7684. On by RSI only. +23 in 2h.
- 10:35 mid. On by CCI and Power. +11 in 1h, −41 by 6.5h.
- 12:30 pause. On by CCI only. −10 in 1h.
- 13:20 weakening. CCI only. −26 in 4h.
- 14:20 last on-bar. −35 in 2h.
- Lesson: the rule stayed "on" while price stopped rising.
- Job B is meant to catch that.

---

## 4. Part 3 — Frozen policy and challenger

### 4.1 Two models per scout
- **FROZEN**: the one the live system uses. It does not change.
- **CHALLENGER**: keeps training on new data.
- The challenger replaces the frozen one only after it passes the checklist.
- Promotion is by hand. Only with Mark's OK (MASTER_PLAN Step 2).

### 4.2 Model ID
- Format: `FT-S<set>-SC<scout>-<YYYYMMDD>-R<run>`.
- Example: `FT-S2-SC01-20261027-R01`.
- Never reuse. Never change. (`training\testing_scouts.md`)
- Saved in the folder name and in `MODEL_MANIFEST.json`.

### 4.3 Promotion checklist
All must be true. Write the pass numbers down BEFORE the test. Mark sets them.
- [ ] Has a model ID and description.
- [ ] Trained only on data before the test period.
- [ ] Gap (purge + embargo) between train and test.
- [ ] Beats the rules-only baseline on unseen time.
- [ ] Beats the rules-only baseline on unseen symbols, in unseen time.
- [ ] Beats the FROZEN model on the same tests.
- [ ] Beats "still on later = on now" for LATER answers.
- [ ] Passes `shortcut_stress_run.py` (rule in scout-fix-guide: 80+ and not "falls apart").
- [ ] Calibration checked (Part 4).
- [ ] No peek-ahead (timing test passes).
- [ ] Report saved in `tests\00_test_regimes\runs\<date>_run-NN`.
- [ ] Mark says yes.

Note: the hard test grades against the OLD rule labels. It needs a version for the new labels. See section 7.

---

## 5. Part 4 — Confidence on every label

### 5.1 How
- Turn SVM margins into probabilities with calibration.
- Use `CalibratedClassifierCV` (sigmoid or isotonic).
- Fit calibration on a later time slice than training. Never on the test.
- Check it: reliability chart and Brier score per scout.

### 5.2 What the boss receives (one line per scout, per symbol)
- Symbol.
- Set (2, 3 or 4).
- Time (bar close, ET shown in UI).
- Scout number and regime name.
- Rule yes/no (the teacher, as written).
- Scout probability (calibrated).
- Confidence band. Mark decides the bands.
- State age (bars since the rule turned on).
- LATER read: chance it is still on at +5m, +15m, +1h, +1d.
- Model ID (FROZEN).
- Data quality flags (gap, warm-up, stale bar).

### 5.3 Where it goes
- One status file (MASTER_PLAN Step 4).
- The scouts write it. The UI and the boss read it.

---

## 6. Part 5 — Boss feedback loop (LATER PHASE)
Mark said: focus on scout regime identification first.
This is a sketch only. Do not build yet.
- The boss acts (or not) on scout labels.
- Each outcome is saved with the scout's model ID and label.
- Outcomes become extra review notes for the challenger.
- They never change the FROZEN model directly.
- They only help through the normal challenger + checklist route.
- Risk: the boss's own choices bias the data. Must be handled.
- COMPLETE line 85: "Boss learns from scout reports. Three learning methods. Mark still has to pick the test."

---

## 7. Part 6 — Changes needed in the SVM repo
Do not change the repo yet. This is the list.

### 7.1 Safety and housekeeping
1. **Price CSV**
   - File: `Handoff projects/.../data_modules/ICICI Minute Data.csv`.
   - Why: folder rule 7. No price data on GitHub.
   - Change: remove from git. Add `*.csv` under `Handoff projects/` and `data/` to `.gitignore`.
   - Note: it stays in git history unless history is rewritten. Mark decides.
2. **Repo is public**
   - Course pages say "© QuantInsti".
   - Making it private is safest. Mark's call.
3. **Order code**
   - File: `Handoff projects/SVM/SVM.py` calls `order_target_percent`.
   - Change: move to an `ARCHIVE/course_original/` folder, or add a clear "DO NOT RUN" header. Never import it.
   - Why: order placement stays off.

### 7.2 Data
4. **MT5 loader** — new `regime_workshop/mt5_loader.py`.
   - Read Mark's MT5 CSV exports, one file per symbol and timeframe.
   - Map to symbol, timeframe, timestamp, OHLCV.
   - Server time to UTC. Winter UTC+2 measured. Summer not checked yet.
   - Flag gaps. Restart warm-up after gaps.
   - Exclude the known index hole (2025-07-16 to 2025-09-09).
5. **Forming-bar alignment** — `regime_workshop/data.py`, `align_timeframes`.
   - Now: bigger charts use closed bars only.
   - Mark's doc: "Use forming candles reconstructed exactly as known at that time." (REGIME_03 line 83)
   - Also: "Never reveal a completed bigger-chart candle early." (line 84)
   - Change: build the forming 30m/1h bar from the 5m bars seen so far. Add a mode switch: `closed` or `forming_as_known`.
   - Keep the existing closed-bar test. Add a forming-bar test.
6. **Indicator features** — new `regime_workshop/indicators.py`.
   - CCI 10/30/100 (typical price) and their SMA 50.
   - RSI 4/14.
   - Bulls/Bears Power 13/33.
   - SMA 20/30/50, StdDev 20/30, ATR 5/14, BB 20.
   - On all 3 charts, with roles: anchor, support 1, support 2.
   - Check values against MT5 output before trusting them.
   - Add to `FEATURE_NAMES` / `feature_names_for_set`.
7. **Timeframe sets** — `config.py`.
   - Set 1 is dropped in Fallout (COMPLETE). Hide it or mark it unused.

### 7.3 Labels
8. **Teacher vs observation split** — `config.py`, `data.py`.
   - Teacher conditions read a teacher-only feature table.
   - The model reads the observation profile.
   - Block rule flags that build the label from the observation list.
9. **Votes across charts** — `config.py`.
   - Add "at least N of 3 charts" (Scout 1 CCI = 2 of 3).
   - Add OR between strategies inside one regime (already per strategy; check it).
10. **Stop auto-active** — `data.py`, `label_examples`.
    - Now: rule match = "active".
    - Docs say teacher labels need review. Code skips that.
    - Change: rule match = "candidate". Then apply the LATER check (Part 2). Else "unknown".
11. **LATER labels** — new `regime_workshop/labels_later.py`.
    - For each bar, is the teacher still on at +5m, +15m, +1h, +1d?
    - Separate outputs for Job A and Job B.
    - Save overlap weights.

### 7.4 Training and tests
12. **Fast SVM options** — `model.py`.
    - Add `LinearSVC`, `SGDClassifier`, and `Nystroem` + linear.
    - Keep RBF only for small data.
    - Add row subsampling with a fixed seed.
13. **Calibration** — `model.py`.
    - Wrap with `CalibratedClassifierCV`, on a later time slice.
    - Fill `confidence` in `predict()`.
14. **Walk-forward with purge and embargo** — `model.py`, replace `_chronological_split`.
    - Several time folds.
    - Purge rows whose LATER window crosses into the test.
    - Embargo gap after each test fold. Length: Mark decides (we can propose from data).
    - Keep the newest 183 days locked until the end.
15. **Unseen symbols in unseen time**
    - Leave one symbol out. Test it only on the locked final stretch.
16. **Baselines** — new `regime_workshop/baselines.py`.
    - Rules-only (teacher as written).
    - "Still on later = on now".
    - XGBoost on the same inputs.
17. **Shortcut test**
    - Port the idea of `shortcut_stress_run.py`.
    - Grade against the new LATER labels, not the old rule labels.
18. **Report** — per scout: precision, recall, F1, counts, by symbol, by time, calibration.

### 7.5 Export and link to Fallout
19. **Model ID and bundle** — `model.py`, `export_bundle`.
    - Bundle name = `FT-S<set>-SC<scout>-<YYYYMMDD>-R<run>`.
    - Write `MODEL_MANIFEST.json` with the description fields from `testing_scouts.md`.
    - Add code commit and data date range.
20. **FROZEN / CHALLENGER pointers**
    - Small file per scout naming the two model IDs.
21. **Status file writer**
    - Output the boss fields from section 5.2.
22. **Docs**
    - Update `README.md` and `docs/architecture.md`. Point to this plan.

---

## 8. Known issues
- **Peek-ahead in Fallout `build.py`.**
  - It uses the final values of a 30m or 1h bar that is still forming.
  - On US500 over 5 weeks, about 3.9% of 5m bars differed from the honest version.
  - In the 2026-09-30 US500 episode it blinked on at 08:00 ET. The honest version started at 08:30 ET.
  - Found by reading code and testing a copy of the rule. Repo code not run.
- **Trend Up has no trigger, exit or invalidation.**
  - REGIME_03 line 135: "Losing a qualifying state is not automatically an exit rule."
  - Lines 130–131 list as open: "Early starting and ending signs." "Entry timing, stops, and exits."
  - Scout 13 close target: "Not defined. Do not invent one."
- **Hard test grades old labels.** It needs a new-label version.
- **Fallout splits** have no purge and no embargo. Unseen-symbol tests use the same dates.
- **Summer MT5 clock** is not checked.
- **SVM repo**: public, holds a price CSV, and holds order code (section 7.1).
- **Folder rule 5**: this plan must not compete with the master plan.

## 9. Decisions only Mark can make
- Switch the scouts to SVM, or keep XGBoost with SVM as a test?
- Which scout is the pilot? (Suggested: Scout 1.)
- Which set first? (Suggested: Set 2.)
- What counts as "active" across the 4 LATER spans?
- How to merge short off-blinks.
- Early start and end signs for each regime.
- Which clues each scout may keep as inputs.
- Embargo length.
- Which 1–2 symbols to hold out.
- Pass numbers for the promotion checklist.
- Confidence bands shown to the boss.
- How to vote across the 3 mini-models.
- One model per set, or shared?
- Repo private? Remove the price CSV? Rewrite history?
- When to start the boss feedback loop.

## 10. Where we left off / start here tomorrow
Done tonight (2026-10-27):
- Read the SVM repo and the Fallout docs.
- Wrote this plan and `SVM_HANDOFF.md`.
- No code changed. Nothing trained.

Next steps, in order:
1. Mark reads sections 8 and 9.
2. Mark decides: repo private, and the ICICI CSV.
3. Mark picks: SVM vs XGBoost role, pilot scout, set.
4. Mark answers the Part 2 label questions (active rule, off-blinks).
5. With Mark's OK: fix the peek-ahead (forming bars as known).
6. With Mark's OK: make the SVM repo changes in section 7, safety items first (7.1).
7. Build the Scout 1 pilot on Set 2. Rules-only baseline first.
8. Run the checklist (section 4.3). Report with a model ID.
9. Mark decides on promotion.
10. Later: other scouts. Later still: boss feedback loop.
