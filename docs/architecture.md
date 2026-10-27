# Architecture, reuse decisions, and limitations

## Repository inventory

The supplied repository contained only a short `README.md` describing the
Quantra course. It had no course notebooks or scripts, no strategy/doctrine
definitions, no historical data, no dependencies, and no existing tests. The
workshop therefore starts with a small synthetic demonstration and does not
invent or reproduce course or user strategy definitions.

The implementation reuses no downloaded course code. It uses scikit-learn
pipelines for scaling and SVM classification, Streamlit for the local UI, and
standard-library CSV/JSON handling. Course material redistribution permissions
were not present in the checkout and have not been assumed.

## Components and saved data

- `regime_workshop/config.py` validates regime, strategy, timeframe-set, and
  grouped-condition configuration; condition expressions are data, never
  evaluated as Python.
- `regime_workshop/data.py` imports canonical OHLCV CSV, validates timestamps,
  numeric values and duplicates, creates candle features, aligns completed
  support bars, and attaches independent per-regime teaching labels.
- `regime_workshop/model.py` fits one preprocessing-plus-SVM pipeline per
  regime, evaluates a chronological holdout, and exports/loads versioned
  estimator bundles.
- `app.py` provides the guided local workflow. `generate_demo.py` creates
  deterministic synthetic candles. `examples/consumer.py` demonstrates
  prediction without teacher rules.
- `workshop_data/workshop.json`, `examples.json`, and `exports/` are local
  runtime files and are intentionally ignored by Git.

Several regimes may be active on the same row. Each regime has its own
`active` / `inactive` / `unknown` target. A teaching-rule match is recorded as
`strategy-generated`; every non-match stays `unknown` until explicitly
reviewed. Only user-reviewed active/inactive examples are included in that
regime's supervised fit and holdout.

## CSV input and time handling

The initial importer supports this canonical header (case-insensitive):

```text
symbol,timeframe,timestamp,open,high,low,close,volume
```

Timeframes must be one of `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, or `1d`.
Timestamps must be ISO 8601 and include a timezone. The importer converts them
to UTC and rejects duplicates and invalid/non-finite OHLCV. The UI asks whether
timestamps represent bar-open or bar-close time; open times are advanced by
the bar duration. Alignment uses an anchor bar and the latest support bars
whose timestamps are no later than the anchor decision time. Old support bars
are excluded by a freshness limit. Official sets are:

| Set | Anchor | Support 1 | Support 2 |
| --- | --- | --- | --- |
| Set 1 | 1m | 15m | 30m |
| Set 2 | 5m | 30m | 1h |
| Set 3 | 15m | 1h | 4h |
| Set 4 | 30m | 4h | 1d |

Generated numerical inputs keep anchor, support 1, and support 2 features
separate: one-bar return, body fraction, upper/lower wick fraction, and range
fraction. A one-hot feature records the selected official timeframe set so
rows from different sets remain distinguishable during training. This baseline
does not import MT5 indicator exports, calculate
technical indicators, make synchronized charts, or validate parity against
MT5. CSV mapping is limited to the documented canonical column names.
Missing or stale support bars and indicator warm-up are handled by omitting
unalignable rows, but a full missing-bar/gap report and interactive import
mapping are not implemented.

## Training, testing, and export

Each regime gets a separate `StandardScaler` + class-balanced linear or RBF
SVC. Unknown targets are excluded, and all scaling is fitted on that regime's
earlier chronological training segment. The later segment is a single
chronological holdout. The report includes precision, recall, F1, confusion
matrix, counts, and the test period. A regime with too few reviewed samples or
only one class is reported as skipped; no successful training is claimed for
it.

The bundle includes estimators, feature order, regime names, strategy versions,
observation profile, a dataset fingerprint, software versions, and the
evaluation report. Existing bundle paths are not overwritten. `predict()` accepts the
listed numerical observation features plus separate symbol/timestamp/timeframe
context; it does not need teaching rules or targets. Callers can pass data
quality flags to make the output unavailable. Raw SVM scores are not exposed
as probabilities.
Load bundles only from trusted sources because Python joblib files can execute
code when loaded.

## Not yet implemented or validated

This is a functioning baseline, not the full research/evaluation system in the
project brief. In particular:

- Strategies currently support grouped numeric comparisons over generated
  candle features; indicator periods, applied prices, parameters, richer
  candle/baseline relationships, transitions, and invalidation definitions
  are not modeled. The UI currently creates one comparison per strategy; the
  validated configuration schema can represent multiple grouped comparisons.
- The observation profile is separate from teaching strategies, but currently
  contains only generated candle features. Indicator history, role/relationship
  adapters, unfamiliar-family transfer, and teacher-overlap dependence tests
  are unsupported.
- The configuration editor requires complete strategies; draft saves for
  incomplete definitions are not implemented.
- Review displays numeric features and supports individual corrections or
  explicit user-selected batch corrections. Synchronized charts, borderline
  suggestions, and model/rule disagreement queues are not implemented.
- Evaluation does not yet include held-out symbols, episode grouping, overlap
  purging/embargo, calibration, uncertainty intervals, near-match testing, or
  explicit comparisons with frequency/rule/persistence baselines.
- Warning forecasts, transition event scoring, live-readiness approval,
  staleness reporting at inference, and calibrated confidence are unavailable.
- Configuration and reviewed examples persist locally. In-memory training
  state does not survive restarting the app until a bundle is exported.
- The synthetic demo exercises mechanics only. No real market performance,
  trading profitability, universal symbol generalization, or external
  trading-floor integration has been demonstrated.

Expected behavior/predictive meaning can be saved as descriptive regime
metadata, but this does not train a future-warning output. Exported bundles
are independent regime models, not a jointly trained combined classifier.
