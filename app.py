"""Streamlit entry point for the guided local regime-model workshop."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from regime_workshop.config import TIMEFRAME_SETS, load_config, save_config
from regime_workshop.data import align_timeframes, feature_names_for_set, label_examples, read_ohlcv_csv
from regime_workshop.model import export_bundle, train_and_evaluate


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workshop_data"
CONFIG_PATH = WORKSPACE / "workshop.json"
EXAMPLES_PATH = WORKSPACE / "examples.json"

st.set_page_config(page_title="Regime Model Workshop", page_icon="📊", layout="wide")
st.title("Regime Model Workshop")
st.caption("Build and test recognition models locally. This workshop does not place trades.")
WORKSPACE.mkdir(exist_ok=True)

try:
    config = load_config(CONFIG_PATH)
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Could not load the saved setup: {error}. Fix or remove {CONFIG_PATH} and try again.")
    st.stop()

if "examples" not in st.session_state:
    try:
        st.session_state.examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.session_state.examples = []

steps = ["Add strategy", "Add data", "Review examples", "Train", "Test", "Export"]
step = st.sidebar.radio("Workshop steps", steps)
if not config["regimes"]:
    st.info("Start with **Add strategy**. Your setup and reviewed examples are saved locally.")

if step == "Add strategy":
    st.header("Add a teaching strategy")
    st.write("A teaching strategy supplies example labels. It is not required by the exported model.")
    with st.form("strategy_form"):
        regime_id = st.text_input("Regime ID", placeholder="trend_up")
        regime_name = st.text_input("Regime name", placeholder="Upward trend")
        description = st.text_area("What does this regime mean?")
        expected = st.text_area("Expected behavior or predictive meaning (optional)")
        strategy_id = st.text_input("Strategy ID", placeholder="trend_up_v1")
        strategy_version = st.text_input("Strategy version", value="1")
        timeframe_set = st.selectbox("Official timeframe set", list(TIMEFRAME_SETS))
        options = feature_names_for_set(timeframe_set)
        feature = st.selectbox("Teaching condition feature", options)
        operator = st.selectbox("Relationship", [">", ">=", "<", "<=", "==", "!="])
        threshold = st.number_input("Threshold", value=0.0, format="%.8f")
        group_logic = st.selectbox("Combine this group's conditions with", ["all", "any"])
        submitted = st.form_submit_button("Save strategy")
    if submitted:
        try:
            if not regime_id.strip() or not regime_name.strip() or not strategy_id.strip():
                raise ValueError("Enter a regime ID, regime name, and strategy ID.")
            existing = next((item for item in config["regimes"] if item["id"] == regime_id.strip()), None)
            regime = existing or {
                "id": regime_id.strip(),
                "name": regime_name.strip(),
                "description": description,
                "expected_behavior": expected,
                "strategies": [],
            }
            if existing and existing["name"] != regime_name.strip():
                raise ValueError("This regime ID already exists with a different name.")
            regime["description"] = description or regime.get("description", "")
            regime["expected_behavior"] = expected or regime.get("expected_behavior", "")
            regime["strategies"].append(
                {
                    "id": strategy_id.strip(),
                    "version": strategy_version.strip(),
                    "timeframe_set": timeframe_set,
                    "groups": [
                        {
                            "logic": group_logic,
                            "conditions": [{"feature": feature, "operator": operator, "value": threshold}],
                        }
                    ],
                    "notes": "",
                    "source_reference": "",
                }
            )
            if not existing:
                config["regimes"].append(regime)
            save_config(config, CONFIG_PATH)
            st.success("Strategy saved. You can add another strategy to the same regime.")
        except (OSError, ValueError) as error:
            st.error(f"Strategy not saved: {error}")
    if config["regimes"]:
        st.subheader("Saved regimes")
        for regime in config["regimes"]:
            st.write(f"**{regime['name']}** (`{regime['id']}`) — {len(regime['strategies'])} teaching strategies")

elif step == "Add data":
    st.header("Add historical candles")
    st.write("Upload a CSV with symbol, timeframe, timestamp, open, high, low, close, and volume.")
    st.caption("Use timezone-aware ISO 8601 timestamps. Completed bars are aligned to the anchor decision time.")
    upload = st.file_uploader("Historical OHLCV CSV", type=["csv"])
    convention = st.radio("Timestamp means", ["close", "open"], format_func=lambda value: f"Bar {value} time")
    if upload is not None:
        try:
            import io

            bars = read_ohlcv_csv(io.StringIO(upload.getvalue().decode("utf-8-sig")), convention)
            sets = sorted(
                {
                    strategy["timeframe_set"]
                    for regime in config["regimes"]
                    for strategy in regime["strategies"]
                }
            )
            examples = []
            counts = {}
            for timeframe_set in sets:
                aligned = align_timeframes(bars, timeframe_set)
                counts[timeframe_set] = len(aligned)
                examples.extend(label_examples(aligned, config["regimes"]))
            if not examples:
                raise ValueError(
                    "No aligned examples were created. Check that the CSV includes enough completed bars "
                    "for every timeframe in your strategy's official set."
                )
            st.write(f"Validated {len(bars)} candles. Aligned examples: {counts}")
            st.dataframe(
                [
                    {key: value for key, value in row.items() if key not in {"labels", "quality_flags"}}
                    for row in examples[:10]
                ],
                use_container_width=True,
            )
            if st.button("Save imported data and generated examples"):
                st.session_state.examples = examples
                EXAMPLES_PATH.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
                st.success(f"Saved {len(examples)} examples. Non-matches are unknown, not inactive.")
        except (UnicodeDecodeError, OSError, ValueError) as error:
            st.error(f"Data was not imported: {error}")

elif step == "Review examples":
    st.header("Review examples and labels")
    rows = st.session_state.examples
    if not rows or not config["regimes"]:
        st.info("Add a strategy and import historical candles first.")
    else:
        st.caption("A rule match suggests active. A non-match stays unknown until you review it.")
        labels = [f"{i}: {row['symbol']} at {row['timestamp']} ({row['timeframe_set']})" for i, row in enumerate(rows)]
        batch_indices = st.multiselect(
            "Optional: select a batch you have independently reviewed",
            range(len(rows)),
            format_func=lambda i: labels[i],
        )
        batch_regime = st.selectbox(
            "Batch regime",
            config["regimes"],
            format_func=lambda item: item["name"],
            key="batch_regime",
        )
        batch_label = st.selectbox("Reviewed label for this batch", ["active", "inactive", "unknown"])
        if st.button("Save batch review", disabled=not batch_indices):
            for batch_index in batch_indices:
                target_label = rows[batch_index]["labels"][batch_regime["id"]]
                rows[batch_index]["labels"][batch_regime["id"]] = {
                    "value": batch_label,
                    "source": "user-reviewed",
                    "strategy_ids": target_label["strategy_ids"],
                }
            EXAMPLES_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            st.success(f"Saved a review for {len(batch_indices)} selected examples.")
        index = st.selectbox("Choose an example", range(len(rows)), format_func=lambda i: labels[i])
        row = rows[index]
        st.write(
            f"**{row['symbol']}** · {row['timestamp']} · {row['timeframe_set']} · "
            f"Data flags: {', '.join(row.get('quality_flags', [])) or 'none'}"
        )
        st.json({key: value for key, value in row.items() if key not in {"labels"}})
        regime = st.selectbox("Regime to review", config["regimes"], format_func=lambda item: item["name"])
        current = row["labels"][regime["id"]]
        st.write(f"Suggested label source: **{current['source']}**; matched strategy IDs: {current['strategy_ids']}")
        new_label = st.selectbox(
            "Your reviewed label",
            ["unknown", "active", "inactive"],
            index=["unknown", "active", "inactive"].index(current["value"]),
        )
        if st.button("Save label review"):
            row["labels"][regime["id"]] = {
                "value": new_label,
                "source": "user-reviewed",
                "strategy_ids": current["strategy_ids"],
            }
            EXAMPLES_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            st.success("Review saved. Each regime keeps its own active/inactive/unknown label.")

elif step == "Train":
    st.header("Train recognition models")
    rows = st.session_state.examples
    sets = sorted({row["timeframe_set"] for row in rows})
    available = sorted({name for set_name in sets for name in feature_names_for_set(set_name)})
    if not rows or not config["regimes"]:
        st.info("Add strategies, import data, and review examples before training.")
    else:
        st.write("Unknown labels are excluded from each regime's supervised training and evaluation.")
        selected = st.multiselect("Observation profile features", available, default=available)
        kernel = st.selectbox("SVM", ["linear", "rbf"])
        test_fraction = st.slider("Final chronological holdout", 0.1, 0.4, 0.2, 0.05)
        if st.button("Train and test"):
            try:
                config["observation_profile"] = {"id": "default", "features": selected}
                save_config(config, CONFIG_PATH)
                models, report = train_and_evaluate(
                    rows, config["regimes"], selected, kernel=kernel, test_fraction=test_fraction
                )
                st.session_state.trained_models = models
                st.session_state.evaluation = report
                st.success("Training completed for the regimes marked trained below.")
                st.json(report)
            except (OSError, ValueError) as error:
                st.error(f"Training did not complete: {error}")

elif step == "Test":
    st.header("Test results")
    report = st.session_state.get("evaluation")
    if not report:
        st.info("No training result is available. Train a model first; skipped tasks are not treated as successes.")
    else:
        st.write("These are chronological holdout scores, not proof of broader generalization or trading performance.")
        st.json(report)
        st.warning(
            "This first version reports one chronological holdout only. It does not yet test held-out symbols, "
            "episode leakage, purging, calibration, or generalization to reviewed near-matches."
        )

elif step == "Export":
    st.header("Export a model bundle")
    models = st.session_state.get("trained_models")
    report = st.session_state.get("evaluation")
    if not models or not report:
        st.info("Train at least one regime first.")
    else:
        default_name = "regime-model-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        bundle_name = st.text_input("New bundle name", value=default_name)
        if st.button("Export bundle"):
            try:
                if not bundle_name.strip() or Path(bundle_name).name != bundle_name:
                    raise ValueError("Use a simple new bundle name without folders.")
                destination = WORKSPACE / "exports" / bundle_name
                target = export_bundle(
                    models,
                    config,
                    report,
                    config["observation_profile"]["features"],
                    destination,
                    st.session_state.examples,
                )
                st.success(f"Exported bundle to {target}. Existing bundles are never overwritten.")
                st.write("Predict uses saved observation features only; teaching rules are not part of inference.")
            except (OSError, ValueError, FileExistsError) as error:
                st.error(f"Export did not complete: {error}")
        st.warning("Export is not approval for live use. Confidence scores and future-warning forecasts are unavailable.")
