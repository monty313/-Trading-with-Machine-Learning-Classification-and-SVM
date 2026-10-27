"""Chronological per-regime SVM training and portable inference bundles."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


KNOWN_LABELS = {"active": 1, "inactive": 0}
RESERVED_FEATURE_PARTS = ("label", "target", "teacher", "strategy_match", "selection_flag")


def validate_feature_names(feature_names: list[str]) -> None:
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Choose one or more unique observation features.")
    for name in feature_names:
        if not isinstance(name, str) or not name.strip() or any(
            part in name.lower() for part in RESERVED_FEATURE_PARTS
        ):
            raise ValueError(f"Feature {name!r} is not an allowed observation input.")


def _chronological_split(rows: list[dict[str, Any]], labels: list[int], test_fraction: float):
    ordered = sorted(
        zip(rows, labels),
        key=lambda item: (item[0].get("timestamp", ""), item[0].get("symbol", "")),
    )
    split_at = max(1, min(len(ordered) - 1, int(len(ordered) * (1.0 - test_fraction))))
    train, test = ordered[:split_at], ordered[split_at:]
    return (
        [item[0] for item in train],
        [item[1] for item in train],
        [item[0] for item in test],
        [item[1] for item in test],
    )


def _metric_report(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def train_and_evaluate(
    rows: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
    feature_names: list[str],
    kernel: str = "linear",
    test_fraction: float = 0.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if kernel not in {"linear", "rbf"}:
        raise ValueError("Kernel must be 'linear' or 'rbf'.")
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("Test fraction must be between 0.1 and 0.4.")
    validate_feature_names(feature_names)
    if any(any(name not in row for name in feature_names) for row in rows):
        raise ValueError("Some selected observation features are missing from the imported examples.")
    models: dict[str, Any] = {}
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kernel": kernel,
        "feature_names": feature_names,
        "regimes": {},
        "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "limitations": ["One chronological holdout; not evidence of symbol generalization or profitability."],
    }
    for regime in regimes:
        known = [
            (row, KNOWN_LABELS[row["labels"][regime["id"]]["value"]])
            for row in rows
            if row.get("labels", {}).get(regime["id"], {}).get("value") in KNOWN_LABELS
        ]
        if not known:
            report["regimes"][regime["id"]] = {"status": "skipped", "reason": "No reviewed active or inactive labels."}
            continue
        known_rows, labels = zip(*known)
        if len(set(labels)) != 2 or len(known_rows) < 5:
            report["regimes"][regime["id"]] = {
                "status": "skipped",
                "reason": "Need at least five reviewed examples and both active and inactive labels.",
                "class_counts": {"inactive": labels.count(0), "active": labels.count(1)},
            }
            continue
        train_rows, train_y, test_rows, test_y = _chronological_split(
            list(known_rows), list(labels), test_fraction
        )
        if len(set(train_y)) != 2:
            report["regimes"][regime["id"]] = {
                "status": "skipped",
                "reason": "The chronological training period must contain both classes.",
            }
            continue
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", SVC(kernel=kernel, class_weight="balanced")),
            ]
        )
        x_train = [[float(row[name]) for name in feature_names] for row in train_rows]
        x_test = [[float(row[name]) for name in feature_names] for row in test_rows]
        if not all(math.isfinite(value) for vector in x_train + x_test for value in vector):
            report["regimes"][regime["id"]] = {"status": "skipped", "reason": "Observation values must be finite."}
            continue
        pipeline.fit(x_train, train_y)
        predicted = pipeline.predict(x_test).tolist()
        models[regime["id"]] = pipeline
        report["regimes"][regime["id"]] = {
            "status": "trained",
            "class_counts": {"inactive": labels.count(0), "active": labels.count(1)},
            "train_count": len(train_y),
            "test_count": len(test_y),
            "test_metrics": _metric_report(test_y, predicted),
            "test_period": [test_rows[0]["timestamp"], test_rows[-1]["timestamp"]],
        }
    if not models:
        raise ValueError(
            "No model was trained. Review active and inactive examples for at least one regime, "
            "including both classes in the earlier training period."
        )
    return models, report


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def export_bundle(
    models: dict[str, Any],
    config: dict[str, Any],
    report: dict[str, Any],
    feature_names: list[str],
    destination: str | Path,
    rows: list[dict[str, Any]],
) -> Path:
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Bundle already exists: {target}. Choose a new bundle name.")
    validate_feature_names(feature_names)
    target.mkdir(parents=True)
    labels = {regime["id"]: regime["name"] for regime in config["regimes"]}
    joblib.dump(models, target / "estimators.joblib")
    manifest = {
        "schema_version": 1,
        "model_id": target.name,
        "feature_names": feature_names,
        "labels": labels,
        "timeframe_sets": {
            regime["id"]: sorted({strategy["timeframe_set"] for strategy in regime["strategies"]})
            for regime in config["regimes"]
        },
        "strategy_versions": {
            strategy["id"]: strategy["version"]
            for regime in config["regimes"]
            for strategy in regime["strategies"]
        },
        "observation_profile": config["observation_profile"],
        "dataset_fingerprint": dataset_fingerprint(rows),
        "evaluation": report,
        "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "confidence": "unavailable; raw SVM margins are not calibrated probabilities",
        "warning_forecasts": "unavailable; transition definitions and forecast targets are not configured",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def load_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model bundle schema.")
    validate_feature_names(manifest["feature_names"])
    models = joblib.load(target / "estimators.joblib")
    return models, manifest


def predict(models: dict[str, Any], manifest: dict[str, Any], observations: dict[str, Any]) -> list[dict[str, Any]]:
    feature_names = manifest["feature_names"]
    missing = [name for name in feature_names if name not in observations]
    if missing:
        raise ValueError("Missing observation features: " + ", ".join(missing))
    values = []
    for name in feature_names:
        value = observations[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Observation {name} must be a finite number.")
        values.append(float(value))
    reports = []
    for regime_id, estimator in models.items():
        active = bool(estimator.predict([values])[0])
        reports.append(
            {
                "model_id": manifest["model_id"],
                "regime_id": regime_id,
                "regime_name": manifest["labels"].get(regime_id, regime_id),
                "status": "active" if active else "inactive",
                "confidence": None,
                "confidence_note": "No calibrated probability is available.",
            }
        )
    return reports
