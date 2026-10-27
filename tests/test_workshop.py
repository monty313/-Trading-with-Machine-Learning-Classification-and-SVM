from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from regime_workshop.config import load_config, new_config, save_config, validate_config
from regime_workshop.data import align_timeframes, label_examples, read_ohlcv_csv
from regime_workshop.model import export_bundle, load_bundle, predict, train_and_evaluate


def make_bar(symbol: str, timeframe: str, timestamp: datetime, close: float) -> dict:
    spread = close * 0.001
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": timestamp,
        "open": close - spread,
        "high": close + spread,
        "low": close - 2 * spread,
        "close": close,
        "volume": 10.0,
    }


def training_fixture() -> tuple[list[dict], dict]:
    config = new_config()
    config["regimes"] = [
        {
            "id": "up",
            "name": "Up move",
            "strategies": [
                {
                    "id": "up-rule",
                    "version": "1",
                    "timeframe_set": "Set 2",
                    "groups": [
                        {
                            "logic": "all",
                            "conditions": [{"feature": "anchor__return_1", "operator": ">=", "value": 0.0}],
                        }
                    ],
                }
            ],
        }
    ]
    rows = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for index in range(40):
        value = float(index + 1)
        rows.append(
            {
                "symbol": "SYNTH",
                "timestamp": (start + timedelta(minutes=5 * index)).isoformat(),
                "anchor__return_1": value if index % 2 else -value,
                "labels": {
                    "up": {
                        "value": "active" if index % 2 else "inactive",
                        "source": "user-reviewed",
                        "strategy_ids": [],
                    }
                },
            }
        )
    return rows, config


class ConfigurationTests(unittest.TestCase):
    def test_grouped_conditions_validate_without_eval(self):
        config = new_config()
        config["regimes"] = [
            {
                "id": "r1",
                "name": "Regime",
                "strategies": [
                    {
                        "id": "s1",
                        "version": "1",
                        "timeframe_set": "Set 1",
                        "groups": [
                            {
                                "logic": "any",
                                "conditions": [{"feature": "anchor__return_1", "operator": ">", "value": 0.0}],
                            }
                        ],
                    }
                ],
            }
        ]
        validate_config(config)
        config["regimes"][0]["strategies"][0]["groups"][0]["conditions"][0]["operator"] = "exec"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_config_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(new_config(), path)
            self.assertEqual(load_config(path), new_config())


class DataTests(unittest.TestCase):
    def test_csv_requires_timezone_and_detects_duplicate_bars(self):
        content = (
            "symbol,timeframe,timestamp,open,high,low,close,volume\n"
            "X,1m,2024-01-01T00:00:00,1,2,0.5,1.5,10\n"
        )
        with self.assertRaisesRegex(ValueError, "timezone"):
            read_ohlcv_csv(content.splitlines())
        duplicate = content.replace("2024-01-01T00:00:00", "2024-01-01T00:00:00Z").replace(
            "X,1m,2024-01-01T00:00:00Z,1,2,0.5,1.5,10\n",
            "X,1m,2024-01-01T00:00:00Z,1,2,0.5,1.5,10\nX,1m,2024-01-01T00:00:00Z,1,2,0.5,1.5,10\n",
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_ohlcv_csv(duplicate.splitlines())

    def test_support_features_use_completed_bars_only(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = []
        for index in range(30):
            bars.append(make_bar("X", "5m", start + timedelta(minutes=5 * (index + 1)), 100 + index))
        for index in range(5):
            bars.append(make_bar("X", "30m", start + timedelta(minutes=30 * (index + 1)), 100 + index * 3))
        for index, close in enumerate((100.0, 110.0, 150.0)):
            bars.append(make_bar("X", "1h", start + timedelta(minutes=60 * (index + 1)), close))
        aligned = align_timeframes(bars, "Set 2")
        self.assertTrue(aligned)
        self.assertAlmostEqual(aligned[0]["support_2__return_1"], 0.1)
        for row in aligned:
            self.assertIn("anchor__return_1", row)
            self.assertIn("support_1__return_1", row)
            self.assertIn("support_2__return_1", row)

    def test_rule_nonmatch_is_unknown_not_inactive(self):
        rows = [
            {"timeframe_set": "Set 1", "anchor__return_1": -0.1},
            {"timeframe_set": "Set 1", "anchor__return_1": 0.1},
        ]
        config = new_config()
        config["regimes"] = [
            {
                "id": "up",
                "name": "Up",
                "strategies": [
                    {
                        "id": "rule",
                        "version": "1",
                        "timeframe_set": "Set 1",
                        "groups": [
                            {
                                "logic": "all",
                                "conditions": [{"feature": "anchor__return_1", "operator": ">", "value": 0.0}],
                            }
                        ],
                    }
                ],
            }
        ]
        result = label_examples(rows, config["regimes"])
        self.assertEqual(result[0]["labels"]["up"]["value"], "unknown")
        self.assertEqual(result[1]["labels"]["up"]["value"], "active")
        wrong_set = label_examples(
            [{"timeframe_set": "Set 2", "anchor__return_1": 0.1}], config["regimes"]
        )
        self.assertEqual(wrong_set[0]["labels"]["up"]["value"], "unknown")


class ModelTests(unittest.TestCase):
    def test_unknowns_excluded_and_scaler_fits_training_period_only(self):
        rows, config = training_fixture()
        rows[0]["labels"]["up"]["value"] = "unknown"
        models, report = train_and_evaluate(rows, config["regimes"], ["anchor__return_1"])
        self.assertEqual(report["regimes"]["up"]["status"], "trained")
        trained_x = [row["anchor__return_1"] for row in rows[1:32]]
        expected_mean = sum(trained_x) / len(trained_x)
        self.assertAlmostEqual(models["up"].named_steps["scale"].mean_[0], expected_mean)
        self.assertEqual(report["regimes"]["up"]["class_counts"]["inactive"], 19)

    def test_bundle_predict_matches_and_requires_observations_only(self):
        rows, config = training_fixture()
        models, report = train_and_evaluate(rows, config["regimes"], ["anchor__return_1"])
        with tempfile.TemporaryDirectory() as directory:
            bundle = export_bundle(
                models, config, report, ["anchor__return_1"], Path(directory) / "model-v1", rows
            )
            loaded, manifest = load_bundle(bundle)
            input_row = {"anchor__return_1": 3.0}
            context = {
                "symbol": "SYNTH",
                "timestamp": "2024-01-03T00:00:00Z",
                "timeframe_set": "Set 2",
            }
            actual = predict(loaded, manifest, input_row, context)
            expected = bool(models["up"].predict([[3.0]])[0])
            self.assertEqual(actual[0]["status"], "active" if expected else "inactive")
            self.assertIsNone(actual[0]["confidence"])
            self.assertEqual(actual[0]["symbol"], "SYNTH")
            self.assertEqual(actual[0]["decision_timestamp"], context["timestamp"])
            self.assertEqual(actual[0]["supporting_observations"], input_row)
            with self.assertRaisesRegex(ValueError, "Missing observation features"):
                predict(loaded, manifest, {}, context)
            self.assertEqual(
                predict(loaded, manifest, input_row, {**context, "data_quality_flags": ["stale"]})[0]["status"],
                "unavailable",
            )
            with self.assertRaises(FileExistsError):
                export_bundle(models, config, report, ["anchor__return_1"], bundle, rows)


if __name__ == "__main__":
    unittest.main()
