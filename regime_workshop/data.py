"""CSV OHLCV validation and completed-bar multi-timeframe alignment."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import TIMEFRAME_SETS


TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}
REQUIRED_COLUMNS = {"symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"}
FEATURE_NAMES = ("return_1", "body_fraction", "upper_wick_fraction", "lower_wick_fraction", "range_fraction")


def read_ohlcv_csv(source: str | Path | Iterable[str], timestamp_convention: str = "close") -> list[dict[str, Any]]:
    if timestamp_convention not in {"open", "close"}:
        raise ValueError("Timestamp convention must be 'open' or 'close'.")
    if isinstance(source, (str, Path)):
        handle = Path(source).open(newline="", encoding="utf-8-sig")
        close_after = True
    else:
        handle = source
        close_after = False
    try:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("The CSV is empty.")
        names = {name.strip().lower() for name in reader.fieldnames if name}
        missing = REQUIRED_COLUMNS - names
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        normalized = {name.strip().lower(): name for name in reader.fieldnames if name}
        bars: list[dict[str, Any]] = []
        seen: set[tuple[str, str, datetime]] = set()
        for line_number, raw in enumerate(reader, start=2):
            symbol = (raw[normalized["symbol"]] or "").strip()
            timeframe = (raw[normalized["timeframe"]] or "").strip()
            if not symbol:
                raise ValueError(f"Row {line_number}: symbol is missing.")
            if timeframe not in TIMEFRAME_MINUTES:
                raise ValueError(f"Row {line_number}: unsupported timeframe {timeframe!r}.")
            try:
                timestamp = datetime.fromisoformat(raw[normalized["timestamp"]].strip().replace("Z", "+00:00"))
            except (AttributeError, ValueError):
                raise ValueError(f"Row {line_number}: timestamp must be ISO 8601 with a timezone.") from None
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"Row {line_number}: timestamp must include a timezone.")
            timestamp = timestamp.astimezone(timezone.utc)
            if timestamp_convention == "open":
                timestamp += timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
            try:
                values = {
                    field: float(raw[normalized[field]])
                    for field in ("open", "high", "low", "close", "volume")
                }
            except (TypeError, ValueError):
                raise ValueError(f"Row {line_number}: OHLCV values must be numbers.") from None
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Row {line_number}: OHLCV values must be finite.")
            if values["low"] > min(values["open"], values["close"]) or values["high"] < max(
                values["open"], values["close"]
            ):
                raise ValueError(f"Row {line_number}: high/low do not contain the open and close.")
            if values["volume"] < 0:
                raise ValueError(f"Row {line_number}: volume cannot be negative.")
            key = (symbol, timeframe, timestamp)
            if key in seen:
                raise ValueError(f"Row {line_number}: duplicate {symbol} {timeframe} bar at {timestamp.isoformat()}.")
            seen.add(key)
            bars.append({"symbol": symbol, "timeframe": timeframe, "timestamp": timestamp, **values})
        if not bars:
            raise ValueError("The CSV has a header but no data rows.")
        bars.sort(key=lambda bar: (bar["symbol"], bar["timeframe"], bar["timestamp"]))
        return bars
    finally:
        if close_after:
            handle.close()


def feature_names_for_set(timeframe_set: str) -> list[str]:
    if timeframe_set not in TIMEFRAME_SETS:
        raise ValueError(f"Unknown timeframe set: {timeframe_set}.")
    return [
        f"{role}__{feature}"
        for role in ("anchor", "support_1", "support_2")
        for feature in FEATURE_NAMES
    ]


def _candle_features(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, float] | None:
    close = current["close"]
    if previous is None or previous["close"] == 0 or current["high"] == current["low"] or close == 0:
        return None
    span = current["high"] - current["low"]
    return {
        "return_1": close / previous["close"] - 1.0,
        "body_fraction": (close - current["open"]) / span,
        "upper_wick_fraction": (current["high"] - max(current["open"], close)) / span,
        "lower_wick_fraction": (min(current["open"], close) - current["low"]) / span,
        "range_fraction": span / abs(close),
    }


def align_timeframes(bars: list[dict[str, Any]], timeframe_set: str) -> list[dict[str, Any]]:
    """Create anchor-time rows using only completed, sufficiently fresh support bars."""
    if timeframe_set not in TIMEFRAME_SETS:
        raise ValueError(f"Unknown timeframe set: {timeframe_set}.")
    anchor_tf, support_1_tf, support_2_tf = TIMEFRAME_SETS[timeframe_set]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for bar in bars:
        grouped.setdefault((bar["symbol"], bar["timeframe"]), []).append(bar)
    for values in grouped.values():
        values.sort(key=lambda bar: bar["timestamp"])
    aligned: list[dict[str, Any]] = []
    for (symbol, timeframe), anchors in grouped.items():
        if timeframe != anchor_tf:
            continue
        supports = []
        missing = False
        for support_tf in (support_1_tf, support_2_tf):
            series = grouped.get((symbol, support_tf), [])
            if len(series) < 2:
                missing = True
            supports.append(series)
        if missing:
            continue
        for anchor_index, anchor in enumerate(anchors):
            anchor_features = _candle_features(
                anchor, anchors[anchor_index - 1] if anchor_index else None
            )
            if anchor_features is None:
                continue
            result: dict[str, Any] = {
                "symbol": symbol,
                "timestamp": anchor["timestamp"].isoformat().replace("+00:00", "Z"),
                "timeframe_set": timeframe_set,
                "quality_flags": [],
            }
            result.update({f"anchor__{name}": value for name, value in anchor_features.items()})
            supports_ok = True
            for role, support_tf, series in zip(("support_1", "support_2"), (support_1_tf, support_2_tf), supports):
                available = [bar for bar in series if bar["timestamp"] <= anchor["timestamp"]]
                if len(available) < 2:
                    supports_ok = False
                    break
                current = available[-1]
                age = anchor["timestamp"] - current["timestamp"]
                max_age = timedelta(minutes=TIMEFRAME_MINUTES[support_tf] + TIMEFRAME_MINUTES[anchor_tf])
                if age < timedelta(0) or age > max_age:
                    supports_ok = False
                    break
                features = _candle_features(current, available[-2])
                if features is None:
                    supports_ok = False
                    break
                result.update({f"{role}__{name}": value for name, value in features.items()})
            if supports_ok:
                aligned.append(result)
    aligned.sort(key=lambda row: (row["timestamp"], row["symbol"]))
    return aligned


def evaluate_condition(condition: dict[str, Any], features: dict[str, Any]) -> bool:
    if condition["feature"] not in features:
        return False
    observed = features[condition["feature"]]
    if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed):
        return False
    expected = condition["value"]
    operator = condition["operator"]
    return {
        ">": observed > expected,
        ">=": observed >= expected,
        "<": observed < expected,
        "<=": observed <= expected,
        "==": observed == expected,
        "!=": observed != expected,
    }[operator]


def strategy_matches(strategy: dict[str, Any], features: dict[str, Any]) -> bool:
    return all(
        (all if group["logic"] == "all" else any)(
            evaluate_condition(condition, features) for condition in group["conditions"]
        )
        for group in strategy["groups"]
    )


def label_examples(
    examples: list[dict[str, Any]], regimes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach independent per-regime targets; rule non-matches remain unknown."""
    labeled = []
    for example in examples:
        row = dict(example)
        row["labels"] = {}
        for regime in regimes:
            matched = [
                strategy["id"]
                for strategy in regime["strategies"]
                if strategy_matches(strategy, example)
            ]
            row["labels"][regime["id"]] = {
                "value": "active" if matched else "unknown",
                "source": "strategy-generated" if matched else "unknown",
                "strategy_ids": matched,
            }
        labeled.append(row)
    return labeled
