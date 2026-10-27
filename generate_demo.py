"""Create a small deterministic synthetic OHLCV file for the workshop demo."""

from __future__ import annotations

import csv
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate(path: str | Path = "demo_ohlcv.csv") -> Path:
    random.seed(23)
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    close = 100.0
    anchor_bars = []
    for index in range(24 * 12):
        phase = (index // 48) % 4
        drift = (0.0007 if phase in (0, 1) else -0.0006) + random.uniform(-0.001, 0.001)
        opening = close
        close = opening * (1 + drift)
        spread = abs(close - opening) + opening * 0.0005
        anchor_bars.append(
            {
                "symbol": "SYNTH",
                "timeframe": "5m",
                "timestamp": (start + timedelta(minutes=5 * (index + 1))).isoformat().replace("+00:00", "Z"),
                "open": opening,
                "high": max(opening, close) + spread,
                "low": min(opening, close) - spread,
                "close": close,
                "volume": 100 + (index % 17) * 3,
            }
        )
    bars = list(anchor_bars)
    for timeframe, step in (("30m", 6), ("1h", 12)):
        for index in range(0, len(anchor_bars), step):
            batch = anchor_bars[index : index + step]
            if len(batch) != step:
                continue
            bars.append(
                {
                    "symbol": "SYNTH",
                    "timeframe": timeframe,
                    "timestamp": batch[-1]["timestamp"],
                    "open": batch[0]["open"],
                    "high": max(bar["high"] for bar in batch),
                    "low": min(bar["low"] for bar in batch),
                    "close": batch[-1]["close"],
                    "volume": sum(bar["volume"] for bar in batch),
                }
            )
    bars.sort(key=lambda bar: (bar["timestamp"], bar["timeframe"]))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(bars)
    return destination


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_ohlcv.csv")
    print(f"Wrote synthetic candles to {generate(output)}")
