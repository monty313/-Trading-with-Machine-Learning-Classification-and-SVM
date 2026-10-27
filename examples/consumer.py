"""Minimal trusted-bundle consumer; provide one row of observation features."""

from __future__ import annotations

import argparse
import json

from regime_workshop.model import load_bundle, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Path to an exported bundle from this workshop")
    parser.add_argument("input", help="JSON object with observation features and prediction context")
    arguments = parser.parse_args()
    models, manifest = load_bundle(arguments.bundle)
    payload = json.loads(arguments.input)
    reports = predict(
        models,
        manifest,
        payload["observations"],
        payload["context"],
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
