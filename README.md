# Regime Model Workshop

This repository is a local model-building workshop inspired by the QuantInsti
Quantra course, [Trading with Machine Learning: Classification and SVM](https://quantra.quantinsti.com/course/trading-machine-learning-classification-svm).
The checkout contains only this short course-description README; no notebooks,
scripts, doctrine files, datasets, or licenses from the course were supplied.
Accordingly, this project does not claim to reuse or redistribute course
materials.

The guided workflow is:

**Add strategy → Add data → Review examples → Train → Test → Export**

The app prepares and tests regime-recognition models. It does not connect to a
broker, place orders, size positions, or make trading decisions.

## Run locally

Requirements: Python 3.10 or newer.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
./run_app.sh
```

Open the local URL printed by Streamlit. Configuration and reviewed examples
are saved under the ignored `workshop_data/` directory. Add at least one
strategy before importing data.

## Try the synthetic demo

Create a deterministic, artificial OHLCV file:

```sh
python generate_demo.py /tmp/regime-demo.csv
```

In the app, add a regime and strategy using **Set 2**, with a simple condition
such as `anchor__return_1 >= 0.0001`. In **Add data**, upload
`/tmp/regime-demo.csv` and confirm the preview. In **Review examples**, review
both active suggestions and some non-matches as inactive; non-matches begin as
unknown and are never treated as negatives without review. Train and inspect
the holdout report, then export a uniquely named bundle.

The demo is synthetic software test data, not market evidence or a claim of
generalization.

Run the focused tests with:

```sh
python -m unittest discover -s tests -v
```

See [the architecture and limitations](docs/architecture.md) for the supported
CSV schema, data flow, and what remains unvalidated. Python bundle consumers can
start from [`examples/consumer.py`](examples/consumer.py), passing a JSON object
with `observations` and `context` keys from the repository root, for example
with `python -m examples.consumer /path/to/bundle '<JSON>'`. Include all
features listed in the bundle manifest.
