# PathSnap: GPS map-matching classifier

Predicts which road segment a vehicle is on for each noisy GPS point (LightGBM candidate-arc classifier).

**Results are in the [`results/`](results/) directory** (metrics, model comparison, CV folds, feature importance, preprocessing reports, trained model, config).

## Run

```bash
uv sync
uv run python scripts/build_dataset.py     # features cached in artifacts/
uv run python scripts/train_lightgbm.py    # writes everything to results/
```
