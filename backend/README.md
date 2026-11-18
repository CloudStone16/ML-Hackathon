# PathSnap backend (FastAPI, laptop 1)

Serves the full prediction pipeline for the evaluation: raw trip files -> candidate generation + features ->
trained model -> HMM/Viterbi -> reconstructed route, plus server-side accuracy against the hidden reference route.

```
UI (laptop 2)  --POST /predict {trip_id}-->  API (laptop 1)  --> pipeline --> predicted route + metrics
```

Default model: LightGBM (`PATHSNAP_MODEL=lightgbm`). The Deep MLP is also supported (`PATHSNAP_MODEL=mlp`).

## Rules this backend enforces
- Prediction reads only `.track`, `.nodes`, `.arcs`. The `.route` file is opened afterwards, inside `score()`, only to compute metrics.
- Responses never contain the reference route or any labels, only the predicted route, the GPS points and metrics.
- Nothing is hardcoded or cached: every request runs the whole pipeline again, and every request is logged to the console.

## Setup (once, on laptop 1)
```bash
pip install -r backend/requirements.txt
```
1. **Model artifacts**
   - LightGBM (default): `results/lightgbm_model.joblib` and `results/config.json` are already in the repo. The decoder settings are in `backend/decoder_config.json`.
   - MLP: copy `pathsnap_mlp_weights.pt`, `feature_scaler.joblib`, `model_config.json` (written by `models/train_mlp.py` into `saved_models/`, which is git-ignored) into `saved_models/`. Also `pip install torch`.
2. **Evaluation trips**: put the evaluator folder at `server_data/evaluator_test/` (one folder per trip containing `<trip>.track/.nodes/.arcs/.route`), or set `PATHSNAP_DATA_DIR`. `server_data/` is git-ignored: never commit it.

## Run
From the repo root:
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
`--host 0.0.0.0` makes it reachable from the second laptop. Keep this terminal visible: the evaluator must see the request logs.

| Environment variable | Meaning | Default |
|---|---|---|
| `PATHSNAP_MODEL` | `lightgbm` or `mlp` | `lightgbm` |
| `PATHSNAP_MODEL_DIR` | artifacts folder | `results/` (lightgbm), `saved_models/` (mlp) |
| `PATHSNAP_DATA_DIR` | evaluation trips folder | `server_data/evaluator_test` |

Log lines per request:
```
12:01:07 | INFO | PREDICT request  trip=00000015 from 100.71.36.2
12:01:08 | INFO | PREDICT done     trip=00000015 points=2503 | top1 HMM=0.9xxx clf-only=0.9xxx | continuity=... | 0.9s
12:01:08 | INFO | 100.71.36.2 POST /predict -> 200 (940 ms)
```

## Connecting the UI laptop (Tailscale)
- On laptop 1 find its Tailscale address: `tailscale ip -4` (Anirudh's MacBook: `100.124.123.106`).
- The UI uses `http://<that-ip>:8000`. Check from laptop 2: `curl http://100.124.123.106:8000/health`
- macOS: when Python first listens, click Allow on the "accept incoming network connections" prompt.

## Endpoints
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, model info, number of trips |
| GET | `/trips` | `[{trip_id, n_gps_points}]` for the UI list |
| POST | `/predict` `{"trip_id": "00000015"}` | predicted route + metrics |
| GET | `/predict/{trip_id}` | same, convenient for a browser/curl |

`/predict` response: `gps_points` `[[lat, lon]]`, `predicted_route` (segments `[[lat, lon], [lat, lon]]`), `predicted_arc_ids` (one per GPS point),
`metrics.hmm_viterbi` and `metrics.classifier_only` (`top1_accuracy`, `route_continuity`, `sequence_jaccard`, `sequence_f1`), `timings_s`.

## HMM decoder (`decoder.py`)
Works with any model's per-candidate probabilities. Emission = alpha * log p, transitions from road-graph hops (same or next arc 0,
skipped arcs a small cost, U-turn onto the reverse twin and unreachable jumps large costs), restart after long time gaps, Viterbi decoding over the
top 10 candidates per point. Its five parameters were tuned on LightGBM's train out-of-fold probabilities only (`python backend/tune_decoder.py --data-root <dataset>`),
results in `backend/decoder_results.json` and `backend/decoder_trials.csv`. Re-run it if you switch models, since probabilities differ between models.

## Test
```bash
python backend/smoke_test.py http://127.0.0.1:8000
```
Runs every trip, checks that no ground-truth fields leak, and prints per-trip accuracy.
