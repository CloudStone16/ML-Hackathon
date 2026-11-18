# PathSnap evaluator UI (laptop 2)

A single static page (`index.html`): no install and no build step. It lists the evaluation trips from the API, sends the
selected trip ID to `POST /predict`, and draws the observed GPS points and the predicted route on a map, next to the
accuracy the API computed.

The UI never receives ground truth. The API scores the prediction against its own reference data and returns only metrics.

## Run (on laptop 2)
```bash
cd ui
python -m http.server 8080
```
Open `http://localhost:8080` in a browser. Type the API address of laptop 1 in the box and press Connect:

| Setup | API address |
|---|---|
| Anirudh's MacBook over Tailscale | `http://100.124.123.106:8000` (default in the box) |
| Same machine, for testing | `http://127.0.0.1:8000` |

You can also pass it in the link: `http://localhost:8080/?api=http://100.124.123.106:8000`. The last address used is remembered.

## Use
1. Press **Connect**. A green dot and the model name confirm the API is reachable.
2. Click any trip. The page shows the predicted route (red) over the observed GPS points (blue), top-1 accuracy, route continuity, and a comparison of classifier only against classifier plus HMM. The model name shown comes from the API, so the page works with whichever model the server runs.
3. **Run all trips** goes through every trip and shows each one's accuracy on its card.
4. The **Request log** at the bottom-left mirrors what the API logs on laptop 1. Show the API terminal to the evaluator as well.

## Notes
- The map uses OpenStreetMap tiles and the Leaflet library from cdnjs, so laptop 2 needs internet. Without it the routes and points still draw, just without background tiles.
- Start the API first (see `backend/README.md`), otherwise the UI shows "Cannot reach ...".
- Screenshots for the report: the UI result page, and the API terminal showing the `PREDICT request` / `PREDICT done` log lines.
