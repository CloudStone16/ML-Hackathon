"""Smoke test for a running PathSnap API.   python backend/smoke_test.py [http://host:8000]

Checks every trip end to end and verifies that no ground-truth fields are exposed in responses.
"""
import json
import sys
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
FORBIDDEN = {"route", "reference_route", "gt_arcs", "ground_truth", "true_arcs", "labels"}


def call(path, payload=None):
    req = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


h = call("/health")
print("health:", h["status"], "|", h["model"]["name"], "| trips:", h["n_trips"])
trips = [t["trip_id"] for t in call("/trips")["trips"]]
print("trips :", trips)
bad = 0
for tid in trips:
    r = call("/predict", {"trip_id": tid})
    leaked = FORBIDDEN & set(r.keys())
    m = r["metrics"]["hmm_viterbi"] if r["metrics"] else None
    print("%s points=%5d segments=%4d top1=%s continuity=%s time=%.2fs %s" % (
        tid, r["n_gps_points"], r["n_route_segments"],
        "%.4f" % m["top1_accuracy"] if m else "n/a", "%.4f" % m["route_continuity"] if m else "n/a",
        r["timings_s"]["total"], "LEAK:" + str(leaked) if leaked else ""))
    bad += bool(leaked) or len(r["predicted_arc_ids"]) != r["n_gps_points"]
print("ALL OK" if not bad else "PROBLEMS: %d" % bad)
sys.exit(1 if bad else 0)
