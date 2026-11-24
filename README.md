# Algae Culture Health Monitor

A system that watches an algae culture (photobioreactor or open raceway pond) and warns you before it crashes. It ingests sensor readings and microscope images, forecasts optical density (OD750) 24 hours ahead, flags contamination and process faults, suggests the next settings to try, and explains alerts in plain English.

It runs **entirely on simulated data first**, so no photobioreactor or lab images are needed. Real sensors and images plug in later through the same API.

> **Provenance:** this repository is a clean reimplementation of a project I originally built (the AzCATI-related algae monitor). The original code was not available, so I rebuilt it from the original design. It has **not** been run against real AzCATI sensor data or real microscope images, and every number below comes from the built-in simulator.

## Components

| Part | What it does | Where |
|---|---|---|
| Simulator | Logistic growth limited by light, temperature and pH, with night respiration and three injected, labeled faults (contamination, CO₂ starvation, heat spike). Constants are plausible, not calibrated to a real strain. | `src/algae/sim/simulator.py` |
| Contamination detector | YOLOv8 detector for algae cells vs. rotifers, ciliates, amoebae, bacterial clumps and debris. Trained on synthetic microscope images; real labeled images drop in using the same YOLO layout. Cell counts convert to cells/mL using a hemocytometer field volume. | `src/algae/sim/synth_microscope.py`, `src/algae/vision/` |
| Forecaster | LightGBM quantile models (p10/p50/p90) predicting OD750 24 h ahead, compared against a persistence baseline. Train/val/test are split **by culture**, never by random rows, so the future cannot leak. | `src/algae/forecast/` |
| Anomaly detection | Two layers: residual rules against healthy baselines learned from fault-free runs, plus an Isolation Forest. An alert needs k consecutive anomalous hours. | `src/algae/anomaly/detector.py` |
| Settings optimizer | Optuna (TPE) suggests the next temperature / pH / light settings. Supports the real-world `ask`/`tell` loop. | `src/algae/optimize/bo.py` |
| Explainer | Claude turns an alert plus the last 48 h of numbers into a plain-English summary, hypotheses and checks, using only the supplied data. | `src/algae/explain/explain.py` |
| API + dashboard | FastAPI backend (SQLite) and a Streamlit dashboard with forecast vs. actual, alerts and an "Explain" button. | `src/algae/api/`, `src/algae/dashboard/` |

## Results on simulated data

Measured by running the code in this repo (20 simulated 30-day cultures, held-out cultures for testing):

- **Forecast (OD750, +24 h):** MAE **0.028** for LightGBM vs. **0.100** for the persistence baseline. The 80% interval (p10–p90) covers **68%** of held-out points, so it is somewhat **over-confident** (nominal 80%). The dashboard should not present these intervals as calibrated.
- **Fault detection:** the residual-rule layer flagged **33 / 33** injected fault events within 24 h of onset; 2 of 10 fault-free cultures raised at least one false alarm. The Isolation Forest alone caught 11 / 33 with the same false-alarm count, which is why the explainable rules are the primary layer. These rules are tuned against this simulator, so they show the pipeline works, not that it will perform the same on a real culture.
- **Optimizer:** after 60 trials it recommends about 28.3 °C, pH 7.4, light scale 0.58 (best simulated 10-day biomass ≈ 2.2 g/L), consistent with the simulator's built-in optimum.

Not yet exercised in this repo: YOLO training and inference (needs `ultralytics`, ideally a GPU), the live Claude explainer (needs an API key), the Streamlit UI, and any real hardware.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python 3.11+
pip install -r requirements.txt && pip install -e .
cp .env.example .env                                  # add ANTHROPIC_API_KEY for the explainer

make seed          # 20 simulated cultures into SQLite
make images        # synthetic microscope images + YOLO labels
make train         # YOLO detector + forecaster
make api           # http://localhost:8000/docs
make ui            # Streamlit dashboard
make test
```

After training the detector, copy the weights to `data/models/yolo_best.pt` (Ultralytics prints the save directory).

Send a reading:

```bash
curl -X POST localhost:8000/readings -H 'content-type: application/json' \
  -d '[{"ts":"2026-02-01T00:00:00Z","culture_id":"pbr0","temp_c":28.1,"ph":7.6,"do_mgl":6.4,"par":0,"od750":1.02}]'
curl localhost:8000/cultures/pbr0/forecast
```

### Going to real hardware
An ESP32 or Raspberry Pi can post the same JSON once an hour. Calibrate each probe against a known standard before trusting a number, and retrain the forecaster and detector baselines on real cultures (keep the split-by-culture rule and re-check interval coverage).

## Roadmap
- Fit simulator constants to real growth curves
- Second strain with a different optimum temperature
- Cellpose for dense cell counting
- Let the explainer cite cultivation papers through a RAG service
