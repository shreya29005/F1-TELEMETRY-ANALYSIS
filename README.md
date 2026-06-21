# F1 Telemetry Analysis

A Streamlit app for micro-sector level analysis of Formula 1 driver telemetry, with driving-style clustering across corners.

**Live app:** https://f1-telemetryanalysis.streamlit.app

## What it does

- Lets you select any Grand Prix, session (Qualifying or Race), and a distance window on the track (a "micro-sector," e.g. a single corner or braking zone)
- Pulls each driver's fastest lap telemetry for that window and computes corner-level features: entry/apex/exit speed, speed loss and recovery, braking percentage, trail-braking index, throttle commitment, full-throttle percentage, and a composite corner aggression score
- Clusters drivers by driving style using KMeans, with an adjustable number of clusters
- Validates cluster quality using Calinski-Harabasz, Davies-Bouldin, and Silhouette scores
- Visualizes results with Plotly

## Coverage

2024 and 2025 seasons, Qualifying and Race sessions, all events.

Telemetry is pre-extracted and shipped with the app (`data/processed/`) rather than fetched live from FastF1 at runtime. F1's data backend blocks requests from cloud-hosting IP ranges, which makes live fetching unreliable on Streamlit Cloud — pre-processing locally and committing the slim extracted dataset sidesteps that entirely.

## Tech stack

- **[FastF1](https://github.com/theOehrly/Fast-F1)** — Python library for accessing official F1 timing and telemetry data (speed, throttle, brake, RPM, distance) at the lap level. Used during local data preparation; the deployed app reads pre-extracted output rather than calling it live.
- **pandas / numpy** — telemetry cleaning, distance-windowed filtering for micro-sectors, and all the derived feature calculations (speed loss, braking percentage, trail-braking index, etc.)
- **scikit-learn**
  - `KMeans` for clustering drivers by driving style based on corner-level features
  - `StandardScaler` / `SimpleImputer` for feature preprocessing before clustering
  - `PCA` for dimensionality reduction in the style-space visualization
  - `calinski_harabasz_score`, `davies_bouldin_score`, `silhouette_score` for validating cluster quality and choosing a sensible number of clusters
- **Plotly** (`plotly.express` and `plotly.graph_objects`) — interactive charts: speed traces over the micro-sector, cluster scatter plots, driver comparisons
- **PyArrow / Parquet** — storage format for the pre-extracted telemetry in `data/processed/`; compact and fast to read on app startup
- **Streamlit** — the web app itself: sidebar controls (GP/session/micro-sector selection, cluster count), landing page, and rendering the analysis results

## Project structure
.

├── app.py                      # Streamlit UI, landing page, app flow

├── data_fetch.py                # FastF1 session/telemetry fetching, cache handling

├── feature_engg.py              # Micro-sector feature extraction, clustering prep

├── export_processed_sessions.py # One-time batch export of telemetry to data/processed/

├── style.css                    # Landing page styling

├── requirements.txt

├── data/

│   └── processed/               # Pre-extracted per-session telemetry (committed)

└── cache/                       # Raw FastF1 cache (local only, gitignored)

## Running locally

```bash
git clone https://github.com/shreya29005/f1-telemetry-analysis.git
cd f1-telemetry-analysis
pip install -r requirements.txt
streamlit run app.py
```

## Extending coverage to new seasons or sessions

1. Run `export_processed_sessions.py` locally with the new year(s)/session types added to the `YEARS` / `SESSION_TYPES` lists.
2. This reads from FastF1 (using your local cache where available, fetching fresh otherwise) and writes compact per-session parquet files to `data/processed/`.
3. Commit `data/processed/` and push — Streamlit Cloud redeploys automatically.

## Deployment

Hosted on Streamlit Community Cloud, auto-deploying from the `main` branch on every push.
