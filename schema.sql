CREATE TABLE IF NOT EXISTS readings (
  ts TEXT NOT NULL,            -- ISO8601 UTC, hourly
  culture_id TEXT NOT NULL,
  temp_c REAL, ph REAL, do_mgl REAL, par REAL, od750 REAL,
  PRIMARY KEY (culture_id, ts)
);
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, culture_id TEXT NOT NULL, path TEXT NOT NULL,
  n_cells INTEGER, cells_per_ml REAL,
  contaminants_json TEXT       -- [{"cls":"rotifer","conf":0.82}, ...]
);
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, culture_id TEXT NOT NULL,
  kind TEXT NOT NULL,          -- anomaly | rule | contamination | forecast_miss
  severity TEXT NOT NULL,      -- info | warn | critical
  detail_json TEXT NOT NULL,
  explanation TEXT,            -- filled by the LLM explainer
  acknowledged INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS forecasts (
  made_at TEXT NOT NULL, culture_id TEXT NOT NULL, target_ts TEXT NOT NULL,
  p10 REAL, p50 REAL, p90 REAL,
  PRIMARY KEY (culture_id, made_at)
);
CREATE TABLE IF NOT EXISTS experiments (
  trial INTEGER PRIMARY KEY, params_json TEXT NOT NULL, result REAL, ts TEXT
);
