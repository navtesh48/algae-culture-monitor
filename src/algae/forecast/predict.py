import joblib, numpy as np, pandas as pd
from algae.forecast.features import make_features, FEATURES
_models = None

def forecast_latest(df_recent: pd.DataFrame) -> dict:   # needs >= 48 hourly rows, one culture
    global _models
    if _models is None:
        _models = joblib.load('data/models/forecast.joblib')
    f = make_features(df_recent).dropna(subset=FEATURES).iloc[[-1]]
    p10, p50, p90 = np.sort([float(_models[k].predict(f[FEATURES])[0]) for k in ('p10', 'p50', 'p90')])
    made_at = pd.to_datetime(f['ts'].iloc[0])
    return {'made_at': made_at.isoformat(), 'target_ts': (made_at + pd.Timedelta(hours=24)).isoformat(),
            'p10': p10, 'p50': p50, 'p90': p90}
