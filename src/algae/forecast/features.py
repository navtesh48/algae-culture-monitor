import numpy as np, pandas as pd

FEATURES = ['od750', 'od_lag1', 'od_lag6', 'od_lag12', 'od_lag24', 'od_d24', 'dli24',
            'temp_c_m24', 'temp_c_s24', 'ph_m24', 'ph_s24', 'do_mgl_m24', 'do_mgl_s24',
            'hour_sin', 'hour_cos']

def make_features(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    g = df.sort_values('ts').reset_index(drop=True).copy()
    for l in (1, 6, 12, 24):
        g[f'od_lag{l}'] = g['od750'].shift(l)
    g['od_d24'] = g['od750'] - g['od750'].shift(24)
    g['dli24'] = g['par'].rolling(24).sum() * 3600 / 1e6        # mol photons m-2 per 24h
    for c in ('temp_c', 'ph', 'do_mgl'):
        g[f'{c}_m24'] = g[c].rolling(24).mean()
        g[f'{c}_s24'] = g[c].rolling(24).std()
    g['hour_sin'] = np.sin(2 * np.pi * g['ts'].dt.hour / 24)
    g['hour_cos'] = np.cos(2 * np.pi * g['ts'].dt.hour / 24)
    g['y'] = g['od750'].shift(-horizon)
    return g
