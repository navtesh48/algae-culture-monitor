"""Anomaly detection in two layers.

1. Residual rules (cheap and explainable): compare each reading with what a *healthy* culture
   does at that hour of day / light level, learned from fault-free runs, and flag large
   deviations: a sustained OD drop, temperature far above the daily norm, or pH far above
   what the light level explains.
2. An Isolation Forest on rolling-window features, also trained only on fault-free runs, as a
   generic "something looks different" score.

An alert fires after `k` consecutive anomalous hours. Raising k or the thresholds lowers false
alarms but delays detection."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from algae.sim.simulator import simulate

COLS = ['temp_c', 'ph', 'do_mgl', 'od750']
OD_DROP_6H = 0.06       # relative OD fall over 6 h
TEMP_OVER = 5.0         # deg C above the healthy value for that hour of day
PH_OVER = 0.5           # pH units above what the light level explains

def feats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values('ts').reset_index(drop=True)
    out = pd.DataFrame(index=g.index)
    for c in COLS:
        out[f'{c}_m6'] = g[c].rolling(6).mean()
        out[f'{c}_s6'] = g[c].rolling(6).std()
    out['od_d6'] = g['od750'].diff(6)
    out['od_d12'] = g['od750'].diff(12)
    return out

def rule_flags(row) -> list[str]:
    """Absolute safety limits, independent of the learned baseline."""
    f = []
    if row.ph > 8.8 or row.ph < 6.5: f.append('ph_out_of_range')
    if row.temp_c > 36: f.append('temp_high')
    if row.do_mgl < 2: f.append('do_low')
    return f

class Detector:
    def fit(self, n_runs=12):
        runs = [simulate(30, f'n{k}', seed=1000 + k, events=False) for k in range(n_runs)]
        healthy = pd.concat(runs)
        # healthy baselines learned from data, not hard-coded
        self.temp_by_hour = healthy.groupby(healthy['ts'].dt.hour)['temp_c'].mean()
        self.ph_slope, self.ph_icpt = np.polyfit(healthy['par'], healthy['ph'], 1)
        X = pd.concat([feats(r) for r in runs]).dropna()
        self.scaler = StandardScaler().fit(X)
        self.model = IsolationForest(n_estimators=300, random_state=0).fit(self.scaler.transform(X))
        self.thr = float(np.percentile(-self.model.score_samples(self.scaler.transform(X)), 99.5))
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Isolation-forest anomaly score (higher = stranger)."""
        F = feats(df); ok = F.notna().all(axis=1)
        s = pd.Series(np.nan, index=F.index)
        s[ok] = -self.model.score_samples(self.scaler.transform(F[ok]))
        return s

    def residual_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.sort_values('ts').reset_index(drop=True)
        od_drop = g['od750'].pct_change(6) < -OD_DROP_6H
        temp_res = g['temp_c'] - g['ts'].dt.hour.map(self.temp_by_hour)
        ph_res = g['ph'] - (self.ph_icpt + self.ph_slope * g['par'])
        return pd.DataFrame({'od_drop': od_drop, 'temp_high': temp_res > TEMP_OVER, 'ph_high': ph_res > PH_OVER})

    def alert_hours(self, df: pd.DataFrame, k: int = 3) -> pd.Series:
        rule = self.residual_flags(df).any(axis=1)
        return rule.astype(int).rolling(k).sum() >= k      # k consecutive anomalous hours

    def model_alert_hours(self, df: pd.DataFrame, k: int = 3) -> pd.Series:
        above = (self.score(df) > self.thr).astype(int)
        return above.rolling(k).sum() >= k
