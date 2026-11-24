"""Synthetic photobioreactor: logistic growth limited by light, temperature and pH,
with night respiration and three injectable, labeled fault types.

The constants are plausible, not calibrated to a real strain. Fit them to real
growth curves if you have them."""
from __future__ import annotations
import numpy as np, pandas as pd

def simulate(days=30, culture_id='pbr1', seed=0, events=True,
             temp_set=28.0, ph_set=7.6, par_scale=1.0, x0=0.15, K=4.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24
    ts = pd.date_range('2026-01-01', periods=n, freq='h')
    hour = ts.hour.to_numpy()

    par = np.clip(1500 * np.sin(np.pi * (hour - 6) / 12), 0, None) * par_scale
    par = par * rng.uniform(0.85, 1.0, size=n)                    # clouds
    temp = temp_set + 3 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.3, n)
    ph = ph_set + 0.4 * (par / 1500) + rng.normal(0, 0.03, n)

    ev = np.array(['none'] * n, dtype=object)
    mu_pen = np.ones(n)
    crash = np.ones(n)
    if events:
        for _ in range(int(rng.integers(0, 3))):
            kind = str(rng.choice(['contamination', 'co2_starvation', 'heat_spike']))
            start = int(rng.integers(24 * 5, n - 48))
            end = min(n, start + int(rng.integers(24, 72)))
            idx = np.arange(start, end)
            ev[idx] = kind
            if kind == 'contamination':
                crash[idx] = 0.985                                 # 1.5 percent die-off per hour
            elif kind == 'co2_starvation':
                ph[idx] += np.linspace(0.3, 1.4, len(idx))         # pH drifts up
                mu_pen[idx] = 0.4
            else:
                temp[idx] += 8

    X = np.empty(n); X[0] = x0                                      # biomass g/L (hidden truth)
    for i in range(1, n):
        I = par[i]
        f_I = (I / 600) * np.exp(1 - I / 600) if I > 0 else 0.0
        f_T = np.exp(-((temp[i] - 30) / 8) ** 2)
        f_pH = np.exp(-((ph[i] - 7.6) / 1.2) ** 2)
        mu = 0.05 * f_I * f_T * f_pH * mu_pen[i]                    # 1/h
        X[i] = max(0.01, (X[i-1] + (mu * (1 - X[i-1] / K) - 0.004) * X[i-1]) * crash[i])

    do = np.clip(6 + 3 * (par / 1500) + 0.8 * X + rng.normal(0, 0.15, n), 0, None)
    od = 0.85 * X + rng.normal(0, 0.01, n)
    return pd.DataFrame({'ts': ts, 'culture_id': culture_id, 'temp_c': temp, 'ph': ph,
                         'do_mgl': do, 'par': par, 'od750': od,
                         'biomass_gl': X, 'event': ev})
