import numpy as np
from algae.anomaly.detector import Detector
from algae.sim.simulator import simulate

def test_detects_faults_and_is_quiet_on_normal_runs():
    d = Detector().fit()
    hit = total = 0
    for s in range(30):
        df = simulate(30, seed=s)
        flags = d.alert_hours(df).fillna(False).to_numpy()
        ev = (df.event != 'none').to_numpy()
        starts = np.where(ev & ~np.r_[False, ev[:-1]])[0]
        for st in starts:
            total += 1; hit += bool(flags[st: st + 24].any())
    assert total > 0 and hit / total >= 0.7
    quiet = sum(d.alert_hours(simulate(30, seed=500 + k, events=False)).fillna(False).any() for k in range(10))
    assert quiet <= 2
