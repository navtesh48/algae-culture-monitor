from algae.forecast.train import run

def test_beats_persistence_and_intervals_sane():
    res = run(save=False)
    assert res['mae_lgbm'] < res['mae_persistence']
    assert 0.6 < res['coverage_80'] < 0.95
